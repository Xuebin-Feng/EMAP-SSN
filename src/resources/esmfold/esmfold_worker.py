# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import getpass
import json
import os
import re
import sys
import time
import urllib.request

# torch is imported where local folding starts (run_predictions), not here: a
# missing or broken torch can crash the import, and main() must have read and
# deleted the private --delete-input file before anything that can.


WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.dirname(WORKER_DIR)
SRC_DIR = os.path.dirname(RESOURCES_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from resources import Biohub_API


# A Viewer that is busy or gone must not stall folding.
NOTIFY_TIMEOUT_SECONDS = 5.0


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def prediction_filename(rec_id, model_suffix=None):
    """Return the PDB filename a record is written to."""
    suffix = f"_{sanitize_filename(model_suffix)}" if model_suffix else ""
    return f"{sanitize_filename(rec_id)}{suffix}.pdb"


def notify_server(node_id, pdb_filename, action_url=None):
    """Tell the Viewer at ``action_url`` about a new structure; no URL, no request."""
    if not action_url:
        return
    payload = {
        "action": "structure_folded",
        "node_id": node_id,
        "pdb_filename": pdb_filename,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        action_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=NOTIFY_TIMEOUT_SECONDS):
            pass
    except Exception as error:
        print(f"Warning: Could not notify main visualizer server: {error}")


def _replace_file(source, target, attempts=10):
    """Move ``source`` over ``target``, waiting for readers to let go of it.

    Windows refuses to replace a file while another process has it open
    without delete sharing, which Python's ``open`` never grants. The readers
    here (the Viewer's web server serving a PDB, its WorkerTracker or an
    external monitor polling the status file) release it quickly.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2)


_portal_path = None
_portal_noninteractive = False
_portal_state = {}

# The Viewer's WorkerTracker reports the job's result only once it reads one of
# these, so they wait longer for readers to close the status file (about 10 s)
# than a progress update (about 2 s), which the next update repeats anyway.
PORTAL_FINAL_STATUSES = frozenset({'succeeded', 'failed', 'cancelled'})
PORTAL_PROGRESS_ATTEMPTS = 10
PORTAL_FINAL_ATTEMPTS = 50


def portal_event(status, message=None, **result):
    if not _portal_path:
        return
    import psutil
    if _portal_state.get('status') == 'cancelled':
        return
    _portal_state.update(pid=os.getpid(), created=psutil.Process().create_time(), status=status)
    if message is not None:
        _portal_state['message'] = message
    if result:
        _portal_state.setdefault('result', {}).update(result)
    temporary = _portal_path + '.partial'
    attempts = PORTAL_FINAL_ATTEMPTS if status in PORTAL_FINAL_STATUSES else PORTAL_PROGRESS_ATTEMPTS
    try:
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(_portal_state, handle)
        _replace_file(temporary, _portal_path, attempts)
    except OSError as error:
        # The status file only reports on the run: failing to write it must not
        # fail a structure that is already saved. _portal_state is cumulative,
        # so the next event's write also carries this one.
        print(f"Warning: Could not update the worker status file ({status}): {error}")
        try:
            os.remove(temporary)
        except OSError:
            pass


def portal_pause(message):
    if not _portal_noninteractive:
        input(message)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run local or Biohub-backed ESM3 structure prediction."
    )
    parser.add_argument("input_json_path")
    parser.add_argument(
        "structures_dir",
        nargs="?",
        default=os.path.join("Cache_Files", "Structures"),
    )
    parser.add_argument("device", nargs="?", default=None)
    parser.add_argument(
        "--mode",
        choices=("local", "large"),
        default="local",
        help="local ESM3 1.4B or remote Biohub ESM3",
    )
    parser.add_argument(
        "--action-url",
        default=None,
        help="Viewer instance endpoint that receives structure-folded notifications; "
        "when omitted, no Viewer is notified",
    )
    parser.add_argument("--portal-status", default=None)
    parser.add_argument("--noninteractive", action="store_true")
    parser.add_argument(
        "--delete-input",
        action="store_true",
        help="delete input_json_path after reading it (for a private temporary file)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep an existing PDB for a record instead of predicting it again",
    )
    return parser.parse_args(argv)


def _terminal_token_prompt(replacement=False):
    if _portal_noninteractive:
        raise RuntimeError("Biohub credentials require terminal input; configure credentials before headless execution.")
    portal_event("awaiting_user_input", "Enter Biohub credentials in the worker terminal")
    action = "Replacement" if replacement else "Biohub"
    try:
        token = getpass.getpass(
            f"{action} API token (input hidden; press Enter to cancel): "
        )
        portal_event("running", "Credential prompt completed")
        if not token:
            portal_event("cancelled", "Credential entry cancelled")
        return token
    except (EOFError, KeyboardInterrupt):
        portal_event("cancelled", "Credential entry cancelled")
        return None


def _load_nodes(input_json_path, delete=False):
    try:
        with open(input_json_path, "r", encoding="utf-8") as handle:
            nodes_to_fold = json.load(handle)
    finally:
        # Only a caller that hands over a private file (the Viewer's esmfold
        # command passes --delete-input) gives up ownership of it.
        if delete:
            try:
                os.remove(input_json_path)
            except OSError:
                pass
    if not isinstance(nodes_to_fold, list):
        raise ValueError("The ESMFold input JSON must contain a list of node records.")
    return nodes_to_fold


def _load_local_model(device):
    import esm.pretrained
    from esm.utils.constants.esm3 import data_root as package_data_root
    from huggingface_hub import snapshot_download
    from pathlib import Path

    def custom_data_root(model_type: str):
        # Only ESM3 is loaded here. Its weights are looked up in the local cache
        # first, so a machine that already has them never needs the network;
        # any other model keeps esm's own repository mapping.
        if not model_type.startswith("esm3"):
            return package_data_root(model_type)
        try:
            return Path(
                snapshot_download(
                    repo_id="biohub/esm3-sm-open-v1",
                    local_files_only=True,
                )
            )
        except Exception:
            print(
                "Model weights not found in local cache. Downloading/resolving "
                "MIT-licensed ESM3 1.4B model weights "
                "(biohub/esm3-sm-open-v1)..."
            )
            return Path(snapshot_download(repo_id="biohub/esm3-sm-open-v1"))

    esm.pretrained.data_root = custom_data_root
    from esm.models.esm3 import ESM3

    print("Loading local MIT-licensed ESM3 1.4B model (biohub/esm3-sm-open-v1)...")
    return ESM3.from_pretrained("esm3_sm_open_v1").to(device)


def _load_large_client(prompt_callback=None):
    from esm.sdk import client

    callback = prompt_callback or (lambda: _terminal_token_prompt(False))
    settings = Biohub_API.load_api_settings(prompt_callback=callback)
    print(
        f"Connecting to remote Biohub ESM3 model {settings['ESM3_MODEL']} at "
        f"{settings['ESM_API_URL']}..."
    )
    model = client(
        model=settings["ESM3_MODEL"],
        url=settings["ESM_API_URL"],
        token=settings["ESM_API_TOKEN"],
    )
    return model, settings


def _api_error_message(result, operation):
    code = getattr(result, "error_code", "unknown")
    detail = getattr(result, "error_msg", str(result))
    return f"Biohub {operation} error (code {code}): {detail}"


def _generate_remote_structure(model, protein, generation_config, auth_state):
    from esm.sdk.api import ESMProteinError

    result = model.generate(protein, generation_config)
    status = Biohub_API.authentication_status(result)
    if status is not None and not auth_state["refresh_attempted"]:
        auth_state["refresh_attempted"] = True
        print("Biohub rejected the saved API token. Enter a replacement to retry once.")
        refreshed = Biohub_API.refresh_api_token(
            auth_state["settings"],
            lambda: _terminal_token_prompt(True),
        )
        auth_state["settings"] = refreshed
        Biohub_API.update_client_token(model, refreshed["ESM_API_TOKEN"])
        result = model.generate(protein, generation_config)

    if isinstance(result, ESMProteinError):
        status = Biohub_API.authentication_status(result)
        if status is not None:
            raise Biohub_API.BiohubAuthenticationError(
                status,
                "Biohub rejected the API token after one replacement attempt.",
            )
        raise RuntimeError(_api_error_message(result, "generation"))
    return result


def _write_prediction(output_protein, rec_id, structures_dir, model_suffix=None):
    pdb_filename = prediction_filename(rec_id, model_suffix)
    pdb_path = os.path.join(structures_dir, pdb_filename)

    # ESMProtein.plddt is on a 0-1 scale and to_pdb_string() already applies
    # esm's PLDDT_B_FACTOR_SCALE (100.0) when it builds the atom array. Do not
    # pre-scale here: a second factor of 100 overflows the 6-column PDB
    # temperature-factor field and biotite rejects the structure.
    pdb_content = output_protein.to_pdb_string()
    # Publish whole files only, so an interrupted run never leaves a truncated
    # PDB that a later --skip-existing run would keep.
    partial_path = f"{pdb_path}.{os.getpid()}.partial"
    try:
        with open(partial_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(pdb_content)
        _replace_file(partial_path, pdb_path)
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return pdb_filename, pdb_path


def run_predictions(
    nodes_to_fold,
    structures_dir,
    *,
    mode,
    target_device=None,
    token_prompt=None,
    notifier=notify_server,
    skip_existing=False,
):
    """Fold each [rec_id, sequence]; return (completed, errors_occurred).

    Records kept because their PDB already exists (``skip_existing``) count
    as completed but are not passed to ``notifier``.
    """
    from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig

    os.makedirs(structures_dir, exist_ok=True)
    auth_state = None
    model = None
    close_model = False

    if mode == "large":
        print("Using remote Biohub inference; local CPU/GPU selection is not applicable.")
        model, remote_settings = _load_large_client(token_prompt)
        auth_state = {
            "settings": remote_settings,
            "refresh_attempted": False,
        }
        close_model = True
    else:
        import torch

        device = torch.device(
            target_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print(f"Using device: {device}")
        model = _load_local_model(device)

    folded_count = 0
    skipped_count = 0
    errors_occurred = False
    total = len(nodes_to_fold)
    claimed_names = {}
    try:
        for index, node_record in enumerate(nodes_to_fold, 1):
            try:
                rec_id, sequence = node_record
            except (TypeError, ValueError):
                print(f"Error: Invalid node record at position {index}: {node_record!r}")
                errors_occurred = True
                continue

            model_suffix = auth_state["settings"]["ESM3_MODEL"] if mode == "large" else None
            pdb_filename = prediction_filename(rec_id, model_suffix)
            pdb_path = os.path.join(structures_dir, pdb_filename)
            first_claim = claimed_names.setdefault(pdb_filename, rec_id)
            if first_claim != rec_id:
                print(f"Warning: {rec_id!r} and {first_claim!r} both map to {pdb_filename}; "
                      "only one structure can be kept under that name.")
            if skip_existing and os.path.exists(pdb_path):
                print(f"\n[{index}/{total}] Keeping existing structure for {rec_id}: {pdb_path}")
                skipped_count += 1
                continue

            print(f"\n[{index}/{total}] Folding sequence: {rec_id} ({len(sequence)} aa)...")
            try:
                protein = ESMProtein(sequence=sequence)
                generation_config = GenerationConfig(track="structure", num_steps=8)
                if mode == "large":
                    output_protein = _generate_remote_structure(
                        model,
                        protein,
                        generation_config,
                        auth_state,
                    )
                else:
                    output_protein = model.generate(protein, generation_config)
                    if isinstance(output_protein, ESMProteinError):
                        raise RuntimeError(_api_error_message(output_protein, "generation"))

                if os.path.exists(pdb_path):
                    print(f"Warning: replacing existing structure {pdb_path}")
                pdb_filename, pdb_path = _write_prediction(
                    output_protein,
                    rec_id,
                    structures_dir,
                    model_suffix=model_suffix,
                )
                print(f"Saved predicted structure to: {pdb_path}")
                notifier(rec_id, pdb_filename)
                folded_count += 1
            except Exception as error:
                print(f"Error folding sequence {rec_id}: {error}")
                errors_occurred = True
        if skipped_count:
            print(f"\nKept {skipped_count} existing structure(s) without predicting them again.")
    finally:
        if close_model and model is not None:
            try:
                model.close()
            except Exception as error:
                print(f"Warning: Could not close the Biohub API client cleanly: {error}")

    return folded_count + skipped_count, errors_occurred


def main(argv=None):
    global _portal_path, _portal_noninteractive, _portal_state
    arguments = parse_arguments(argv)
    _portal_path = arguments.portal_status
    _portal_noninteractive = arguments.noninteractive
    _portal_state = {"artifacts": []}
    if _portal_path:
        import sys
        class WorkerOutput:
            def __init__(self, stream, name):
                self.stream = stream
                self.path = _portal_path + '.' + name
            def write(self, text):
                with open(self.path, 'ab') as handle:
                    handle.write(text.encode('utf-8', errors='replace'))
                return self.stream.write(text)
            def flush(self): return self.stream.flush()
            def __getattr__(self, name): return getattr(self.stream, name)
        sys.stdout = WorkerOutput(sys.stdout, 'stdout')
        sys.stderr = WorkerOutput(sys.stderr, 'stderr')
    portal_event("running", "Worker started")
    os.makedirs(arguments.structures_dir, exist_ok=True)

    try:
        nodes_to_fold = _load_nodes(arguments.input_json_path, delete=arguments.delete_input)
    except Exception as error:
        print(f"Error reading input JSON {arguments.input_json_path}: {error}")
        portal_event("failed", str(error))
        portal_pause("\nError occurred. Press Enter to close this window...")
        return 1

    if not nodes_to_fold:
        print("No nodes to fold found in input JSON.")
        portal_event("succeeded", "No nodes to fold", succeeded=0, total=0)
        return 0

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="esm")
    try:
        def notifier(node_id, pdb_filename):
            notify_server(node_id, pdb_filename, arguments.action_url)
            _portal_state['artifacts'].append(os.path.abspath(os.path.join(arguments.structures_dir, pdb_filename)))
            portal_event('running', f'Folded {node_id}', succeeded=len(_portal_state['artifacts']), total=len(nodes_to_fold))
        folded_count, errors_occurred = run_predictions(
            nodes_to_fold,
            arguments.structures_dir,
            mode=arguments.mode,
            target_device=arguments.device,
            notifier=notifier,
            skip_existing=arguments.skip_existing,
        )
    except Exception as error:
        print(f"Error initializing ESM3 prediction: {error}")
        portal_event("failed", str(error))
        portal_pause("\nError occurred. Press Enter to close this window...")
        return 1

    print(
        f"\nSuccessfully completed {folded_count} / {len(nodes_to_fold)} "
        "structure prediction(s)."
    )
    if errors_occurred or folded_count < len(nodes_to_fold):
        portal_event("failed", "One or more structures failed", succeeded=folded_count, total=len(nodes_to_fold))
        portal_pause("\nErrors occurred. Press Enter to close this window...")
        return 1
    portal_event("succeeded", "Structure prediction completed", succeeded=folded_count, total=len(nodes_to_fold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
