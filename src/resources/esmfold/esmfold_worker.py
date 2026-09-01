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
import urllib.request

import torch


WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.dirname(WORKER_DIR)
SRC_DIR = os.path.dirname(RESOURCES_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from resources import Biohub_API


DEFAULT_ACTION_URL = "http://127.0.0.1:8000/api/action"


def sanitize_filename(name):
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def notify_server(node_id, pdb_filename, action_url=DEFAULT_ACTION_URL):
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
        with urllib.request.urlopen(request):
            pass
    except Exception as error:
        print(f"Warning: Could not notify main visualizer server: {error}")


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
        default=DEFAULT_ACTION_URL,
        help="Viewer instance endpoint that receives structure-folded notifications",
    )
    return parser.parse_args(argv)


def _terminal_token_prompt(replacement=False):
    action = "Replacement" if replacement else "Biohub"
    try:
        return getpass.getpass(
            f"{action} API token (input hidden; press Enter to cancel): "
        )
    except (EOFError, KeyboardInterrupt):
        return None


def _load_nodes(input_json_path):
    try:
        with open(input_json_path, "r", encoding="utf-8") as handle:
            nodes_to_fold = json.load(handle)
    finally:
        try:
            os.remove(input_json_path)
        except OSError:
            pass
    if not isinstance(nodes_to_fold, list):
        raise ValueError("The ESMFold input JSON must contain a list of node records.")
    return nodes_to_fold


def _load_local_model(device):
    import esm.pretrained
    from huggingface_hub import snapshot_download
    from pathlib import Path

    def custom_data_root(model_type: str):
        if model_type.startswith("esm3"):
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
        if model_type.startswith("esmc-300"):
            try:
                return Path(
                    snapshot_download(
                        repo_id="EvolutionaryScale/esmc-300m-2024-12",
                        local_files_only=True,
                    )
                )
            except Exception:
                return Path(
                    snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12")
                )
        if model_type.startswith("esmc-600"):
            try:
                return Path(
                    snapshot_download(
                        repo_id="EvolutionaryScale/esmc-600m-2024-12",
                        local_files_only=True,
                    )
                )
            except Exception:
                return Path(
                    snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12")
                )
        raise ValueError(f"{model_type=} is an invalid model name.")

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
    clean_identifier = sanitize_filename(rec_id)
    suffix = f"_{sanitize_filename(model_suffix)}" if model_suffix else ""
    pdb_filename = f"{clean_identifier}{suffix}.pdb"
    pdb_path = os.path.join(structures_dir, pdb_filename)

    if output_protein.plddt is not None:
        output_protein.plddt = output_protein.plddt * 100.0
    pdb_content = output_protein.to_pdb_string()
    with open(pdb_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(pdb_content)
    return pdb_filename, pdb_path


def run_predictions(
    nodes_to_fold,
    structures_dir,
    *,
    mode,
    target_device=None,
    token_prompt=None,
    notifier=notify_server,
):
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
        device = torch.device(
            target_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print(f"Using device: {device}")
        model = _load_local_model(device)

    folded_count = 0
    errors_occurred = False
    total = len(nodes_to_fold)
    try:
        for index, node_record in enumerate(nodes_to_fold, 1):
            try:
                rec_id, sequence = node_record
            except (TypeError, ValueError):
                print(f"Error: Invalid node record at position {index}: {node_record!r}")
                errors_occurred = True
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
                    model_suffix = auth_state["settings"]["ESM3_MODEL"]
                else:
                    output_protein = model.generate(protein, generation_config)
                    if isinstance(output_protein, ESMProteinError):
                        raise RuntimeError(_api_error_message(output_protein, "generation"))
                    model_suffix = None

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
    finally:
        if close_model and model is not None:
            try:
                model.close()
            except Exception as error:
                print(f"Warning: Could not close the Biohub API client cleanly: {error}")

    return folded_count, errors_occurred


def main(argv=None):
    arguments = parse_arguments(argv)
    os.makedirs(arguments.structures_dir, exist_ok=True)

    try:
        nodes_to_fold = _load_nodes(arguments.input_json_path)
    except Exception as error:
        print(f"Error reading input JSON {arguments.input_json_path}: {error}")
        input("\nError occurred. Press Enter to close this window...")
        return 1

    if not nodes_to_fold:
        print("No nodes to fold found in input JSON.")
        return 0

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="esm")
    try:
        notifier = lambda node_id, pdb_filename: notify_server(
            node_id,
            pdb_filename,
            arguments.action_url,
        )
        folded_count, errors_occurred = run_predictions(
            nodes_to_fold,
            arguments.structures_dir,
            mode=arguments.mode,
            target_device=arguments.device,
            notifier=notifier,
        )
    except Exception as error:
        print(f"Error initializing ESM3 prediction: {error}")
        input("\nError occurred. Press Enter to close this window...")
        return 1

    print(
        f"\nSuccessfully completed {folded_count} / {len(nodes_to_fold)} "
        "structure prediction(s)."
    )
    if errors_occurred or folded_count < len(nodes_to_fold):
        input("\nErrors occurred. Press Enter to close this window...")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
