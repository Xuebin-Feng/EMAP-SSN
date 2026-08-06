# Copyright 2026 Xuebin Feng
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

import os
import sys
import json
import re
import urllib.request
import torch

def sanitize_filename(name):
    # Replace any character that is not alphanumeric, a dash, dot, or underscore with '_'
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def notify_server(node_id, pdb_filename):
    url = "http://localhost:8000/api/action"
    payload = {
        "action": "structure_folded",
        "node_id": node_id,
        "pdb_filename": pdb_filename
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            pass
    except Exception as e:
        print(f"Warning: Could not notify main visualizer server: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: esmfold_worker.py <input_json_path> [structures_dir] [device]")
        sys.exit(1)

    input_json_path = sys.argv[1]
    structures_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("Cache_Files", "Structures")
    target_device = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(structures_dir, exist_ok=True)

    errors_occurred = False

    # 1. Read input nodes
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            nodes_to_fold = json.load(f)
        try:
            os.remove(input_json_path)
        except Exception:
            pass
    except Exception as e:
        print(f"Error reading input JSON {input_json_path}: {e}")
        input("\nError occurred. Press Enter to close this window...")
        sys.exit(1)

    if not nodes_to_fold:
        print("No nodes to fold found in input JSON.")
        sys.exit(0)

    # 2. Suppress library-level user warnings from esm library
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")

    # 3. Detect device
    device = torch.device(target_device)
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"Using device: {device}")

    # 4. Lazy Load ESM3 model with Registry Bypass
    try:
        import esm.pretrained
        from pathlib import Path
        from huggingface_hub import snapshot_download

        # Dynamic patch to redirect downloads/weights loading to biohub/esm3-sm-open-v1
        def custom_data_root(model_type: str):
            if model_type.startswith("esm3"):
                try:
                    # First try reading directly from local cache to prevent network checks and progress bar clutter
                    path = Path(snapshot_download(
                        repo_id="biohub/esm3-sm-open-v1",
                        local_files_only=True
                    ))
                    return path
                except Exception:
                    # Fallback to online download if not cached locally
                    print("Model weights not found in local cache. Downloading/resolving MIT-licensed ESM3 1.4B model weights (biohub/esm3-sm-open-v1)...")
                    path = Path(snapshot_download(
                        repo_id="biohub/esm3-sm-open-v1"
                    ))
                    return path
            elif model_type.startswith("esmc-300"):
                try:
                    return Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12", local_files_only=True))
                except Exception:
                    return Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
            elif model_type.startswith("esmc-600"):
                try:
                    return Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12", local_files_only=True))
                except Exception:
                    return Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
            else:
                raise ValueError(f"{model_type=} is an invalid model name.")

        esm.pretrained.data_root = custom_data_root

        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, GenerationConfig

        print("Loading local MIT-licensed ESM3 1.4B model (biohub/esm3-sm-open-v1)...")
        model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
    except Exception as e:
        print(f"Error loading ESM3 model: {e}")
        input("\nError occurred. Press Enter to close this window...")
        sys.exit(1)

    # 5. Perform Predictions
    folded_count = 0
    total = len(nodes_to_fold)
    for idx, (rec_id, sequence) in enumerate(nodes_to_fold, 1):
        print(f"\n[{idx}/{total}] Folding sequence: {rec_id} ({len(sequence)} aa)...")
        try:
            # Construct ESMProtein object
            protein = ESMProtein(sequence=sequence)
            
            # Configure structure generation track
            generation_config = GenerationConfig(track="structure", num_steps=8)
            
            # Generate predicted coordinates
            output_protein = model.generate(protein, generation_config)
            
            # Save structure file to structures directory
            clean_filename = sanitize_filename(rec_id)
            pdb_filename = f"{clean_filename}.pdb"
            pdb_path = os.path.join(structures_dir, pdb_filename)
            
            # Scale pLDDT confidence scores to 0-100 and write via to_pdb_string to preserve B-factors
            if output_protein.plddt is not None:
                output_protein.plddt = output_protein.plddt * 100.0
                
            pdb_content = output_protein.to_pdb_string()
            with open(pdb_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(pdb_content)
                
            print(f"Saved predicted structure to: {pdb_path}")
            
            # Notify main visualizer process
            notify_server(rec_id, pdb_filename)
            folded_count += 1
            
        except Exception as e:
            print(f"Error folding sequence {rec_id}: {e}")
            errors_occurred = True

    print(f"\nSuccessfully completed {folded_count} / {total} structure prediction(s).")
    if errors_occurred or folded_count < total:
        input("\nErrors occurred. Press Enter to close this window...")

if __name__ == "__main__":
    main()
