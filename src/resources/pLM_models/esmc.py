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

import re

SUPPORTED_MODELS = ["esmc_300m", "esmc_600m"]
MODEL_EXECUTION_MODES = {
    "esmc_300m": "local",
    "esmc_600m": "local",
}

# ESM's sequence tokenizer has distinct tokens for the standard amino acids,
# the ambiguity codes X/B/U/Z/O, and the alignment symbols "." and "-".
# J is not in the vocabulary and is represented as X instead.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence for the native ESM sequence vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def _patch_esm_pretrained_registry():
    """
    Patches the local model registry in esm==3.2.1 for ESMC models.
    In upstream esm, ESMC model builders pass the parent repository snapshot directory
    to huggingface_hub.load_torch_model rather than the checkpoint weight file path
    under data/weights/*.pth, and fail to pass assign=True to materialize meta tensors.
    """
    import torch
    from accelerate import init_empty_weights
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from esm.utils.constants.esm3 import data_root
    from esm.pretrained import register_local_model

    configs = {
        "esmc_300m": (960, 15, 30, "esmc-300", "esmc_300m_2024_12_v0.pth"),
        "esmc_600m": (1152, 18, 36, "esmc-600", "esmc_600m_2024_12_v0.pth"),
        "esmc_6b": (2560, 40, 80, "esmc-6b", "esmc_6b_2024_12_v0.pth"),
    }

    for name, (d_model, n_heads, n_layers, repo_name, weight_file) in configs.items():
        def _make_builder(_d=d_model, _h=n_heads, _l=n_layers, _r=repo_name, _w=weight_file):
            def _builder(device="cpu", use_flash_attn=True):
                with init_empty_weights():
                    model = ESMC(
                        d_model=_d,
                        n_heads=_h,
                        n_layers=_l,
                        tokenizer=get_esmc_model_tokenizers(),
                        use_flash_attn=use_flash_attn,
                    ).eval()
                weight_path = data_root(_r) / "data" / "weights" / _w
                state_dict = torch.load(weight_path, map_location="cpu")
                model.load_state_dict(state_dict, assign=True)
                return model.to(device)
            return _builder

        register_local_model(name, _make_builder())


def load_model(model_name, device):
    """
    Loads the ESMC model on the specified device.
    """
    from esm.models.esmc import ESMC
    _patch_esm_pretrained_registry()
    print(f"Loading {model_name} ...")
    client = ESMC.from_pretrained(model_name, device=device)
    return client

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ESMC model.
    """
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        protein_tensor = model_obj.encode(ESMProtein(sequence=seq))
        logits = model_obj.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        # Slice out the start/end special tokens and convert to target precision.
        # ESMC returns bfloat16 tensors, which NumPy cannot represent, so upcast
        # to float32 before leaving PyTorch and then apply the storage dtype.
        embeddings = logits.embeddings.squeeze(0)[1:-1]
        return embeddings.to(torch.float32).cpu().numpy().astype(target_dtype)
