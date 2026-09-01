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

import re

SUPPORTED_MODELS = [
    "esm2_t6_8m",
    "esm2_t12_35m",
    "esm2_t30_150m",
    "esm2_t33_650m",
    "esm2_t36_3b",
    "esm2_t48_15b"
]
MODEL_EXECUTION_MODES = {
    "esm2_t6_8m": "local",
    "esm2_t12_35m": "local",
    "esm2_t30_150m": "local",
    "esm2_t33_650m": "local",
    "esm2_t36_3b": "local",
    "esm2_t48_15b": "local",
}

# ESM-2 uses the same sequence alphabet as ESM-C: X/B/U/Z/O, ".", and
# "-" are native tokens, while J must be represented as X.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence for the native ESM-2 tokenizer vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def load_model(model_name, device):
    """
    Loads the ESM-2 model and tokenizer from Hugging Face on the specified device.
    """
    from transformers import AutoTokenizer, AutoModel
    
    hf_mappings = {
        "esm2_t6_8m": "facebook/esm2_t6_8M_UR50D",
        "esm2_t12_35m": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150m": "facebook/esm2_t30_150M_UR50D",
        "esm2_t33_650m": "facebook/esm2_t33_650M_UR50D",
        "esm2_t36_3b": "facebook/esm2_t36_3B_UR50D",
        "esm2_t48_15b": "facebook/esm2_t48_15B_UR50D"
    }
    
    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ESM-2 model.
    """
    import torch
    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        # ESM-2 takes continuous unspaced sequences
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # ESM-2 tokenizer automatically prepends <cls> and appends <eos>, so we slice 1:-1
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
