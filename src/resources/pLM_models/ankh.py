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

SUPPORTED_MODELS = [
    "ankh_base",
    "ankh_large"
]
MODEL_EXECUTION_MODES = {
    "ankh_base": "local",
    "ankh_large": "local",
}

# The Ankh tokenizer recognizes X/B/U/Z/O directly. It does not have native
# J, ".", or "-" residue tokens, so those are represented as X.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence for the native Ankh tokenizer vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def load_model(model_name, device):
    """
    Loads the Ankh model (encoder-only) and tokenizer on the specified device.
    """
    from transformers import AutoTokenizer, T5EncoderModel
    
    hf_mappings = {
        "ankh_base": "ElnaggarLab/ankh-base",
        "ankh_large": "ElnaggarLab/ankh-large"
    }
    
    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = T5EncoderModel.from_pretrained(hf_id).to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded Ankh model.
    """
    import torch
    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        # Ankh tokenizer takes unspaced sequences and appends only </s> at the end.
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slicing :-1 drops the trailing </s> and retains the exact residue representations.
        return outputs.last_hidden_state[0, :-1].cpu().numpy().astype(target_dtype)
