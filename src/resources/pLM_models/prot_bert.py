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

from tools.tool_helpers.Model_Plugins import safetensors_conversion_disabled

SUPPORTED_MODELS = ["prot_bert"]
MODEL_EXECUTION_MODES = {"prot_bert": "local"}
# What the Tools dropdowns show; the identifier above stays what settings and files hold.
MODEL_DISPLAY_NAMES = {"prot_bert": "ProtBERT"}

# ProtBERT's published preprocessing maps the rare/ambiguous B/Z/U/O codes to
# X. J and alignment punctuation are also outside that input contract.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence using the published ProtBERT input contract."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def load_model(model_name, device):
    """
    Loads the ProtBERT model and tokenizer on the specified device.
    """
    import torch
    from transformers import BertTokenizer, BertModel
    print(f"Loading {model_name} ...")
    tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
    # Stated explicitly rather than inherited from the checkpoint config.
    # Transformers 5 defaults to dtype="auto", which adopts whatever type
    # the publisher uploaded, so the compute precision would otherwise be
    # theirs to change rather than ours to declare.
    # The repository holds only pytorch_model.bin, so Transformers would also
    # fetch the converter bot's safetensors copy (as large again) in the
    # background; the weights come from the .bin either way.
    with safetensors_conversion_disabled():
        model = BertModel.from_pretrained(
            f"Rostlab/{model_name}", dtype=torch.float32
        )
    model = model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProtBERT model.
    """
    import torch
    from tools.tool_helpers.Embedding_Batching import extract_single_embedding

    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.inference_mode():
        spaced_seq = " ".join(seq)
        inputs = tokenizer(spaced_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slice out start/end special tokens ([CLS]/[SEP]) and convert to target precision
        return extract_single_embedding(outputs.last_hidden_state[0, 1:-1], target_dtype)


def get_embeddings_batch(sequences, model_obj, device, target_dtype):
    """Generate strict FP32 embeddings for a batch of sequences."""
    import torch
    from tools.tool_helpers.Embedding_Batching import extract_batch_embeddings

    if not sequences:
        return []
    tokenizer, model = model_obj

    spaced = []
    lengths = []
    for seq in sequences:
        clean = _clean_sequence(seq)
        if not clean:
            raise ValueError("Sequence contains no supported amino-acid characters.")
        spaced.append(" ".join(clean))
        lengths.append(len(clean))

    non_blocking = getattr(device, "type", "") == "cuda"
    with torch.inference_mode():
        inputs = tokenizer(spaced, return_tensors="pt", padding=True)
        inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in inputs.items()}
        outputs = model(**inputs)
        return extract_batch_embeddings(outputs.last_hidden_state, lengths, 1, target_dtype)
