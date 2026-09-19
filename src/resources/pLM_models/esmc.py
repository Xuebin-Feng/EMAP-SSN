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


def load_model(model_name, device):
    """Load the ESMC encoder and tokenizer on the specified device, in float32.

    This loads the bare encoder rather than the deprecated ``ESMC`` wrapper.
    That wrapper's ``from_pretrained`` forces ``dtype=torch.bfloat16`` on every
    non-CPU device, and its ``logits()`` additionally runs the forward pass
    under bfloat16 autocast on CUDA, so the same sequence came back at a
    different precision on CPU than on GPU. It also carries the masked-LM head
    and stacks every hidden state, neither of which this pipeline reads.

    ``dtype`` is stated explicitly rather than inherited from the checkpoint
    config, so the compute precision is a property of this plugin and cannot
    change under us when a publisher re-uploads weights in a narrower type.
    """
    import torch
    from esm.models.esmc import EsmcModel, EsmcTokenizer

    hf_mappings = {
        "esmc_300m": "biohub/ESMC-300M",
        "esmc_600m": "biohub/ESMC-600M",
    }

    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) ...")
    # The tokenizer is built from the packaged ESMC vocabulary, not downloaded.
    tokenizer = EsmcTokenizer()
    model = EsmcModel.from_pretrained(hf_id, device=device, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ESMC model.
    """
    import torch
    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        # ESMC takes continuous unspaced sequences
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # The ESMC tokenizer wraps every sequence in <cls> ... <eos>, so we
        # slice 1:-1. The encoder runs in float32, so no upcast is needed
        # before NumPy; only the selected storage dtype is applied.
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
