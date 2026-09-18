# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
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

from tools.tool_helpers.Model_Plugins import require_model_license_acceptance

SUPPORTED_MODELS = [
    "ankh_base",
    "ankh_large"
]
MODEL_EXECUTION_MODES = {
    "ankh_base": "local",
    "ankh_large": "local",
}
MODEL_USAGE_TERMS = {
    "ankh_base": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh-base",
        "license_id": "CC-BY-NC-SA-4.0",
        "license_url": (
            "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en"
        ),
        "restriction": (
            "The publisher permits non-commercial use and requires attribution "
            "and ShareAlike distribution under CC BY-NC-SA 4.0."
        ),
        "requires_acknowledgement": True,
    },
    "ankh_large": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh-large",
        "license_id": "CC-BY-NC-SA-4.0",
        "license_url": (
            "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en"
        ),
        "restriction": (
            "The publisher permits non-commercial use and requires attribution "
            "and ShareAlike distribution under CC BY-NC-SA 4.0."
        ),
        "requires_acknowledgement": True,
    },
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
    require_model_license_acceptance(
        model_name,
        MODEL_USAGE_TERMS.get(model_name),
    )
    from transformers import AutoTokenizer, T5EncoderModel
    
    hf_mappings = {
        "ankh_base": "ElnaggarLab/ankh-base",
        "ankh_large": "ElnaggarLab/ankh-large"
    }
    
    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) ...")
    tokenizer = _residue_aligned_tokenizer(AutoTokenizer.from_pretrained(hf_id))
    model = T5EncoderModel.from_pretrained(hf_id).to(device)
    model.eval()
    return tokenizer, model


# One bare residue per token, plus the single trailing </s> that get_embedding
# slices off. Anything else means the tokenizer is not residue-aligned.
_ALIGNMENT_PROBE = "MKTA"


def _tokenized_residue_count(tokenizer) -> int:
    """Return how many rows get_embedding would keep for the probe sequence."""
    return len(tokenizer(_ALIGNMENT_PROBE)["input_ids"]) - 1


def _residue_aligned_tokenizer(tokenizer):
    """Ensure Ankh tokenizes to exactly one token per residue.

    Ankh's vocabulary holds bare amino acids and has no sentencepiece word-start
    marker. Transformers 5 rebuilt T5Tokenizer on the Rust ``tokenizers``
    backend and gives it a Metaspace pre-tokenizer with
    ``prepend_scheme="always"``, so an unspaced sequence gains a leading "▁"
    that is absent from the vocabulary and encodes as <unk>. That produced one
    row more than the sequence has residues. Transformers 4 did not do this.

    The prefix is dropped only when the probe actually shows the extra token, so
    a future tokenizer that is already correct is left untouched. Alignment is
    re-checked afterwards and a failure is raised here, at load time, rather
    than surfacing later as a row-count error during embedding generation.
    """
    if _tokenized_residue_count(tokenizer) == len(_ALIGNMENT_PROBE):
        return tokenizer

    try:
        from tokenizers.pre_tokenizers import Metaspace, Sequence, WhitespaceSplit

        tokenizer._tokenizer.pre_tokenizer = Sequence([
            WhitespaceSplit(),
            Metaspace(replacement="▁", prepend_scheme="never", split=True),
        ])
    except Exception as error:
        raise RuntimeError(
            "Ankh tokenizer emits more tokens than residues and the "
            f"word-start prefix could not be removed: {error}"
        ) from error

    actual = _tokenized_residue_count(tokenizer)
    if actual != len(_ALIGNMENT_PROBE):
        raise RuntimeError(
            "Ankh tokenizer is not residue-aligned: probe "
            f"{_ALIGNMENT_PROBE!r} produced {actual} residue tokens, expected "
            f"{len(_ALIGNMENT_PROBE)}."
        )
    return tokenizer

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
