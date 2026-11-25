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

from tools.tool_helpers.Model_Plugins import (
    require_model_license_acceptance,
    safetensors_conversion_disabled,
)

SUPPORTED_MODELS = [
    "ankh3_large",
    "ankh3_large_s2s",
    "ankh3_xl",
    "ankh3_xl_s2s",
]
MODEL_EXECUTION_MODES = {
    "ankh3_large": "local",
    "ankh3_large_s2s": "local",
    "ankh3_xl": "local",
    "ankh3_xl_s2s": "local",
}
# What the Tools dropdowns show; the identifiers above stay what settings and files hold.
MODEL_DISPLAY_NAMES = {
    "ankh3_large": "Ankh3 Large",
    "ankh3_large_s2s": "Ankh3 Large (S2S)",
    "ankh3_xl": "Ankh3 XL",
    "ankh3_xl_s2s": "Ankh3 XL (S2S)",
}
MODEL_USAGE_TERMS = {
    "ankh3_large": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh3-large",
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
    "ankh3_large_s2s": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh3-large",
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
    "ankh3_xl": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh3-xl",
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
    "ankh3_xl_s2s": {
        "source_url": "https://huggingface.co/ElnaggarLab/ankh3-xl",
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

_HF_MAPPINGS = {
    "ankh3_large": "ElnaggarLab/ankh3-large",
    "ankh3_large_s2s": "ElnaggarLab/ankh3-large",
    "ankh3_xl": "ElnaggarLab/ankh3-xl",
    "ankh3_xl_s2s": "ElnaggarLab/ankh3-xl",
}
_PREFIX_MAPPINGS = {
    "ankh3_large": "[NLU]",
    "ankh3_large_s2s": "[S2S]",
    "ankh3_xl": "[NLU]",
    "ankh3_xl_s2s": "[S2S]",
}

# The Ankh3 spiece vocabulary holds bare amino acids including X/B/U/Z/O,
# matching Ankh v1. J, ".", and "-" are absent and mapped to X.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence for the native Ankh3 tokenizer vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )


# Leading mode token ([NLU] or [S2S]), one bare residue per token, and one
# trailing </s> that get_embedding slices off. Anything else means the
# tokenizer is not residue-aligned.
_ALIGNMENT_PROBE = "MKTA"


def _probe_ids(tokenizer, prefix):
    """Return input_ids emitted for ``prefix + _ALIGNMENT_PROBE``."""
    return tokenizer(prefix + _ALIGNMENT_PROBE)["input_ids"]


def _is_residue_aligned(tokenizer, prefix) -> bool:
    """Return True when ``prefix + _ALIGNMENT_PROBE`` maps to [prefix, *residues, </s>]."""
    ids = _probe_ids(tokenizer, prefix)
    if len(ids) != len(_ALIGNMENT_PROBE) + 2:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and (ids[0] == unk_id or any(tok_id == unk_id for tok_id in ids[1:-1])):
        return False
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and ids[-1] != eos_id:
        return False
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        prefix_id = convert(prefix)
        if isinstance(prefix_id, int) and prefix_id != unk_id and ids[0] != prefix_id:
            return False
    return True


def _residue_aligned_tokenizer(tokenizer, prefix):
    """Ensure Ankh3 tokenizes to [prefix, *residues, </s>] with no leading <unk>.

    Ankh3's SentencePiece vocabulary holds bare amino acids plus the task
    prefixes ``[NLU]`` and ``[S2S]``, and has no word-start ``▁`` token
    (``add_prefix_space: false`` in ``tokenizer_config.json``). Transformers 5
    rebuilt ``T5Tokenizer`` on the Rust ``tokenizers`` backend and attaches
    ``Metaspace(prepend_scheme="always")``, which prepends a ``▁`` before the
    prefix that encodes as ``<unk>`` and shifts every residue position by one.

    The pre-tokenizer is updated only when the probe shows misalignment, so a
    tokenizer that is already aligned is left untouched. Alignment is
    re-checked immediately so any mismatch fails at load time rather than
    silently shifting residue rows during embedding generation.
    """
    if _is_residue_aligned(tokenizer, prefix):
        return tokenizer

    try:
        from tokenizers.pre_tokenizers import Metaspace, Sequence, WhitespaceSplit

        tokenizer._tokenizer.pre_tokenizer = Sequence([
            WhitespaceSplit(),
            Metaspace(replacement="▁", prepend_scheme="never", split=True),
        ])
    except Exception as error:
        raise RuntimeError(
            "Ankh3 tokenizer is not residue-aligned and the "
            f"word-start prefix could not be removed: {error}"
        ) from error

    if not _is_residue_aligned(tokenizer, prefix):
        ids = _probe_ids(tokenizer, prefix)
        raise RuntimeError(
            "Ankh3 tokenizer is not residue-aligned: probe "
            f"{(prefix + _ALIGNMENT_PROBE)!r} produced {len(ids)} tokens "
            f"({ids}), expected {len(_ALIGNMENT_PROBE) + 2} "
            f"(prefix {prefix!r} + {len(_ALIGNMENT_PROBE)} residues + </s>)."
        )
    return tokenizer


def load_model(model_name, device):
    """Load the Ankh3 encoder, T5Tokenizer, and task prefix on ``device``."""
    require_model_license_acceptance(
        model_name,
        MODEL_USAGE_TERMS.get(model_name),
    )
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    hf_id = _HF_MAPPINGS.get(model_name, model_name)
    prefix = _PREFIX_MAPPINGS.get(model_name, "[NLU]")
    print(f"Loading {model_name} ({hf_id}, prefix {prefix}) ...")
    # Per the Ankh3 model card, always load with T5Tokenizer rather than AutoTokenizer.
    tokenizer = _residue_aligned_tokenizer(T5Tokenizer.from_pretrained(hf_id), prefix)
    # Stated explicitly rather than inherited from the checkpoint config.
    # Both ElnaggarLab/ankh3-large and ElnaggarLab/ankh3-xl publish .bin
    # weights without safetensors on main, so disable Transformers' background
    # safetensors-conversion download thread while loading.
    with safetensors_conversion_disabled():
        model = T5EncoderModel.from_pretrained(hf_id, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return tokenizer, model, prefix


def _unpack_model(model_obj):
    if len(model_obj) == 3:
        return model_obj
    tokenizer, model = model_obj
    return tokenizer, model, "[NLU]"


def get_embedding(seq, model_obj, device, target_dtype):
    """Generate per-residue embeddings for one sequence using the loaded Ankh3 model."""
    import torch
    from tools.tool_helpers.Embedding_Batching import extract_single_embedding

    tokenizer, model, prefix = _unpack_model(model_obj)

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.inference_mode():
        # Ankh3 takes unspaced sequences prefixed with [NLU] or [S2S] and appends </s>.
        inputs = tokenizer(prefix + seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slicing 1:-1 drops the leading task prefix token and trailing </s>.
        return extract_single_embedding(outputs.last_hidden_state[0, 1:-1], target_dtype)


def get_embeddings_batch(sequences, model_obj, device, target_dtype):
    """Generate strict FP32 per-residue embeddings for a batch of sequences."""
    import torch
    from tools.tool_helpers.Embedding_Batching import extract_batch_embeddings

    if not sequences:
        return []
    tokenizer, model, prefix = _unpack_model(model_obj)

    prefixed = []
    lengths = []
    for seq in sequences:
        clean = _clean_sequence(seq)
        if not clean:
            raise ValueError("Sequence contains no supported amino-acid characters.")
        prefixed.append(prefix + clean)
        lengths.append(len(clean))

    non_blocking = getattr(device, "type", "") == "cuda"
    with torch.inference_mode():
        inputs = tokenizer(prefixed, return_tensors="pt", padding=True)
        inputs = {k: v.to(device, non_blocking=non_blocking) for k, v in inputs.items()}
        outputs = model(**inputs)
        return extract_batch_embeddings(outputs.last_hidden_state, lengths, 1, target_dtype)
