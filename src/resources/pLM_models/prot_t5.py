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

SUPPORTED_MODELS = ["prot_t5"]
MODEL_EXECUTION_MODES = {"prot_t5": "local"}
# What the Tools dropdowns show; the identifier above stays what settings and files hold.
MODEL_DISPLAY_NAMES = {"prot_t5": "ProtT5"}

# ProtT5-XL-U50's published preprocessing maps the rare/ambiguous B/Z/U/O codes
# to X. Its vocabulary still lists B/Z/U/O, but training never showed them, so
# they are mapped as published. J and alignment punctuation have no token at
# all (they would encode as <unk>), so they are represented as X too.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence using the published ProtT5 input contract."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def load_model(model_name, device):
    """
    Loads the ProtT5-XL-U50 encoder and tokenizer on the specified device.
    """
    import torch
    from transformers import T5Tokenizer, T5EncoderModel

    hf_mappings = {
        "prot_t5": "Rostlab/prot_t5_xl_uniref50"
    }

    hf_id = hf_mappings.get(model_name, model_name)
    print(f"Loading {model_name} ({hf_id}) ...")
    tokenizer = T5Tokenizer.from_pretrained(hf_id, do_lower_case=False)
    # Stated explicitly rather than inherited from the checkpoint config.
    # Transformers 5 defaults to dtype="auto", which adopts whatever type
    # the publisher uploaded, so the compute precision would otherwise be
    # theirs to change rather than ours to declare.
    # Rostlab/prot_t5_xl_uniref50 provides the unrounded float32 weights of the
    # whole encoder-decoder (an 11.3 GB download). T5EncoderModel keeps only
    # the encoder, about 1.2 B parameters or 4.8 GB in float32, and drops the
    # decoder. Rostlab/prot_t5_xl_half_uniref50-enc is that encoder rounded to
    # float16: it halves the memory but gives different embeddings, and an
    # embedding file records only the model name, so repointing hf_mappings
    # would silently mix checkpoints under prot_t5; add a separate model entry
    # instead.
    # The repository holds only pytorch_model.bin, so Transformers would also
    # fetch the converter bot's safetensors copy (as large again) in the
    # background; the weights come from the .bin either way.
    with safetensors_conversion_disabled():
        model = T5EncoderModel.from_pretrained(hf_id, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProtT5 encoder.
    """
    import torch
    from tools.tool_helpers.Embedding_Batching import extract_single_embedding

    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.inference_mode():
        # The vocabulary holds each residue as a word ("▁M"), so residues are
        # separated by spaces; an unspaced sequence would encode as <unk>.
        spaced_seq = " ".join(seq)
        inputs = tokenizer(spaced_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # ProtT5 adds no start token, only the trailing </s>: slicing :-1 keeps
        # one row per residue. Convert to the target precision.
        return extract_single_embedding(outputs.last_hidden_state[0, :-1], target_dtype)


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
        return extract_batch_embeddings(outputs.last_hidden_state, lengths, 0, target_dtype)
