import re

SUPPORTED_MODELS = ["ProstT5"]
MODEL_EXECUTION_MODES = {"ProstT5": "local"}

# ProstT5's published amino-acid preprocessing maps B/Z/U/O to X. J and
# alignment punctuation are also outside the model's documented AA input.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)


def _clean_sequence(seq):
    """Normalize one sequence using the published ProstT5 AA contract."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )

def load_model(model_name, device):
    """
    Loads the ProstT5 model and tokenizer on the specified device.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    print(f"Loading {model_name} on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}_fp16", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}_fp16").to(device)
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProstT5 model.
    """
    import torch
    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        spaced_seq = " ".join(list(seq))
        input_seq = "<AA2fold> " + spaced_seq
        inputs = tokenizer(input_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slice out start/end special tokens and convert to target precision
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
