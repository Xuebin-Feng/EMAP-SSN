import re

SUPPORTED_MODELS = ["prot_bert"]
MODEL_EXECUTION_MODES = {"prot_bert": "local"}

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
    from transformers import BertTokenizer, BertModel
    print(f"Loading {model_name} on {device}...")
    tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
    model = BertModel.from_pretrained(f"Rostlab/{model_name}").to(device)
    model.eval()
    return tokenizer, model

def get_embedding(seq, model_obj, device, target_dtype):
    """
    Generates embedding for a sequence using the loaded ProtBERT model.
    """
    import torch
    tokenizer, model = model_obj

    seq = _clean_sequence(seq)
    if not seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    with torch.no_grad():
        spaced_seq = " ".join(list(seq))
        inputs = tokenizer(spaced_seq, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Slice out start/end special tokens ([CLS]/[SEP]) and convert to target precision
        return outputs.last_hidden_state[0, 1:-1].cpu().numpy().astype(target_dtype)
