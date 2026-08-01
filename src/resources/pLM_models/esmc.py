import re

SUPPORTED_MODELS = ["esmc_300m", "esmc_600m"]

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
    """
    Loads the ESMC model on the specified device.
    """
    from esm.models.esmc import ESMC
    print(f"Loading {model_name} on {device}...")
    client = ESMC.from_pretrained(model_name).to(device)
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
        # Slice out the start/end special tokens and convert to target precision
        return logits.embeddings.squeeze(0)[1:-1].cpu().numpy().astype(target_dtype)
