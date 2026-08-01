import json
import os
import re
import time
import warnings

import numpy as np


SUPPORTED_MODELS = ["esmc_6b"]

# ESM's sequence tokenizer has distinct tokens for the standard amino acids,
# the ambiguity codes X/B/U/Z/O, and the alignment symbols "." and "-".
# J is not in the vocabulary and is represented as X instead.
SUPPORTED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO.-")
_RESIDUE_BOUNDARY_PATTERN = re.compile(
    r"[ACDEFGHIKLMNPQRSTVWYBZJXUO].*[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
    r"|[ACDEFGHIKLMNPQRSTVWYBZJXUO]"
)

API_MODEL_MAPPINGS = {
    "esmc_6b": "esmc-6b-2024-12",
}

DEFAULT_API_URL = "https://biohub.ai"
MAX_API_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

API_KEYS_FILE = os.path.join(
    os.path.dirname(__file__),
    "esmc_6b_api_key.json",
)


def _load_api_settings():
    """Load the ESM API token without keeping credentials in tracked source code."""
    settings = {}
    if os.path.exists(API_KEYS_FILE):
        try:
            with open(API_KEYS_FILE, "r", encoding="utf-8") as handle:
                settings = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read valid JSON from the ESM API key file: {API_KEYS_FILE}"
            ) from exc

        if not isinstance(settings, dict):
            raise ValueError(
                f"ESM API key file must contain a JSON object: {API_KEYS_FILE}"
            )

    token = settings.get("ESM_API_TOKEN")
    if not token:
        token = os.environ.get("ESM_API_KEY")

    if not token or not str(token).strip():
        raise ValueError(
            "ESMC 6B requires an API token. Add ESM_API_TOKEN to "
            f"'{API_KEYS_FILE}' or set the ESM_API_KEY environment variable."
        )

    api_url = settings.get("ESM_API_URL", DEFAULT_API_URL)
    return str(token).strip(), str(api_url).strip()


def load_model(model_name, device):
    """Initialize the remote ESMC 6B inference client."""
    from esm.sdk.forge import ESMCForgeInferenceClient

    try:
        api_model_name = API_MODEL_MAPPINGS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No Biohub API model mapping is configured for '{model_name}'."
        ) from exc

    token, api_url = _load_api_settings()
    print(
        f"Initializing remote {model_name} ({api_model_name}) API client at {api_url} "
        f"(local device '{device}' is not used for inference)..."
    )
    return ESMCForgeInferenceClient(
        model=api_model_name,
        url=api_url,
        token=token,
    )


def _clean_sequence(seq):
    """Normalize one sequence for the native ESM sequence vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )


def _request_embedding(seq, client, target_dtype):
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    with warnings.catch_warnings():
        # ESM 3.2.x reconstructs some already-tensor response fields with
        # torch.tensor(), which emits a harmless tensor-copy advisory.
        warnings.filterwarnings(
            "ignore",
            message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.",
            category=UserWarning,
            module=r"esm\.utils\.misc",
        )

        protein_tensor = client.encode(ESMProtein(sequence=seq))
        if isinstance(protein_tensor, ESMProteinError):
            raise RuntimeError(
                f"API encode error (code {protein_tensor.error_code}): "
                f"{protein_tensor.error_msg}"
            )

        logits = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    if isinstance(logits, ESMProteinError):
        raise RuntimeError(
            f"API logits error (code {logits.error_code}): {logits.error_msg}"
        )

    embeddings = logits.embeddings
    # NumPy cannot directly convert PyTorch bfloat16 tensors. Upcast first,
    # then apply the user-selected float16/float32 storage dtype below.
    if hasattr(embeddings, "to"):
        import torch

        embeddings = embeddings.to(torch.float32)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()
    if hasattr(embeddings, "cpu"):
        embeddings = embeddings.cpu()
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
    else:
        embeddings = np.asarray(embeddings)

    if embeddings.ndim == 3:
        embeddings = embeddings[0]
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise RuntimeError(
            f"API returned an unexpected embedding shape: {embeddings.shape}"
        )

    # Remove the start and stop token rows to preserve the per-residue HDF5 contract.
    embeddings = embeddings[1:-1]
    if embeddings.shape[0] != len(seq):
        raise RuntimeError(
            "API embedding length does not match the cleaned sequence length "
            f"({embeddings.shape[0]} != {len(seq)})."
        )

    return embeddings.astype(target_dtype)


def get_embedding(seq, model_obj, device, target_dtype):
    """Generate one residue-level embedding through the ESMC 6B API."""
    del device  # The plugin contract supplies this, but inference is remote.
    cleaned_seq = _clean_sequence(seq)
    if not cleaned_seq:
        raise ValueError("Sequence contains no supported amino-acid characters.")

    last_error = None
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            return _request_embedding(cleaned_seq, model_obj, target_dtype)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_API_ATTEMPTS:
                break

            wait_seconds = attempt * RETRY_DELAY_SECONDS
            print(
                f"\nTemporary ESMC 6B API error "
                f"(attempt {attempt}/{MAX_API_ATTEMPTS}): {exc}. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"ESMC 6B API request failed after {MAX_API_ATTEMPTS} attempts."
    ) from last_error
