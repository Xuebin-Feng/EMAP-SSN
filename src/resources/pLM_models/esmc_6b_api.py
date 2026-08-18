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

import getpass
import re
import time
import warnings

import numpy as np
from resources import Biohub_API


SUPPORTED_MODELS = ["esmc_6b"]
MODEL_EXECUTION_MODES = {"esmc_6b": "remote_api"}

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

MAX_API_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


def _terminal_token_prompt(replacement=False):
    action = "Replacement" if replacement else "Biohub"
    try:
        return getpass.getpass(
            f"{action} API token (input hidden; press Enter to cancel): "
        )
    except (EOFError, KeyboardInterrupt):
        return None


def load_model(model_name, device):
    """Initialize the remote ESMC 6B inference client."""
    from esm.sdk.forge import ESMCForgeInferenceClient

    try:
        api_model_name = API_MODEL_MAPPINGS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No Biohub API model mapping is configured for '{model_name}'."
        ) from exc

    settings = Biohub_API.load_api_settings(
        prompt_callback=lambda: _terminal_token_prompt(False),
    )
    token = settings["ESM_API_TOKEN"]
    api_url = settings["ESM_API_URL"]
    print(
        f"Initializing remote {model_name} ({api_model_name}) API client at {api_url} "
        "(remote inference; no local device is used)..."
    )
    client = ESMCForgeInferenceClient(
        model=api_model_name,
        url=api_url,
        token=token,
    )
    client._ssn_biohub_settings = settings
    client._ssn_auth_refresh_attempted = False
    return client


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
            status = Biohub_API.authentication_status(protein_tensor)
            if status is not None:
                raise Biohub_API.BiohubAuthenticationError(status)
            raise RuntimeError(
                f"API encode error (code {protein_tensor.error_code}): "
                f"{protein_tensor.error_msg}"
            )

        logits = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )

    if isinstance(logits, ESMProteinError):
        status = Biohub_API.authentication_status(logits)
        if status is not None:
            raise Biohub_API.BiohubAuthenticationError(status)
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
            while True:
                try:
                    return _request_embedding(cleaned_seq, model_obj, target_dtype)
                except Biohub_API.BiohubAuthenticationError as auth_error:
                    if getattr(model_obj, "_ssn_auth_refresh_attempted", False):
                        raise Biohub_API.BiohubAuthenticationError(
                            auth_error.error_code,
                            "Biohub rejected the API token after one replacement attempt.",
                        ) from auth_error
                    model_obj._ssn_auth_refresh_attempted = True
                    print(
                        "\nBiohub rejected the saved API token. "
                        "Enter a replacement to retry once."
                    )
                    settings = getattr(model_obj, "_ssn_biohub_settings", None)
                    if settings is None:
                        settings = Biohub_API.load_api_settings()
                    refreshed = Biohub_API.refresh_api_token(
                        settings,
                        lambda: _terminal_token_prompt(True),
                    )
                    model_obj._ssn_biohub_settings = refreshed
                    Biohub_API.update_client_token(
                        model_obj,
                        refreshed["ESM_API_TOKEN"],
                    )
        except Biohub_API.BiohubAuthenticationError:
            raise
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
