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

import getpass
import re
import threading
import time
import warnings

import numpy as np
from resources import Biohub_API
from tools.tool_helpers.Remote_Requests import (
    RateLimitedError,
    notify,
    rate_limits_signalled,
)

# ESM 3.2.x reconstructs some already-tensor response fields with
# torch.tensor(), which emits a harmless tensor-copy advisory. The filter is
# added once, here, because warnings.catch_warnings() changes process-wide
# state and is not safe while several threads send requests.
warnings.filterwarnings(
    "ignore",
    message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.",
    category=UserWarning,
    module=r"esm\.utils\.misc",
)

# Only one thread at a time may replace a rejected token (and ask for it).
_TOKEN_REPLACEMENT_LOCK = threading.Lock()


SUPPORTED_MODELS = ["esmc_6b"]
MODEL_EXECUTION_MODES = {"esmc_6b": "remote_api"}
# What the Tools dropdowns show; the identifier above stays what settings and files hold.
MODEL_DISPLAY_NAMES = {"esmc_6b": "ESM C 6B"}

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
    # Counts token replacements, so a thread whose request used the old token
    # can tell that another thread has already replaced it.
    client._ssn_auth_generation = 0
    # The SDK makes its HTTP client on first use; make it here, once, before
    # several threads use the client.
    getattr(client, "client", None)
    return client


def _clean_sequence(seq):
    """Normalize one sequence for the native ESM sequence vocabulary."""
    seq = seq.upper()
    match = _RESIDUE_BOUNDARY_PATTERN.search(seq)
    core_seq = match.group(0) if match else ""
    return "".join(
        code if code in SUPPORTED_RESIDUE_CODES else "X" for code in core_seq
    )


def _raise_api_error(call, error):
    """Raise the error a server reply (an ESMProteinError) stands for."""
    status = Biohub_API.authentication_status(error)
    if status is not None:
        raise Biohub_API.BiohubAuthenticationError(status)
    # RateLimitedError is a RuntimeError, so it reads as before when the
    # caller does not ask for it by name.
    error_type = RateLimitedError if error.error_code == 429 else RuntimeError
    raise error_type(
        f"API {call} error (code {error.error_code}): {error.error_msg}"
    )


def _request_embedding(seq, client, target_dtype):
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    protein_tensor = client.encode(ESMProtein(sequence=seq))
    if isinstance(protein_tensor, ESMProteinError):
        _raise_api_error("encode", protein_tensor)

    logits = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    if isinstance(logits, ESMProteinError):
        _raise_api_error("logits", logits)

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


def _wait_before_retry(attempt, error):
    wait_seconds = attempt * RETRY_DELAY_SECONDS
    notify(
        f"Temporary ESMC 6B API error "
        f"(attempt {attempt}/{MAX_API_ATTEMPTS}): {error}. "
        f"Retrying in {wait_seconds} seconds..."
    )
    time.sleep(wait_seconds)


def _replace_rejected_token(model_obj, auth_error, seen_generation):
    """Replace a rejected token once, from one thread; return to retry.

    seen_generation is the token generation the failed request was sent with.
    A thread that waited for the lock while another replaced the token finds a
    newer generation and simply retries. A rejection of the replacement itself
    ends the run.
    """
    with _TOKEN_REPLACEMENT_LOCK:
        if getattr(model_obj, "_ssn_auth_generation", 0) != seen_generation:
            return
        if getattr(model_obj, "_ssn_auth_refresh_attempted", False):
            raise Biohub_API.BiohubAuthenticationError(
                auth_error.error_code,
                "Biohub rejected the API token after one replacement attempt.",
            ) from auth_error
        model_obj._ssn_auth_refresh_attempted = True
        notify(
            "Biohub rejected the saved API token. "
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
        model_obj._ssn_auth_generation = seen_generation + 1


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
                token_generation = getattr(model_obj, "_ssn_auth_generation", 0)
                try:
                    return _request_embedding(cleaned_seq, model_obj, target_dtype)
                except Biohub_API.BiohubAuthenticationError as auth_error:
                    _replace_rejected_token(model_obj, auth_error, token_generation)
        except Biohub_API.BiohubAuthenticationError:
            raise
        except Exception as exc:
            if isinstance(exc, RateLimitedError) and rate_limits_signalled():
                # The caller keeps several requests in flight and lowers their
                # number itself; this is not a failed attempt.
                raise
            last_error = exc
            if attempt == MAX_API_ATTEMPTS:
                break
            _wait_before_retry(attempt, exc)

    raise RuntimeError(
        f"ESMC 6B API request failed after {MAX_API_ATTEMPTS} attempts."
    ) from last_error
