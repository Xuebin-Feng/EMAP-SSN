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

"""Shared, UI-independent Biohub API credential storage."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from urllib.parse import urlparse


DEFAULT_API_URL = "https://biohub.ai"
# The model a fold through Biohub uses: the SDK's ESMFOLD2 (esm.utils.constants.models). A literal, so that
# this UI-independent module never imports esm; a test keeps the two equal.
DEFAULT_FOLD_MODEL = "esmfold2-2026-05"
FOLD_MODEL_PREFIX = "esmfold2"
# How many fold requests `esmfold large multi` (and a remote Predict Structures run) keeps in flight.
DEFAULT_FOLD_PARALLEL_REQUESTS = 4
MAX_FOLD_PARALLEL_REQUESTS = 16
# Settings the file may hold besides the token and the URL. They are written only when given, so that a
# default is read from here each time and a later change of it reaches every user.
OPTIONAL_KEYS = ("FOLD_MODEL", "FOLD_PARALLEL_REQUESTS")

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
API_SETTINGS_FILE = os.path.join(RESOURCE_DIR, "Biohub_API.json")
LEGACY_ESMC_SETTINGS_FILE = os.path.join(
    RESOURCE_DIR,
    "pLM_models",
    "esmc_6b_api_key.json",
)


class BiohubSettingsError(ValueError):
    """Raised when Biohub settings are missing, malformed, or unusable."""


class BiohubPromptCancelled(BiohubSettingsError):
    """Raised when an interactive token prompt is cancelled or left blank."""


class BiohubAuthenticationError(RuntimeError):
    """Credential error retaining the HTTP status without exposing the token."""

    def __init__(self, status_code, message="Biohub rejected the API token."):
        super().__init__(message)
        self.error_code = int(status_code)


def validate_api_settings(settings: Mapping) -> dict[str, str]:
    """Validate and normalize the shared JSON contract.

    The result holds the token, the URL and only those of FOLD_MODEL and FOLD_PARALLEL_REQUESTS that
    ``settings`` gave (resolved_settings fills the defaults in for a reader). An ESM3_MODEL, which an older
    version wrote into every file, is ignored: nothing uses it, and it is not written back.
    """
    if not isinstance(settings, Mapping):
        raise BiohubSettingsError("Biohub API settings must contain a JSON object.")

    token = settings.get("ESM_API_TOKEN")
    if not isinstance(token, str) or not token.strip():
        raise BiohubSettingsError(
            "Biohub API settings must contain a nonblank ESM_API_TOKEN."
        )

    api_url = settings.get("ESM_API_URL", DEFAULT_API_URL)
    if not isinstance(api_url, str) or not api_url.strip():
        raise BiohubSettingsError("ESM_API_URL must be a nonblank HTTP(S) URL.")
    api_url = api_url.strip().rstrip("/")
    parsed_url = urlparse(api_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise BiohubSettingsError("ESM_API_URL must be a valid HTTP(S) URL.")

    normalized = {
        "ESM_API_TOKEN": token.strip(),
        "ESM_API_URL": api_url,
    }

    if "FOLD_MODEL" in settings:
        fold_model = settings["FOLD_MODEL"]
        if not isinstance(fold_model, str) or not fold_model.strip():
            raise BiohubSettingsError("FOLD_MODEL must be a nonblank model identifier.")
        fold_model = fold_model.strip()
        if not fold_model.startswith(FOLD_MODEL_PREFIX):
            raise BiohubSettingsError(
                f"FOLD_MODEL must begin with '{FOLD_MODEL_PREFIX}'; ESM3 models are no longer used for folding."
            )
        normalized["FOLD_MODEL"] = fold_model

    if "FOLD_PARALLEL_REQUESTS" in settings:
        parallel = settings["FOLD_PARALLEL_REQUESTS"]
        if (
            isinstance(parallel, bool)
            or not isinstance(parallel, int)
            or not 1 <= parallel <= MAX_FOLD_PARALLEL_REQUESTS
        ):
            raise BiohubSettingsError(
                f"FOLD_PARALLEL_REQUESTS must be a whole number from 1 to {MAX_FOLD_PARALLEL_REQUESTS}."
            )
        normalized["FOLD_PARALLEL_REQUESTS"] = parallel

    return normalized


def resolved_settings(settings: Mapping) -> dict:
    """Validated settings with every default filled in: what a run reads (never what is written)."""
    resolved = validate_api_settings(settings)
    resolved.setdefault("FOLD_MODEL", DEFAULT_FOLD_MODEL)
    resolved.setdefault("FOLD_PARALLEL_REQUESTS", DEFAULT_FOLD_PARALLEL_REQUESTS)
    return resolved


# A write saves to a temporary sibling, .<name>.<random>.tmp, and renames it
# over the settings file. Only a hard kill in between leaves the sibling
# behind; a write takes milliseconds, so one older than this belongs to no
# write in progress.
_STALE_TEMPORARY_SECONDS = 60


def _remove_stale_temporaries(path):
    """Delete temporary siblings of path that an interrupted write left behind.

    They hold the API token, so they are removed unread.
    """
    directory = os.path.dirname(os.path.abspath(path))
    prefix = f".{os.path.basename(path)}."
    cutoff = time.time() - _STALE_TEMPORARY_SECONDS
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        if not (entry.name.startswith(prefix) and entry.name.endswith(".tmp")):
            continue
        try:
            if entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                os.unlink(entry.path)
        except OSError:
            pass


def read_api_settings(path=API_SETTINGS_FILE) -> dict[str, str] | None:
    """Read one settings file, returning ``None`` only when it is absent."""
    _remove_stale_temporaries(path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BiohubSettingsError(
            f"Could not read valid Biohub API JSON from '{path}'."
        ) from error
    try:
        return validate_api_settings(settings)
    except BiohubSettingsError as error:
        raise BiohubSettingsError(f"Invalid Biohub API settings in '{path}': {error}") from error


def credentials_state(path=API_SETTINGS_FILE, environ=None):
    """What a run through Biohub has for credentials, without asking anyone: ("saved", None), ("missing", None) or ("invalid", error).

    Saved means a settings file that holds a token, or ESM_API_KEY in the environment. A settings file that
    is there but unusable is not "missing": asking for a token would overwrite what the user wrote in it.
    The token itself is not returned.
    """
    try:
        if read_api_settings(path) is not None:
            return ("saved", None)
    except BiohubSettingsError as error:
        return ("invalid", error)
    token = (os.environ if environ is None else environ).get("ESM_API_KEY")
    return ("saved", None) if isinstance(token, str) and token.strip() else ("missing", None)


def remote_model_name(path=API_SETTINGS_FILE) -> str:
    """The ESMFold 2 model a fold through Biohub uses, which names the structure files of a remote run.

    The saved settings' FOLD_MODEL, else the default. A settings file that cannot be read gives the default.
    """
    try:
        settings = read_api_settings(path)
    except BiohubSettingsError:
        settings = None
    return (settings or {}).get("FOLD_MODEL", DEFAULT_FOLD_MODEL)


def write_api_settings(settings: Mapping, path=API_SETTINGS_FILE) -> dict[str, str]:
    """Validate and atomically publish settings without logging credentials."""
    normalized = validate_api_settings(settings)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    _remove_stale_temporaries(path)

    descriptor = None
    temporary_path = None
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            os.chmod(temporary_path, 0o600)
        except OSError:
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            json.dump(normalized, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError as error:
        raise BiohubSettingsError(
            f"Could not write Biohub API settings to '{path}'."
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    return normalized


def migrate_legacy_settings(
    path=API_SETTINGS_FILE,
    legacy_path=LEGACY_ESMC_SETTINGS_FILE,
) -> dict[str, str] | None:
    """Move a valid legacy ESMC token into the shared settings file."""
    current = read_api_settings(path)
    if current is not None or not os.path.exists(legacy_path):
        return current

    try:
        with open(legacy_path, "r", encoding="utf-8") as handle:
            legacy = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BiohubSettingsError(
            f"Could not migrate valid JSON from '{legacy_path}'."
        ) from error
    if not isinstance(legacy, dict):
        raise BiohubSettingsError(
            f"Legacy Biohub settings must contain a JSON object: '{legacy_path}'."
        )

    migrated = write_api_settings(
        {
            "ESM_API_TOKEN": legacy.get("ESM_API_TOKEN"),
            "ESM_API_URL": legacy.get("ESM_API_URL", DEFAULT_API_URL),
        },
        path,
    )
    verified = read_api_settings(path)
    if verified != migrated:
        raise BiohubSettingsError(
            "The shared Biohub API file could not be verified after migration."
        )
    try:
        os.unlink(legacy_path)
    except OSError as error:
        raise BiohubSettingsError(
            f"The shared Biohub settings were created, but '{legacy_path}' "
            "could not be removed."
        ) from error
    return verified


def load_api_settings(
    prompt_callback: Callable[[], str | None] | None = None,
    *,
    path=API_SETTINGS_FILE,
    legacy_path=LEGACY_ESMC_SETTINGS_FILE,
    environ=None,
) -> dict[str, str]:
    """Load, migrate, use an environment fallback, or interactively create settings.

    The result has every default filled in (resolved_settings); what is written holds only what was given.
    """
    settings = migrate_legacy_settings(path=path, legacy_path=legacy_path)
    if settings is not None:
        return resolved_settings(settings)

    environment = os.environ if environ is None else environ
    environment_token = environment.get("ESM_API_KEY")
    if isinstance(environment_token, str) and environment_token.strip():
        return resolved_settings({"ESM_API_TOKEN": environment_token})

    if prompt_callback is None:
        raise BiohubSettingsError(
            "Biohub API credentials are required. Create "
            f"'{path}' or set ESM_API_KEY."
        )
    token = prompt_callback()
    if not isinstance(token, str) or not token.strip():
        raise BiohubPromptCancelled("Biohub API token entry was cancelled.")
    return resolved_settings(write_api_settings({"ESM_API_TOKEN": token}, path))


def refresh_api_token(
    settings: Mapping,
    prompt_callback: Callable[[], str | None],
    *,
    path=API_SETTINGS_FILE,
) -> dict[str, str]:
    """Replace a rejected token while preserving the selected URL and the FOLD_* settings that were given."""
    normalized = validate_api_settings(settings)
    # A caller may hold resolved settings (defaults filled in): a default the file does not hold is not
    # written, so that it stays a default.
    try:
        stored = read_api_settings(path) or {}
    except BiohubSettingsError:
        stored = {}
    defaults = {"FOLD_MODEL": DEFAULT_FOLD_MODEL, "FOLD_PARALLEL_REQUESTS": DEFAULT_FOLD_PARALLEL_REQUESTS}
    for key, default in defaults.items():
        if key not in stored and normalized.get(key) == default:
            normalized.pop(key)
    token = prompt_callback()
    if not isinstance(token, str) or not token.strip():
        raise BiohubPromptCancelled("Biohub API token replacement was cancelled.")
    normalized["ESM_API_TOKEN"] = token.strip()
    return write_api_settings(normalized, path)


def authentication_status(error) -> int | None:
    """Return 401/403 from an SDK result, exception, or chained cause."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "error_code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        if code in {401, 403}:
            return code
        current = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
    return None


def update_client_token(client, token: str):
    """Update an existing ESM SDK client's in-memory authorization header."""
    clean_token = str(token).strip()
    client.token = clean_token
    headers = dict(getattr(client, "headers", {}) or {})
    headers["Authorization"] = f"Bearer {clean_token}"
    client.headers = headers

