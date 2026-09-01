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
from collections.abc import Callable, Mapping
from urllib.parse import urlparse


DEFAULT_API_URL = "https://biohub.ai"
DEFAULT_ESM3_MODEL = "esm3-large-2024-03"

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
    """Validate and normalize the shared JSON contract."""
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

    esm3_model = settings.get("ESM3_MODEL", DEFAULT_ESM3_MODEL)
    if not isinstance(esm3_model, str) or not esm3_model.strip():
        raise BiohubSettingsError("ESM3_MODEL must be a nonblank model identifier.")
    esm3_model = esm3_model.strip()
    if not esm3_model.startswith("esm3-"):
        raise BiohubSettingsError("ESM3_MODEL must begin with 'esm3-'.")

    return {
        "ESM_API_TOKEN": token.strip(),
        "ESM_API_URL": api_url,
        "ESM3_MODEL": esm3_model,
    }


def read_api_settings(path=API_SETTINGS_FILE) -> dict[str, str] | None:
    """Read one settings file, returning ``None`` only when it is absent."""
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


def write_api_settings(settings: Mapping, path=API_SETTINGS_FILE) -> dict[str, str]:
    """Validate and atomically publish settings without logging credentials."""
    normalized = validate_api_settings(settings)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

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
            "ESM3_MODEL": legacy.get("ESM3_MODEL", DEFAULT_ESM3_MODEL),
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
    """Load, migrate, use an environment fallback, or interactively create settings."""
    settings = migrate_legacy_settings(path=path, legacy_path=legacy_path)
    if settings is not None:
        return settings

    environment = os.environ if environ is None else environ
    environment_token = environment.get("ESM_API_KEY")
    if isinstance(environment_token, str) and environment_token.strip():
        return validate_api_settings({"ESM_API_TOKEN": environment_token})

    if prompt_callback is None:
        raise BiohubSettingsError(
            "Biohub API credentials are required. Create "
            f"'{path}' or set ESM_API_KEY."
        )
    token = prompt_callback()
    if not isinstance(token, str) or not token.strip():
        raise BiohubPromptCancelled("Biohub API token entry was cancelled.")
    return write_api_settings({"ESM_API_TOKEN": token}, path)


def refresh_api_token(
    settings: Mapping,
    prompt_callback: Callable[[], str | None],
    *,
    path=API_SETTINGS_FILE,
) -> dict[str, str]:
    """Replace a rejected token while preserving the selected URL and ESM3 model."""
    normalized = validate_api_settings(settings)
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

