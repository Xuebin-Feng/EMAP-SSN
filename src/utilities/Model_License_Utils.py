# Copyright 2026 Xuebin Feng
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

"""Local acknowledgement records for separately licensed model weights."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ACCEPTANCE_FILE = os.path.join(
    PROJECT_ROOT,
    "src",
    "resources",
    "pLM_models",
    "ankh_license.json",
)


class ModelLicenseAcceptanceRequired(PermissionError):
    """Raised before model access when required terms have not been accepted."""


def model_terms_fingerprint(model_name, terms):
    """Return a stable fingerprint that changes with the model's declared terms."""
    payload = {
        "schema": 1,
        "model_name": model_name,
        "terms": terms,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_acceptance_store(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def is_model_license_accepted(model_name, terms, path=DEFAULT_ACCEPTANCE_FILE):
    """Return true only for an exact, current model-and-terms acknowledgement."""
    if not terms or not terms.get("requires_acknowledgement", False):
        return True
    record = _read_acceptance_store(path).get(model_name)
    return bool(
        isinstance(record, dict)
        and record.get("terms_fingerprint")
        == model_terms_fingerprint(model_name, terms)
    )


def record_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
):
    """Atomically record acceptance without collecting identity or other PII."""
    if not terms or not terms.get("requires_acknowledgement", False):
        return

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    data = _read_acceptance_store(path)
    data[model_name] = {
        "license_id": terms["license_id"],
        "source_url": terms["source_url"],
        "license_url": terms["license_url"],
        "terms_fingerprint": model_terms_fingerprint(model_name, terms),
        "accepted_at": datetime.now(timezone.utc).isoformat(),
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".model-license-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def format_model_usage_terms(model_name, terms):
    """Create the common human-readable notice used by GUI and CLI paths."""
    return (
        f"Model: {model_name}\n"
        f"Weights license: {terms['license_id']}\n"
        f"Restriction: {terms['restriction']}\n"
        f"Model source: {terms['source_url']}\n"
        f"License information: {terms['license_url']}\n\n"
        "The SSN Viewer integration code is Apache-2.0, but the separately "
        "downloaded model weights are not."
    )


def format_model_selector_label(model_name, terms):
    """Label separately licensed weights without changing the model identifier."""
    if not terms:
        return model_name
    restriction = terms["restriction"].lower()
    if "non-commercial" in restriction:
        return f"{model_name} [non-commercial]"
    return f"{model_name} [separate terms]"


def prompt_for_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
    input_func=input,
    output_func=print,
):
    """Review terms and optionally persist an exact terminal acknowledgement."""
    output_func(format_model_usage_terms(model_name, terms))
    response = input_func(
        "\nType I ACCEPT to record acceptance, or press Enter to cancel: "
    )
    if response.strip() != "I ACCEPT":
        return False
    record_model_license_acceptance(model_name, terms, path)
    return True


def require_model_license_acceptance(
    model_name,
    terms,
    path=DEFAULT_ACCEPTANCE_FILE,
):
    """Fail closed before model loading if current terms need acknowledgement."""
    if is_model_license_accepted(model_name, terms, path):
        return
    notice = format_model_usage_terms(model_name, terms)
    raise ModelLicenseAcceptanceRequired(
        notice
        + "\n\nNo model files were accessed. To review and accept these terms "
        "from a terminal, run:\n"
        f"  python src/tools/Generate_Embeddings.py "
        f"--accept-model-license {model_name}"
    )
