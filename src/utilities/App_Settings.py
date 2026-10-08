# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Settings for the whole program, kept apart from any one window's profile.

app_settings.json, in the project folder, holds what every window shares,
such as the LANGUAGE the windows show. It can't live in viewer_settings.json:
loading a Config profile replaces that file. SSN_APP_SETTINGS_PATH points
somewhere else, as the tests do.

A file that can't be read is reported and left alone: saving over it would
discard everything else it holds. Saves are atomic: a temporary sibling is
renamed into place, so a reader never sees half a file.

Nothing here imports Qt.
"""

import json
import os
from pathlib import Path
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SETTINGS_NAME = "app_settings.json"
APP_SETTINGS_VARIABLE = "SSN_APP_SETTINGS_PATH"

# A save writes .<name>.<random>.partial beside the file and renames it into
# place. Only a hard kill in between leaves the sibling behind; a save takes
# milliseconds, so one older than this belongs to no save in progress.
_STALE_PARTIAL_SECONDS = 60


class AppSettingsError(Exception):
    """app_settings.json exists but can't be read as a JSON object."""


def app_settings_path(environment=None):
    """Where the program's own settings live."""
    environment = os.environ if environment is None else environment
    override = environment.get(APP_SETTINGS_VARIABLE, "").strip()
    return Path(override) if override else PROJECT_ROOT / APP_SETTINGS_NAME


def read_app_settings(path=None):
    """The saved settings, as a dict; a missing file counts as empty.

    Raises AppSettingsError for a file that can't be read, isn't valid JSON
    or doesn't hold a JSON object.
    """
    path = app_settings_path() if path is None else Path(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            settings = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError) as error:
        raise AppSettingsError(f"Could not read settings file '{path}': {error}") from error
    if not isinstance(settings, dict):
        raise AppSettingsError(f"Settings file '{path}' does not hold a JSON object.")
    return settings


def _remove_stale_partials(path):
    prefix, cutoff = f".{path.name}.", time.time() - _STALE_PARTIAL_SECONDS
    try:
        entries = list(os.scandir(path.parent))
    except OSError:
        return
    for entry in entries:
        if not (entry.name.startswith(prefix) and entry.name.endswith(".partial")):
            continue
        try:
            if entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                os.unlink(entry.path)
        except OSError:
            pass


def save_app_setting(key, value, path=None):
    """Set one setting, keeping the others, in an atomic save.

    Raises AppSettingsError, and changes nothing, when the existing file
    can't be read.
    """
    path = app_settings_path() if path is None else Path(path)
    settings = read_app_settings(path)
    settings[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_partials(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return settings
