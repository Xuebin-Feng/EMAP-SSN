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

"""Shared duplicate-safe opening for the bundled browser pages."""

import time
import webbrowser

from PySide6.QtWidgets import QMessageBox


PENDING_OPEN_SECONDS = 10.0


def _set_console_message(viewer, message):
    console_text = getattr(viewer, "console_text", None)
    if console_text is not None:
        console_text.text = message
        update_background = getattr(viewer, "update_console_background", None)
        if callable(update_background):
            update_background()


def _report_existing_page(viewer, message, show_dialog):
    _set_console_message(viewer, message)
    if show_dialog:
        QMessageBox.information(
            getattr(viewer, "main_window", None),
            "Browser Page Already Open",
            message,
        )


def _pending_opens(viewer):
    pending = getattr(viewer, "_web_ui_pending_opens", None)
    if not isinstance(pending, dict):
        pending = {}
        viewer._web_ui_pending_opens = pending
    return pending


def open_browser_page(
    viewer,
    path,
    label,
    client_id,
    *,
    show_existing_dialog=True,
):
    """Open one bundled page unless that client is connected or opening."""
    pending = _pending_opens(viewer)
    now = time.monotonic()

    web_server = getattr(viewer, "web_server", None)
    has_event_client = getattr(web_server, "has_event_client", None)
    if callable(has_event_client) and has_event_client(client_id):
        pending.pop(client_id, None)
        _report_existing_page(
            viewer,
            f"{label} is already open in your browser.",
            show_existing_dialog,
        )
        return False

    pending_deadline = pending.get(client_id, 0.0)
    if pending_deadline > now:
        _report_existing_page(
            viewer,
            f"{label} is already being opened in your browser.",
            show_existing_dialog,
        )
        return False
    pending.pop(client_id, None)

    try:
        url = viewer.get_web_url(path)
    except RuntimeError as error:
        _set_console_message(viewer, f"{label} unavailable: {error}")
        return False

    pending[client_id] = now + PENDING_OPEN_SECONDS
    try:
        opened = bool(webbrowser.open(url, new=2))
    except Exception as error:
        pending.pop(client_id, None)
        _set_console_message(viewer, f"Could not open {label}: {error}")
        return False

    if not opened:
        pending.pop(client_id, None)
        _set_console_message(viewer, f"Could not open {label}: {url}")
        return False

    _set_console_message(viewer, f"{label} opened at {url}")
    return True
