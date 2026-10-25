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

"""Shared duplicate-safe opening for the bundled browser pages."""

import time

from PySide6.QtWidgets import QMessageBox

from desktop.Desktop_App import translate
from desktop.Viewer_Window import viewer_shows_windows
from utilities import Desktop_Open
from utilities.Localization import Message, display_text


PENDING_OPEN_SECONDS = 10.0


def is_headless(viewer):
    """Whether viewer shows no windows: a headless Viewer, or a hidden one while its window is not shown.

    It has no screen to show a page, a dialog or a terminal on.
    """
    return not viewer_shows_windows(viewer)


def page_address(viewer, path):
    """The URL viewer serves ``path`` at, or None while its web server is unavailable."""
    try:
        return viewer.get_web_url(path)
    except RuntimeError:
        return None


def _set_console_message(viewer, message):
    if getattr(viewer, "console_text", None) is not None:
        import Command_Engine

        Command_Engine.show_status(viewer, message)
        update_background = getattr(viewer, "update_console_background", None)
        if callable(update_background):
            update_background()


def _report_existing_page(viewer, message, show_dialog):
    _set_console_message(viewer, message)
    # Nobody is at a Viewer that shows no windows to close the dialog.
    if show_dialog and not is_headless(viewer):
        QMessageBox.information(
            getattr(viewer, "main_window", None),
            translate("Viewer", "Browser Page Already Open"),
            display_text(message),
        )


def _pending_opens(viewer):
    pending = getattr(viewer, "_web_ui_pending_opens", None)
    if not isinstance(pending, dict):
        pending = {}
        viewer._web_ui_pending_opens = pending
    return pending


def page_is_open(viewer, client_id):
    """Whether client_id's page is connected to the Viewer or is being opened."""
    web_server = getattr(viewer, "web_server", None)
    has_event_client = getattr(web_server, "has_event_client", None)
    if callable(has_event_client) and has_event_client(client_id):
        return True
    return _pending_opens(viewer).get(client_id, 0.0) > time.monotonic()


def open_browser_page(
    viewer,
    path,
    label,
    client_id,
    *,
    show_existing_dialog=True,
):
    """Open one bundled page unless that client is connected or opening.

    label names the page in the messages; a Message shows translated. A
    headless Viewer opens no browser; its console line gives the address.
    """
    pending = _pending_opens(viewer)
    now = time.monotonic()

    web_server = getattr(viewer, "web_server", None)
    has_event_client = getattr(web_server, "has_event_client", None)
    if callable(has_event_client) and has_event_client(client_id):
        pending.pop(client_id, None)
        _report_existing_page(
            viewer,
            Message("{page} is already open in your browser.", page=label),
            show_existing_dialog,
        )
        return False

    pending_deadline = pending.get(client_id, 0.0)
    if pending_deadline > now:
        _report_existing_page(
            viewer,
            Message("{page} is already being opened in your browser.", page=label),
            show_existing_dialog,
        )
        return False
    pending.pop(client_id, None)

    try:
        url = viewer.get_web_url(path)
    except RuntimeError as error:
        _set_console_message(viewer, Message("{page} unavailable: {error}", page=label, error=error))
        return False

    if is_headless(viewer):
        _set_console_message(viewer, Message("Could not open {page}: {url}", page=label, url=url))
        return False

    pending[client_id] = now + PENDING_OPEN_SECONDS
    try:
        opened = Desktop_Open.open_url(url, new_tab=True)
    except Exception as error:
        pending.pop(client_id, None)
        _set_console_message(viewer, Message("Could not open {page}: {error}", page=label, error=error))
        return False

    if not opened:
        pending.pop(client_id, None)
        _set_console_message(viewer, Message("Could not open {page}: {url}", page=label, url=url))
        return False

    _set_console_message(viewer, Message("{page} opened at {url}", page=label, url=url))
    return True
