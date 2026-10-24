# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""The dialog that asks for the Biohub API token, and the saving of what it is given.

``esmfold large`` folds through Biohub and needs the token. A fold has no window to ask
in, so the Viewer asks: before the fold is queued when no token is saved, and while it
runs when Biohub rejected the one that is. The text is typed hidden. The token goes from
the dialog to the settings file (Biohub_API.write_api_settings) and nowhere else: it is
in no message, event, record or log, and a command never takes it as an argument.
"""

from PySide6 import QtWidgets

from desktop.Desktop_App import translate
from resources import Biohub_API


def make_dialog(parent, *, replacement=False):
    """The dialog, not yet shown: one line of hidden text, with Biohub's reason for asking above it."""
    dialog = QtWidgets.QInputDialog(parent)
    dialog.setInputMode(QtWidgets.QInputDialog.InputMode.TextInput)
    dialog.setTextEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
    dialog.setWindowTitle(translate("Viewer", "Biohub API Token"))
    if replacement:
        dialog.setLabelText(translate(
            "Viewer",
            "Biohub rejected the saved API token. Enter a replacement to fold the remaining "
            "structures; it replaces the saved token.",
        ))
    else:
        dialog.setLabelText(translate(
            "Viewer",
            "Folding with Biohub needs your Biohub API token. It is saved for later folds "
            "and is not shown again.",
        ))
    return dialog


def ask_token(parent, *, replacement=False, shown=None):
    """Show the dialog and return the token typed, or None if it was cancelled or left blank.

    ``shown(dialog)`` is called with the dialog before it opens, for a caller that may
    have to close it from elsewhere.
    """
    dialog = make_dialog(parent, replacement=replacement)
    if shown is not None:
        shown(dialog)
    try:
        accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
        token = dialog.textValue().strip() if accepted else ""
    finally:
        dialog.deleteLater()
    return token or None


def save_token(token, *, replacement=False):
    """Save ``token`` as the Biohub API token, keeping the URL and model of the settings already saved.

    Raises Biohub_API.BiohubSettingsError, which never holds the token, when the file cannot be written.
    """
    path = Biohub_API.API_SETTINGS_FILE
    if replacement:
        settings = Biohub_API.read_api_settings(path)
        if settings is not None:
            return Biohub_API.refresh_api_token(settings, lambda: token, path=path)
    return Biohub_API.write_api_settings({"ESM_API_TOKEN": token}, path)
