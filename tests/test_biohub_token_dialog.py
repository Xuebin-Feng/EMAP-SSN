# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""The dialog that asks for the Biohub API token, and the saving of what it is given."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from desktop import Biohub_Token_Dialog as dialog_module
from resources import Biohub_API


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def type_and(self, text, accept=True):
        """Open the dialog as ask_token does, type text in it, and press OK (or Cancel); return what ask_token gave."""
        seen = {}

        def shown(dialog):
            seen["dialog"] = dialog
            seen["echo"] = dialog.textEchoMode()
            seen["label"] = dialog.labelText()
            seen["title"] = dialog.windowTitle()

            def answer():
                dialog.setTextValue(text)
                dialog.accept() if accept else dialog.reject()

            QtCore.QTimer.singleShot(0, answer)

        self.replacement = seen
        return dialog_module.ask_token(None, replacement=getattr(self, "asking_replacement", False), shown=shown), seen

    def test_the_text_is_typed_hidden_and_the_token_comes_back_trimmed(self):
        token, seen = self.type_and("  secret-token-1  ")
        self.assertEqual(token, "secret-token-1")
        self.assertEqual(seen["echo"], QtWidgets.QLineEdit.EchoMode.Password)
        self.assertIn("Biohub API token", seen["label"])
        self.assertEqual(seen["title"], "Biohub API Token")

    def test_cancelling_or_leaving_it_blank_gives_nothing(self):
        self.assertIsNone(self.type_and("secret", accept=False)[0])
        self.assertIsNone(self.type_and("   ")[0])
        self.assertIsNone(self.type_and("")[0])

    def test_a_replacement_says_why_it_is_asked_for(self):
        self.asking_replacement = True
        try:
            _token, seen = self.type_and("new-token")
        finally:
            del self.asking_replacement
        self.assertIn("Biohub rejected the saved API token", seen["label"])

    def test_the_dialog_is_not_left_behind(self):
        _token, seen = self.type_and("secret")
        self.app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        import shiboken6
        self.assertFalse(shiboken6.isValid(seen["dialog"]), "the dialog was deleted once it closed")


class SaveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = os.path.join(directory.name, "Biohub_API.json")
        patcher = mock.patch.object(Biohub_API, "API_SETTINGS_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def saved(self):
        with open(self.path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_a_new_token_is_saved_with_the_default_url_and_model(self):
        dialog_module.save_token("brand-new")
        self.assertEqual(self.saved(), {
            "ESM_API_TOKEN": "brand-new", "ESM_API_URL": Biohub_API.DEFAULT_API_URL,
            "ESM3_MODEL": Biohub_API.DEFAULT_ESM3_MODEL})

    def test_a_replacement_keeps_the_url_and_model_of_the_saved_settings(self):
        Biohub_API.write_api_settings(
            {"ESM_API_TOKEN": "old", "ESM_API_URL": "https://biohub.example", "ESM3_MODEL": "esm3-large-2024-03"}, self.path)
        dialog_module.save_token("new", replacement=True)
        self.assertEqual(self.saved(), {
            "ESM_API_TOKEN": "new", "ESM_API_URL": "https://biohub.example", "ESM3_MODEL": "esm3-large-2024-03"})

    def test_a_replacement_with_nothing_saved_yet_writes_the_defaults(self):
        dialog_module.save_token("from-the-environment-case", replacement=True)
        self.assertEqual(self.saved()["ESM_API_TOKEN"], "from-the-environment-case")

    def test_a_settings_file_that_cannot_be_read_is_not_written_over_by_a_replacement(self):
        Path(self.path).write_text("{not json", encoding="utf-8")
        with self.assertRaises(Biohub_API.BiohubSettingsError) as caught:
            dialog_module.save_token("secret-token", replacement=True)
        self.assertNotIn("secret-token", str(caught.exception))
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "{not json")

    def test_an_error_names_no_token(self):
        with mock.patch.object(Biohub_API, "write_api_settings",
                               side_effect=Biohub_API.BiohubSettingsError("Could not write Biohub API settings.")):
            with self.assertRaises(Biohub_API.BiohubSettingsError) as caught:
                dialog_module.save_token("secret-token")
        self.assertNotIn("secret-token", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
