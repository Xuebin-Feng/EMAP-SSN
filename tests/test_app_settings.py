# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The program's own settings file (utilities.App_Settings).

app_settings.json holds what every window shares, such as the language. A
save keeps the other settings, is atomic, and refuses to overwrite a file it
can't read, which would lose whatever else that file holds.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utilities import App_Settings
from utilities.App_Settings import AppSettingsError, app_settings_path, read_app_settings, save_app_setting


class AppSettingsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.path = self.folder / "app_settings.json"

    def test_the_file_lives_in_the_project_unless_the_environment_says_otherwise(self):
        self.assertEqual(app_settings_path({}), App_Settings.PROJECT_ROOT / "app_settings.json")
        self.assertEqual((App_Settings.PROJECT_ROOT / "src").is_dir(), True)
        self.assertEqual(app_settings_path({"SSN_APP_SETTINGS_PATH": str(self.path)}), self.path)

    def test_the_tests_never_use_the_developers_file(self):
        self.assertNotEqual(app_settings_path(), App_Settings.PROJECT_ROOT / "app_settings.json")

    def test_a_missing_file_counts_as_empty(self):
        self.assertEqual(read_app_settings(self.path), {})

    def test_a_save_adds_one_setting_and_keeps_the_others(self):
        self.path.write_text(json.dumps({"OTHER": [1, 2]}), encoding="utf-8")
        save_app_setting("LANGUAGE", "zh_CN", self.path)
        self.assertEqual(read_app_settings(self.path), {"OTHER": [1, 2], "LANGUAGE": "zh_CN"})
        save_app_setting("LANGUAGE", "system", self.path)
        self.assertEqual(read_app_settings(self.path), {"OTHER": [1, 2], "LANGUAGE": "system"})
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), ["app_settings.json"])

    def test_a_file_that_cannot_be_read_is_reported_and_left_alone(self):
        for broken in ("{not json", "[1, 2]", "\udcff"):
            with self.subTest(content=broken):
                self.path.write_bytes(broken.encode("utf-8", "surrogateescape"))
                before = self.path.read_bytes()
                with self.assertRaises(AppSettingsError):
                    read_app_settings(self.path)
                with self.assertRaises(AppSettingsError):
                    save_app_setting("LANGUAGE", "en", self.path)
                self.assertEqual(self.path.read_bytes(), before)

    def test_an_interrupted_save_leaves_the_file_whole(self):
        save_app_setting("LANGUAGE", "en", self.path)
        with mock.patch.object(App_Settings.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_app_setting("LANGUAGE", "de", self.path)
        self.assertEqual(read_app_settings(self.path), {"LANGUAGE": "en"})
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), ["app_settings.json"])

    def test_a_save_clears_old_partial_files_but_not_a_save_in_progress(self):
        old = self.folder / ".app_settings.json.old.partial"
        fresh = self.folder / ".app_settings.json.fresh.partial"
        for partial in (old, fresh):
            partial.write_text("{}", encoding="utf-8")
        past = time.time() - 3600
        os.utime(old, (past, past))
        save_app_setting("LANGUAGE", "en", self.path)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())


if __name__ == "__main__":
    unittest.main()
