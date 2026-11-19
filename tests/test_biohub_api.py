"""Tests for shared Biohub credential storage and migration."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resources import Biohub_API
from tests.git_availability import git_executable_for


class BiohubAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shared_path = self.root / "Biohub_API.json"
        self.legacy_path = self.root / "pLM_models" / "esmc_6b_api_key.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def load(self, prompt=None, environ=None):
        return Biohub_API.load_api_settings(
            prompt_callback=prompt,
            path=self.shared_path,
            legacy_path=self.legacy_path,
            environ={} if environ is None else environ,
        )

    def test_first_prompt_writes_the_token_and_url_and_reads_the_fold_defaults(self):
        settings = self.load(prompt=lambda: "  secret-token  ")

        self.assertEqual(settings["ESM_API_TOKEN"], "secret-token")
        self.assertEqual(settings["ESM_API_URL"], Biohub_API.DEFAULT_API_URL)
        self.assertEqual(settings["FOLD_MODEL"], Biohub_API.DEFAULT_FOLD_MODEL)
        self.assertEqual(settings["FOLD_PARALLEL_REQUESTS"], Biohub_API.DEFAULT_FOLD_PARALLEL_REQUESTS)
        # A default is read each time, never written: a later change of it must reach this file too.
        self.assertEqual(
            json.loads(self.shared_path.read_text(encoding="utf-8")),
            {"ESM_API_TOKEN": "secret-token", "ESM_API_URL": Biohub_API.DEFAULT_API_URL},
        )

    def test_the_default_fold_model_is_the_sdks_esmfold2(self):
        try:
            from esm.utils.constants.models import ESMFOLD2
        except ImportError:
            self.skipTest("the esm package is not installed")
        self.assertEqual(Biohub_API.DEFAULT_FOLD_MODEL, ESMFOLD2)
        self.assertEqual(Biohub_API.DEFAULT_FOLD_MODEL, "esmfold2-2026-05")

    def write_raw(self, document):
        self.shared_path.write_text(json.dumps(document), encoding="utf-8")

    def test_a_file_that_holds_only_the_old_esm3_model_stays_valid_and_folds_with_esmfold2(self):
        self.write_raw({"ESM_API_TOKEN": "token", "ESM_API_URL": "https://biohub.ai", "ESM3_MODEL": "esm3-large-2024-03"})

        self.assertEqual(Biohub_API.credentials_state(self.shared_path, environ={}), ("saved", None))
        self.assertEqual(Biohub_API.remote_model_name(self.shared_path), "esmfold2-2026-05")
        self.assertNotIn("ESM3_MODEL", Biohub_API.read_api_settings(self.shared_path))
        loaded = self.load()
        self.assertEqual((loaded["FOLD_MODEL"], loaded["FOLD_PARALLEL_REQUESTS"]), ("esmfold2-2026-05", 4))

    def test_an_esm3_model_is_ignored_whatever_it_holds(self):
        for value in ("esm3-medium-2024-08", "", 7, None, "not-esm3"):
            with self.subTest(value=value):
                self.write_raw({"ESM_API_TOKEN": "token", "ESM3_MODEL": value})
                self.assertEqual(Biohub_API.credentials_state(self.shared_path, environ={}), ("saved", None))
                self.assertEqual(Biohub_API.remote_model_name(self.shared_path), Biohub_API.DEFAULT_FOLD_MODEL)

    def test_the_esm3_model_is_dropped_by_the_next_write_and_not_copied_by_the_migration(self):
        self.write_raw({"ESM_API_TOKEN": "old", "ESM_API_URL": "https://example.test", "ESM3_MODEL": "esm3-large-2024-03"})
        stored = Biohub_API.read_api_settings(self.shared_path)
        Biohub_API.refresh_api_token(stored, lambda: "new", path=self.shared_path)
        self.assertEqual(
            json.loads(self.shared_path.read_text(encoding="utf-8")),
            {"ESM_API_TOKEN": "new", "ESM_API_URL": "https://example.test"},
        )

        self.shared_path.unlink()
        self.legacy_path.parent.mkdir(parents=True)
        self.legacy_path.write_text(json.dumps({"ESM_API_TOKEN": "legacy", "ESM3_MODEL": "esm3-large-2024-03"}), encoding="utf-8")
        self.load()
        self.assertEqual(json.loads(self.shared_path.read_text(encoding="utf-8")),
                         {"ESM_API_TOKEN": "legacy", "ESM_API_URL": Biohub_API.DEFAULT_API_URL})

    def test_fold_model_and_parallel_requests_are_validated(self):
        valid = {"ESM_API_TOKEN": "token", "FOLD_MODEL": " esmfold2-fast-2026-05 ", "FOLD_PARALLEL_REQUESTS": 16}
        settings = Biohub_API.validate_api_settings(valid)
        self.assertEqual((settings["FOLD_MODEL"], settings["FOLD_PARALLEL_REQUESTS"]), ("esmfold2-fast-2026-05", 16))
        for key, value in (
            ("FOLD_MODEL", "esm3-large-2024-03"), ("FOLD_MODEL", ""), ("FOLD_MODEL", 5), ("FOLD_MODEL", None),
            ("FOLD_PARALLEL_REQUESTS", 0), ("FOLD_PARALLEL_REQUESTS", 17), ("FOLD_PARALLEL_REQUESTS", True),
            ("FOLD_PARALLEL_REQUESTS", "4"), ("FOLD_PARALLEL_REQUESTS", 2.0),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(Biohub_API.BiohubSettingsError) as captured:
                    Biohub_API.validate_api_settings({"ESM_API_TOKEN": "token", key: value})
                self.assertIn(key, str(captured.exception))
        self.assertEqual(Biohub_API.validate_api_settings({"ESM_API_TOKEN": "token", "FOLD_PARALLEL_REQUESTS": 1})
                         ["FOLD_PARALLEL_REQUESTS"], 1)

    def test_a_file_with_an_unusable_fold_setting_is_invalid_not_missing(self):
        self.write_raw({"ESM_API_TOKEN": "token", "FOLD_MODEL": "esm3-large-2024-03"})
        state, error = Biohub_API.credentials_state(self.shared_path, environ={"ESM_API_KEY": "x"})
        self.assertEqual(state, "invalid")
        self.assertIn("FOLD_MODEL", str(error))
        self.assertEqual(Biohub_API.remote_model_name(self.shared_path), Biohub_API.DEFAULT_FOLD_MODEL)

    def test_the_fold_settings_a_file_holds_are_read_and_kept_by_a_token_replacement(self):
        self.write_raw({"ESM_API_TOKEN": "old", "FOLD_MODEL": "esmfold2-fast-2026-05", "FOLD_PARALLEL_REQUESTS": 8})
        self.assertEqual(Biohub_API.remote_model_name(self.shared_path), "esmfold2-fast-2026-05")
        self.assertEqual(self.load()["FOLD_PARALLEL_REQUESTS"], 8)
        Biohub_API.refresh_api_token(Biohub_API.read_api_settings(self.shared_path), lambda: "new", path=self.shared_path)
        self.assertEqual(
            json.loads(self.shared_path.read_text(encoding="utf-8")),
            {"ESM_API_TOKEN": "new", "ESM_API_URL": Biohub_API.DEFAULT_API_URL,
             "FOLD_MODEL": "esmfold2-fast-2026-05", "FOLD_PARALLEL_REQUESTS": 8},
        )

    def test_a_token_replacement_given_resolved_settings_does_not_write_the_defaults(self):
        self.write_raw({"ESM_API_TOKEN": "old"})
        resolved = self.load()
        self.assertEqual(resolved["FOLD_MODEL"], Biohub_API.DEFAULT_FOLD_MODEL)
        Biohub_API.refresh_api_token(resolved, lambda: "new", path=self.shared_path)
        self.assertEqual(
            json.loads(self.shared_path.read_text(encoding="utf-8")),
            {"ESM_API_TOKEN": "new", "ESM_API_URL": Biohub_API.DEFAULT_API_URL},
        )

    def test_the_environment_fallback_has_the_fold_defaults(self):
        settings = self.load(environ={"ESM_API_KEY": "environment-token"})
        self.assertEqual((settings["FOLD_MODEL"], settings["FOLD_PARALLEL_REQUESTS"]), ("esmfold2-2026-05", 4))

    def test_environment_fallback_is_not_persisted(self):
        settings = self.load(environ={"ESM_API_KEY": "environment-token"})

        self.assertEqual(settings["ESM_API_TOKEN"], "environment-token")
        self.assertFalse(self.shared_path.exists())

    def test_cancelled_prompt_does_not_create_file(self):
        with self.assertRaises(Biohub_API.BiohubPromptCancelled):
            self.load(prompt=lambda: None)
        self.assertFalse(self.shared_path.exists())

    def test_legacy_file_is_moved_and_deleted_after_verification(self):
        self.legacy_path.parent.mkdir(parents=True)
        self.legacy_path.write_text(
            json.dumps({"ESM_API_TOKEN": "legacy-token"}),
            encoding="utf-8",
        )

        settings = self.load(prompt=lambda: self.fail("prompt should not run"))

        self.assertEqual(settings["ESM_API_TOKEN"], "legacy-token")
        self.assertTrue(self.shared_path.exists())
        self.assertFalse(self.legacy_path.exists())

    def test_existing_optional_url_and_model_are_validated(self):
        settings = Biohub_API.write_api_settings(
            {
                "ESM_API_TOKEN": "token",
                "ESM_API_URL": "https://example.test/api/",
                "FOLD_MODEL": "esmfold2-fast-2026-05",
            },
            self.shared_path,
        )

        self.assertEqual(settings["ESM_API_URL"], "https://example.test/api")
        self.assertEqual(settings["FOLD_MODEL"], "esmfold2-fast-2026-05")

    def test_invalid_settings_do_not_echo_the_token(self):
        token = "do-not-print-this-token"
        with self.assertRaises(Biohub_API.BiohubSettingsError) as captured:
            Biohub_API.write_api_settings(
                {
                    "ESM_API_TOKEN": token,
                    "ESM_API_URL": "not-a-url",
                },
                self.shared_path,
            )
        self.assertNotIn(token, str(captured.exception))
        self.assertFalse(self.shared_path.exists())

    def test_refresh_preserves_url_and_model(self):
        original = {
            "ESM_API_TOKEN": "old-token",
            "ESM_API_URL": "https://example.test",
            "FOLD_MODEL": "esmfold2-fast-2026-05",
        }
        refreshed = Biohub_API.refresh_api_token(
            original,
            lambda: "new-token",
            path=self.shared_path,
        )

        self.assertEqual(refreshed["ESM_API_TOKEN"], "new-token")
        self.assertEqual(refreshed["ESM_API_URL"], original["ESM_API_URL"])
        self.assertEqual(refreshed["FOLD_MODEL"], original["FOLD_MODEL"])

    def test_authentication_detection_and_client_refresh(self):
        chained = RuntimeError("outer")
        cause = RuntimeError("inner")
        cause.error_code = 403
        chained.__cause__ = cause
        self.assertEqual(Biohub_API.authentication_status(chained), 403)

        client = SimpleNamespace(token="old", headers={"Existing": "header"})
        Biohub_API.update_client_token(client, "new")
        self.assertEqual(client.token, "new")
        self.assertEqual(client.headers["Authorization"], "Bearer new")
        self.assertEqual(client.headers["Existing"], "header")

    def test_failed_replace_removes_temporary_file(self):
        with mock.patch.object(
            Biohub_API.os,
            "replace",
            side_effect=OSError("blocked"),
        ):
            with self.assertRaises(Biohub_API.BiohubSettingsError):
                Biohub_API.write_api_settings(
                    {"ESM_API_TOKEN": "token"},
                    self.shared_path,
                )
        self.assertEqual(list(self.root.glob(".Biohub_API.json.*.tmp")), [])

    def interrupted_write(self):
        """Stop a write as a hard kill would, before the rename; return the
        temporary copy it leaves behind, token included."""
        class Killed(BaseException):
            pass

        before = set(self.root.iterdir())
        with mock.patch.object(Biohub_API.os, "replace", side_effect=Killed), \
                mock.patch.object(Biohub_API.os, "unlink"), self.assertRaises(Killed):
            Biohub_API.write_api_settings({"ESM_API_TOKEN": "secret-token"}, self.shared_path)
        [leftover] = set(self.root.iterdir()) - before
        return leftover

    def test_an_interrupted_write_leaves_its_copy_only_until_it_is_stale(self):
        Biohub_API.write_api_settings({"ESM_API_TOKEN": "kept-token"}, self.shared_path)
        stale = self.interrupted_write()
        recent = self.interrupted_write()
        old = time.time() - 2 * Biohub_API._STALE_TEMPORARY_SECONDS
        os.utime(stale, (old, old))

        with mock.patch("builtins.open", wraps=open) as opened:
            settings = Biohub_API.read_api_settings(self.shared_path)
        self.assertEqual(settings["ESM_API_TOKEN"], "kept-token")
        self.assertFalse(stale.exists())
        self.assertTrue(recent.exists())
        self.assertNotIn(stale.name, str(opened.call_args_list))

        os.utime(recent, (old, old))
        Biohub_API.write_api_settings({"ESM_API_TOKEN": "new-token"}, self.shared_path)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["Biohub_API.json"])

    def test_an_interrupted_write_copy_cannot_be_committed(self):
        relative = f"src/resources/{self.interrupted_write().name}"
        git = git_executable_for(PROJECT_ROOT)
        try:
            result = subprocess.run(
                [git, "-C", str(PROJECT_ROOT), "check-ignore", "--no-index", "--quiet", "--", relative],
                capture_output=True,
            )
        except OSError:
            self.skipTest("git is not installed")
        if result.returncode not in (0, 1):
            self.skipTest("not a git checkout")
        self.assertEqual(result.returncode, 0, f"{relative} is not git-ignored")


if __name__ == "__main__":
    unittest.main()
