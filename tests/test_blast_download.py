# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""BLAST_Download: the pinned BLAST+ install (download, size and SHA-256 checks,
unpacking only the listed files, the swap with an existing install, the check
that blastp starts) and the pin file itself. No test reaches the network: the
archives are small fabricated tar.gz files and a pin that matches them."""

import copy
import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilities import BLAST_Download as download  # noqa: E402
from utilities import BLAST_Executables as blast  # noqa: E402

PREFIX = "ncbi-blast-9.9.9+/bin/"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob(label, size):
    """Bytes that do not compress, so that the archive is several chunks long."""
    data = b""
    while len(data) < size:
        data += hashlib.sha256(label.encode() + len(data).to_bytes(4, "big")).digest()
    return data[:size]


UNIX_FILES = {"blastp": blob("blastp", 2000), "makeblastdb": blob("makeblastdb", 1500)}
WINDOWS_FILES = {
    "blastp.exe": blob("blastp.exe", 2000), "makeblastdb.exe": blob("makeblastdb.exe", 1500),
    "nghttp2.dll": blob("nghttp2.dll", 300),
}


def build_archive(path, files, prefix=PREFIX, mode=0o755, links=(), extras=None):
    """A tar.gz of files (name -> bytes) under prefix; links are stored as symlinks instead of files."""
    extras = {
        "ncbi-blast-9.9.9+/LICENSE": b"public domain",
        "ncbi-blast-9.9.9+/bin/blastn": blob("blastn", 3000),
        "ncbi-blast-9.9.9+/bin/../../escape.txt": b"never written",
    } if extras is None else extras
    with tarfile.open(path, "w:gz") as tar:
        for name, data in extras.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for name, data in files.items():
            info = tarfile.TarInfo(prefix + name)
            info.mode = mode
            if name in links:
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


def make_pin(archive, files, key=download.LINUX, prefix=PREFIX, mode="0755"):
    """A pin whose one platform describes archive exactly."""
    data = Path(archive).read_bytes()
    members = [
        {"archive_path": prefix + name, "install_name": name, "size": len(content),
         "sha256": sha256(content), "mode": mode}
        for name, content in files.items()
    ]
    entry = {
        "url": "https://example.invalid/blast-9.9.9.tar.gz", "size": len(data), "md5": hashlib.md5(data).hexdigest(),
        "sha256": sha256(data), "strip_prefix": prefix, "members": members,
        "installed_size": sum(member["size"] for member in members), "requires": {},
    }
    return {"version": "9.9.9", "source": "https://example.invalid/", "platforms": {key: entry}}


class InstallTests(unittest.TestCase):
    key = download.LINUX
    files = UNIX_FILES

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.destination = self.root / "tools"
        self.archive = self.root / "blast.tar.gz"
        build_archive(self.archive, self.files)
        self.pin = make_pin(self.archive, self.files, self.key)
        self.checked = []  # what the start check saw: (folder name, whether the program was there)
        self.blastp = next(name for name in self.files if name.startswith("blastp"))
        self.makeblastdb = next(name for name in self.files if name.startswith("makeblastdb"))

        def query_version(path, timeout=30):
            self.checked.append((Path(path).parent.name, os.path.isfile(path)))
            return "blastp: 9.9.9+"

        self.enterContext(mock.patch.object(blast, "query_version", side_effect=query_version))

    @property
    def entry(self):
        return self.pin["platforms"][self.key]

    def install(self, **options):
        options.setdefault("source", str(self.archive))
        return download.install_blast(self.destination, pin=self.pin, host=self.key, **options)

    def assertNothingLeft(self):
        self.assertEqual(os.listdir(self.destination), [])

    def install_old(self, marker="old.txt"):
        old = self.destination / "blast-9.9.9+"
        old.mkdir(parents=True)
        (old / marker).write_text("the previous install", encoding="utf-8")
        return old

    def test_only_the_listed_files_are_installed_flattened_and_checked(self):
        result = self.install()
        final = self.destination / "blast-9.9.9+"
        self.assertEqual(result, download.BlastInstall(str(final), "blastp: 9.9.9+"))
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(sorted(os.listdir(final)), sorted(self.files))
        for name, data in self.files.items():
            self.assertEqual((final / name).read_bytes(), data)
        self.assertEqual(list(self.root.rglob("escape.txt")), [])
        # blastp was run from the unpacked folder, before it was put in place.
        self.assertEqual(self.checked, [("blast-9.9.9+.partial", True)])

    def test_the_result_is_a_folder_the_lookup_accepts(self):
        result = self.install()
        with mock.patch.object(blast, "is_windows", return_value=self.key == download.WINDOWS), \
                mock.patch.object(blast, "find_automatic", side_effect=AssertionError("fell through")):
            self.assertEqual(blast.find_executables(result.folder).directory, result.folder)

    def test_progress_runs_through_the_stages_to_the_totals(self):
        events = []
        with mock.patch.object(download, "CHUNK_BYTES", 128):
            self.install(progress=lambda done, total, stage: events.append((done, total, stage)))
        stages = [stage for _, _, stage in events]
        self.assertEqual(stages, sorted(stages, key={"download": 0, "extract": 1, "check": 2}.get))
        self.assertEqual(set(stages), {"download", "extract", "check"})
        downloads = [event for event in events if event[2] == "download"]
        self.assertEqual(downloads[0], (0, self.entry["size"], "download"))
        self.assertEqual(downloads[-1], (self.entry["size"], self.entry["size"], "download"))
        self.assertGreater(len(downloads), 3)
        self.assertEqual(
            [event for event in events if event[2] == "extract"][-1],
            (self.entry["installed_size"], self.entry["installed_size"], "extract"),
        )

    def test_the_default_source_is_the_pinned_https_url(self):
        data = self.archive.read_bytes()
        with mock.patch.object(download.urllib.request, "urlopen", return_value=io.BytesIO(data)) as urlopen:
            self.install(source=None)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, self.entry["url"])
        self.assertEqual(urlopen.call_args.kwargs["timeout"], download.CONNECTION_TIMEOUT)

    def test_a_network_failure_is_a_readable_error_and_leaves_nothing(self):
        with mock.patch.object(download.urllib.request, "urlopen", side_effect=urllib.error.URLError("no route")):
            with self.assertRaises(download.BlastDownloadError) as caught:
                self.install(source=None)
        self.assertIn(f"The download from {self.entry['url']} failed", str(caught.exception))
        self.assertIn("no route", str(caught.exception))
        self.assertNothingLeft()

    def test_a_source_that_is_neither_https_nor_a_file_is_refused(self):
        for source in ("http://example.invalid/blast.tar.gz", "ftp://example.invalid/blast.tar.gz",
                       str(self.root / "missing.tar.gz")):
            with self.subTest(source=source):
                with self.assertRaisesRegex(download.BlastDownloadError, "neither an https URL nor a file"):
                    self.install(source=source)
                self.assertNothingLeft()

    # -- refusals ----------------------------------------------------------------

    def test_an_archive_larger_than_the_pin_says_is_refused_while_downloading(self):
        self.entry["size"] -= 1
        with self.assertRaisesRegex(download.BlastDownloadError, "larger than the"):
            self.install()
        self.assertNothingLeft()

    def test_an_archive_shorter_than_the_pin_says_is_refused(self):
        self.entry["size"] += 5
        with self.assertRaisesRegex(download.BlastDownloadError, "ended after"):
            self.install()
        self.assertNothingLeft()

    def test_an_archive_with_the_wrong_sha256_is_refused_and_nothing_is_unpacked(self):
        self.entry["sha256"] = "0" * 64
        with mock.patch.object(download.tarfile, "open", side_effect=AssertionError("unpacked")):
            with self.assertRaisesRegex(download.BlastDownloadError, "failed its SHA-256 check"):
                self.install()
        self.assertNothingLeft()

    def test_a_member_with_the_wrong_sha256_is_refused(self):
        self.entry["members"][1]["sha256"] = "f" * 64
        with self.assertRaisesRegex(download.BlastDownloadError, "makeblastdb\\S* failed its SHA-256 check"):
            self.install()
        self.assertNothingLeft()

    def test_a_member_of_another_size_is_refused(self):
        self.entry["members"][0]["size"] += 1
        self.entry["installed_size"] += 1
        with self.assertRaisesRegex(download.BlastDownloadError, "is not the file the pin file describes"):
            self.install()
        self.assertNothingLeft()

    def test_a_member_that_is_a_link_is_refused(self):
        build_archive(self.archive, self.files, links=(self.blastp,))
        self.pin = make_pin(self.archive, self.files, self.key)
        with self.assertRaisesRegex(download.BlastDownloadError, "is not the file the pin file describes"):
            self.install()
        self.assertNothingLeft()

    def test_a_missing_member_is_refused(self):
        kept = {name: data for name, data in self.files.items() if name != self.makeblastdb}
        build_archive(self.archive, kept)
        self.pin = make_pin(self.archive, kept, self.key)
        self.entry["members"].append(
            {"archive_path": PREFIX + self.makeblastdb, "install_name": self.makeblastdb, "size": 10,
             "sha256": "a" * 64, "mode": "0755"})
        with self.assertRaisesRegex(download.BlastDownloadError, f"does not hold {self.makeblastdb}"):
            self.install()
        self.assertNothingLeft()

    def test_a_damaged_archive_is_a_readable_error(self):
        self.archive.write_bytes(b"not a tar file at all " * 50)
        self.pin = make_pin(self.archive, self.files, self.key)
        with self.assertRaisesRegex(download.BlastDownloadError, "could not be unpacked"):
            self.install()
        self.assertNothingLeft()

    def test_too_little_disk_space_is_refused_before_anything_is_downloaded(self):
        with mock.patch.object(download.shutil, "disk_usage", return_value=mock.Mock(free=10)):
            with self.assertRaisesRegex(download.BlastDownloadError, "not enough free disk space"):
                self.install()
        self.assertNothingLeft()

    def test_a_blank_destination_is_refused(self):
        for blank in ("", "  ", None):
            with self.subTest(blank=blank), self.assertRaisesRegex(download.BlastDownloadError, "Choose a folder"):
                download.install_blast(blank, pin=self.pin, host=self.key, source=str(self.archive))

    def test_a_pin_without_the_hosts_platform_is_refused(self):
        with self.assertRaisesRegex(download.BlastDownloadError, "no download for macos-arm64"):
            download.install_blast(self.destination, pin=self.pin, host=download.MACOS, source=str(self.archive))

    def test_an_unsupported_host_is_refused_with_its_reason(self):
        with mock.patch.object(download, "_detect_platform", return_value=(None, "No build for this computer.")):
            with self.assertRaises(download.BlastDownloadError) as caught:
                download.install_blast(self.destination, pin=self.pin, source=str(self.archive))
        self.assertIn("No build for this computer.", str(caught.exception))
        self.assertIn("BLASTP Directory", str(caught.exception))
        self.assertFalse(self.destination.exists())

    # -- cancel ------------------------------------------------------------------

    def test_a_cancel_while_downloading_leaves_nothing(self):
        cancel = threading.Event()
        events = []

        def progress(done, total, stage):
            events.append((done, stage))
            if stage == "download" and done > 0:
                cancel.set()

        with mock.patch.object(download, "CHUNK_BYTES", 128):
            with self.assertRaises(download.BlastDownloadCancelled) as caught:
                self.install(progress=progress, cancel=cancel)
        self.assertIsInstance(caught.exception, download.BlastDownloadError)
        # It stopped the download itself: the archive was not read to the end first.
        self.assertEqual({stage for _, stage in events}, {"download"})
        self.assertLess(max(done for done, _ in events), self.entry["size"])
        self.assertNothingLeft()

    def test_a_cancel_while_unpacking_leaves_nothing(self):
        cancel = threading.Event()

        def progress(done, total, stage):
            if stage == "extract" and done > 0:
                cancel.set()

        with self.assertRaises(download.BlastDownloadCancelled):
            self.install(progress=progress, cancel=cancel)
        self.assertNothingLeft()

    def test_a_cancel_after_the_check_keeps_the_previous_install(self):
        old = self.install_old()
        cancel = threading.Event()
        with self.assertRaises(download.BlastDownloadCancelled):
            self.install(progress=lambda done, total, stage: cancel.set() if stage == "check" else None,
                         cancel=cancel)
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(os.listdir(old), ["old.txt"])

    def test_an_event_that_is_not_set_changes_nothing(self):
        self.install(cancel=threading.Event())
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])

    # -- the swap ----------------------------------------------------------------

    def test_an_existing_install_is_replaced_and_its_leftover_removed(self):
        self.install_old()
        self.install()
        final = self.destination / "blast-9.9.9+"
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(sorted(os.listdir(final)), sorted(self.files))

    def test_a_failed_swap_puts_the_previous_install_back(self):
        old = self.install_old()
        real_rename = os.rename

        def rename(source, target):
            if str(source).endswith(".partial") and not str(target).endswith((".previous", ".partial")):
                raise PermissionError("a file is in use")
            return real_rename(source, target)

        with mock.patch.object(download.os, "rename", rename):
            with self.assertRaises(download.BlastDownloadError) as caught:
                self.install()
        self.assertIn("could not be put in place", str(caught.exception))
        self.assertIn("a file is in use", str(caught.exception))
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(os.listdir(old), ["old.txt"])

    def test_a_program_that_will_not_start_leaves_the_previous_install_as_it_was(self):
        old = self.install_old()
        blast.query_version.side_effect = blast.BlastError("Unable to query BLASTP version: boom")
        with self.assertRaises(download.BlastDownloadError) as caught:
            self.install()
        self.assertIn("boom", str(caught.exception))
        self.assertIn("Nothing was installed.", str(caught.exception))
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(os.listdir(old), ["old.txt"])

    def test_the_start_check_runs_before_the_swap(self):
        old = self.install_old()
        seen = []
        blast.query_version.side_effect = lambda path, timeout=30: (
            seen.append(os.listdir(old)) or "blastp: 9.9.9+")
        self.install()
        self.assertEqual(seen, [["old.txt"]])

    def test_an_install_that_was_killed_between_its_two_renames_is_put_back_first(self):
        previous = self.destination / "blast-9.9.9+.previous"
        previous.mkdir(parents=True)
        (previous / "old.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(download.BlastDownloadError):
            self.install(source=str(self.root / "missing.tar.gz"))
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(os.listdir(self.destination / "blast-9.9.9+"), ["old.txt"])

    def test_a_leftover_beside_a_complete_install_is_removed(self):
        old = self.install_old()
        previous = self.destination / "blast-9.9.9+.previous"
        previous.mkdir()
        (previous / "older.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(download.BlastDownloadError):
            self.install(source=str(self.root / "missing.tar.gz"))
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])
        self.assertEqual(os.listdir(old), ["old.txt"])

    def test_partial_files_of_an_earlier_attempt_are_replaced(self):
        self.destination.mkdir()
        (self.destination / "blast-9.9.9+.download.partial").write_bytes(b"half")
        (self.destination / "blast-9.9.9+.partial").mkdir()
        (self.destination / "blast-9.9.9+.partial" / "stale").write_bytes(b"x")
        self.install()
        self.assertEqual(os.listdir(self.destination), ["blast-9.9.9+"])

    # -- modes -------------------------------------------------------------------

    def test_files_are_unpacked_with_the_data_filter(self):
        real_extract = tarfile.TarFile.extract
        filters = []

        def spy(tar, member, *args, **kwargs):
            filters.append(kwargs.get("filter"))
            return real_extract(tar, member, *args, **kwargs)

        with mock.patch.object(download.tarfile.TarFile, "extract", spy):
            self.install()
        self.assertEqual(filters, ["data"] * len(self.files))

    def test_the_data_filter_keeps_the_owner_execute_bit_and_clears_group_and_other_write(self):
        # The filter is plain Python, so this holds on every host.
        info = tarfile.TarInfo("blastp")
        info.mode = 0o4777
        kept = tarfile.data_filter(info, str(self.root))
        self.assertEqual(kept.mode & 0o7777, 0o755)

    def test_a_program_stored_without_the_execute_bit_is_given_the_pinned_mode_where_files_have_one(self):
        if self.key == download.WINDOWS:
            self.skipTest("Windows reports an execute bit for every .exe, so there is nothing to give")
        build_archive(self.archive, self.files, mode=0o644)
        self.pin = make_pin(self.archive, self.files, self.key, mode="0755")
        with mock.patch.object(download, "_has_modes", return_value=True), \
                mock.patch.object(download.Path, "chmod", autospec=True) as chmod:
            self.install()
        self.assertEqual(sorted(call.args[0].name for call in chmod.call_args_list), sorted(self.files))
        self.assertEqual({call.args[1] for call in chmod.call_args_list}, {0o755})

    @unittest.skipIf(os.name == "nt", "Windows has no execute bit")
    def test_a_program_that_already_has_the_execute_bit_is_left_alone(self):
        with mock.patch.object(download.Path, "chmod", autospec=True) as chmod:
            self.install()
        chmod.assert_not_called()

    @unittest.skipIf(os.name == "nt", "Windows has no execute bit")
    def test_the_programs_keep_their_owner_execute_bit_and_lose_group_write(self):
        build_archive(self.archive, self.files, mode=0o775)
        self.pin = make_pin(self.archive, self.files, self.key)
        self.install()
        for name in self.files:
            mode = stat.S_IMODE((self.destination / "blast-9.9.9+" / name).stat().st_mode)
            self.assertTrue(mode & stat.S_IXUSR, name)
            self.assertFalse(mode & (stat.S_IWGRP | stat.S_IWOTH), name)

    @unittest.skipIf(os.name == "nt", "Windows has no execute bit")
    def test_a_program_the_archive_stored_without_the_execute_bit_gets_the_pinned_mode(self):
        build_archive(self.archive, self.files, mode=0o644)
        self.pin = make_pin(self.archive, self.files, self.key, mode="0755")
        self.install()
        for name in self.files:
            self.assertTrue(os.access(self.destination / "blast-9.9.9+" / name, os.X_OK), name)


class WindowsInstallTests(InstallTests):
    """The same install for the Windows archive: three files, and its own hint when blastp will not start."""

    key = download.WINDOWS
    files = WINDOWS_FILES

    def test_a_missing_runtime_names_the_visual_cpp_redistributable_and_its_link(self):
        blast.query_version.side_effect = blast.BlastError("Unable to query BLASTP version: exit status 3221225781")
        with self.assertRaises(download.BlastDownloadError) as caught:
            self.install()
        message = str(caught.exception)
        self.assertIn("Visual C++ 2015-2022 Redistributable (x64)", message)
        self.assertIn("https://aka.ms/vs/17/release/vc_redist.x64.exe", message)
        self.assertIn("vcruntime140_1.dll", message)
        self.assertNothingLeft()

    def test_the_dll_is_installed_beside_the_programs(self):
        self.install()
        self.assertIn("nghttp2.dll", os.listdir(self.destination / "blast-9.9.9+"))


class StartHintTests(unittest.TestCase):
    def test_each_platform_says_what_blast_needs_to_start(self):
        self.assertIn("libgomp1", download.START_HINTS[download.LINUX])
        self.assertIn("glibc 2.28", download.START_HINTS[download.LINUX])
        self.assertIn("macOS 14", download.START_HINTS[download.MACOS])
        self.assertEqual(set(download.START_HINTS), set(download.PLATFORMS))


class PlatformTests(unittest.TestCase):
    def detect(self, system, machine):
        with mock.patch.object(download.sys, "platform", system), \
                mock.patch.object(download.platform, "machine", return_value=machine):
            return download.platform_key(), download.unsupported_reason()

    def test_the_three_supported_hosts(self):
        self.assertEqual(self.detect("win32", "AMD64"), (download.WINDOWS, None))
        self.assertEqual(self.detect("linux", "x86_64"), (download.LINUX, None))
        self.assertEqual(self.detect("darwin", "arm64"), (download.MACOS, None))

    def test_every_other_host_has_no_key_and_a_reason(self):
        for system, machine in (("win32", "ARM64"), ("linux", "aarch64"), ("darwin", "x86_64"),
                                ("freebsd14", "amd64"), ("linux", "")):
            with self.subTest(system=system, machine=machine):
                key, reason = self.detect(system, machine)
                self.assertIsNone(key)
                self.assertIn("BLAST+", reason)
                self.assertTrue(reason.endswith("."))

    def test_this_host_is_a_supported_key_or_has_a_reason(self):
        key, reason = download.platform_key(), download.unsupported_reason()
        self.assertTrue((key in download.PLATFORMS) != (reason is not None))


class PinFileTests(unittest.TestCase):
    """src/resources/blast/blast_pin.json, as shipped."""

    @classmethod
    def setUpClass(cls):
        cls.pin = download.read_pin()

    def test_it_pins_one_version_with_its_source_and_licence_and_usage_notes(self):
        self.assertEqual(self.pin["version"], "2.17.0")
        self.assertEqual(self.pin["source"], "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/")
        self.assertIn("PUBLIC DOMAIN", self.pin["license"]["summary"])
        self.assertEqual(self.pin["usage_reporting"]["opt_out_env"], {blast.USAGE_REPORT_VARIABLE: "false"})

    def test_the_install_folder_is_named_for_the_version(self):
        self.assertEqual(download.install_folder_name(self.pin), f"blast-{self.pin['version']}+")
        self.assertEqual(download.install_folder_name(), "blast-2.17.0+")

    def test_the_three_supported_platforms_are_there(self):
        self.assertEqual(set(self.pin["platforms"]), set(download.PLATFORMS))

    def test_every_platform_pins_its_archive_by_size_and_hash(self):
        for key, entry in self.pin["platforms"].items():
            with self.subTest(key=key):
                self.assertTrue(entry["url"].replace("%2B", "+").startswith(self.pin["source"]))
                self.assertTrue(entry["url"].endswith(".tar.gz"))
                self.assertGreater(entry["size"], 0)
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(entry["md5"], r"^[0-9a-f]{32}$")
                self.assertEqual(entry["strip_prefix"], "ncbi-blast-2.17.0+/bin/")

    def test_every_platform_lists_blastp_and_makeblastdb_each_with_size_hash_and_mode(self):
        for key, entry in self.pin["platforms"].items():
            with self.subTest(key=key):
                names = {member["install_name"] for member in entry["members"]}
                suffix = ".exe" if key == download.WINDOWS else ""
                self.assertTrue({"blastp" + suffix, "makeblastdb" + suffix} <= names)
                for member in entry["members"]:
                    self.assertEqual(member["archive_path"], entry["strip_prefix"] + member["install_name"])
                    self.assertGreater(member["size"], 0)
                    self.assertRegex(member["sha256"], r"^[0-9a-f]{64}$")
                    self.assertEqual(member["mode"], "0755")
                self.assertEqual(entry["installed_size"], sum(member["size"] for member in entry["members"]))

    def test_windows_also_ships_nghttp2_and_the_others_ship_only_the_two_programs(self):
        windows = {member["install_name"] for member in self.pin["platforms"][download.WINDOWS]["members"]}
        self.assertEqual(windows, {"blastp.exe", "makeblastdb.exe", "nghttp2.dll"})
        for key in (download.LINUX, download.MACOS):
            names = {member["install_name"] for member in self.pin["platforms"][key]["members"]}
            self.assertEqual(names, {"blastp", "makeblastdb"})

    def test_each_platform_says_what_it_needs_to_run(self):
        for key, entry in self.pin["platforms"].items():
            with self.subTest(key=key):
                self.assertTrue(entry["requires"]["os"])
        self.assertIn("Visual C++", " ".join(self.pin["platforms"][download.WINDOWS]["requires"]["not_in_default_os"]))


class PinValidationTests(unittest.TestCase):
    def setUp(self):
        self.pin = copy.deepcopy(download.read_pin())
        self.windows = self.pin["platforms"][download.WINDOWS]

    def refused(self, reason):
        with self.assertRaises(download.BlastDownloadError) as caught:
            download.check_pin(self.pin)
        self.assertIn("The BLAST+ pin file is not valid", str(caught.exception))
        self.assertIn(reason, str(caught.exception))

    def test_the_shipped_pin_is_valid(self):
        self.assertIs(download.check_pin(self.pin), self.pin)

    def test_each_kind_of_damage_is_refused_with_its_reason(self):
        damage = {
            "'version'": lambda: self.pin.update(version="latest"),
            "'platforms'": lambda: self.pin.update(platforms={}),
            "https URL": lambda: self.windows.update(url="http://ftp.ncbi.nlm.nih.gov/x.tar.gz"),
            "positive 'size'": lambda: self.windows.update(size=0),
            "SHA-256": lambda: self.windows.update(sha256="ABC"),
            "'strip_prefix'": lambda: self.windows.pop("strip_prefix"),
            "lists no members": lambda: self.windows.update(members=[]),
            "not a plain file name": lambda: self.windows["members"][0].update(install_name="..\\evil.exe"),
            "strip_prefix plus its name": lambda: self.windows["members"][0].update(archive_path="elsewhere/blastp.exe"),
            "positive size": lambda: self.windows["members"][0].update(size=-1),
            "octal mode": lambda: self.windows["members"][0].update(mode="rwx"),
            "lists a file twice": lambda: self.windows["members"].append(dict(self.windows["members"][0])),
            "does not include blastp and makeblastdb": lambda: self.windows["members"].pop(0),
        }
        for reason, break_it in damage.items():
            with self.subTest(reason=reason):
                self.pin = copy.deepcopy(download.read_pin())
                self.windows = self.pin["platforms"][download.WINDOWS]
                break_it()
                self.refused(reason)

    def test_a_pin_that_is_not_an_object_is_refused(self):
        self.pin = ["not", "an", "object"]
        self.refused("not a JSON object")

    def test_an_unreadable_or_invalid_file_is_a_readable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.json"
            broken = Path(folder) / "broken.json"
            broken.write_text("{ not json", encoding="utf-8")
            for path in (missing, broken):
                with self.subTest(path=path.name), self.assertRaises(download.BlastDownloadError) as caught:
                    download.read_pin(path)
                self.assertIn(str(path), str(caught.exception))
            valid = Path(folder) / "valid.json"
            valid.write_text(json.dumps(download.read_pin()), encoding="utf-8")
            self.assertEqual(download.read_pin(valid), download.read_pin())


if __name__ == "__main__":
    unittest.main()
