"""Output names stay in the folder the command writes to, whoever issues it."""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Metadata_Core  # noqa: E402
from commands import export as export_command, logo, meta as meta_command  # noqa: E402
from utilities import Output_Names  # noqa: E402
from utilities.Output_Names import (  # noqa: E402
    first_case_collision,
    folder_ignores_case,
    new_partial_file,
    validate_output_basename,
)
from utilities.Sequence_Utils import write_fasta_atomic  # noqa: E402
from tests.command_fixtures import reported_outcomes  # noqa: E402
from tests.folder_case_fixtures import ignores_case, make_case_sensitive  # noqa: E402


class OutputBasenameTests(unittest.TestCase):
    def test_plain_names_are_returned_without_surrounding_space(self):
        for name, expected in [
            ("hits", "hits"),
            (" hits.txt ", "hits.txt"),
            ("my traits.csv", "my traits.csv"),
            (".hidden", ".hidden"),
            ("...", "..."),
        ]:
            with self.subTest(name=name):
                self.assertEqual(validate_output_basename(name), expected)

    def test_paths_are_refused(self):
        # os.path.join(folder, name) drops the folder for each of these, and on
        # Windows os.path.isabs misses "C:x" and "\x" altogether.
        for name in [
            r"C:\Users\Public\x.csv",
            r"\x.csv",
            r"\\host\share\x.csv",
            "/tmp/x.csv",
            r"..\x.csv",
            "../x.csv",
            r"sub\x.csv",
            "sub/x.csv",
            ".",
            "..",
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "must not include a directory or path separators"
            ):
                validate_output_basename(name)

    def test_drive_and_stream_names_are_refused(self):
        # "C:x" is relative to drive C's current directory; "x.txt:hidden"
        # writes an NTFS alternate data stream inside x.txt.
        for name in ["C:x.csv", "x.txt:hidden", "a<b", "a>b", 'a"b', "a|b",
                     "a?b", "a*b", "a\x00b", "a\tb"]:
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "unsupported characters"
            ):
                validate_output_basename(name)

    def test_empty_names_are_refused(self):
        for name in ("", "   "):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "cannot be empty"
            ):
                validate_output_basename(name)

    def test_logo_keeps_its_extension_rule(self):
        self.assertEqual(logo._normalize_logo_filename("motif"), "motif.svg")
        self.assertEqual(logo._normalize_logo_filename("motif.PNG"), "motif.PNG")
        with self.assertRaisesRegex(ValueError, "path separators"):
            logo._normalize_logo_filename(r"..\outside.svg")


class CaseCollisionTests(unittest.TestCase):
    def test_the_first_pair_differing_only_in_case_is_returned_in_order(self):
        self.assertEqual(
            first_case_collision(["beta.fasta", "Alpha.fasta", "ALPHA.fasta", "alpha.fasta"]),
            ("Alpha.fasta", "ALPHA.fasta"),
        )

    def test_names_that_differ_beyond_case_do_not_collide(self):
        self.assertIsNone(first_case_collision(["alpha.fasta", "beta.fasta", "alpha.fa"]))
        self.assertIsNone(first_case_collision([]))


class FolderCaseTests(unittest.TestCase):
    """folder_ignores_case asks the file system, not the OS name."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name

    def on_the_other_platform(self, folder_ignores_case):
        """Pose as an OS whose default is the opposite of this folder's answer,
        so a probe that falls back on the platform gets it wrong."""
        os_name, platform = ("posix", "linux") if folder_ignores_case else ("nt", "win32")
        return mock.patch.object(Output_Names.os, "name", os_name),             mock.patch.object(Output_Names.sys, "platform", platform)

    def test_probe_answers_for_the_folder_not_the_os_and_leaves_nothing_behind(self):
        expected = ignores_case(self.root)
        probed = os.path.join(self.root, "probed")
        os.mkdir(probed)

        os_name, platform = self.on_the_other_platform(expected)
        with os_name, platform:
            self.assertIs(folder_ignores_case(probed), expected)
        self.assertEqual(os.listdir(probed), [])

    def test_a_case_sensitive_folder_is_told_apart(self):
        # Windows' and macOS's usual folders ignore case; this one doesn't.
        make_case_sensitive(self, self.root)
        self.assertIs(folder_ignores_case(self.root), False)
        self.assertEqual(os.listdir(self.root), [])

    def test_a_folder_not_made_yet_is_answered_by_its_nearest_existing_folder(self):
        expected = ignores_case(self.root)
        existing = os.path.join(self.root, "existing")
        os.mkdir(existing)

        not_made = os.path.join(existing, "source_[test]", "Score0.5_GROUPS")
        os_name, platform = self.on_the_other_platform(expected)
        with os_name, platform:
            self.assertIs(folder_ignores_case(not_made), expected)
        self.assertEqual(os.listdir(existing), [])

    def test_a_folder_that_cannot_be_probed_gets_its_platform_default(self):
        for os_name, platform, expected in (
            ("nt", "win32", True),
            ("posix", "darwin", True),
            ("posix", "linux", False),
        ):
            with self.subTest(platform=platform),                     mock.patch.object(Output_Names.tempfile, "mkstemp",
                                      side_effect=PermissionError("read-only")),                     mock.patch.object(Output_Names.os, "name", os_name),                     mock.patch.object(Output_Names.sys, "platform", platform):
                self.assertIs(folder_ignores_case(self.root), expected)


class MetadataDownloadPathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.meta_dir = directory.name

    def test_requested_names_stay_in_the_metadata_directory(self):
        for name, expected in [
            ("traits", "traits.csv"),
            ("traits.CSV", "traits.CSV"),
            ("traits.xlsx", "traits.xlsx"),
            ("traits.XLSX", "traits.XLSX"),
            ("my traits.csv", "my traits.csv"),
        ]:
            with self.subTest(name=name):
                self.assertEqual(
                    Metadata_Core.metadata_download_path(self.meta_dir, name),
                    os.path.join(self.meta_dir, expected),
                )

    def test_other_extensions_and_paths_are_refused(self):
        for name, message in [
            ("traits.xls", "only be downloaded as .csv or .xlsx, not '.xls'"),
            ("traits.xlsm", "only be downloaded as .csv or .xlsx"),
            ("traits.txt", "only be downloaded as .csv or .xlsx"),
            ("traits.", "only be downloaded as .csv or .xlsx"),
            (r"..\traits.csv", "path separators"),
            ("C:traits.csv", "unsupported characters"),
        ]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                Metadata_Core.metadata_download_path(self.meta_dir, name)

    def test_automatic_names_skip_existing_files(self):
        first = Metadata_Core.metadata_download_path(self.meta_dir)
        self.assertEqual(first, os.path.abspath(os.path.join(self.meta_dir, "metadata.csv")))
        Path(first).write_text("taken", encoding="utf-8")
        self.assertEqual(
            Metadata_Core.metadata_download_path(self.meta_dir, ""),
            os.path.abspath(os.path.join(self.meta_dir, "metadata1.csv")),
        )


class MetaDownloadCommandTests(unittest.TestCase):
    """The desktop `meta download` command, as the console, agent and MCP run it."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.meta_dir = self.root / "Meta_Data"
        self.outside = self.root / "outside"
        self.outside.mkdir()
        patcher = mock.patch.object(
            meta_command.cfg, "METADATA_DIR", str(self.meta_dir), create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = SimpleNamespace(
            n_nodes=2,
            full_headers=["node-1", "node-2"],
            metadata={"Score": {"type": "number", "values": np.array([1.0, 2.5])}},
        )

    def download(self, *names):
        engine = meta_command.Command_Engine
        with mock.patch.object(engine, "print_help"), \
                mock.patch.object(engine, "command_artifact"), \
                reported_outcomes() as (succeeded, failed):
            meta_command.run(self.viewer, ["download", *names])
        return succeeded, failed

    def test_a_path_is_refused_and_nothing_is_written(self):
        # os.path.join(METADATA_DIR, name) used to drop the directory for an
        # absolute name, and "..\" climbed out of it.
        for name in (str(self.outside / "escaped.csv"), r"..\escaped.csv",
                     "../escaped.csv"):
            with self.subTest(name=name):
                succeeded, failed = self.download(name)
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn("path separators", failed.call_args.args[1])
        self.assertEqual(list(self.outside.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["Meta_Data", "outside"])
        self.assertEqual(list(self.meta_dir.iterdir()), [])

    def test_names_with_spaces_and_upper_case_extensions_are_written(self):
        # "book.XLSX" used to fail inside pandas: "No engine for filetype: 'XLSX'".
        for names, written in [(("my", "traits"), "my traits.csv"),
                               (("upper.CSV",), "upper.CSV"),
                               (("book.XLSX",), "book.XLSX")]:
            with self.subTest(names=names):
                succeeded, failed = self.download(*names)
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertTrue((self.meta_dir / written).is_file())
        workbook = pd.read_excel(self.meta_dir / "book.XLSX", header=None, engine="openpyxl")
        self.assertEqual(workbook.iloc[2:, 0].tolist(), ["node-1", "node-2"])
        self.assertEqual(workbook.iloc[2:, 1].tolist(), [1, 2.5])

    def test_other_extensions_are_refused_before_writing(self):
        for name in ("traits.xls", "traits.xlsm", "traits.txt"):
            with self.subTest(name=name):
                succeeded, failed = self.download(name)
                succeeded.assert_not_called()
                self.assertIn("only be downloaded as .csv or .xlsx", failed.call_args.args[1])
        self.assertEqual(list(self.meta_dir.iterdir()), [])


class ExportCacheNameTests(unittest.TestCase):
    """`export` names files after group labels and a folder after the clustering
    parameters. The Viewer's cache loader restores both as stored
    (``[set(g) for g in json.loads(...)]`` and ``tuple(json.loads(...))``), so a
    hand-made layout cache can turn either into a path."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.outside = self.root / "outside"
        self.outside.mkdir()
        cfg = export_command.cfg
        metadata = SimpleNamespace(model_name="test", network_type="blast")
        for patcher in (
            mock.patch.object(cfg, "NODE_FASTA_FILE", str(self.root / "source.fasta")),
            mock.patch.object(cfg, "INPUT_HDF5", "network.h5"),
            mock.patch.object(cfg, "TOP_EDGE_PERCENT", None),
            mock.patch.object(cfg, "SIMILARITY_THRESHOLD", 0.5),
            mock.patch.object(export_command, "SEQUENCE_EXPORT_DIRECTORY",
                              str(self.root / "exports")),
            mock.patch.object(export_command.cache_manifest, "validate_network_schema",
                              return_value=metadata),
            # A successful export opens its folder in the system file manager.
            mock.patch.object(export_command, "open_in_file_manager"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_viewer(self, groups, cluster_params=None):
        """One node per entry of `groups`, every node in cluster 0."""
        headers = [f"node_{index}" for index in range(len(groups))]
        return SimpleNamespace(
            n_nodes=len(headers),
            full_headers=headers,
            cluster_labels=np.zeros(len(headers), dtype=int),
            group_labels=[set(labels) for labels in groups],
            last_cluster_params=cluster_params,
            _selected_fasta_records=[
                (header, "ACDEFGHIK"[: index + 2]) for index, header in enumerate(headers)
            ],
            console_text=SimpleNamespace(text=""),
        )

    def export(self, viewer, *args):
        engine = export_command.Command_Engine
        with mock.patch.object(engine, "command_artifact"), \
                reported_outcomes() as (succeeded, failed), \
                redirect_stdout(io.StringIO()):
            export_command.run(viewer, list(args))
        return succeeded, failed

    def written(self):
        """Every file under the test folder, relative to it."""
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*") if path.is_file()
        )

    def assert_refused(self, viewer, succeeded, failed, refused):
        # Refused before anything was created, in the export folder or outside it.
        self.assertEqual(self.written(), [])
        self.assertEqual([path.name for path in self.root.iterdir()], ["outside"])
        succeeded.assert_not_called()
        failed.assert_called_once()
        message = failed.call_args.args[1]
        self.assertTrue(message.startswith(f"Error: Export refused {refused}: "), message)
        self.assertIn("path separators", message)
        self.assertEqual(viewer.console_text.text, message)

    def test_group_labels_that_are_paths_are_refused(self):
        for label in (
            str(self.outside / "escaped"),
            os.path.join("..", "..", "..", "outside", "escaped"),
            r"..\escaped",
        ):
            # The plain label is mapped first and must not be written either.
            for args in (["groups"], ["#alpha#", f"#{label}#"]):
                with self.subTest(label=label, args=args):
                    viewer = self.make_viewer([{"alpha"}, {label}])
                    succeeded, failed = self.export(viewer, *args)
                    self.assert_refused(viewer, succeeded, failed, f"group label '{label}'")

    def test_clustering_parameters_that_are_paths_are_refused(self):
        # The folder name starts "Score0.5_LEIDEN_1.0_Min", which absorbs one '..'.
        climb = os.path.join("..", "..", "..", "..", "outside", "escaped")
        for threshold, params, args in (
            # A UMAP layout has neither a threshold nor a top-edge percentage, so
            # the parameters alone name the folder, and an absolute one replaces
            # the export folder altogether.
            (None, (str(self.outside / "escaped"), 20), ["clusters"]),
            (0.5, ("LEIDEN_1.0", climb), ["clusters"]),
            (0.5, ("LEIDEN_1.0", climb), ["#cluster_0#"]),
        ):
            with self.subTest(params=params, args=args), mock.patch.object(
                export_command.cfg, "SIMILARITY_THRESHOLD", threshold
            ):
                viewer = self.make_viewer([set(), set()], cluster_params=params)
                succeeded, failed = self.export(viewer, *args)
                self.assert_refused(
                    viewer, succeeded, failed,
                    f"clustering parameters ({params[0]}, {params[1]})",
                )

    def test_plain_group_labels_are_exported_as_before(self):
        # The group command accepts '.' and '..'; their files are the plain
        # names '..fasta' and '...fasta'. Clustering parameters name only
        # cluster folders, so a group export never uses them.
        viewer = self.make_viewer(
            [{"alpha"}, {".."}, {"."}, {"alpha", ".."}],
            cluster_params=(str(self.outside / "escaped"), 20),
        )
        succeeded, failed = self.export(viewer, "groups")
        failed.assert_not_called()
        succeeded.assert_called_once()
        folder = "exports/source_[test]/Score0.5_GROUPS"
        self.assertEqual(
            self.written(),
            sorted(f"{folder}/{name}" for name in ("alpha.fasta", "...fasta", "..fasta")),
        )
        self.assertEqual(
            (self.root / folder / "...fasta").read_text(encoding="utf-8"),
            ">node_1\nACD\n>node_3\nACDEF\n",
        )

    def test_plain_clustering_parameters_name_the_folder_as_before(self):
        for threshold, folder in ((0.5, "Score0.5_LEIDEN_1.0_Min20"),
                                  (None, "LEIDEN_1.0_Min20")):
            with self.subTest(threshold=threshold), mock.patch.object(
                export_command.cfg, "SIMILARITY_THRESHOLD", threshold
            ):
                viewer = self.make_viewer([set(), set()], cluster_params=("LEIDEN_1.0", 20))
                succeeded, failed = self.export(viewer, "clusters")
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertIn(
                    f"exports/source_[test]/{folder}/Cluster_0.fasta", self.written()
                )


@unittest.skipIf(sys.platform == "win32", "Windows has no file mode bits for the umask to decide")
class PartialFileModeTests(unittest.TestCase):
    """A partial file, and so the output os.replace makes of it, is made as a saved file is: the umask decides.

    tempfile's files are for their owner alone (0600), which made logos, label workbooks and exported FASTA
    files private while a printed PNG beside them was not.
    """

    def setUp(self):
        self.previous_umask = os.umask(0o027)
        self.addCleanup(os.umask, self.previous_umask)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = directory.name

    def test_the_partial_file_is_new_empty_hidden_and_follows_the_umask(self):
        first = new_partial_file(self.folder, ".logo.", ".partial.svg")
        second = new_partial_file(self.folder, ".logo.", ".partial.svg")
        self.assertNotEqual(first, second)
        for path in (first, second):
            name = os.path.basename(path)
            self.assertTrue(name.startswith(".logo.") and name.endswith(".partial.svg"), name)
            self.assertEqual(os.path.getsize(path), 0)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o640)

    def test_an_exported_fasta_follows_the_umask(self):
        path = write_fasta_atomic(os.path.join(self.folder, "Cluster_1.fasta"), ["a", "b"], ["MK", "MV"])
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o640)
        self.assertEqual(Path(path).read_text(encoding="utf-8"), ">a\nMK\n>b\nMV\n")
        self.assertEqual(os.listdir(self.folder), ["Cluster_1.fasta"])


if __name__ == "__main__":
    unittest.main()
