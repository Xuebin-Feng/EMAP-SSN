# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's per-residue layers.

esmfold writes numbers for each residue beside a structure's PDB
(<name>.residues.json), and the page can colour the structure by any of them.
These tests cover the three parts of that:

* web_ui/residue_layers.py reads the file and makes the layer listing and the
  MolViewSpec annotation of a layer;
* the Viewer's /api/esmfold/layers/ route serves both, under the same-origin
  and containment rules of every other route;
* the files the real writer makes: esmfold's own records (commands/esmfold.py),
  the Residue_Data providers over an alignment the production loader read, and
  the worker's run_predictions with a stand-in model, nothing folded, read back
  through the route;
* esmfold.html, in Qt WebEngine over the real server and the real bundled
  Mol*, attaches a layer to a structure it has already loaded and takes it off
  again: pLDDT stays the default and comes back exactly, each structure has
  its own layer, the superposition and a restored session are untouched, and
  a hovered residue shows its tooltip.

The structures are a small helix drawn by tests/esmfold_layer_fixtures.py, and
the residue data files are written by hand in the format esmfold writes. The
offscreen engine has no WebGL, so the page is checked through Mol*'s state:
the colours the colour theme gives each residue and the labels it gives a
hovered one, not pixels.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Set before Qt WebEngine starts Chromium, as tests/test_agent_composer.py does.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtWidgets import QApplication

from commands import esmfold as esmfold_command
from resources.esmfold import esmfold_worker
from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_page_harness import FoldViewPageMixin, http_get, make_viewer, page_headers
from tests.sparse_alignment import load_manager, write_fasta
from tests.test_esmfold_worker import FakeClient, FakeOutput, fake_esm_modules
from tests.translation_fixtures import pseudo_language
from web_ui import residue_layers
from web_ui.residue_layers import (
    COLOR_SCALES,
    LAYER_STYLES,
    ResidueDataError,
    layer_annotation,
    layer_listing,
    read_residue_data,
    scale_color,
)

FIRST = "protein_a.pdb"
SECOND = "protein_b.pdb"
THIRD = "protein_c.pdb"
SCORE = "fixture_score"


def score_property(values, *, label="Fixture Score", unit="", value_range=(0, 1), source=None):
    """A property of my own: unitless by default, from 0 to 1, with no value for some residues."""
    return fixtures.residue_property(label, unit, value_range, values, source)


class ResidueLayerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.outer = folder.name
        # The folder esmfold writes to, with a folder around it that nothing may reach.
        self.directory = os.path.join(self.outer, "structures")
        os.makedirs(self.directory)

    def write(self, name=FIRST, sequence=fixtures.FIRST_SEQUENCE, **options):
        return fixtures.write_structure(self.directory, name, sequence, **options)

    def sidecar_path(self, name=FIRST):
        return os.path.join(self.directory, name[:-len(".pdb")] + ".residues.json")

    def structure_path(self, name=FIRST):
        return os.path.join(self.directory, name)

    def listing(self, name=FIRST):
        return layer_listing(name, self.structure_path(name), self.sidecar_path(name))

    def data(self, name=FIRST):
        return read_residue_data(name, self.structure_path(name), self.sidecar_path(name))

    def layer(self, listing, layer_id):
        return next(entry for entry in listing["layers"] if entry["id"] == layer_id)


class ResidueDataTests(ResidueLayerTestCase):
    def test_every_property_of_the_file_is_a_layer_in_file_order_after_plddt(self):
        self.write()
        listing = self.listing()
        self.assertEqual([entry["id"] for entry in listing["layers"]],
                         ["plddt", "information_content", "same_residue_pct"])
        self.assertEqual(listing["unavailable"], "")
        information = self.layer(listing, "information_content")
        self.assertEqual((information["label"], information["unit"], information["range"]),
                         ("Information Content", "%", [0, 100]))
        # The writer leaves no residue out of information content, and the non-standard ones out of agreement.
        self.assertEqual((information["residues"], information["covered"]), (33, 33))
        agreement = self.layer(listing, "same_residue_pct")
        self.assertEqual((agreement["residues"], agreement["covered"]), (33, 32))
        self.assertEqual(information["source"], "From fixture_alignment.fasta, computed when folded, using 12 sequences")
        self.assertTrue(information["description"])
        self.assertTrue(information["annotation"].startswith(
            "/api/esmfold/layers/protein_a.pdb/information_content?v="))

    def test_a_property_nothing_names_in_this_module_is_offered_in_file_order(self):
        # No LAYER_STYLES entry, no page code: the file alone adds a layer.
        properties = {
            "zeta_score": score_property([0.1] * 33, label="Zeta"),
            "alpha_score": score_property([0.9] * 33, label="Alpha"),
        }
        self.write(properties=properties)
        self.assertEqual([entry["id"] for entry in self.listing()["layers"]], ["plddt", "zeta_score", "alpha_score"])
        self.assertNotIn("zeta_score", LAYER_STYLES)

    def test_plddt_alone_is_offered_with_the_reason_when_a_structure_has_no_usable_data(self):
        self.write(name=FIRST, sidecar=False)
        cases = {
            "no file": ("No per-residue data was saved with this structure.", None),
            "not json": ("cannot be read", lambda path: Path(path).write_text("{", encoding="utf-8")),
            "another format": ("not in a known format", {"format": "other"}),
            "a newer version": ("unsupported version", {"version": 2}),
            "another structure": ("belongs to another structure", {"structure": "other.pdb"}),
            "no numbering": ("does not say how the residues are numbered", {"residue_numbering": {"chain": "A"}}),
            "no sequence": ("does not hold the sequence", {"sequence": None}),
            "another protein": ("different protein", {"sequence": fixtures.SECOND_SEQUENCE}),
            "a shorter protein": ("different protein", {"sequence": fixtures.FIRST_SEQUENCE[:-1]}),
            "another chain": ("different protein", {"residue_numbering": {"chain": "B", "first": 1}}),
            "other residue numbers": ("not numbered as", {"residue_numbering": {"chain": "A", "first": 5}}),
            "no properties": ("holds no usable property", {"properties": {}}),
            "only unusable ones": ("holds no usable property", {"properties": {
                "all_null": score_property([None] * 33), "not_a_list": {"values": 5},
                "plddt": score_property([0.5] * 33), "bad id": score_property([0.5] * 33)}}),
        }
        for name, (reason, damage) in cases.items():
            with self.subTest(name):
                if isinstance(damage, dict):
                    fixtures.write_sidecar(self.directory, FIRST, {
                        **fixtures.residue_data(FIRST, fixtures.FIRST_SEQUENCE), **damage})
                elif damage is not None:
                    damage(self.sidecar_path())
                listing = self.listing()
                self.assertEqual([entry["id"] for entry in listing["layers"]], ["plddt"])
                self.assertIn(reason, listing["unavailable"])
                self.assertEqual(listing["layers"][0]["legend"]["kind"], "bands")

    def test_what_cannot_be_compared_and_a_chain_the_file_is_not_about_do_not_refuse_it(self):
        # The PDB has UNK at residue 4 and the file an X at residue 7, in lower case: neither can be compared.
        pdb_sequence = fixtures.FIRST_SEQUENCE[:3] + "X" + fixtures.FIRST_SEQUENCE[4:]
        file_sequence = fixtures.FIRST_SEQUENCE[:6] + "X" + fixtures.FIRST_SEQUENCE[7:]
        fixtures.write_structure(self.directory, FIRST, pdb_sequence, sidecar=False)
        fixtures.write_sidecar(self.directory, FIRST, fixtures.residue_data(FIRST, file_sequence.lower()))
        self.assertEqual([entry["id"] for entry in self.listing()["layers"]],
                         ["plddt", "information_content", "same_residue_pct"])
        # Another chain in the PDB is not what the file is about.
        both = fixtures.helix_pdb(pdb_sequence, fixtures.plddt_values(33)).replace("TER\nEND\n", "")
        both += fixtures.helix_pdb(fixtures.SECOND_SEQUENCE, fixtures.plddt_values(33), chain="B")
        Path(self.structure_path()).write_text(both, encoding="utf-8")
        self.assertEqual(len(self.listing()["layers"]), 3)

    def test_a_file_too_large_to_read_is_refused(self):
        self.write()
        with mock.patch.object(residue_layers, "MAX_SIDECAR_BYTES", 100):
            with self.assertRaisesRegex(ResidueDataError, "too large"):
                self.data()

    def test_a_property_for_another_number_of_residues_than_the_sequence_is_left_out(self):
        properties = {"short": score_property([0.5] * 30), "whole": score_property([0.5] * 33)}
        self.write(properties=properties)
        self.assertEqual(list(self.data().properties), ["whole"])

    def test_the_annotation_selects_residues_by_the_pdbs_chain_and_numbering(self):
        self.write(chain="B", first=17)
        rows = layer_annotation(self.data(), "same_residue_pct")
        # Residue 4 of the structure has no value, so it has no row.
        self.assertEqual(len(rows), 32)
        self.assertEqual([row["auth_seq_id"] for row in rows[:4]], [17, 18, 19, 21])
        self.assertEqual({row["auth_asym_id"] for row in rows}, {"B"})
        for row in rows:
            self.assertEqual(set(row), {"auth_asym_id", "auth_seq_id", "color", "value", "tooltip"})
            self.assertRegex(row["color"], r"^#[0-9a-f]{6}$")
        # The rows name numbers the PDB itself carries.
        pdb = fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), chain="B", first=17)
        residues = {(line[21], int(line[22:26])) for line in pdb.splitlines() if line.startswith("ATOM")}
        self.assertLessEqual({(row["auth_asym_id"], row["auth_seq_id"]) for row in rows}, residues)

    def test_the_tooltip_names_the_value_its_unit_and_the_one_based_alignment_column(self):
        self.write()
        rows = {row["auth_seq_id"]: row for row in layer_annotation(self.data(), "information_content")}
        # Residue 1 sits in 0-based column 100 and residue 2 in column 102.
        self.assertEqual(rows[1]["tooltip"], "Information Content 0.5 % (MSA column 101)")
        self.assertEqual(rows[2]["tooltip"], "Information Content 7.5 % (MSA column 103)")
        self.assertEqual(rows[1]["value"], 0.5)

    def test_a_residue_whose_column_is_null_has_a_tooltip_without_one(self):
        columns = [None if index == 1 else 100 + index for index in range(33)]
        fixtures.write_structure(self.directory, FIRST, fixtures.FIRST_SEQUENCE, columns=columns)
        rows = {row["auth_seq_id"]: row for row in layer_annotation(self.data(), "information_content")}
        self.assertEqual(rows[1]["tooltip"], "Information Content 0.5 % (MSA column 101)")
        self.assertEqual(rows[2]["tooltip"], "Information Content 7.5 %")

    def test_colours_run_along_the_scale_across_the_range_the_file_gives(self):
        properties = {
            "narrow": score_property([-2.0, 0.0, 2.0, 5.0, -9.0], value_range=(-2, 2)),
            "plain": fixtures.residue_property("Plain", "%", (0, 100), [0, 50, 100, None, 25]),
        }
        self.write(sequence="MKTAY", properties=properties, plddt=[50] * 5)
        data = self.data()
        scale = COLOR_SCALES[residue_layers.DEFAULT_SCALE]
        colors = [row["color"] for row in layer_annotation(data, "narrow")]
        # The low end, the middle and the high end of the range; values outside it clamp.
        self.assertEqual(colors[:3], [scale[0], scale[2], scale[-1]])
        self.assertEqual((colors[3], colors[4]), (scale[-1], scale[0]))
        plain = layer_annotation(data, "plain")
        self.assertEqual([row["auth_seq_id"] for row in plain], [1, 2, 3, 5])
        self.assertEqual(plain[0]["color"], scale[0])
        self.assertEqual(plain[3]["color"], scale[1])

    def test_scale_colour_interpolates_between_its_stops_and_clamps(self):
        scale = COLOR_SCALES["blue-red"]
        self.assertEqual(scale_color("blue-red", -1), scale[0])
        self.assertEqual(scale_color("blue-red", 2), scale[-1])
        self.assertEqual(scale_color("blue-red", 0.5), scale[2])
        self.assertEqual(scale_color("blue-red", 0.125), "#5a96c5")

    def test_a_layer_style_picks_a_scale_for_a_property_id_and_nothing_else_is_needed(self):
        self.write(properties={"hydrophobicity": score_property([0.0, 1.0] * 16 + [0.5])})
        before = layer_annotation(self.data(), "hydrophobicity")
        self.assertEqual(before[0]["color"], COLOR_SCALES[residue_layers.DEFAULT_SCALE][0])
        with mock.patch.dict(LAYER_STYLES, {"hydrophobicity": {"scale": "blue-red"}}):
            after = layer_annotation(self.data(), "hydrophobicity")
            legend = self.layer(self.listing(), "hydrophobicity")["legend"]
        self.assertEqual((after[0]["color"], after[1]["color"]),
                         (COLOR_SCALES["blue-red"][0], COLOR_SCALES["blue-red"][-1]))
        self.assertEqual([stop[1] for stop in legend["stops"]], list(COLOR_SCALES["blue-red"]))
        # A style naming a scale that does not exist falls back to the default.
        with mock.patch.dict(LAYER_STYLES, {"hydrophobicity": {"scale": "nonexistent"}}):
            self.assertEqual(layer_annotation(self.data(), "hydrophobicity"), before)

    def test_a_unitless_property_has_no_unit_in_its_tooltip_or_legend_and_nulls_are_no_data(self):
        values = [0.82, None, 0.5, float("nan"), True, "x", 1.0] + [0.25] * 26
        self.write(properties={SCORE: score_property(values)})
        listing = self.listing()
        layer = self.layer(listing, SCORE)
        self.assertEqual((layer["unit"], layer["range"], layer["residues"]), ("", [0, 1], 33))
        # Null, NaN, a boolean and a string are no data.
        self.assertEqual(layer["covered"], 29)
        self.assertEqual(listing["background"], residue_layers.BACKGROUND_COLOR)
        self.assertEqual(listing["texts"]["no_data"], "No data")
        rows = layer_annotation(self.data(), SCORE)
        self.assertEqual([row["auth_seq_id"] for row in rows[:3]], [1, 3, 7])
        self.assertEqual(rows[0]["tooltip"], "Fixture Score 0.82")
        self.assertEqual(rows[1]["tooltip"], "Fixture Score 0.5")
        self.assertEqual(layer["legend"]["stops"][0][0], 0)
        self.assertEqual(layer["legend"]["stops"][-1][0], 1)

    def test_a_range_the_file_does_not_give_is_the_range_of_its_values(self):
        entry = score_property([2.0, 4.0, None, 3.0])
        del entry["range"]
        self.write(sequence="MKTA", properties={SCORE: entry}, plddt=[50] * 4)
        layer = self.layer(self.listing(), SCORE)
        self.assertEqual(layer["range"], [2.0, 4.0])
        self.assertEqual(layer["legend"]["min"], 2.0)

    def test_the_source_line_says_where_the_numbers_came_from(self):
        properties = {
            "with_count": score_property([0.5] * 33, source={
                "kind": "msa", "file": "family.fasta", "alignment_id": "x", "columns": None, "sequences": 1}),
            "no_count": score_property([0.5] * 33, source={"kind": "msa", "file": "family.fasta"}),
            "no_file": score_property([0.5] * 33, source={"kind": "embedding"}),
            "no_source": score_property([0.5] * 33),
        }
        self.write(properties=properties)
        listing = self.listing()
        self.assertEqual(self.layer(listing, "with_count")["source"],
                         "From family.fasta, computed when folded, using 1 sequence")
        self.assertEqual(self.layer(listing, "no_count")["source"], "From family.fasta, computed when folded")
        self.assertEqual(self.layer(listing, "no_file")["source"], "")
        self.assertEqual(self.layer(listing, "no_source")["source"], "")
        # Without alignment columns the tooltip has no column.
        self.assertEqual(layer_annotation(self.data(), "with_count")[0]["tooltip"], "Fixture Score 0.5")

    def test_a_layers_address_changes_when_the_file_does(self):
        self.write()
        first = self.layer(self.listing(), "same_residue_pct")["annotation"]
        self.assertEqual(first, self.layer(self.listing(), "same_residue_pct")["annotation"])
        path = Path(self.sidecar_path())
        data = json.loads(path.read_text(encoding="utf-8"))
        data["properties"]["same_residue_pct"]["values"][0] = 99.0
        path.write_text(json.dumps(data), encoding="utf-8")
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        second = self.layer(self.listing(), "same_residue_pct")["annotation"]
        self.assertNotEqual(first, second)
        self.assertEqual(layer_annotation(self.data(), "same_residue_pct")[0]["value"], 99.0)

    def test_a_layer_the_file_does_not_have_has_no_annotation(self):
        self.write()
        self.assertIsNone(layer_annotation(self.data(), "plddt"))
        self.assertIsNone(layer_annotation(self.data(), "missing"))

    def test_the_names_a_sidecar_and_a_pdb_give_each_other(self):
        self.assertEqual(residue_layers.sidecar_name("XP_007399765.1.pdb"), "XP_007399765.1.residues.json")
        self.assertEqual(residue_layers.sidecar_name("..x.PDB"), "..x.residues.json")
        for name in ("model.cif", "model", "", None, 5):
            self.assertIsNone(residue_layers.sidecar_name(name))

    def test_every_text_it_makes_comes_in_the_viewers_language(self):
        self.write()
        english = self.listing()
        english_rows = layer_annotation(self.data(), "information_content")
        pseudo_language(self, self.app)
        listing = self.listing()
        rows = layer_annotation(self.data(), "information_content")
        information = self.layer(listing, "information_content")
        plddt = listing["layers"][0]
        data = ("Information Content", "fixture_alignment.fasta", "12", "pLDDT")
        shown = [listing["texts"]["color_by"], listing["texts"]["no_data"], plddt["description"],
                 information["source"], rows[0]["tooltip"]] + [band["label"] for band in plddt["legend"]["bands"]]
        for text in shown:
            with self.subTest(text=text):
                self.assertNotEqual(text, "")
                remaining = re.sub("|".join(map(re.escape, data[:3])) + r"|\d+(\.\d+)?", "", text)
                self.assertEqual(re.findall("[A-Za-z]", remaining), [], text)
        self.assertNotEqual(listing["texts"]["color_by"], english["texts"]["color_by"])
        self.assertNotEqual(rows[0]["tooltip"], english_rows[0]["tooltip"])
        # The property's own label and unit and the file name are data, shown as they are.
        self.assertIn("Information Content", rows[0]["tooltip"])
        self.assertIn("fixture_alignment.fasta", information["source"])
        self.assertEqual(plddt["label"], "pLDDT")
        os.remove(self.sidecar_path())
        reason = self.listing()["unavailable"]
        self.assertNotEqual(reason, "No per-residue data was saved with this structure.")
        self.assertEqual(re.findall("[A-Za-z]", reason), [], reason)
        # So is each reason a file is not shown on a structure it does not describe.
        for damage in ({"sequence": fixtures.SECOND_SEQUENCE}, {"sequence": None},
                       {"residue_numbering": {"chain": "A", "first": 5}}):
            fixtures.write_sidecar(self.directory, FIRST, {
                **fixtures.residue_data(FIRST, fixtures.FIRST_SEQUENCE), **damage})
            reason = self.listing()["unavailable"]
            with self.subTest(damage=list(damage)):
                self.assertNotEqual(reason, "")
                self.assertEqual(re.findall("[A-Za-z]", reason), [], reason)


# The alignment of the writer tests: five nodes of two families, 36 columns wide, whose rows hold their residues
# in other columns than their positions (gaps), and node_c has a residue the providers do not count (X).
THIRD_SEQUENCE = fixtures.FIRST_SEQUENCE[:3] + "X" + fixtures.FIRST_SEQUENCE[4:]
SEQUENCES = {
    "node_a": fixtures.FIRST_SEQUENCE,
    "node_b": fixtures.SECOND_SEQUENCE,
    "node_c": THIRD_SEQUENCE,
    "node_d": fixtures.FIRST_SEQUENCE[:9] + "E" + fixtures.FIRST_SEQUENCE[10:],
    "node_e": fixtures.SECOND_SEQUENCE[:5] + "L" + fixtures.SECOND_SEQUENCE[6:],
}
GAPS = {"node_a": {3, 10, 11}, "node_b": {0, 20, 21}, "node_c": {3, 10, 11}, "node_d": {3, 10, 11}, "node_e": {0, 20, 21}}
LARGE_MODEL = "esm3-large-2024-03"


def aligned(sequence, gaps, width=36):
    """``sequence`` laid into ``width`` columns with a gap at each column of ``gaps``."""
    residues = iter(sequence)
    return "".join("-" if column in gaps else next(residues) for column in range(width))


ROWS = [(name, aligned(sequence, GAPS[name])) for name, sequence in SEQUENCES.items()]


def residue_columns(name):
    """The 0-based column of each residue of a node: where its aligned row has a symbol."""
    return [column for column, symbol in enumerate(dict(ROWS)[name]) if symbol != "-"]


class WriterPathMixin:
    """Folds nodes through the real writer path with a stand-in model, into self.directory.

    The records are the ones `esmfold` puts in the worker's input file
    (commands.esmfold._worker_records over the Viewer's alignment and the
    Residue_Data providers); the worker's run_predictions writes the PDB the
    stand-in model returns and the sidecar beside it. Nothing is folded.
    """

    def viewer_for(self, rows=ROWS, name="family.fasta"):
        """A stand-in Viewer whose alignment the production loader read from a FASTA file of ``rows``."""
        path = os.path.join(self.outer, name)
        write_fasta(path, rows)
        headers = [header for header, _ in rows]
        return SimpleNamespace(alignment=load_manager(path, headers), full_headers=headers)

    def fold(self, names, *, viewer=None, data=True, mode="local", sequences=None, numbering=None):
        """Fold ``names``; return the written files' names.

        ``data`` false gives the worker records of the old pair form, as when no alignment is loaded.
        ``numbering`` gives a node's PDB another chain and first residue.
        """
        sequences = sequences or {name: SEQUENCES[name] for name in names}
        nodes = [(name, sequences[name]) for name in names]
        if data:
            viewer = viewer or self.viewer_for()
            with contextlib.redirect_stdout(io.StringIO()):
                records, _ = esmfold_command._worker_records(
                    viewer, [viewer.full_headers.index(name) for name in names], nodes)
        else:
            records = [list(node) for node in nodes]
        # The worker reads the records from a JSON file.
        records = json.loads(json.dumps(records))
        options = {sequences[name]: (numbering or {}).get(name, {}) for name in names}

        def structure_of(sequence):
            return FakeOutput(pdb=fixtures.helix_pdb(sequence, fixtures.plddt_values(len(sequence)), **options[sequence]))

        before = set(os.listdir(self.directory))
        # A local run reads its device from torch, which only needs to name one: the stand-in is no torch, so
        # the real one is not imported (a second import of it in this process would crash it).
        torch = types.ModuleType("torch")
        torch.device = lambda name: SimpleNamespace(type=name, __str__=lambda self: name)
        with (mock.patch.dict(sys.modules, {**fake_esm_modules(), "torch": torch}),
              contextlib.redirect_stdout(io.StringIO())):
            if mode == "large":
                client = FakeClient([structure_of(sequence) for _, sequence in nodes])
                settings = {"ESM_API_TOKEN": "hidden", "ESM_API_URL": "https://biohub.ai", "ESM3_MODEL": LARGE_MODEL}
                with mock.patch.object(esmfold_worker, "_load_large_client", return_value=(client, settings)):
                    result = esmfold_worker.run_predictions(records, self.directory, mode="large", notifier=mock.Mock())
            else:
                model = mock.Mock()
                model.generate.side_effect = lambda protein, config: structure_of(protein.sequence)
                with mock.patch.object(esmfold_worker, "_load_local_model", return_value=model):
                    result = esmfold_worker.run_predictions(
                        records, self.directory, mode="local", target_device="cpu", notifier=mock.Mock())
        self.assertEqual(result, (len(names), False))
        return sorted(set(os.listdir(self.directory)) - before)

    def pdb_residues(self, pdb_name):
        """(chain, number) of each residue of a PDB in the folder, in file order."""
        residues = []
        with open(os.path.join(self.directory, pdb_name), encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("ATOM  ") and (not residues or residues[-1] != (line[21], int(line[22:26]))):
                    residues.append((line[21], int(line[22:26])))
        return residues


class ServedLayerTestCase(ResidueLayerTestCase):
    """The real web server over a folder of fixture structures."""

    def setUp(self):
        super().setUp()
        from web_ui import Web_Server

        sessions = tempfile.TemporaryDirectory()
        self.addCleanup(sessions.cleanup)
        environment = mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": sessions.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.viewer = make_viewer()
        self.server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)
        self.server.static_routes["/structures/"] = self.directory
        self.server.static_routes["/esmfold/"] = str(SRC / "resources" / "esmfold")

    def get_json(self, path, headers=None):
        status, response_headers, body = http_get(
            self.server, path, page_headers(self.server) if headers is None else headers)
        try:
            return status, response_headers, json.loads(body)
        except ValueError:
            return status, response_headers, body


class LayerRouteTests(ServedLayerTestCase):
    def test_the_listing_and_each_layers_annotation_are_served_as_json_that_is_not_kept(self):
        self.write()
        status, headers, listing = self.get_json("/api/esmfold/layers/protein_a.pdb")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(listing, json.loads(json.dumps(self.listing())))
        for layer in listing["layers"][1:]:
            status, headers, rows = self.get_json(layer["annotation"])
            self.assertEqual(status, 200, layer["id"])
            self.assertEqual(rows, json.loads(json.dumps(layer_annotation(self.data(), layer["id"]))))

    def test_a_structure_without_residue_data_still_has_plddt(self):
        self.write(sidecar=False)
        status, _, listing = self.get_json("/api/esmfold/layers/protein_a.pdb")
        self.assertEqual(status, 200)
        self.assertEqual([layer["id"] for layer in listing["layers"]], ["plddt"])
        self.assertTrue(listing["unavailable"])
        # Its annotations do not exist.
        self.assertEqual(self.get_json("/api/esmfold/layers/protein_a.pdb/information_content")[0], 404)

    def test_a_cross_site_page_or_a_rebound_name_is_refused_like_on_every_other_route(self):
        self.write()
        port = self.server.server_address[1]
        refused = [
            {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"},
            {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "image"},
            {"Origin": f"http://127.0.0.1:{port + 1}", "Sec-Fetch-Site": "same-site"},
            {"Host": "rebound.example"},
        ]
        paths = ["/api/esmfold/layers/protein_a.pdb", "/api/esmfold/layers/protein_a.pdb/information_content"]
        for path in paths:
            for headers in refused:
                with self.subTest(path=path, headers=headers):
                    status, _, body = self.get_json(path, headers)
                    self.assertEqual(status, 403)
                    self.assertEqual(list(body), ["error"])
        # A local caller with no browser headers is allowed, like the structures route.
        self.assertEqual(self.get_json(paths[0], {})[0], 200)

    def test_nothing_outside_the_structures_folder_can_be_reached(self):
        # A structure and residue data in the folder around it, which no spelling may reach.
        fixtures.write_structure(self.outer, "outside.pdb", fixtures.FIRST_SEQUENCE)
        fixtures.write_structure(self.outer, "..sneaky.pdb", fixtures.FIRST_SEQUENCE)
        self.write()
        spellings = [
            "../outside.pdb", "..%2Foutside.pdb", "%2e%2e%2foutside.pdb", "..%5Coutside.pdb", "..\\outside.pdb",
            "%2e%2e/outside.pdb", "sub/../../outside.pdb", "C:outside.pdb", "C:..\\outside.pdb", "/outside.pdb",
            "%5Coutside.pdb", "outside.pdb%00", "nul.pdb", "protein_a.pdb/../../outside.pdb", "..%2F..sneaky.pdb",
        ]
        for spelling in spellings:
            for suffix in ("", "/information_content"):
                path = f"/api/esmfold/layers/{spelling}{suffix}"
                with self.subTest(path=path):
                    status, _, body = self.get_json(path)
                    self.assertIn(status, (403, 404), body)
                    self.assertNotIn("layers", body if isinstance(body, dict) else {})
        # A name that merely starts with two dots is a name, as for the structures route.
        self.write(name="..x.pdb")
        self.assertEqual(self.get_json("/api/esmfold/layers/..x.pdb")[0], 200)
        # A structure in a folder inside it is not named by a path.
        os.makedirs(os.path.join(self.directory, "inner"))
        fixtures.write_structure(os.path.join(self.directory, "inner"), "nested.pdb", fixtures.FIRST_SEQUENCE)
        self.assertEqual(self.get_json("/api/esmfold/layers/inner%2Fnested.pdb")[0], 404)

    def test_a_structure_or_its_residue_data_that_is_a_link_out_of_the_folder_is_refused(self):
        fixtures.write_structure(self.outer, "elsewhere.pdb", fixtures.FIRST_SEQUENCE)
        fixtures.write_structure(self.directory, "linked_data.pdb", fixtures.FIRST_SEQUENCE, sidecar=False)
        fixtures.write_structure(self.directory, "linked_structure.pdb", fixtures.FIRST_SEQUENCE)
        os.remove(os.path.join(self.directory, "linked_structure.pdb"))
        fixtures.write_sidecar(self.outer, "linked_data.pdb", fixtures.residue_data("linked_data.pdb", fixtures.FIRST_SEQUENCE))
        try:
            # Residue data that lies elsewhere, and a structure that does.
            os.symlink(os.path.join(self.outer, "linked_data.residues.json"),
                       os.path.join(self.directory, "linked_data.residues.json"))
            os.symlink(os.path.join(self.outer, "elsewhere.pdb"), os.path.join(self.directory, "linked_structure.pdb"))
        except (OSError, NotImplementedError):
            self.skipTest("this account cannot create symbolic links")
        for name in ("linked_data.pdb", "linked_structure.pdb"):
            with self.subTest(name):
                status, _, body = self.get_json(f"/api/esmfold/layers/{name}")
                self.assertEqual((status, body), (403, {"error": "Forbidden"}))
        # A link inside the folder, to a file inside it, is still a structure.
        self.write(name="real.pdb")
        os.symlink(os.path.join(self.directory, "real.pdb"), os.path.join(self.directory, "alias.pdb"))
        os.symlink(os.path.join(self.directory, "real.residues.json"), os.path.join(self.directory, "alias.residues.json"))
        self.assertEqual(self.get_json("/api/esmfold/layers/alias.pdb")[0], 200)

    def test_what_is_not_a_layer_of_a_structure_is_not_found(self):
        self.write()
        for path in (
            "/api/esmfold/layers/", "/api/esmfold/layers/protein_a.pdb/", "/api/esmfold/layers/missing.pdb",
            "/api/esmfold/layers/protein_a", "/api/esmfold/layers/protein_a.cif",
            "/api/esmfold/layers/protein_a.pdb/information_content/extra",
            "/api/esmfold/layers/protein_a.pdb/plddt", "/api/esmfold/layers/protein_a.pdb/nope",
            "/api/esmfold/layers/protein_a.residues.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.get_json(path)[0], 404)
        del self.server.static_routes["/structures/"]
        self.assertEqual(self.get_json("/api/esmfold/layers/protein_a.pdb")[0], 404)

    def test_the_structures_route_still_serves_the_files_beside_it(self):
        self.write()
        status, _, body = http_get(self.server, "/structures/protein_a.residues.json", page_headers(self.server))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["structure"], FIRST)
        self.assertEqual(http_get(self.server, "/structures/protein_a.pdb", page_headers(self.server))[0], 200)


class WriterOutputTests(WriterPathMixin, ServedLayerTestCase):
    """What the real writer makes, read back through the route and the listing."""

    def test_the_files_the_worker_writes_are_listed_and_annotated_as_they_describe(self):
        # node_b's PDB is chain B from residue 7, which the writer reads from the PDB itself.
        numbering = {"node_b": {"chain": "B", "first": 7}}
        viewer = self.viewer_for()
        names = self.fold(["node_a", "node_c"], viewer=viewer) + self.fold(
            ["node_b"], viewer=viewer, mode="large", numbering=numbering)
        self.assertEqual(names, [
            "node_a.pdb", "node_a.residues.json", "node_c.pdb", "node_c.residues.json",
            f"node_b_{LARGE_MODEL}.pdb", f"node_b_{LARGE_MODEL}.residues.json",
        ])
        for pdb_name, node, expected in (
                ("node_a.pdb", "node_a", ("A", 1)), ("node_c.pdb", "node_c", ("A", 1)),
                (f"node_b_{LARGE_MODEL}.pdb", "node_b", ("B", 7))):
            with self.subTest(pdb_name):
                self.check_a_structure(pdb_name, node, *expected)

    def check_a_structure(self, pdb_name, node, chain, first):
        sidecar_name = residue_layers.sidecar_name(pdb_name)
        text = Path(self.directory, sidecar_name).read_text(encoding="utf-8")
        sidecar = json.loads(text)
        # Compact JSON in the writer's key order, named for the PDB's own stem, model suffix included.
        self.assertTrue(text.startswith('{"format":"emapssn-residue-data","version":1,"structure":"' + pdb_name + '","header":'))
        self.assertEqual(list(sidecar), ["format", "version", "structure", "header", "sequence", "residue_numbering", "properties"])
        self.assertEqual(list(sidecar["properties"]), ["information_content", "same_residue_pct"])
        self.assertEqual(sidecar["sequence"], SEQUENCES[node])
        self.assertEqual(sidecar["residue_numbering"], {"chain": chain, "first": first})
        self.assertEqual(self.pdb_residues(pdb_name), [(chain, first + index) for index in range(33)])

        status, _, listing = self.get_json(f"/api/esmfold/layers/{pdb_name}")
        self.assertEqual(status, 200)
        self.assertEqual(listing["unavailable"], "")
        self.assertEqual([(layer["id"], layer["label"], layer["unit"], layer.get("range")) for layer in listing["layers"]], [
            ("plddt", "pLDDT", "", None),
            ("information_content", "Information Content", "%", [0, 100]),
            ("same_residue_pct", "Residue Agreement", "%", [0, 100]),
        ])
        information, agreement = listing["layers"][1:]
        self.assertEqual((information["residues"], information["covered"]), (33, 33))
        standard = [letter != "X" for letter in SEQUENCES[node]]
        self.assertEqual((agreement["residues"], agreement["covered"]), (33, sum(standard)))
        self.assertEqual(information["source"], "From family.fasta, computed when folded, using 5 sequences")
        for layer in (information, agreement):
            self.assertTrue(layer["description"])
            self.assertEqual(layer["annotation"].split("?")[0], f"/api/esmfold/layers/{pdb_name}/{layer['id']}")

        columns = residue_columns(node)
        for layer, entry in zip((information, agreement), sidecar["properties"].values()):
            status, _, rows = self.get_json(layer["annotation"])
            self.assertEqual(status, 200)
            self.assertEqual(entry["source"]["columns"], columns)
            # A row for each residue with a value, selecting it by the PDB's own chain and number...
            expected = [index for index, value in enumerate(entry["values"]) if value is not None]
            self.assertEqual([(row["auth_asym_id"], row["auth_seq_id"]) for row in rows],
                             [(chain, first + index) for index in expected])
            for row, index in zip(rows, expected):
                value = entry["values"][index]
                self.assertEqual(row["value"], value)
                # ...with the tooltip naming the alignment column of that residue, counted from 1.
                self.assertEqual(row["tooltip"], f"{layer['label']} {round(value, 2):g} % (MSA column {columns[index] + 1})")
                self.assertEqual(row["color"], residue_layers.property_color(self.data(pdb_name).properties[layer["id"]], value))

        # Information content is never null; residue agreement is null at the residue the providers do not count.
        values = sidecar["properties"]["information_content"]["values"]
        self.assertTrue(all(isinstance(value, (int, float)) for value in values))
        agreement_values = sidecar["properties"]["same_residue_pct"]["values"]
        self.assertEqual([index for index, value in enumerate(agreement_values) if value is None],
                         [index for index, ok in enumerate(standard) if not ok])

    def test_a_refold_that_gets_no_data_leaves_a_file_that_is_not_shown_on_another_protein(self):
        self.fold(["node_a"])
        self.assertEqual([layer["id"] for layer in self.get_json("/api/esmfold/layers/node_a.pdb")[2]["layers"]],
                         ["plddt", "information_content", "same_residue_pct"])
        # The same name folded again with another protein and no data: the worker leaves the old file where it is.
        self.fold(["node_a"], data=False, sequences={"node_a": SEQUENCES["node_b"]})
        self.assertTrue(os.path.exists(self.sidecar_path("node_a.pdb")))
        status, _, listing = self.get_json("/api/esmfold/layers/node_a.pdb")
        self.assertEqual(status, 200)
        self.assertEqual([layer["id"] for layer in listing["layers"]], ["plddt"])
        self.assertIn("different protein", listing["unavailable"])
        status, _, body = self.get_json("/api/esmfold/layers/node_a.pdb/information_content")
        self.assertEqual(status, 404)
        self.assertIn("different protein", body["error"])

        # The same protein folded again without data is the protein the old file describes, so it is shown,
        # and its source line names the alignment it came from.
        self.fold(["node_a"], data=False)
        listing = self.get_json("/api/esmfold/layers/node_a.pdb")[2]
        self.assertEqual([layer["id"] for layer in listing["layers"]], ["plddt", "information_content", "same_residue_pct"])
        self.assertIn("family.fasta", listing["layers"][1]["source"])
        # A newer file from another alignment replaces it and the page sees the change.
        before = listing["layers"][1]["annotation"]
        smaller = self.viewer_for([row for row in ROWS if row[0] in ("node_a", "node_d")], name="smaller.fasta")
        self.fold(["node_a"], viewer=smaller)
        layer = self.get_json("/api/esmfold/layers/node_a.pdb")[2]["layers"][1]
        self.assertEqual(layer["source"], "From smaller.fasta, computed when folded, using 2 sequences")
        self.assertNotEqual(layer["annotation"], before)

    def test_a_structure_whose_residues_were_renumbered_since_is_not_given_the_numbers_of_the_old_ones(self):
        self.fold(["node_a"])
        pdb = Path(self.directory, "node_a.pdb")
        pdb.write_text(fixtures.helix_pdb(SEQUENCES["node_a"], fixtures.plddt_values(33), first=5), encoding="utf-8")
        listing = self.get_json("/api/esmfold/layers/node_a.pdb")[2]
        self.assertEqual([layer["id"] for layer in listing["layers"]], ["plddt"])
        self.assertIn("not numbered as", listing["unavailable"])

    def test_a_node_the_alignment_lacks_gets_no_file_and_so_plddt_alone(self):
        viewer = self.viewer_for([row for row in ROWS if row[0] != "node_e"])
        viewer.full_headers = viewer.full_headers + ["node_e"]
        viewer.alignment.network_headers = list(viewer.full_headers)
        # The alignment has no row for node_e, so the command writes it no data, and so no file.
        names = self.fold(["node_a"], viewer=viewer) + self.fold(["node_e"], viewer=viewer)
        self.assertEqual(names, ["node_a.pdb", "node_a.residues.json", "node_e.pdb"])
        listing = self.get_json("/api/esmfold/layers/node_e.pdb")[2]
        self.assertEqual([layer["id"] for layer in listing["layers"]], ["plddt"])
        self.assertIn("No per-residue data was saved", listing["unavailable"])


class LayerPageTests(FoldViewPageMixin, WriterPathMixin, ServedLayerTestCase):
    """esmfold.html in Qt WebEngine over the real server, with the real Mol*."""

    GREY = "#dcdcdc"
    BANDS = ((90, "#0053d6"), (70, "#65cbf3"), (50, "#ffdb13"), (-1, "#ff7d45"))

    def band(self, value):
        return next(color for floor, color in self.BANDS if value > floor)

    def test_plddt_is_the_default_and_a_layer_can_be_switched_to_and_from_exactly(self):
        self.write(name=FIRST)
        self.open_page()
        self.push(FIRST)
        start = self.script("return {state: layerState(0), colors: colors(0, 'plddt-confidence', {}), card: await cardNow()}")
        # pLDDT is how the structure starts, coloured by Mol*'s own theme from the B-factors.
        self.assertEqual(start["state"]["active"], "plddt")
        self.assertEqual(start["state"]["themes"], [{"name": "plddt-confidence", "params": {}}])
        self.assertFalse(start["state"]["isMvs"])
        self.assertEqual(start["state"]["annotations"], [])
        plddt = fixtures.plddt_values(33)
        for number in range(1, 34):
            self.assertEqual(start["colors"][str(number)], self.band(plddt[number - 1]), number)
        # The card offers pLDDT, the structure's own color and each property of the file, with a legend for the chosen one.
        self.assertEqual(start["card"]["chips"], [
            ["builtin:structure", "Structure color", ""], ["plddt", "pLDDT", ""], ["information_content", "Information Content", ""],
            ["same_residue_pct", "Residue Agreement", ""]])
        self.assertEqual(start["card"]["selected"], "plddt")
        self.assertEqual(start["card"]["title"], "Color by")
        self.assertEqual(len(start["card"]["keys"]), 4)
        self.assertEqual(start["card"]["compact"]["shown"], "1")

        layered = self.script("""
            await applyLayer(loadedStructures()[0], 'information_content');
            const rows = await (await fetch('/api/esmfold/layers/protein_a.pdb/information_content')).json();
            const theme = reprThemes()[0];
            deriveColorMode();
            return {state: layerState(0), theme, rows, colors: colors(0, 'mvs-annotation', theme.params),
                    tooltip: await tooltip(0, 5), card: await cardNow()};
            """)
        self.assertEqual(layered["state"]["active"], "information_content")
        self.assertEqual(layered["theme"]["name"], "mvs-annotation")
        self.assertTrue(layered["state"]["isMvs"])
        self.assertEqual([a["id"] for a in layered["state"]["annotations"]], ["information_content"])
        # Each residue the file has a number for takes the colour of its row.
        by_number = {str(row["auth_seq_id"]): row["color"] for row in layered["rows"]}
        self.assertEqual(len(by_number), 33)
        for number in range(1, 34):
            self.assertEqual(layered["colors"][str(number)], by_number[str(number)], number)
        self.assertNotEqual(layered["colors"], start["colors"])
        self.assertEqual(layered["card"]["selected"], "information_content")
        self.assertTrue(layered["card"]["bar"])
        self.assertEqual(layered["card"]["ends"], ["0 %", "100 %"])
        self.assertEqual(layered["card"]["compact"]["ends"], ["0 %", "100 %"])
        # Information content has a value for every residue, so its legend has no "No data".
        self.assertEqual(layered["card"]["keys"], [])
        self.assertIn("From fixture_alignment.fasta, computed when folded", layered["card"]["source"])
        # Hovering residue 6 shows Mol*'s own label and the layer's tooltip for it.
        self.assertIn("Information Content 35.5 % (MSA column 111)", layered["tooltip"])

        # Residue agreement has no value for residue 4: grey, with a legend entry for it.
        agreement = self.script("""
            await applyLayer(loadedStructures()[0], 'same_residue_pct');
            const rows = await (await fetch('/api/esmfold/layers/protein_a.pdb/same_residue_pct')).json();
            const theme = reprThemes()[0];
            deriveColorMode();
            return {state: layerState(0), rows, colors: colors(0, 'mvs-annotation', theme.params),
                    tooltip: await tooltip(0, 5), card: await cardNow()};
            """)
        self.assertEqual([a["id"] for a in agreement["state"]["annotations"]], ["same_residue_pct"])
        by_number = {str(row["auth_seq_id"]): row["color"] for row in agreement["rows"]}
        self.assertEqual(len(by_number), 32)
        for number in range(1, 34):
            self.assertEqual(agreement["colors"][str(number)], by_number.get(str(number), self.GREY), number)
        self.assertEqual(agreement["colors"]["4"], self.GREY)
        self.assertEqual(agreement["card"]["keys"], ["No data"])
        self.assertIn("Residue Agreement 65 % (MSA column 111)", agreement["tooltip"])

        restored = self.script("""
            await restorePlddt(loadedStructures()[0]);
            deriveColorMode();
            return {state: layerState(0), colors: colors(0, 'plddt-confidence', {}), tooltip: await tooltip(0, 5),
                    card: await cardNow()};
            """)
        # Back to exactly what it was: the theme, the properties, the colours; no tooltip but Mol*'s.
        self.assertEqual(restored["state"], start["state"])
        self.assertEqual(restored["colors"], start["colors"])
        self.assertEqual(restored["card"]["selected"], "plddt")
        self.assertEqual(len(restored["tooltip"]), 1, restored["tooltip"])
        self.assertNotIn("Information Content", restored["tooltip"][0])

    def test_structures_the_writer_made_get_both_layers_and_no_data_where_it_has_none(self):
        self.fold(["node_a", "node_c"])
        self.open_page()
        self.push("node_a.pdb")
        self.push("node_c.pdb")
        sidecar = json.loads(Path(self.sidecar_path("node_c.pdb")).read_text(encoding="utf-8"))
        columns = residue_columns("node_c")
        result = self.script("""
            const out = {options: card().chips, states: [0, 1].map(layerState)};
            for (const id of ['information_content', 'same_residue_pct']) {
                // The scene's choice: both structures have the layer.
                await setColorMode(id);
                const rows = await (await fetch('/api/esmfold/layers/node_c.pdb/' + id)).json();
                out[id] = {rows, colors: colors(1, 'mvs-annotation', reprThemes()[1].params),
                           tooltip: await tooltip(1, 5), card: await cardNow(), fit: fit(1)};
            }
            return out;
            """)
        self.assertEqual(result["options"], [
            ["builtin:structure", "Structure color", ""], ["plddt", "pLDDT", ""], ["information_content", "Information Content", ""],
            ["same_residue_pct", "Residue Agreement", ""]])
        self.assertEqual([state["active"] for state in result["states"]], ["plddt", "plddt"])
        for layer_id, label, rows_expected, swatches in (
                ("information_content", "Information Content", 33, []), ("same_residue_pct", "Residue Agreement", 32, ["No data"])):
            shown = result[layer_id]
            by_number = {str(row["auth_seq_id"]): row["color"] for row in shown["rows"]}
            self.assertEqual(len(by_number), rows_expected, layer_id)
            for number in range(1, 34):
                self.assertEqual(shown["colors"][str(number)], by_number.get(str(number), self.GREY), (layer_id, number))
            self.assertEqual(shown["card"]["keys"], swatches, layer_id)
            self.assertEqual(shown["card"]["selected"], layer_id)
            self.assertIn("From family.fasta, computed when folded, using 5 sequences", shown["card"]["source"])
            # The hovered residue (the 6th) shows the writer's value and the alignment column it came from.
            value = sidecar["properties"][layer_id]["values"][5]
            self.assertIn(f"{label} {round(value, 2):g} % (MSA column {columns[5] + 1})", shown["tooltip"], layer_id)
            self.assertTrue(shown["fit"]["transformed"])
        # The residue the writer has no agreement for is grey, and the information content of it is not.
        self.assertEqual(result["same_residue_pct"]["colors"]["4"], self.GREY)
        self.assertNotEqual(result["information_content"]["colors"]["4"], self.GREY)

    def test_the_card_switches_a_layer_and_plddt_again(self):
        self.write(name=FIRST)
        self.open_page()
        self.push(FIRST)
        after = self.script("""
            const choose = async value => {
                document.querySelector('#color-body [data-coloring="' + value + '"]').click();
                await structureQueue;
                await cardNow();
                return [layerState(0).active, card().selected];
            };
            const ids = () => layerState(0).annotations.map(a => a.id);
            // What is attached each time the colours move: the old layer must outlast the move.
            const attachedAtColouring = [];
            const colour = themeStructure;
            themeStructure = (structure, theme) => { attachedAtColouring.push(ids()); return colour(structure, theme); };
            const steps = [];
            // From one layer straight to another, and back to pLDDT: only the chosen layer stays attached.
            for (const value of ['same_residue_pct', 'information_content', 'plddt']) {
                steps.push([...await choose(value), ids(), reprThemes()[0].name,
                            reprThemes()[0].params.annotationId || null]);
            }
            themeStructure = colour;
            return {steps, attachedAtColouring};
            """)
        self.assertEqual(after["steps"], [
            ["same_residue_pct", "same_residue_pct", ["same_residue_pct"], "mvs-annotation", "same_residue_pct"],
            ["information_content", "information_content", ["information_content"], "mvs-annotation", "information_content"],
            ["plddt", "plddt", [], "plddt-confidence", None],
        ])
        # Choosing a first layer attaches it before colouring; choosing another keeps the first until the
        # colours have moved; going back to pLDDT colours first, then lets go.
        self.assertEqual(after["attachedAtColouring"], [
            ["same_residue_pct"], ["information_content", "same_residue_pct"], ["information_content"]])

    def test_a_layer_the_viewer_does_not_offer_cannot_be_applied(self):
        self.write(name=FIRST)
        self.open_page()
        self.push(FIRST)
        result = self.script("""
            let message = null;
            try { await applyLayer(loadedStructures()[0], 'missing'); } catch (error) { message = error.message; }
            return {message, state: layerState(0)};
            """)
        self.assertIn("no layer missing", result["message"])
        self.assertEqual(result["state"]["active"], "plddt")

    def test_each_structure_has_its_own_layer_and_the_superposition_is_untouched(self):
        self.write(name=FIRST)
        self.write(name=SECOND, sequence=fixtures.SECOND_SEQUENCE, shift=(14.0, 6.0, -3.0), wobble=0.3, offset=20,
                   plddt=fixtures.plddt_values(33, 17))
        self.write(name=THIRD, sidecar=False, shift=(-12.0, 4.0, 2.0))
        self.open_page()
        self.push(FIRST)
        self.script("await applyLayer(loadedStructures()[0], 'same_residue_pct'); return 1")
        # Loading another structure does not recolour the first, and superposes the new one.
        self.push(SECOND)
        loaded = self.script("return {first: layerState(0), second: layerState(1), fit: fit(1)}")
        self.assertEqual(loaded["first"]["active"], "same_residue_pct")
        self.assertEqual(loaded["second"]["active"], "plddt")
        self.assertTrue(loaded["fit"]["transformed"])
        self.assertLess(loaded["fit"]["rmsd"], 0.6)
        pose = loaded["fit"]["matrix"]

        both = self.script("""
            await applyLayer(loadedStructures()[1], 'information_content');
            const rowsFirst = await (await fetch('/api/esmfold/layers/protein_a.pdb/same_residue_pct')).json();
            const rowsSecond = await (await fetch('/api/esmfold/layers/protein_b.pdb/information_content')).json();
            const themes = reprThemes();
            return {first: layerState(0), second: layerState(1), themes, rowsFirst, rowsSecond, fit: fit(1),
                    firstColors: colors(0, 'mvs-annotation', themes[0].params),
                    secondColors: colors(1, 'mvs-annotation', themes[1].params),
                    tooltips: [await tooltip(0, 5), await tooltip(1, 5)]};
            """)
        self.assertEqual((both["first"]["active"], both["second"]["active"]), ("same_residue_pct", "information_content"))
        # Each structure is coloured from its own file, though both have residue 6.
        self.assertNotEqual(both["rowsFirst"], both["rowsSecond"])
        for colors, rows in ((both["firstColors"], both["rowsFirst"]), (both["secondColors"], both["rowsSecond"])):
            by_number = {str(row["auth_seq_id"]): row["color"] for row in rows}
            for number in range(1, 34):
                self.assertEqual(colors[str(number)], by_number.get(str(number), self.GREY))
        self.assertIn("Residue Agreement 65 % (MSA column 111)", both["tooltips"][0])
        self.assertIn("Information Content 55.5 % (MSA column 111)", both["tooltips"][1])
        # The pose is exactly what the page computed before any layer.
        self.assertEqual(both["fit"]["matrix"], pose)
        self.assertLess(both["fit"]["rmsd"], 0.6)

        # A structure with no residue data has pLDDT only, and leaves the others as they are.
        self.push(THIRD)
        third = self.script("return {states: [0, 1, 2].map(layerState), card: await cardNow(), fit: fit(2), names: loadedStructures().map(structureName)}")
        self.assertEqual([state["active"] for state in third["states"]], ["same_residue_pct", "information_content", "plddt"])
        self.assertEqual(third["names"], ["protein_a", "protein_b", "protein_c"])
        self.assertTrue(third["fit"]["transformed"])
        self.assertTrue(any("protein_c" in note and "No per-residue data" in note for note in third["card"]["layerNotes"]),
                        third["card"]["layerNotes"])

    def test_a_saved_session_brings_its_layers_back_and_drops_one_that_is_gone(self):
        self.write(name=FIRST)
        self.write(name=SECOND, sequence=fixtures.SECOND_SEQUENCE, shift=(14.0, 6.0, -3.0), wobble=0.3, offset=20,
                   plddt=fixtures.plddt_values(33, 17))
        self.open_page()
        self.push(FIRST)
        self.push(SECOND)
        # Let the saves the loads asked for come first, so that what arrives next is the layers' own.
        self.pump(2.5)
        self.saved.clear()
        self.script("""
            await applyLayer(loadedStructures()[0], 'information_content');
            await applyLayer(loadedStructures()[1], 'same_residue_pct');
            return 1""")
        inspect = ("return {states: [0, 1].map(layerState), fit: fit(1),"
                   " colors: [0, 1].map(i => colors(i, reprThemes()[i].name, reprThemes()[i].params))}")
        before = self.script(inspect)
        # The page saves its session after a layer changes, as it does after a load.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not self.saved:
            self.pump(0.1)
        self.assertTrue(self.saved, "the page did not save its session after the layers changed")
        snapshot = json.loads(json.dumps(self.saved[-1]))
        self.assertIn('"id": "information_content"', json.dumps(snapshot))
        self.assertIn('"id": "same_residue_pct"', json.dumps(snapshot))

        self.session = snapshot
        self.open_page()
        again = self.script(inspect[:-1] + ", n: loadedStructures().length, card: await cardNow(), tooltip: await tooltip(1, 5)}")
        self.assertEqual(again["n"], 2)
        self.assertEqual([state["active"] for state in again["states"]], ["information_content", "same_residue_pct"])
        self.assertEqual(again["colors"], before["colors"])
        self.assertEqual(again["fit"]["matrix"], before["fit"]["matrix"])
        # The scene's choice is worked out from the structures; with one each, the first one's.
        self.assertEqual(again["card"]["selected"], "information_content")
        self.assertIn("Residue Agreement 85 % (MSA column 111)", again["tooltip"])

        # The Viewer's data for a structure changed since: one property is gone, so its layer is dropped.
        data = json.loads(Path(self.sidecar_path(FIRST)).read_text(encoding="utf-8"))
        del data["properties"]["information_content"]
        fixtures.write_sidecar(self.directory, FIRST, data)
        self.open_page()
        gone = self.script("return {states: [0, 1].map(layerState), themes: reprThemes().map(t => t.name), card: await cardNow()}")
        self.assertEqual([state["active"] for state in gone["states"]], ["plddt", "same_residue_pct"])
        self.assertEqual(gone["themes"], ["plddt-confidence", "mvs-annotation"])
        self.assertFalse(gone["states"][0]["isMvs"])
        self.assertEqual(gone["states"][0]["annotations"], [])
        self.assertEqual(gone["card"]["selected"], "plddt")


if __name__ == "__main__":
    unittest.main()
