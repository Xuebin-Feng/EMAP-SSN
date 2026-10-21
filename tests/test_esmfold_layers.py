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

import http.client
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
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

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from tests import esmfold_layer_fixtures as fixtures
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

    def listing(self, name=FIRST):
        return layer_listing(name, self.sidecar_path(name))

    def data(self, name=FIRST):
        return read_residue_data(name, self.sidecar_path(name))

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
        self.assertEqual((information["residues"], information["covered"]), (33, 32))
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
        rows = layer_annotation(self.data(), "information_content")
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
        # Residue 1 sits in 0-based column 100; residue 2's column is not known.
        self.assertEqual(rows[1]["tooltip"], "Information Content 0.5 % (MSA column 101)")
        self.assertEqual(rows[2]["tooltip"], "Information Content 7.5 %")
        self.assertEqual(rows[1]["value"], 0.5)

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


def http_get(server, path, headers=None):
    """GET path from the real server; returns (status, headers, body bytes)."""
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        for name, value in {"Host": f"127.0.0.1:{port}", **(headers or {})}.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def page_headers(server):
    """What Chromium sends for a same-origin fetch from a bundled page."""
    port = server.server_address[1]
    return {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}


def make_viewer():
    """The little a Viewer must have for the real web server to run over it."""
    from desktop.Viewer_Inspection import ViewerInspectionService
    from web_ui import Web_Server

    viewer = SimpleNamespace(
        n_nodes=2, full_headers=["a", "b"], metadata={}, edges=[], visible_mask=[True, True], selected_indices=[],
        group_labels=[[], []], cluster_labels=None, web_plugin_registry=None)
    viewer.viewer_inspection = ViewerInspectionService(viewer)
    viewer.communicator = Web_Server.QtCommunicator(viewer)
    viewer.get_initial_web_state = lambda: {"rows": []}
    viewer.handle_web_action = lambda data: None
    return viewer


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


class LayerPageTests(ServedLayerTestCase):
    """esmfold.html in Qt WebEngine over the real server, with the real Mol*."""

    GREY = "#dcdcdc"
    BANDS = ((90, "#0053d6"), (70, "#65cbf3"), (50, "#ffdb13"), (-1, "#ff7d45"))

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.view = QWebEngineView()
        cls.view.resize(1300, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()

    def setUp(self):
        super().setUp()
        self.session = None
        self.saved = []
        self.viewer.handle_web_action = self.handle_action

    def tearDown(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.3)

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

    def handle_action(self, data):
        """The Viewer's side of the page's session handshake and its saves."""
        if data.get("action") == "load_molstar_session":
            self.broadcast({"type": "restore_session", "session": self.session})
        elif data.get("action") == "save_molstar_session":
            self.saved.append(data["session"])

    def broadcast(self, event):
        with self.server.queues_lock:
            for queue in list(self.server.event_queues):
                queue.put(event)

    def js(self, source, timeout=20):
        result = []
        self.view.page().runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(result, source)
        return result[0]

    def script(self, body, timeout=60):
        """The JSON result of the async JavaScript `body`, which may await."""
        self.js("window.__result = null; (async () => { try { window.__result = JSON.stringify(await (async () => {"
                + body + "})()); } catch (error) { window.__result = 'ERROR ' + error.message; } })()")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.js("window.__result")
            if value:
                self.assertFalse(value.startswith("ERROR"), value)
                return json.loads(value)
            self.pump(0.05)
        self.fail("Timed out: " + body)

    def open_page(self):
        loaded = []
        self.view.loadFinished.connect(loaded.append)
        self.view.load(QUrl(f"http://127.0.0.1:{self.server.server_address[1]}/esmfold.html"))
        deadline = time.monotonic() + 30
        while not loaded and time.monotonic() < deadline:
            self.pump(0.02)
        self.view.loadFinished.disconnect(loaded.append)
        self.assertEqual(loaded, [True])
        # Mol* is up and the page has had its session answer.
        self.script("""
            for (let i = 0; i < 600 && !(typeof plugin !== 'undefined' && plugin && !isRestoring); i++) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            if (isRestoring) throw new Error('the page never finished restoring');
            return 1;
            """, timeout=90)
        self.js(HELPERS)

    def push(self, name):
        """Send the page a folded structure, as the Viewer does, and wait until it is loaded."""
        before = self.script("return loadedStructures().length")
        self.broadcast({"type": "esmfold_pdb", "node_id": name[:-4], "pdb_url": f"/structures/{name}"})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.script("await structureQueue; return loadedStructures().length") > before:
                break
            self.pump(0.1)
        else:
            self.fail(f"{name} was not loaded")
        self.script("await structureQueue; await panelNow(); return 1")

    def band(self, value):
        return next(color for floor, color in self.BANDS if value > floor)

    def test_plddt_is_the_default_and_a_layer_can_be_switched_to_and_from_exactly(self):
        self.write(name=FIRST)
        self.open_page()
        self.push(FIRST)
        start = self.script("return {state: layerState(0), colors: colors(0, 'plddt-confidence', {}), panel: await panelNow()}")
        # pLDDT is how the structure starts, coloured by Mol*'s own theme from the B-factors.
        self.assertEqual(start["state"]["active"], "plddt")
        self.assertEqual(start["state"]["themes"], [{"name": "plddt-confidence", "params": {}}])
        self.assertFalse(start["state"]["isMvs"])
        self.assertEqual(start["state"]["annotations"], [])
        plddt = fixtures.plddt_values(33)
        for number in range(1, 34):
            self.assertEqual(start["colors"][str(number)], self.band(plddt[number - 1]), number)
        # The card offers pLDDT and each property of the file, with a legend for the active layer.
        self.assertEqual(start["panel"]["options"], [["plddt", "pLDDT"], ["information_content", "Information Content"],
                                                      ["same_residue_pct", "Residue Agreement"]])
        self.assertEqual(start["panel"]["selected"], ["plddt"])
        self.assertEqual(start["panel"]["title"], "Color by")
        self.assertEqual(len(start["panel"]["legend_swatches"]), 4)
        self.assertEqual(start["panel"]["shown"], "block")

        layered = self.script("""
            await applyLayer(loadedStructures()[0], 'information_content');
            const rows = await (await fetch('/api/esmfold/layers/protein_a.pdb/information_content')).json();
            const theme = reprThemes()[0];
            return {state: layerState(0), theme, rows, colors: colors(0, 'mvs-annotation', theme.params),
                    tooltip: await tooltip(0, 5), panel: await panelNow()};
            """)
        self.assertEqual(layered["state"]["active"], "information_content")
        self.assertEqual(layered["theme"]["name"], "mvs-annotation")
        self.assertTrue(layered["state"]["isMvs"])
        self.assertEqual([a["id"] for a in layered["state"]["annotations"]], ["information_content"])
        # Each residue the file has a number for takes the colour of its row; the rest are grey.
        by_number = {str(row["auth_seq_id"]): row["color"] for row in layered["rows"]}
        for number in range(1, 34):
            self.assertEqual(layered["colors"][str(number)], by_number.get(str(number), self.GREY), number)
        self.assertEqual(layered["colors"]["4"], self.GREY)
        self.assertNotEqual(layered["colors"], start["colors"])
        self.assertEqual(layered["panel"]["selected"], ["information_content"])
        self.assertTrue(layered["panel"]["legend_bar"])
        self.assertEqual(layered["panel"]["legend_ends"], ["0 %", "100 %"])
        self.assertEqual(layered["panel"]["legend_swatches"], ["No data"])
        self.assertIn("From fixture_alignment.fasta, computed when folded", layered["panel"]["note"])
        # Hovering residue 6 shows Mol*'s own label and the layer's tooltip for it.
        self.assertIn("Information Content 35.5 % (MSA column 111)", layered["tooltip"])

        restored = self.script("""
            await restorePlddt(loadedStructures()[0]);
            return {state: layerState(0), colors: colors(0, 'plddt-confidence', {}), tooltip: await tooltip(0, 5),
                    panel: await panelNow()};
            """)
        # Back to exactly what it was: the theme, the properties, the colours; no tooltip but Mol*'s.
        self.assertEqual(restored["state"], start["state"])
        self.assertEqual(restored["colors"], start["colors"])
        self.assertEqual(restored["panel"]["selected"], ["plddt"])
        self.assertEqual(len(restored["tooltip"]), 1, restored["tooltip"])
        self.assertNotIn("Information Content", restored["tooltip"][0])

    def test_the_card_switches_a_layer_and_plddt_again(self):
        self.write(name=FIRST)
        self.open_page()
        self.push(FIRST)
        after = self.script("""
            const choose = async value => {
                const select = layerPanel.querySelector('select');
                select.value = value;
                select.dispatchEvent(new Event('change'));
                await structureQueue;
                await panelNow();
                return [layerState(0).active, panel().selected[0]];
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
        third = self.script("return {states: [0, 1, 2].map(layerState), panel: await panelNow(), fit: fit(2)}")
        self.assertEqual([state["active"] for state in third["states"]], ["same_residue_pct", "information_content", "plddt"])
        self.assertEqual(third["panel"]["names"], ["protein_a", "protein_b", "protein_c"])
        self.assertEqual(third["panel"]["selects"], 2)
        self.assertTrue(third["fit"]["transformed"])
        self.assertTrue(any("No per-residue data" in note for note in third["panel"]["notes"]), third["panel"]["notes"])

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
        again = self.script(inspect[:-1] + ", n: loadedStructures().length, panel: await panelNow(), tooltip: await tooltip(1, 5)}")
        self.assertEqual(again["n"], 2)
        self.assertEqual([state["active"] for state in again["states"]], ["information_content", "same_residue_pct"])
        self.assertEqual(again["colors"], before["colors"])
        self.assertEqual(again["fit"]["matrix"], before["fit"]["matrix"])
        self.assertEqual(again["panel"]["selected"], ["information_content", "same_residue_pct"])
        self.assertIn("Residue Agreement 85 % (MSA column 111)", again["tooltip"])

        # The Viewer's data for a structure changed since: one property is gone, so its layer is dropped.
        data = json.loads(Path(self.sidecar_path(FIRST)).read_text(encoding="utf-8"))
        del data["properties"]["information_content"]
        fixtures.write_sidecar(self.directory, FIRST, data)
        self.open_page()
        gone = self.script("return {states: [0, 1].map(layerState), themes: reprThemes().map(t => t.name), panel: await panelNow()}")
        self.assertEqual([state["active"] for state in gone["states"]], ["plddt", "same_residue_pct"])
        self.assertEqual(gone["themes"], ["plddt-confidence", "mvs-annotation"])
        self.assertFalse(gone["states"][0]["isMvs"])
        self.assertEqual(gone["states"][0]["annotations"], [])
        self.assertEqual(gone["panel"]["selected"], ["plddt", "same_residue_pct"])


# Test-side readers of the page's Mol* state. Everything they return is plain JSON.
HELPERS = """
// Without WebGL, Mol* has no canvas, and a snapshot that includes the camera cannot be taken.
(() => {
    const snapshot = plugin.state.getSnapshot.bind(plugin.state);
    plugin.state.getSnapshot = params => snapshot({camera: !!plugin.canvas3d, ...params});
})();
window.layerState = function (index) {
    const structure = loadedStructures()[index];
    const model = structure.model.properties.cell.transform.params;
    const props = structure.properties.cell.transform.params;
    return {active: activeLayerId(structure), file: structureFile(structure),
            isMvs: model.properties['mvs-is-mvs-model'].isMvs,
            annotations: model.properties['mvs-annotations'].annotations.map(
                a => ({id: a.id, url: a.source.params.url, schema: a.schema})),
            modelAttach: model.autoAttach, tooltips: props.properties['mvs-annotation-tooltips'].tooltips,
            structureAttach: props.autoAttach,
            themes: structure.components.flatMap(c => c.representations.map(r => r.cell.transform.params.colorTheme))};
};
window.reprThemes = function () {
    return loadedStructures().map(s => s.components[0].representations[0].cell.transform.params.colorTheme);
};
window.colors = function (index, themeName, params) {
    const L = molstar.lib.structure;
    const data = loadedStructures()[index].cell.obj.data;
    const theme = plugin.representation.structure.themes.colorThemeRegistry.create(themeName, {structure: data}, params);
    const location = L.StructureElement.Location.create(data);
    const out = {};
    for (const unit of data.units) {
        location.unit = unit;
        for (const element of unit.elements) {
            location.element = element;
            if (L.StructureProperties.atom.label_atom_id(location) !== 'CA') continue;
            out[L.StructureProperties.residue.auth_seq_id(location)] =
                '#' + theme.color(location, false).toString(16).padStart(6, '0');
        }
    }
    return out;
};
// What Mol* shows for the pointer over residue `group` (counted from 0) of a structure.
window.tooltip = async function (index, group) {
    const repr = loadedStructures()[index].components[0].representations[0].cell.obj.data.repr;
    const loci = repr.getLoci({objectId: repr.renderObjects[0].id, instanceId: 0, groupId: group});
    plugin.managers.interactivity.lociHighlights.highlightOnly({loci, repr});
    await new Promise(resolve => setTimeout(resolve, 200));
    const labels = plugin.managers.lociLabels.getLabels().map(String);
    plugin.managers.interactivity.lociHighlights.clearHighlights();
    return labels;
};
window.fit = function (index) {
    const structure = loadedStructures()[index];
    const trace = s => readCaTrace((s.transform || s).cell.obj.data);
    const a = trace(loadedStructures()[0]), b = trace(structure);
    let sum = 0;
    const n = Math.min(a.coords.length, b.coords.length);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) sum += (a.coords[i][d] - b.coords[i][d]) ** 2;
    const node = structure.transform ? structure.transform.cell.transform.params : null;
    return {transformed: !!structure.transform, rmsd: Math.sqrt(sum / n), matrix: node ? Array.from(node.transform.params.data) : null};
};
window.panel = function () {
    const rows = [...layerPanel.querySelectorAll('.layer-row')];
    const first = layerPanel.querySelector('select');
    return {
        rows: rows.length, selects: layerPanel.querySelectorAll('select').length,
        title: (layerPanel.querySelector('.layer-title') || {}).textContent,
        names: rows.map(r => r.querySelector('.layer-name').textContent),
        options: first ? [...first.options].map(o => [o.value, o.textContent]) : [],
        selected: [...layerPanel.querySelectorAll('select')].map(s => s.value),
        legend_bar: !!layerPanel.querySelector('.layer-bar'),
        legend_ends: [...layerPanel.querySelectorAll('.layer-ends span')].map(s => s.textContent),
        legend_swatches: [...layerPanel.querySelectorAll('.layer-key')].map(k => k.textContent),
        note: [...layerPanel.querySelectorAll('.layer-note')].map(n => n.textContent).join(' | '),
        notes: [...layerPanel.querySelectorAll('.layer-note')].map(n => n.textContent),
        shown: layerPanel.style.display
    };
};
window.panelNow = async function () { await renderLayerPanel(); return panel(); };
"""


if __name__ == "__main__":
    unittest.main()
