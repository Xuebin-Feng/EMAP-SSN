"""PaCMAP as a layout method beside UMAP: its settings, its cache identity and what reads them.

LAYOUT_METHOD ("umap" or "pacmap", UMAP is the default) says which manifold method UMAP_MODE
("a manifold layout instead of physics") uses; PACMAP_NEIGHBORS (2 to 50) is the near pairs per node and the
edges kept and drawn per node. Everything written before they existed stays valid and means what it did:
settings documents and files, profiles, manifests, caches and their provenance. These tests cover the settings
documents, the layout generator's validation and dispatch, the manifest and folder name, the provenance the
Viewer reads back (2D and 3D), the headless export and summary, and the MCP layout contract.
"""

import asyncio
import contextlib
import copy
import io
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest
from desktop import Viewer_State
from desktop.Viewer_State import (
    DEFAULTS, INPUT_PROFILE_DEFAULTS, LAYOUT_SECTIONS, PROFILE_ENUM_VALUES, PROFILE_RANGES, ViewerSettingsError,
    decode_document, knn_topology_layout, layout_method, normalize_viewer_settings,
    prepare_network, resolve_cache_settings, resolve_selected_cache, resolve_viewer_document, strongest_edge_limit,
    validate_viewer_document,
)
from Layout_Cache_Generator import LayoutGenerationError, LayoutGenerationSettings, generate_layout_cache
from tests.layout_fixtures import HeadlessSettingsFixture, make_compatibility, preload_network_preparation, \
    settings_document, write_inputs
from tests.network_fixtures import write_network
from utilities.Headless_Settings import export_config_settings, layout_summary

PACMAP = dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=12)


class ManifestTests(unittest.TestCase):
    def test_a_pacmap_manifest_names_the_method_its_revision_and_its_neighbours(self):
        compatibility = make_compatibility(**PACMAP)
        self.assertEqual(compatibility["layout_mode"], "pacmap")
        self.assertEqual(compatibility["pacmap_revision"], Cache_Manifest.PACMAP_REVISION)
        self.assertEqual(compatibility["edge_filter"], {"mode": "pacmap_neighbors", "value": 12})
        self.assertNotIn("umap_knn_revision", compatibility)
        self.assertEqual(make_compatibility(**PACMAP, layout_dimensions=3)["layout_mode"], "pacmap_3d")

    def test_umap_and_physics_manifests_are_what_they_were(self):
        # The manifest id of every cache made before PaCMAP existed must come out the same.
        umap = make_compatibility(umap_mode=True, umap_neighbors=25)
        self.assertEqual(umap, {
            "sequence_sha256": "a" * 64, "network_sha256": "b" * 64, "network_type": "alignment",
            "alignment_score": "global", "normalization": "alignment_length", "layout_mode": "umap",
            "umap_knn_revision": 2, "edge_filter": {"mode": "umap_neighbors", "value": 25},
        })
        # The method is only read when the layout is a manifold layout; its neighbours never are otherwise.
        self.assertEqual(make_compatibility(umap_mode=True, umap_neighbors=25, layout_method="umap",
                                            pacmap_neighbors=3), umap)
        physics = make_compatibility()
        self.assertEqual(make_compatibility(layout_method="pacmap", pacmap_neighbors=3), physics)
        self.assertEqual(physics["layout_mode"], "physics")
        self.assertNotIn("pacmap_revision", physics)

    def test_the_manifest_tells_umap_from_pacmap_and_one_neighbour_count_from_another(self):
        ids = {Cache_Manifest.calculate_manifest_id(make_compatibility(**kwargs)) for kwargs in (
            dict(umap_mode=True, umap_neighbors=12),
            dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=12),
            dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=13),
            dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=12, layout_dimensions=3),
        )}
        self.assertEqual(len(ids), 4)
        # UMAP_NEIGHBORS means nothing to PaCMAP, so it is not part of a PaCMAP cache's identity.
        self.assertEqual(make_compatibility(**PACMAP), make_compatibility(**PACMAP, umap_neighbors=99))

    def test_a_missing_or_unknown_method(self):
        self.assertEqual(make_compatibility(umap_mode=True, layout_method=None)["layout_mode"], "umap")
        self.assertEqual(make_compatibility(umap_mode=True, layout_method=" PaCMAP ")["layout_mode"], "pacmap")
        with self.assertRaisesRegex(Cache_Manifest.CacheManifestError, "layout method"):
            make_compatibility(umap_mode=True, layout_method="tsne")

    def test_the_folder_name(self):
        with tempfile.TemporaryDirectory() as temp:
            write_inputs(pathlib.Path(temp))
            fasta, network = str(pathlib.Path(temp) / "set.fasta"), str(pathlib.Path(temp) / "network.h5")
            names = {}
            for label, kwargs in (
                ("umap", dict(umap_mode=True, umap_neighbors=15)),
                ("pacmap", dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=12)),
                ("pacmap_3d", dict(umap_mode=True, layout_method="pacmap", pacmap_neighbors=12, layout_dimensions=3)),
                ("physics", dict(similarity_threshold=0.5, layout_method="pacmap")),
            ):
                names[label] = Cache_Manifest.build_canonical_cache_name(
                    fasta, network, "alignment", "global", "alignment_length", **kwargs)
        self.assertEqual(names["umap"], "set_[model]_alignment_length_global_UMAP_k15")
        self.assertEqual(names["pacmap"], "set_[model]_alignment_length_global_PACMAP_k12")
        self.assertEqual(names["pacmap_3d"], "set_[model]_alignment_length_global_PACMAP_k12_3D")
        self.assertEqual(names["physics"], "set_[model]_alignment_length_global_Score0.5")


class SettingsDocumentTests(unittest.TestCase):
    def test_the_defaults_and_ranges(self):
        self.assertEqual(INPUT_PROFILE_DEFAULTS["LAYOUT_METHOD"], "umap")
        self.assertEqual(INPUT_PROFILE_DEFAULTS["PACMAP_NEIGHBORS"], 10)
        self.assertEqual(DEFAULTS["LAYOUT_METHOD"], "umap")
        self.assertEqual(PROFILE_ENUM_VALUES["LAYOUT_METHOD"], {"umap", "pacmap"})
        self.assertEqual(PROFILE_RANGES["PACMAP_NEIGHBORS"], (2, 50))
        self.assertIn("LAYOUT_METHOD", LAYOUT_SECTIONS["layout"])
        self.assertIn("PACMAP_NEIGHBORS", LAYOUT_SECTIONS["layout"])

    def layout_document(self, temp):
        return settings_document(pathlib.Path(temp))

    def test_documents_written_before_the_keys_existed_stay_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            document = self.layout_document(temp)
            old = copy.deepcopy(document)
            for key in ("LAYOUT_METHOD", "PACMAP_NEIGHBORS"):
                del old["layout"][key]
            decoded = decode_document(old, "layout")
            self.assertEqual((decoded["LAYOUT_METHOD"], decoded["PACMAP_NEIGHBORS"]), ("umap", 10))
            self.assertEqual(decode_document(document, "layout"), decoded)
            # A partial overlay does not invent them either: decode leaves optional keys to its caller.
            self.assertEqual(decode_document({"schema_version": 2, "kind": "layout"}, "layout", partial=True)
                             ["LAYOUT_METHOD"], "umap")

    def test_unknown_and_misplaced_fields_are_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            document = self.layout_document(temp)
            document["layout"]["LAYOUT_METHODS"] = "umap"
            with self.assertRaisesRegex(ValueError, "unknown or misplaced"):
                decode_document(document, "layout")
            document = self.layout_document(temp)
            document["network"]["LAYOUT_METHOD"] = "pacmap"
            with self.assertRaisesRegex(ValueError, "unknown or misplaced"):
                decode_document(document, "layout")

    def test_the_settings_survive_a_document_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            document = self.layout_document(temp)
            document["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=17)
            document["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
            root = pathlib.Path(temp)
            settings = LayoutGenerationSettings.from_document(document, project_root=root)
            self.assertEqual((settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS), ("pacmap", 17))
            again = settings.to_document(project_root=root)
            self.assertEqual(again["layout"]["LAYOUT_METHOD"], "pacmap")
            self.assertEqual(again["layout"]["PACMAP_NEIGHBORS"], 17)
            second = LayoutGenerationSettings.from_document(again, project_root=root)
            for key in ("UMAP_MODE", "LAYOUT_METHOD", "PACMAP_NEIGHBORS", "UMAP_NEIGHBORS", "UMAP_MIN_DIST",
                        "LAYOUT_SEED", "LAYOUT_DIMENSIONS", "BOX_SCALE"):
                self.assertEqual(getattr(second, key), getattr(settings, key), key)

    def test_a_flat_viewer_document_names_the_neighbour_count_of_its_method(self):
        base = {**DEFAULTS, "TARGET_CACHE_PATH": "x.h5", "NODE_FASTA_FILE": "a.fasta", "INPUT_HDF5": "b.h5",
                "MSA_FILE": "", "SIMILARITY_THRESHOLD": None, "TOP_EDGE_PERCENT": None, "UMAP_MODE": True}
        for method, missing, present in (("pacmap", "PACMAP_NEIGHBORS", "UMAP_NEIGHBORS"),
                                         ("umap", "UMAP_NEIGHBORS", "PACMAP_NEIGHBORS")):
            document = {key: value for key, value in base.items() if key != missing}
            document["LAYOUT_METHOD"] = method
            with self.assertRaisesRegex(ViewerSettingsError, f"{missing}: required"):
                normalize_viewer_settings(document, ROOT)
            document[missing] = DEFAULTS[missing]
            self.assertEqual(normalize_viewer_settings(document, ROOT)["LAYOUT_METHOD"], method)
        with self.assertRaisesRegex(ViewerSettingsError, "LAYOUT_METHOD"):
            normalize_viewer_settings({**base, "LAYOUT_METHOD": "tsne"}, ROOT)
        with self.assertRaisesRegex(ViewerSettingsError, "PACMAP_NEIGHBORS"):
            normalize_viewer_settings({**base, "LAYOUT_METHOD": "pacmap", "PACMAP_NEIGHBORS": 51}, ROOT)

    def test_the_viewer_schema_lists_the_keys_and_requires_the_methods_own_count(self):
        schema = Viewer_State._flat_viewer_settings_schema()
        self.assertEqual(schema["properties"]["LAYOUT_METHOD"]["enum"], ["pacmap", "umap"])
        self.assertEqual((schema["properties"]["PACMAP_NEIGHBORS"]["minimum"],
                          schema["properties"]["PACMAP_NEIGHBORS"]["maximum"]), (2, 50))
        manifold = schema["allOf"][0]["then"]
        self.assertEqual(manifold["then"]["required"], ["PACMAP_NEIGHBORS"])
        self.assertEqual(manifold["else"]["required"], ["UMAP_NEIGHBORS"])


class PredicateTests(unittest.TestCase):
    def test_the_layout_method_and_the_edge_limit(self):
        for settings, method, limit in (
            ({}, "umap", 15),
            ({"LAYOUT_METHOD": None}, "umap", 15),
            ({"LAYOUT_METHOD": "pacmap", "PACMAP_NEIGHBORS": 7, "UMAP_NEIGHBORS": 30}, "pacmap", 7),
            ({"LAYOUT_METHOD": "umap", "PACMAP_NEIGHBORS": 7, "UMAP_NEIGHBORS": 30}, "umap", 30),
            ({"LAYOUT_METHOD": "bogus", "UMAP_NEIGHBORS": 30}, "umap", 30),
            ({"LAYOUT_METHOD": "pacmap"}, "pacmap", 10),
        ):
            with self.subTest(settings=settings):
                namespace = SimpleNamespace(**settings)
                self.assertEqual(layout_method(namespace), method)
                self.assertEqual(strongest_edge_limit(namespace), limit)

    def test_a_manifold_layout_is_either_method(self):
        self.assertTrue(knn_topology_layout(SimpleNamespace(UMAP_MODE=True, LAYOUT_METHOD="pacmap")))
        self.assertFalse(knn_topology_layout(SimpleNamespace(UMAP_MODE=False, LAYOUT_METHOD="pacmap")))


class GeneratorSettingsTests(unittest.TestCase):
    def settings(self, **changes):
        with tempfile.TemporaryDirectory() as temp:
            document = settings_document(pathlib.Path(temp))
            document["layout"].update(UMAP_MODE=True)
            document["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
            for key, value in changes.items():
                section = next(name for name, keys in LAYOUT_SECTIONS.items() if key in keys)
                document[section][key] = value
            if not document["layout"]["UMAP_MODE"]:
                document["network"]["SIMILARITY_THRESHOLD"] = 0.5          # a physics layout needs a filter
            return LayoutGenerationSettings.from_document(document, project_root=temp)

    def test_validation(self):
        self.assertEqual(self.settings().LAYOUT_METHOD, "umap")
        for value in (2, 10, 50):
            self.assertEqual(self.settings(LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=value).PACMAP_NEIGHBORS, value)
        for changes, message in (
            ({"LAYOUT_METHOD": "tsne"}, "LAYOUT_METHOD must be umap or pacmap"),
            ({"LAYOUT_METHOD": 1}, "LAYOUT_METHOD must be umap or pacmap"),
            ({"PACMAP_NEIGHBORS": 1}, "PACMAP_NEIGHBORS must be an integer from 2 to 50"),
            ({"PACMAP_NEIGHBORS": 51}, "PACMAP_NEIGHBORS must be an integer from 2 to 50"),
            ({"PACMAP_NEIGHBORS": True}, "PACMAP_NEIGHBORS must be an integer from 2 to 50"),
            ({"PACMAP_NEIGHBORS": 10.5}, "PACMAP_NEIGHBORS must be an integer from 2 to 50"),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(LayoutGenerationError, message):
                self.settings(**changes)

    def test_from_namespace_reads_the_gui_values_and_defaults_missing_ones(self):
        values = dict(SAVED_LAYOUT_DIR="layouts", NODE_FASTA_FILE="set.fasta", INPUT_HDF5="net.h5", UMAP_MODE="True",
                      LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS="23", SIMILARITY_THRESHOLD=None)
        settings = LayoutGenerationSettings.from_namespace(SimpleNamespace(**values), cache_filename="version_00.h5",
                                                           project_root=ROOT)
        self.assertEqual((settings.UMAP_MODE, settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS), (True, "pacmap", 23))
        del values["LAYOUT_METHOD"], values["PACMAP_NEIGHBORS"]
        settings = LayoutGenerationSettings.from_namespace(SimpleNamespace(**values), cache_filename="version_00.h5",
                                                           project_root=ROOT)
        self.assertEqual((settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS), ("umap", 10))

    def test_only_a_pacmap_layout_records_the_new_settings_in_its_provenance(self):
        legacy = {"UMAP_MODE", "UMAP_NEIGHBORS", "UMAP_MIN_DIST", "LAYOUT_DIMENSIONS", "LAYOUT_SEED", "BOX_SCALE"}
        for changes, recorded in (
            ({}, False),                                                       # UMAP
            ({"LAYOUT_METHOD": "pacmap", "UMAP_MODE": False}, False),          # physics: the method means nothing
            ({"LAYOUT_METHOD": "pacmap"}, True),
        ):
            with self.subTest(changes=changes):
                settings = self.settings(**changes)
                params = settings.engine_params()
                self.assertTrue(legacy <= set(params))
                self.assertEqual({"LAYOUT_METHOD", "PACMAP_NEIGHBORS"} <= set(params), recorded)
                self.assertEqual("LAYOUT_METHOD" in params, recorded)
                self.assertEqual(settings.uses_pacmap(), recorded)

    def test_the_legacy_parameter_set_is_unchanged(self):
        # engine_params feeds layout_compatibility_id; for UMAP and physics layouts it is the set it was.
        settings = self.settings()
        self.assertEqual(sorted(settings.engine_params()), sorted([
            "AUTO_DT", "BOX_SCALE", "COULOMB_CUTOFF", "COULOMB_K", "DAMPING", "DT", "ENABLE_PROGRESSIVE_SIMULATION",
            "LAYOUT_DEVICE_SELECTION", "LAYOUT_DIMENSIONS", "LAYOUT_SEED", "MAX_FORCE_LIMIT", "MAX_STEPS",
            "MAX_TOTAL_REPULSION_FORCE", "PACKING_GEOMETRY", "PACKING_GRID_SIZE", "PACKING_PADDING",
            "PERCENTAGE_DROP_THRESHOLD", "RMSD_THRESHOLD", "RMSD_WINDOW", "SIMILARITY_THRESHOLD", "SPRING_K",
            "UMAP_MIN_DIST", "UMAP_MODE", "UMAP_NEIGHBORS",
        ]))


class GenerationTests(HeadlessSettingsFixture, unittest.TestCase):
    """A cache generated by each method, with a stand-in engine so that nothing is optimised."""

    def setUp(self):
        super().setUp()
        self.saved_config()

    def generate(self, document):
        """Generation with a stand-in engine that places the two nodes in the dimensions asked for."""
        import pandas  # noqa: F401  (imported before the sys.modules window, as the fixture does)
        preload_network_preparation()
        engine = SimpleNamespace(calculate_layout=lambda connectivity, count, params: (
            np.arange(count * int(params["LAYOUT_DIMENSIONS"]), dtype=np.float32).reshape(count, -1), 12.0))
        modules = {f"Layout_Engine_{name}": engine for name in ("SSN", "UMAP", "PaCMAP")}
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            return generate_layout_cache(LayoutGenerationSettings.from_document(document, project_root=self.root))

    def make(self, **changes):
        return self.cache_and_viewer(UMAP_MODE=True, SIMILARITY_THRESHOLD=None, **changes)

    def test_a_pacmap_layout_goes_to_the_pacmap_engine_with_its_mid_near_pairs(self):
        preload_network_preparation()
        seen = {}
        engines = {name: SimpleNamespace(calculate_layout=(lambda name: lambda connectivity, count, params: (
            seen.update({name: (connectivity, count, params)}) or (np.array([[0, 0], [1, 1]], dtype=np.float32), 12.0)
        ))(name)) for name in ("SSN", "UMAP", "PaCMAP")}
        layout = export_config_settings("layout", self.root)["settings_document"]
        layout["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7)
        layout["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        modules = {f"Layout_Engine_{name}": engine for name, engine in engines.items()}
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            result = generate_layout_cache(LayoutGenerationSettings.from_document(layout, project_root=self.root))
        self.assertEqual(list(seen), ["PaCMAP"])
        connectivity, count, params = seen["PaCMAP"]
        self.assertEqual(count, 2)
        self.assertEqual((params["LAYOUT_METHOD"], params["PACMAP_NEIGHBORS"]), ("pacmap", 7))
        # Two nodes are too few for a mid-near partner (six sampled nodes each); the pairs are looked up all the same.
        self.assertEqual(np.asarray(params["PACMAP_MID_NEAR"]).shape, (0, 2))
        self.assertEqual(result.manifest["compatibility"]["layout_mode"], "pacmap")
        self.assertEqual(pathlib.Path(result.cache_path).parent.name, "set_[model]_alignment_length_global_PACMAP_k7")
        with h5py.File(result.cache_path) as cache:
            recorded = json.loads(cache.attrs["layout_compatibility_json"])
        self.assertEqual((recorded["LAYOUT_METHOD"], recorded["PACMAP_NEIGHBORS"]), ("pacmap", 7))

    def test_umap_and_physics_layouts_do_not_look_up_pair_scores(self):
        preload_network_preparation()
        from utilities import PaCMAP_Pairs
        with mock.patch.object(PaCMAP_Pairs, "mid_near_pairs", side_effect=AssertionError("looked up")):
            self.make()
            self.make(UMAP_NEIGHBORS=5)
            self.cache_and_viewer(SIMILARITY_THRESHOLD=0.1)

    def test_the_viewer_reads_the_method_back_from_the_cache(self):
        path, viewer = self.make(LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7)
        settings = resolve_cache_settings(path)
        self.assertTrue(settings["UMAP_MODE"])
        self.assertEqual((settings["LAYOUT_METHOD"], settings["PACMAP_NEIGHBORS"]), ("pacmap", 7))
        self.assertIsNone(settings["SIMILARITY_THRESHOLD"])
        runtime = resolve_viewer_document(viewer, self.root)
        self.assertEqual((runtime["LAYOUT_METHOD"], runtime["PACMAP_NEIGHBORS"], runtime["UMAP_MODE"]),
                         ("pacmap", 7, True))
        # Viewer preferences do not carry the layout method; it comes from the cache.
        flat = decode_document(validate_viewer_document(viewer, self.root), "viewer")
        self.assertNotIn("LAYOUT_METHOD", flat)
        self.assertNotIn("PACMAP_NEIGHBORS", flat)

    def test_old_caches_open_as_umap_or_physics(self):
        path, _ = self.make(UMAP_NEIGHBORS=4)
        umap = resolve_cache_settings(path)
        self.assertEqual((umap["UMAP_MODE"], umap["LAYOUT_METHOD"], umap["UMAP_NEIGHBORS"]), (True, "umap", 4))
        self.assertEqual(umap["PACMAP_NEIGHBORS"], 10)
        with h5py.File(path) as cache:
            self.assertNotIn("LAYOUT_METHOD", json.loads(cache.attrs["layout_compatibility_json"]))
        physics, _ = self.cache_and_viewer(SIMILARITY_THRESHOLD=0.2)
        settings = resolve_cache_settings(physics)
        self.assertEqual((settings["UMAP_MODE"], settings["LAYOUT_METHOD"], settings["SIMILARITY_THRESHOLD"]),
                         (False, "umap", 0.2))

    def rewrite_provenance(self, path, change_parameters=None, change_manifest=None):
        """A cache whose provenance (and, where asked, folder manifest) is edited and re-hashed, as a forger would."""
        import hashlib
        folder = pathlib.Path(path).parent
        if change_parameters is not None:
            with h5py.File(path, "a") as cache:
                parameters = json.loads(cache.attrs["layout_compatibility_json"])
                change_parameters(parameters)
                text = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                cache.attrs["layout_compatibility_json"] = text
                cache.attrs["layout_compatibility_id"] = hashlib.sha256(text.encode()).hexdigest()
        if change_manifest is not None:
            manifest_path = folder / Cache_Manifest.MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text())
            change_manifest(manifest["compatibility"])
            manifest["manifest_id"] = Cache_Manifest.calculate_manifest_id(manifest["compatibility"])
            manifest_path.write_text(json.dumps(manifest))
            with h5py.File(path, "a") as cache:
                cache.attrs["cache_manifest_id"] = manifest["manifest_id"]

    def test_provenance_that_contradicts_itself_is_refused(self):
        for label, parameters, manifest, message in (
            ("method says umap", lambda p: p.update(LAYOUT_METHOD="umap"), None, "conflicting LAYOUT_METHOD"),
            ("neighbours out of range", lambda p: p.update(PACMAP_NEIGHBORS=51), None, "invalid PACMAP_NEIGHBORS"),
            ("neighbours missing", lambda p: p.pop("PACMAP_NEIGHBORS"), None, "invalid PACMAP_NEIGHBORS"),
            ("filter differs", None, lambda c: c["edge_filter"].update(value=8), "conflicting PaCMAP neighbor filter"),
            ("filter is UMAP's", None, lambda c: c.update(edge_filter={"mode": "umap_neighbors", "value": 7}),
             "conflicting PaCMAP neighbor filter"),
        ):
            with self.subTest(label):
                path, _ = self.make(LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7)
                self.rewrite_provenance(path, parameters, manifest)
                with self.assertRaisesRegex(ViewerSettingsError, message):
                    resolve_cache_settings(path)
        # And a UMAP cache that names the other method.
        path, _ = self.make()
        self.rewrite_provenance(path, lambda p: p.update(LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7))
        with self.assertRaisesRegex(ViewerSettingsError, "conflicting LAYOUT_METHOD"):
            resolve_cache_settings(path)
        # And a physics cache that says it is PaCMAP in its mode only.
        path, _ = self.cache_and_viewer(SIMILARITY_THRESHOLD=0.2)
        self.rewrite_provenance(path, None, lambda c: c.update(layout_mode="pacmap"))
        with self.assertRaises(ViewerSettingsError):
            resolve_cache_settings(path)

    def test_a_3d_pacmap_cache_is_for_the_vr_viewer_only(self):
        layout = export_config_settings("layout", self.root)["settings_document"]
        layout["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7, LAYOUT_DIMENSIONS=3)
        layout["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        path = self.generate(layout).cache_path
        with self.assertRaisesRegex(ViewerSettingsError, "pacmap_3d"):
            resolve_cache_settings(path)
        settings = resolve_cache_settings(path, allow_3d=True)
        self.assertEqual((settings["LAYOUT_METHOD"], settings["PACMAP_NEIGHBORS"]), ("pacmap", 7))
        self.assertEqual(pathlib.Path(path).parent.name, "set_[model]_alignment_length_global_PACMAP_k7_3D")

    def test_the_selected_cache_folder_names_the_method(self):
        settings = SimpleNamespace(
            NODE_FASTA_FILE=str(self.root / "set.fasta"), INPUT_HDF5=str(self.root / "network.h5"),
            SAVED_LAYOUT_DIR=str(self.root / "layouts"), ALIGNMENT_SCORE="global", NORM_MODE="alignment_length",
            UMAP_MODE=True, UMAP_NEIGHBORS=15, SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        self.assertEqual(pathlib.Path(resolve_selected_cache(settings)).parent.name,
                         "set_[model]_alignment_length_global_UMAP_k15")
        settings.LAYOUT_METHOD, settings.PACMAP_NEIGHBORS = "pacmap", 9
        self.assertEqual(pathlib.Path(resolve_selected_cache(settings)).parent.name,
                         "set_[model]_alignment_length_global_PACMAP_k9")
        self.assertEqual(pathlib.Path(resolve_selected_cache(settings, layout_dimensions=3)).parent.name,
                         "set_[model]_alignment_length_global_PACMAP_k9_3D")

    def test_the_summary_and_the_export_name_the_method(self):
        preload_network_preparation()
        layout = export_config_settings("layout", self.root)["settings_document"]
        layout["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=7)
        layout["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        with contextlib.redirect_stdout(io.StringIO()):
            settings = LayoutGenerationSettings.from_document(layout, project_root=self.root)
            engine = SimpleNamespace(calculate_layout=lambda *a: (np.array([[0, 0], [1, 1]], dtype=np.float32), 12.0))
            with mock.patch.dict(sys.modules, {"Layout_Engine_PaCMAP": engine}):
                generated = generate_layout_cache(settings)
        summary = layout_summary(generated, settings)
        self.assertEqual(summary["layout_mode"], "pacmap")
        self.assertEqual(summary["edge_filter"], {"mode": "pacmap_neighbors", "value": 7})
        self.assertIsNone(summary["effective_similarity_threshold"])

    def test_the_export_keeps_the_saved_method_when_an_overlay_does_not_name_it(self):
        saved = json.loads((self.root / "viewer_settings.json").read_text())
        saved.update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=33)
        (self.root / "viewer_settings.json").write_text(json.dumps(saved))
        document = export_config_settings("layout", self.root)["settings_document"]
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("pacmap", 33))
        overlay = {"schema_version": 2, "kind": "layout", "layout": {"LAYOUT_SEED": 7}}
        document = export_config_settings("layout", self.root, overlay=overlay)["settings_document"]
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("pacmap", 33))
        self.assertEqual(document["layout"]["LAYOUT_SEED"], 7)
        overlay["layout"] = {"LAYOUT_METHOD": "umap"}
        document = export_config_settings("layout", self.root, overlay=overlay)["settings_document"]
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("umap", 33))

    def test_settings_files_written_before_the_keys_existed_export_the_defaults(self):
        saved = json.loads((self.root / "viewer_settings.json").read_text())
        for key in ("LAYOUT_METHOD", "PACMAP_NEIGHBORS"):
            saved.pop(key, None)
        (self.root / "viewer_settings.json").write_text(json.dumps(saved))
        document = export_config_settings("layout", self.root)["settings_document"]
        self.assertEqual((document["layout"]["LAYOUT_METHOD"], document["layout"]["PACMAP_NEIGHBORS"]), ("umap", 10))


class PrepareNetworkTests(unittest.TestCase):
    """prepare_network keeps and draws PACMAP_NEIGHBORS edges per node in PaCMAP mode."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = pathlib.Path(self.temp.name) / "net.h5"
        write_network(self.path, 30, np.random.default_rng(0))
        self.headers = [f"N{i}" for i in range(30)]

    def run_prepare(self, **settings):
        values = dict(NODE_FASTA_FILE="", ALIGNMENT_SCORE="global", NORM_MODE="alignment_length",
                      SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None, UMAP_MODE=True, UMAP_NEIGHBORS=3,
                      LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=2)
        values.update(settings)
        callbacks = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output), h5py.File(self.path) as network:
            _, edges, scores = prepare_network(network, settings=SimpleNamespace(**values),
                                               selected_fasta_headers=self.headers,
                                               with_pair_source=lambda *found: callbacks.append(found))
        return edges, scores, callbacks, output.getvalue()

    def test_the_method_decides_how_many_edges_are_kept(self):
        edges, _, _, printed = self.run_prepare()
        self.assertIn("PaCMAP Mode enabled", printed)
        self.assertIn("(max 2 per node direction)", printed)
        umap_edges, _, _, printed = self.run_prepare(LAYOUT_METHOD="umap")
        self.assertIn("UMAP Mode enabled", printed)
        self.assertIn("(max 3 per node direction)", printed)
        self.assertLess(len(edges), len(umap_edges))
        # Every node's two strongest edges are there.
        degree = np.bincount(np.concatenate(edges), minlength=30)
        self.assertTrue((degree >= 2).all())
        # Without LAYOUT_METHOD the edges are UMAP's, as for every settings object that predates it.
        legacy = {key: value for key, value in vars(SimpleNamespace(UMAP_MODE=True, UMAP_NEIGHBORS=3)).items()}
        with contextlib.redirect_stdout(io.StringIO()), h5py.File(self.path) as network:
            legacy_edges = prepare_network(network, settings=SimpleNamespace(
                NODE_FASTA_FILE="", ALIGNMENT_SCORE="global", NORM_MODE="alignment_length", SIMILARITY_THRESHOLD=None,
                TOP_EDGE_PERCENT=None, **legacy), selected_fasta_headers=self.headers)[1]
        np.testing.assert_array_equal(legacy_edges, umap_edges)

    def test_the_pair_source_is_handed_over_with_the_kept_edges_while_the_file_is_open(self):
        edges, scores, callbacks, _ = self.run_prepare()
        (source, index_map, node_count, called_edges, called_scores), = callbacks
        self.assertEqual(node_count, 30)
        self.assertEqual(source.edge_count, 30 * 29 // 2)
        np.testing.assert_array_equal(called_edges, edges)
        np.testing.assert_array_equal(called_scores, scores)
        np.testing.assert_array_equal(index_map, np.arange(30))

    def test_physics_never_calls_it(self):
        _, _, callbacks, _ = self.run_prepare(UMAP_MODE=False, SIMILARITY_THRESHOLD=0.0)
        self.assertEqual(callbacks, [])

    def test_a_subset_of_nodes_maps_the_stored_numbers(self):
        self.headers = [f"N{i}" for i in range(0, 30, 2)]
        edges, _, callbacks, _ = self.run_prepare()
        (source, index_map, node_count, _, _), = callbacks
        self.assertEqual(node_count, 15)
        np.testing.assert_array_equal(index_map[::2], np.arange(15))
        self.assertTrue((edges < 15).all())


class RealGenerationTests(unittest.TestCase):
    """The real engine and lookup, from a network file to a published cache (2D and 3D)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        count = 60
        write_network(self.root / "network.h5", count, np.random.default_rng(1))
        rng = np.random.default_rng(2)
        alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
        sequences = ["".join(rng.choice(alphabet, 12)) for _ in range(count)]
        (self.root / "set.fasta").write_text("".join(f">N{i}\n{seq}\n" for i, seq in enumerate(sequences)))

    def generate(self, dimensions, **layout):
        document = settings_document(self.root)
        document["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=4, LAYOUT_DIMENSIONS=dimensions,
                                  **layout)
        document["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
        from utilities import PaCMAP_Pairs
        real = PaCMAP_Pairs.mid_near_pairs
        calls = []

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append((args, result))
            return result

        settings = LayoutGenerationSettings.from_document(document, project_root=self.root)
        with mock.patch.object(PaCMAP_Pairs, "mid_near_pairs", side_effect=spy), \
                contextlib.redirect_stdout(io.StringIO()):
            return generate_layout_cache(settings), calls

    def test_a_cache_in_two_and_three_dimensions(self):
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                result, calls = self.generate(dimensions)
                with h5py.File(result.cache_path) as cache:
                    positions = cache["positions"][:]
                    self.assertEqual(int(cache.attrs["layout_dimensions"]), dimensions)
                self.assertEqual(positions.shape, (60, dimensions))
                self.assertTrue(np.isfinite(positions).all())
                self.assertGreater(float(np.ptp(positions, axis=0).max()), 1.0)
                self.assertEqual(result.manifest["compatibility"]["layout_mode"], "pacmap" + ("_3d" if dimensions == 3 else ""))
                # The lookup read the whole stored pair file once, and found a partner for every slot.
                (args, pairs_found), = calls
                self.assertEqual(args[0].edge_count, 60 * 59 // 2)
                self.assertEqual(args[3].tolist(), list(range(60)))
                self.assertEqual(pairs_found.shape, (60 * 2, 2))              # 4 neighbours -> 2 mid-near each
                settings = resolve_cache_settings(result.cache_path, allow_3d=True)
                self.assertEqual((settings["LAYOUT_METHOD"], settings["PACMAP_NEIGHBORS"]), ("pacmap", 4))

    def test_the_same_seed_gives_the_same_cache_positions_and_another_does_not(self):
        first, _ = self.generate(2, LAYOUT_SEED=7)
        shutil_root = self.root / "layouts"
        positions = []
        for seed, name in ((7, "again.h5"), (8, "other.h5")):
            document = settings_document(self.root, cache_filename=name)
            document["layout"].update(UMAP_MODE=True, LAYOUT_METHOD="pacmap", PACMAP_NEIGHBORS=4, LAYOUT_SEED=seed)
            document["network"].update(SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None)
            with contextlib.redirect_stdout(io.StringIO()):
                result = generate_layout_cache(LayoutGenerationSettings.from_document(document, project_root=self.root))
            with h5py.File(result.cache_path) as cache:
                positions.append(cache["positions"][:])
        with h5py.File(first.cache_path) as cache:
            original = cache["positions"][:]
        np.testing.assert_array_equal(original, positions[0])
        self.assertFalse(np.array_equal(original, positions[1]))


class McpContractTests(unittest.TestCase):
    def test_the_contract_describes_both_fields(self):
        from mcp_server.pipeline.Pipeline_Operations import get_layout_schema, layout_defaults
        schema = get_layout_schema()["fields"]
        self.assertEqual(schema["LAYOUT_METHOD"]["enum"], ["umap", "pacmap"])
        self.assertEqual(layout_defaults()["LAYOUT_METHOD"], "umap")
        self.assertEqual(layout_defaults()["PACMAP_NEIGHBORS"], 10)
        self.assertEqual((schema["PACMAP_NEIGHBORS"]["minimum"], schema["PACMAP_NEIGHBORS"]["maximum"]), (2, 50))
        self.assertEqual(schema["LAYOUT_METHOD"]["section"], "layout")
        self.assertIn("PACMAP_NEIGHBORS", schema["LAYOUT_METHOD"]["description"])
        self.assertIn("PaCMAP", schema["UMAP_MODE"]["description"])
        self.assertIn("PaCMAP only", schema["PACMAP_NEIGHBORS"]["description"])

    def run_start(self, **arguments):
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_server.pipeline import Pipeline_Operations
        captured = {}

        def capture(document, project_root):
            captured.update(document)
            raise ValueError("captured")

        with tempfile.TemporaryDirectory() as temp, mock.patch(
                "Layout_Cache_Generator.LayoutGenerationSettings.from_document", side_effect=capture):
            with self.assertRaises(ToolError) as caught:
                asyncio.run(Pipeline_Operations.start_layout_job(
                    None, node_fasta_file="a.fasta", input_hdf5="b.h5", directories={"SAVED_LAYOUT_DIR": temp},
                    **arguments))
        return captured, str(caught.exception)

    def test_individual_arguments_choose_pacmap_through_parameters(self):
        document, _ = self.run_start(umap_mode=True, parameters={"layout_method": "pacmap", "pacmap_neighbors": 20})
        values = decode_document(document, "layout")
        self.assertEqual((values["UMAP_MODE"], values["LAYOUT_METHOD"], values["PACMAP_NEIGHBORS"]), (True, "pacmap", 20))
        document, _ = self.run_start()
        values = decode_document(document, "layout")
        self.assertEqual((values["UMAP_MODE"], values["LAYOUT_METHOD"], values["PACMAP_NEIGHBORS"]), (False, "umap", 10))

    def test_a_wrong_value_is_refused_before_any_work(self):
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_server.pipeline import Pipeline_Operations
        with tempfile.TemporaryDirectory() as temp:
            for parameters, message in (({"LAYOUT_METHOD": "tsne"}, "LAYOUT_METHOD must be umap or pacmap"),
                                        ({"PACMAP_NEIGHBORS": 80}, "PACMAP_NEIGHBORS must be an integer from 2 to 50"),
                                        ({"PACMAP_NEIGHBOR": 8}, "did you mean PACMAP_NEIGHBORS")):
                with self.subTest(parameters=parameters), self.assertRaisesRegex(ToolError, message):
                    asyncio.run(Pipeline_Operations.start_layout_job(
                        SimpleNamespace(), node_fasta_file="a.fasta", input_hdf5="b.h5", umap_mode=True,
                        parameters=parameters, directories={"SAVED_LAYOUT_DIR": temp}))


if __name__ == "__main__":
    unittest.main()
