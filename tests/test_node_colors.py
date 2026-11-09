"""The color key (desktop/Node_Colors.py): what the node colors mean.

Commands that paint nodes declare their colors; the record is cacheable Viewer attributes, so undo, redo, save
and load carry it. What each command declares is tested with the command (test_cluster_command and the others);
here are the helpers, their takeover and pruning rules, undo and redo, and the shown language."""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import matplotlib as mpl
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from desktop import Node_Colors  # noqa: E402
from desktop.Node_Colors import clear_key, declare_categories, declare_gradient, key_state  # noqa: E402
from utilities import Localization  # noqa: E402
from utilities.Localization import Message  # noqa: E402

RED = (1.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
GRAY = (0.5, 0.5, 0.5, 1.0)


def plain_viewer(n=6):
    """A viewer with no history and no cacheable attributes, as the VR viewer and test fakes are."""
    return SimpleNamespace(n_nodes=n, current_colors=np.tile(GRAY, (n, 1)).astype(np.float32))


def paint(viewer, nodes, color):
    viewer.current_colors[nodes] = color


def history_viewer(n=6):
    """A MainViewer with its real undo history, for the attributes the key registers."""
    from EMAPSSN_Viewer import MainViewer

    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = n
    viewer.full_headers = [f"n{node}" for node in range(n)]
    viewer.pos = np.zeros((n, 2), dtype=np.float32)
    viewer.visible_mask = np.ones(n, dtype=bool)
    viewer.current_colors = np.tile(GRAY, (n, 1)).astype(np.float32)
    viewer.current_sizes = np.full(n, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(n, "disc", dtype=object)
    viewer.node_render_order = np.arange(n, dtype=np.int32)
    viewer.cluster_labels = None
    viewer.last_cluster_params = None
    viewer.group_labels = [set() for _ in range(n)]
    viewer.metadata = {}
    viewer.selected_indices = []
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.update_nodes = mock.Mock()
    viewer.update_selection_visual = mock.Mock()
    viewer.update_edges = mock.Mock()
    viewer.broadcast_event = mock.Mock()
    from EMAPSSN_Viewer import CUSTOM_ATTRIBUTES_INIT  # what the Viewer registers at startup
    viewer._cacheable_attrs = set(CUSTOM_ATTRIBUTES_INIT)
    for name, default in CUSTOM_ATTRIBUTES_INIT.items():
        setattr(viewer, name, default)
    return viewer


class CategoriesTests(unittest.TestCase):
    def test_a_layer_records_its_categories_and_who_owns_which_node(self):
        viewer = plain_viewer()
        paint(viewer, [0, 1, 2], RED)
        paint(viewer, [3], BLUE)
        declare_categories(viewer, Message("Clusters"), [("Cluster 1", RED), (Message("Noise"), BLUE)],
                           [0, 0, 0, 1], nodes=[0, 1, 2, 3])

        (layer,) = viewer.color_key_layers
        self.assertEqual((layer["id"], layer["kind"]), (1, "categories"))
        self.assertEqual(layer["title"], {"template": "Clusters", "values": {}})
        self.assertEqual([entry["rgba"] for entry in layer["categories"]], [list(RED), list(BLUE)])
        for name, dtype in (("owner", np.int32), ("category", np.int32), ("rgba", np.uint8), ("values", np.float32)):
            self.assertEqual(getattr(viewer, f"color_key_{name}").dtype, dtype)
        state = key_state(viewer)
        self.assertEqual(state["layers"][0]["title"], "Clusters")
        self.assertEqual([c["label"] for c in state["layers"][0]["categories"]], ["Cluster 1", "Noise"])
        self.assertEqual(state["owner"].tolist(), [1, 1, 1, 1, -1, -1])
        self.assertEqual(state["category"].tolist(), [0, 0, 0, 1, -1, -1])
        self.assertEqual(state["rgba"][0].tolist(), [255, 0, 0, 255])
        self.assertEqual(state["labelled"].tolist(), [True] * 4 + [False] * 2)
        self.assertTrue(np.isnan(state["values"]).all())

    def test_the_layers_are_json_safe(self):
        viewer = plain_viewer()
        declare_categories(viewer, Message("Subclusters of cluster {cluster}", cluster=np.int64(3)),
                           [("a", RED)], np.zeros(6, dtype=int))
        self.assertEqual(json.loads(json.dumps(viewer.color_key_layers)), viewer.color_key_layers)

    def test_a_later_layer_takes_over_its_nodes_and_leaves_the_others(self):
        viewer = plain_viewer()
        paint(viewer, slice(0, 6), RED)
        declare_categories(viewer, "first", [("red", RED)], np.zeros(6, dtype=int))
        paint(viewer, [4, 5], BLUE)
        declare_categories(viewer, "second", [("blue", BLUE)], [0, 0], nodes=[4, 5])

        state = key_state(viewer)
        self.assertEqual([layer["title"] for layer in state["layers"]], ["first", "second"])
        self.assertEqual(state["owner"].tolist(), [1, 1, 1, 1, 2, 2])
        self.assertTrue(state["labelled"].all())

    def test_a_layer_that_no_longer_owns_a_node_is_dropped_and_its_id_not_reused(self):
        viewer = plain_viewer()
        declare_categories(viewer, "first", [("gray", GRAY)], np.zeros(6, dtype=int))
        paint(viewer, slice(0, 6), BLUE)
        declare_categories(viewer, "second", [("blue", BLUE)], np.zeros(6, dtype=int))
        self.assertEqual([layer["id"] for layer in viewer.color_key_layers], [2])
        paint(viewer, slice(0, 6), GREEN)
        declare_categories(viewer, "third", [("green", GREEN)], np.zeros(6, dtype=int))
        self.assertEqual([layer["id"] for layer in viewer.color_key_layers], [3])

    def test_a_node_painted_without_a_declaration_reads_as_unlabelled(self):
        viewer = plain_viewer()
        paint(viewer, slice(0, 6), RED)
        declare_categories(viewer, "first", [("red", RED)], np.zeros(6, dtype=int))
        paint(viewer, [2], BLUE)
        paint(viewer, [3], (1.0, 0.0, 0.5, 1.0))  # another color
        self.assertEqual(key_state(viewer)["labelled"].tolist(), [True, True, False, False, True, True])
        paint(viewer, [2], RED)  # painted back to what the key expects
        self.assertTrue(key_state(viewer)["labelled"][2])

    def test_a_change_of_alpha_alone_keeps_the_node_labelled(self):
        viewer = plain_viewer()
        paint(viewer, slice(0, 6), RED)
        declare_categories(viewer, "first", [("red", RED)], np.zeros(6, dtype=int))
        paint(viewer, [3], (1.0, 0.0, 0.0, 0.5))  # still red, only fainter
        paint(viewer, [4], (1.0, 0.0, 0.0, 0.0))
        paint(viewer, [5], (1.0, 0.0, 0.01, 0.5))  # a different color as well
        state = key_state(viewer)
        self.assertEqual(state["labelled"].tolist(), [True, True, True, True, True, False])
        self.assertEqual(state["rgba"][3].tolist(), [255, 0, 0, 255])  # the recorded alpha stays

    def test_colors_are_compared_as_the_page_gets_them_rounded_to_a_byte(self):
        viewer = plain_viewer()
        paint(viewer, slice(0, 6), (0.4, 0.4, 0.4, 1.0))
        declare_categories(viewer, "first", [("gray", GRAY)], np.zeros(6, dtype=int))
        paint(viewer, [0], (0.4 + 0.001, 0.4, 0.4, 1.0))  # under half a byte away
        self.assertTrue(key_state(viewer)["labelled"].all())

    def test_clearing_some_nodes_prunes_a_layer_that_owns_none_and_clearing_all_empties_the_key(self):
        viewer = plain_viewer()
        declare_categories(viewer, "first", [("gray", GRAY)], [0, 0], nodes=[0, 1])
        declare_categories(viewer, "second", [("gray", GRAY)], [0, 0], nodes=[2, 3])
        clear_key(viewer, nodes=[0, 1])
        self.assertEqual([layer["id"] for layer in viewer.color_key_layers], [2])
        self.assertEqual(key_state(viewer)["owner"].tolist(), [-1, -1, 2, 2, -1, -1])
        clear_key(viewer, nodes=np.array([True, True, True, True, False, False]))
        self.assertIsNone(key_state(viewer))
        declare_categories(viewer, "third", [("gray", GRAY)], [0])
        clear_key(viewer)
        self.assertIsNone(key_state(viewer))
        self.assertTrue(all(getattr(viewer, name) is None for name in Node_Colors.KEY_ATTRIBUTES))

    def test_clearing_a_viewer_that_never_declared_does_nothing_wrong(self):
        viewer = plain_viewer()
        clear_key(viewer)
        clear_key(viewer, nodes=[0])
        self.assertIsNone(key_state(viewer))


class GradientTests(unittest.TestCase):
    def test_a_gradient_keeps_its_stops_range_values_and_extras(self):
        viewer = plain_viewer()
        cmap = mpl.colormaps["viridis"]
        paint(viewer, [0, 1], cmap([0.0, 1.0]))
        paint(viewer, [2], (0.8, 0.8, 0.8, 0.4))
        declare_gradient(viewer, "Length", "viridis", cmap, 10.0, 20.0, [0, 1], [10.0, 20.0],
                         extras=[(Message("No value"), (0.8, 0.8, 0.8, 0.4), [2])])

        state = key_state(viewer)
        (layer,) = state["layers"]
        self.assertEqual((layer["kind"], layer["title"], layer["scheme"], layer["min"], layer["max"]),
                         ("gradient", "Length", "viridis", 10.0, 20.0))
        self.assertEqual(len(layer["stops"]), 16)
        np.testing.assert_allclose(layer["stops"][0], cmap(0.0)[:3])
        np.testing.assert_allclose(layer["stops"][15], cmap(1.0)[:3])
        self.assertEqual([entry["label"] for entry in layer["extras"]], ["No value"])
        self.assertEqual(state["owner"].tolist(), [1, 1, 1, -1, -1, -1])
        self.assertEqual(state["category"].tolist(), [-1, -1, 0, -1, -1, -1])
        self.assertEqual(state["values"][:2].tolist(), [10.0, 20.0])
        self.assertTrue(np.isnan(state["values"][2:]).all())
        self.assertEqual(state["labelled"].tolist(), [True, True, True, False, False, False])
        json.dumps(viewer.color_key_layers)

    def test_a_gradient_that_colors_only_its_extras_survives(self):
        viewer = plain_viewer()
        paint(viewer, [0], RED)
        declare_gradient(viewer, "t", "viridis", mpl.colormaps["viridis"], None, None, [], [],
                         extras=[("Selected", RED, [0])])
        layer = key_state(viewer)["layers"][0]
        self.assertEqual((layer["min"], layer["max"]), (None, None))
        self.assertEqual(key_state(viewer)["owner"].tolist(), [1, -1, -1, -1, -1, -1])

    def test_a_gradient_taken_over_loses_its_values(self):
        viewer = plain_viewer()
        cmap = mpl.colormaps["viridis"]
        paint(viewer, [0, 1], cmap([0.0, 1.0]))
        declare_gradient(viewer, "t", "viridis", cmap, 0, 1, [0, 1], [0.0, 1.0])
        paint(viewer, [1], RED)
        declare_categories(viewer, "c", [("red", RED)], [0], nodes=[1])
        state = key_state(viewer)
        self.assertTrue(np.isnan(state["values"][1]))
        self.assertEqual(state["values"][0], 0.0)


class PlainViewerTests(unittest.TestCase):
    def test_a_viewer_without_cacheable_attributes_just_gets_the_attributes(self):
        viewer = plain_viewer()
        declare_categories(viewer, "t", [("a", GRAY)], np.zeros(6, dtype=int))
        self.assertFalse(hasattr(viewer, "_cacheable_attrs"))
        self.assertEqual(key_state(viewer)["labelled"].tolist(), [True] * 6)

    def test_a_key_of_another_size_reads_as_no_key(self):
        viewer = plain_viewer()
        declare_categories(viewer, "t", [("a", GRAY)], np.zeros(6, dtype=int))
        viewer.n_nodes = 7
        viewer.current_colors = np.tile(GRAY, (7, 1))
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(key_state(viewer))


class UndoTests(unittest.TestCase):
    """The key rides on the Viewer's cacheable attributes, so every undo step holds it."""

    def paint_and_declare(self, viewer, nodes, color, title):
        viewer._save_state()
        paint(viewer, nodes, color)
        declare_categories(viewer, title, [("c", color)], np.zeros(len(nodes), dtype=int), nodes=nodes)

    def undo(self, viewer):
        with redirect_stdout(io.StringIO()):
            return viewer._do_undo()

    def redo(self, viewer):
        with redirect_stdout(io.StringIO()):
            return viewer._do_redo()

    def test_undoing_the_first_coloring_of_a_session_brings_back_an_empty_key(self):
        viewer = history_viewer()
        self.assertIsNone(viewer.color_key_layers)
        self.paint_and_declare(viewer, [0, 1], RED, "first")
        self.assertEqual(len(viewer.color_key_layers), 1)

        self.assertTrue(self.undo(viewer))
        np.testing.assert_allclose(viewer.current_colors, np.tile(GRAY, (6, 1)))
        self.assertIsNone(key_state(viewer))
        self.assertIsNone(viewer.color_key_owner)

        self.assertTrue(self.redo(viewer))
        np.testing.assert_allclose(viewer.current_colors[0], RED)
        state = key_state(viewer)
        self.assertEqual(state["owner"].tolist(), [1, 1, -1, -1, -1, -1])
        self.assertEqual(state["labelled"].tolist(), [True, True, False, False, False, False])

    def test_undo_and_redo_restore_the_colors_and_the_key_together_and_ids_are_never_reused(self):
        viewer = history_viewer()
        self.paint_and_declare(viewer, [0, 1], RED, "first")
        self.paint_and_declare(viewer, [1, 2], BLUE, "second")
        before = key_state(viewer)
        self.assertEqual(before["owner"].tolist(), [1, 2, 2, -1, -1, -1])

        self.undo(viewer)
        state = key_state(viewer)
        self.assertEqual([layer["title"] for layer in state["layers"]], ["first"])
        self.assertEqual(state["owner"].tolist(), [1, 1, -1, -1, -1, -1])
        self.assertTrue(state["labelled"][:2].all())

        # A new declaration after the undo does not take the undone layer's id.
        self.paint_and_declare(viewer, [4], GREEN, "third")
        self.assertEqual([layer["id"] for layer in viewer.color_key_layers], [1, 3])

    def test_a_snapshot_holds_the_key_before_the_first_declaration_too(self):
        viewer = history_viewer()
        snapshot = viewer._get_custom_attributes_snapshot()
        for name in Node_Colors.KEY_ATTRIBUTES:
            self.assertIn(name, snapshot)
            self.assertIsNone(snapshot[name])

    def test_a_reset_of_the_colors_is_one_undo_step_with_its_key(self):
        viewer = history_viewer()
        self.paint_and_declare(viewer, [0, 1], RED, "first")
        viewer._save_state()
        paint(viewer, slice(0, 6), GRAY)
        clear_key(viewer)
        self.assertIsNone(key_state(viewer))
        self.undo(viewer)
        self.assertEqual(key_state(viewer)["owner"].tolist(), [1, 1, -1, -1, -1, -1])


def good_key_viewer():
    viewer = plain_viewer()
    paint(viewer, slice(0, 6), RED)
    declare_categories(viewer, Message("Clusters"), [(Message("Cluster {number}", number=1), RED)],
                       np.zeros(6, dtype=int))
    return viewer


class MalformedKeyTests(unittest.TestCase):
    """A layout cache is shared and untrusted: whatever it put in the key's attributes reads as no key or a trimmed one."""

    def state_after(self, damage):
        viewer = good_key_viewer()
        damage(viewer)
        with redirect_stdout(io.StringIO()):
            return viewer, key_state(viewer)

    def test_layers_that_are_not_a_list_of_layers_read_as_no_key(self):
        for bad in ("text", 7, {"id": 1}, ["text"], [None], [{"kind": "categories"}]):
            with self.subTest(bad=bad):
                _, state = self.state_after(lambda v: setattr(v, "color_key_layers", bad))
                self.assertIsNone(state)

    def test_a_layer_that_is_malformed_is_dropped_and_a_good_one_beside_it_stays(self):
        def damage(viewer):
            good = viewer.color_key_layers[0]
            viewer.color_key_layers = ["text", {**good, "id": None}, {**good, "id": True}, {**good, "kind": "other"},
                                       {**good, "categories": [{"label": "a", "rgba": [1, 0, 0]}]},  # 3 numbers
                                       {**good, "categories": [{"rgba": [1, 0, 0, 1]}]},  # no label
                                       {**good, "categories": "none"},
                                       {**good, "title": 5},
                                       {**good, "id": 9, "title": {"values": {}}},  # a dict title with no template
                                       good]
        viewer, state = self.state_after(damage)
        self.assertEqual([layer["id"] for layer in state["layers"]], [1])
        self.assertTrue(state["labelled"].all())

    def test_a_gradient_with_a_bad_range_or_stops_is_dropped(self):
        viewer = plain_viewer()
        declare_gradient(viewer, "t", "viridis", mpl.colormaps["viridis"], 0, 1, [0], [0.5])
        good = viewer.color_key_layers[0]
        for name, bad in (("min", "low"), ("max", [1]), ("stops", [[1, 2]]), ("stops", "x"), ("extras", None),
                          ("scheme", 3)):
            with self.subTest(name=name):
                viewer.color_key_layers = [{**good, name: bad}]
                with redirect_stdout(io.StringIO()):
                    self.assertIsNone(key_state(viewer))
        viewer.color_key_layers = [{**good, "min": None, "max": None}]
        self.assertIsNotNone(key_state(viewer))

    def test_a_template_cannot_read_attributes_or_items_or_a_missing_value(self):
        def title_of(template, values=None):
            def damage(viewer):
                viewer.color_key_layers[0]["title"] = {
                    "template": template, "values": {"x": [7], "number": 3} if values is None else values}
            return self.state_after(damage)[1]["layers"][0]["title"]
        self.assertEqual(title_of("{x.__class__}"), "{x.__class__}")
        self.assertEqual(title_of("{x[0]}"), "{x[0]}")
        self.assertEqual(title_of("{missing}"), "{missing}")
        self.assertEqual(title_of("{number!r}"), "{number!r}")
        self.assertEqual(title_of("{number:>5}"), "{number:>5}")
        self.assertEqual(title_of("{}"), "{}")
        self.assertEqual(title_of("{unclosed"), "{unclosed")
        self.assertEqual(title_of("n={number} {{literal}}"), "n=3 {literal}")
        self.assertEqual(title_of("{x}"), "[7]")  # a value that is not text shows as str()
        self.assertEqual(title_of("{number}", values="not a dict"), "{number}")

    def test_a_label_that_is_a_dict_without_a_template_drops_the_layer_and_never_raises(self):
        def damage(viewer):
            viewer.color_key_layers[0]["categories"][0] = {"label": {"values": {}}, "rgba": [1, 0, 0, 1]}
        self.assertIsNone(self.state_after(damage)[1])
        self.assertEqual(Node_Colors.shown({"values": {}}), "")
        self.assertEqual(Node_Colors.shown(None), "None")

    def test_arrays_of_the_wrong_shape_or_dtype_read_as_no_key(self):
        cases = {
            "color_key_owner": [np.zeros(5, np.int32), np.zeros((6, 1), np.int32), np.array(["a"] * 6),
                                np.full(6, 2 ** 40, dtype=np.int64), np.zeros(6, np.float32), "x", [1] * 6],
            "color_key_category": [np.zeros(7, np.int32), np.zeros(6, object)],
            "color_key_rgba": [np.zeros((6, 3), np.uint8), np.zeros(6, np.uint8), np.full((6, 4), 300, np.int64),
                               np.full((6, 4), -1, np.int64)],
            "color_key_values": [np.zeros(2, np.float32), np.array(["a"] * 6)],
        }
        for name, bad_arrays in cases.items():
            for bad in bad_arrays:
                with self.subTest(name=name, bad=repr(bad)[:30]):
                    self.assertIsNone(self.state_after(lambda v: setattr(v, name, bad))[1])

    def test_a_compatible_dtype_is_converted(self):
        def damage(viewer):
            viewer.color_key_owner = viewer.color_key_owner.astype(np.int64)
            viewer.color_key_rgba = viewer.color_key_rgba.astype(np.int64)
            viewer.color_key_values = viewer.color_key_values.astype(np.float64)
        state = self.state_after(damage)[1]
        self.assertTrue(state["labelled"].all())
        self.assertEqual((state["owner"].dtype, state["rgba"].dtype, state["values"].dtype),
                         (np.int32, np.uint8, np.float32))

    def test_a_category_the_layer_does_not_have_is_unlabelled(self):
        def damage(viewer):
            viewer.color_key_category[:2] = [5, -3]
        state = self.state_after(damage)[1]
        self.assertEqual(state["labelled"].tolist(), [False, False, True, True, True, True])

    def test_one_warning_line_is_printed_once_for_a_damaged_key_and_none_for_a_sound_one(self):
        viewer = good_key_viewer()
        with redirect_stdout(io.StringIO()) as printed:
            key_state(viewer)
        self.assertEqual(printed.getvalue(), "")
        viewer.color_key_layers = ["text"]
        with redirect_stdout(io.StringIO()) as printed:
            key_state(viewer)
            key_state(viewer)
        self.assertEqual(printed.getvalue().count("Warning: the color key"), 1)

    def test_a_declaration_after_a_bad_key_starts_afresh_or_keeps_what_was_readable(self):
        for name, bad in (("color_key_layers", "text"), ("color_key_layers", [None, 5]), ("color_key_owner", "x"),
                          ("color_key_rgba", np.zeros((2, 2))), ("color_key_values", None), ("color_key_category", 3)):
            with self.subTest(name=name):
                viewer = good_key_viewer()
                setattr(viewer, name, bad)
                paint(viewer, [2], BLUE)
                with redirect_stdout(io.StringIO()):
                    declare_categories(viewer, "again", [("blue", BLUE)], [0], nodes=[2])
                    state = key_state(viewer)
                self.assertEqual(state["layers"][-1]["title"], "again")
                self.assertEqual(state["owner"][2], state["layers"][-1]["id"])
                self.assertTrue(state["labelled"][2])

    def test_a_node_owned_by_a_layer_that_was_dropped_does_not_pass_to_a_new_layer_with_its_id(self):
        viewer = good_key_viewer()
        viewer.color_key_layers = [{**viewer.color_key_layers[0], "id": 1, "title": 5}]  # unreadable, so dropped
        viewer._color_key_last_id = 0
        paint(viewer, [0], BLUE)
        with redirect_stdout(io.StringIO()):
            declare_categories(viewer, "new", [("blue", BLUE)], [0], nodes=[0])
        state = key_state(viewer)
        self.assertEqual(state["layers"][0]["id"], 1)
        self.assertEqual(state["owner"].tolist(), [1, -1, -1, -1, -1, -1])

    def test_clearing_nodes_of_a_bad_key_does_not_raise(self):
        viewer = good_key_viewer()
        viewer.color_key_owner = "x"
        with redirect_stdout(io.StringIO()):
            clear_key(viewer, nodes=[0])
        self.assertIsNone(key_state(viewer))


class TranslationTests(unittest.TestCase):
    def test_a_title_and_a_fixed_label_are_shown_in_the_current_language_and_data_labels_are_not(self):
        from desktop.Desktop_App import message_translator

        viewer = plain_viewer()
        declare_categories(viewer, Message("Clusters"), [("Cluster 3", RED), (Message("Noise"), GRAY)],
                           [0, 0, 0, 1, 1, 1])
        declare_categories(viewer, Message("Residue at {position}", position="53"), [(Message("Gap"), GRAY)],
                           [0], nodes=[0])
        english = key_state(viewer)["layers"]
        self.assertEqual((english[0]["title"], english[0]["categories"][1]["label"]), ("Clusters", "Noise"))

        previous = Localization.set_translator(message_translator("zh_CN"))
        self.addCleanup(Localization.set_translator, previous)
        chinese = key_state(viewer)["layers"]
        self.assertEqual(chinese[0]["title"], "聚类")
        self.assertEqual([entry["label"] for entry in chinese[0]["categories"]], ["Cluster 3", "噪声"])
        self.assertEqual(chinese[1]["title"], "位置 53 的残基")
        numbered = plain_viewer()
        declare_categories(numbered, "t", [(Message("Cluster {number}", number=3), RED)], [0], nodes=[0])
        self.assertEqual(key_state(numbered)["layers"][0]["categories"][0]["label"], "聚类 3")
        self.assertEqual(chinese[1]["categories"][0]["label"], "空位")
        # What is stored stays English, so the cache reads in whichever language is shown when it is read.
        self.assertEqual(viewer.color_key_layers[0]["title"]["template"], "Clusters")


if __name__ == "__main__":
    unittest.main()
