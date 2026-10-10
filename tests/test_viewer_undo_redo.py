# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Viewer full-state undo/redo (EMAPSSN_Viewer.MainViewer): _save_state,
_do_undo, _do_redo, _apply_state and the 50-entry history. Compact metadata
history entries are covered in test_metadata, the restored render order in
test_node_render_order."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Command_Engine  # noqa: E402
from commands import redo as redo_command, undo as undo_command  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402


BLUE = [0.0, 0.0, 1.0, 1.0]
RED = [1.0, 0.0, 0.0, 1.0]


def make_viewer():
    """A four-node viewer holding every field a full history state captures."""
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 4
    viewer.full_headers = ["n0", "n1", "n2", "n3"]
    viewer.pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    viewer.visible_mask = np.ones(4, dtype=bool)
    viewer.current_colors = np.tile(BLUE, (4, 1)).astype(np.float32)
    viewer.current_sizes = np.full(4, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(4, "disc", dtype=object)
    viewer.node_render_order = np.arange(4, dtype=np.int32)
    viewer.cluster_labels = None
    viewer.last_cluster_params = None
    viewer.group_labels = [set(), set(), set(), set()]
    viewer.metadata = {
        "Length": {"type": "number", "values": np.array([10, 20, 30, 40])}
    }
    viewer._cacheable_attrs = {"sidebar_buttons_to_persist", "custom_scores"}
    viewer.sidebar_buttons_to_persist = []
    viewer.custom_scores = np.array([1.0, 2.0, 3.0, 4.0])
    viewer.selected_indices = []
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.update_selection_visual = mock.Mock()
    viewer.update_edges = mock.Mock()
    viewer.broadcast_event = mock.Mock()
    return viewer


def change_every_field(viewer):
    """Change what colour, hide, group, cluster, order, layout, metadata and
    custom-attribute commands change, in place where those commands do."""
    viewer.current_colors[1] = RED
    viewer.current_sizes[1] = 20.0
    viewer.current_shapes[1] = "star"
    viewer.visible_mask[2] = False
    viewer.group_labels[3].add("kinases")
    viewer.cluster_labels = np.array([0, 0, 1, -1], dtype=np.int32)
    viewer.last_cluster_params = {"method": "leiden"}
    viewer.node_render_order = np.array([0, 2, 3, 1], dtype=np.int32)
    viewer.pos[0] = [5.0, 5.0]
    viewer.metadata["Length"]["values"][0] = 99
    viewer.metadata["Note"] = {
        "type": "text",
        "values": np.array(["a", "b", "c", "d"], dtype=object),
    }
    viewer.custom_scores[0] = 99.0


class FakeMetadataHud:
    """The metadata HUD's display: what it shows, and whether it is showing."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.visible = False
        self.text = ""
        self.redrawn_for = []

    def show(self, text):
        self.visible = True
        self.text = text

    def hide(self):
        self.visible = False
        self.text = ""

    def on_node_clicked(self, node_idx):
        # As the real one: the displayed property's value for the clicked node.
        self.redrawn_for.append(node_idx)
        prop = self.viewer.meta_display_prop
        self.show(f"{prop}: {self.viewer.metadata[prop]['values'][node_idx]}")


def quietly(action):
    """Run an undo/redo step without its console print reaching the test log."""
    with redirect_stdout(io.StringIO()):
        return action()


def reset(viewer, *targets):
    """Run the reset command's execute_reset on a make_viewer() viewer."""
    viewer.tooltip = SimpleNamespace(text="")
    viewer.update_nodes = mock.Mock()
    viewer.update_console_background = mock.Mock()
    return quietly(lambda: Command_Engine.execute_reset(viewer, list(targets)))


class ViewerUndoRedoTests(unittest.TestCase):
    def assert_original_state(self, viewer):
        np.testing.assert_array_equal(viewer.current_colors, np.tile(BLUE, (4, 1)))
        np.testing.assert_array_equal(viewer.current_sizes, [10.0, 10.0, 10.0, 10.0])
        self.assertEqual(viewer.current_shapes.tolist(), ["disc"] * 4)
        np.testing.assert_array_equal(viewer.visible_mask, [True, True, True, True])
        self.assertEqual(viewer.group_labels, [set(), set(), set(), set()])
        self.assertIsNone(viewer.cluster_labels)
        self.assertIsNone(viewer.last_cluster_params)
        np.testing.assert_array_equal(viewer.node_render_order, [0, 1, 2, 3])
        np.testing.assert_array_equal(viewer.pos[0], [0.0, 0.0])
        self.assertEqual(list(viewer.metadata), ["Length"])
        np.testing.assert_array_equal(
            viewer.metadata["Length"]["values"], [10, 20, 30, 40]
        )
        np.testing.assert_array_equal(viewer.custom_scores, [1.0, 2.0, 3.0, 4.0])

    def assert_changed_state(self, viewer):
        np.testing.assert_array_equal(
            viewer.current_colors, [BLUE, RED, BLUE, BLUE]
        )
        np.testing.assert_array_equal(viewer.current_sizes, [10.0, 20.0, 10.0, 10.0])
        self.assertEqual(
            viewer.current_shapes.tolist(), ["disc", "star", "disc", "disc"]
        )
        np.testing.assert_array_equal(viewer.visible_mask, [True, True, False, True])
        self.assertEqual(viewer.group_labels, [set(), set(), set(), {"kinases"}])
        np.testing.assert_array_equal(viewer.cluster_labels, [0, 0, 1, -1])
        self.assertEqual(viewer.last_cluster_params, {"method": "leiden"})
        np.testing.assert_array_equal(viewer.node_render_order, [0, 2, 3, 1])
        np.testing.assert_array_equal(viewer.pos[0], [5.0, 5.0])
        self.assertEqual(list(viewer.metadata), ["Length", "Note"])
        np.testing.assert_array_equal(
            viewer.metadata["Length"]["values"], [99, 20, 30, 40]
        )
        self.assertEqual(
            viewer.metadata["Note"]["values"].tolist(), ["a", "b", "c", "d"]
        )
        np.testing.assert_array_equal(viewer.custom_scores, [99.0, 2.0, 3.0, 4.0])

    def test_undo_restores_every_saved_field_and_redo_reapplies_the_change(self):
        viewer = make_viewer()
        viewer._save_state()
        change_every_field(viewer)

        self.assertTrue(quietly(viewer._do_undo))
        self.assert_original_state(viewer)
        self.assertEqual(viewer.console_text.text, "Undo successful.")
        self.assertEqual(
            (len(viewer.position_history), len(viewer.redo_stack)), (0, 1)
        )

        self.assertTrue(quietly(viewer._do_redo))
        self.assert_changed_state(viewer)
        self.assertEqual(viewer.console_text.text, "Redo successful.")
        self.assertEqual(
            (len(viewer.position_history), len(viewer.redo_stack)), (1, 0)
        )

        # Redo saved the state it replaced, so undo goes back again.
        self.assertTrue(quietly(viewer._do_undo))
        self.assert_original_state(viewer)

        self.assertEqual(viewer.update_selection_visual.call_count, 3)
        self.assertEqual(viewer.update_edges.call_count, 3)
        self.assertEqual(viewer.broadcast_event.call_count, 3)
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["type"], "state_updated")
        self.assertEqual(event["visible_mask"], [True, True, True, True])
        self.assertEqual(event["columns"], ["Node ID", "Length"])

    def test_undo_and_redo_leave_the_sidebar_list_alone(self):
        # A `save` writes this list into the cache. Restoring an older one
        # would drop a button that a later `meta` call added.
        viewer = make_viewer()
        viewer._save_state()
        viewer.sidebar_buttons_to_persist.append("meta")
        viewer.custom_scores[0] = 99.0

        self.assertTrue(quietly(viewer._do_undo))
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])
        # Any other custom attribute is still restored.
        np.testing.assert_array_equal(viewer.custom_scores, [1.0, 2.0, 3.0, 4.0])

        viewer.sidebar_buttons_to_persist.append("agent")
        self.assertTrue(quietly(viewer._do_redo))
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta", "agent"])
        np.testing.assert_array_equal(viewer.custom_scores, [99.0, 2.0, 3.0, 4.0])

        # It stays a cacheable attribute, so `save` still writes it.
        self.assertIn("sidebar_buttons_to_persist", viewer._cacheable_attrs)

    def test_the_snapshot_holds_every_custom_attribute_but_the_sidebar_list(self):
        viewer = make_viewer()
        viewer.sidebar_buttons_to_persist.append("meta")
        viewer._save_state()
        self.assertEqual(
            list(viewer.position_history[-1]["_custom_data"]), ["custom_scores"]
        )

    def test_new_change_after_undo_discards_the_redo_branch(self):
        viewer = make_viewer()
        viewer._save_state()
        viewer.current_colors[0] = RED
        quietly(viewer._do_undo)
        self.assertEqual(len(viewer.redo_stack), 1)

        viewer._save_state()
        viewer.current_sizes[0] = 30.0

        self.assertEqual(viewer.redo_stack, [])
        self.assertEqual(len(viewer.position_history), 1)
        self.assertFalse(quietly(viewer._do_redo))
        self.assertEqual(viewer.console_text.text, "Nothing to redo.")
        np.testing.assert_array_equal(viewer.current_colors[0], BLUE)
        self.assertEqual(viewer.current_sizes[0], 30.0)

    def test_history_keeps_only_the_fifty_most_recent_states(self):
        viewer = make_viewer()
        for size in range(1, 52):
            viewer.current_sizes[0] = float(size)
            viewer._save_state()

        self.assertEqual(len(viewer.position_history), 50)
        self.assertEqual(
            [state["sizes"][0] for state in viewer.position_history],
            list(range(2, 52)),
        )
        # Fifty undos walk back to the oldest kept state; nothing is left after.
        for _ in range(50):
            self.assertTrue(quietly(viewer._do_undo))
        self.assertEqual(viewer.current_sizes[0], 2.0)
        self.assertFalse(quietly(viewer._do_undo))

    def test_a_reset_with_nothing_to_reset_saves_no_history_entry(self):
        # A no-op reset used to push an entry, which at 50 pushed out the oldest real one.
        viewer = make_viewer()
        for size in range(1, 51):
            viewer.current_sizes[0] = float(size)
            viewer._save_state()
        quietly(viewer._do_undo)
        history, redo = list(viewer.position_history), list(viewer.redo_stack)

        reset(viewer, "hide", "shapes", "order", "groups", "clusters", "network", "hidden")

        self.assertEqual(viewer.position_history, history)
        self.assertEqual(viewer.redo_stack, redo)

        # A reset that has something to reset is still one undo step.
        viewer.visible_mask[2] = False
        reset(viewer, "hide")

        self.assertEqual(len(viewer.position_history), len(history) + 1)
        self.assertEqual(viewer.redo_stack, [])
        quietly(viewer._do_undo)
        np.testing.assert_array_equal(viewer.visible_mask, [True, True, False, True])

    def test_every_target_with_something_to_reset_saves_one_history_entry(self):
        for target, change in (
                ("colors", lambda v: v.current_colors.__setitem__(1, RED)),
                ("sizes", lambda v: v.current_sizes.__setitem__(1, 99.0)),
                ("shapes", lambda v: v.current_shapes.__setitem__(1, "star")),
                ("clusters", lambda v: setattr(v, "cluster_labels", np.array([0, 0, 1, -1]))),
                ("groups", lambda v: v.group_labels[3].add("kinases")),
                ("hide", lambda v: v.visible_mask.__setitem__(2, False)),
                ("network", lambda v: (setattr(v, "original_pos", v.pos.copy()), v.pos.__setitem__(0, [5.0, 5.0]))),
                ("order", lambda v: setattr(v, "node_render_order", np.array([0, 2, 3, 1], dtype=np.int32)))):
            with self.subTest(target=target):
                viewer = make_viewer()
                change(viewer)
                reset(viewer, target)
                self.assertEqual(len(viewer.position_history), 1)

    def test_clearing_clusters_clears_their_parameters_and_undo_restores_both(self):
        viewer = make_viewer()
        change_every_field(viewer)

        reset(viewer, "clusters")

        self.assertIsNone(viewer.cluster_labels)
        self.assertIsNone(viewer.last_cluster_params)
        self.assertTrue(quietly(viewer._do_undo))
        np.testing.assert_array_equal(viewer.cluster_labels, [0, 0, 1, -1])
        self.assertEqual(viewer.last_cluster_params, {"method": "leiden"})

    def test_undo_and_redo_with_empty_history_change_nothing(self):
        viewer = make_viewer()
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertFalse(viewer._do_undo())
            undo_message = viewer.console_text.text
            self.assertFalse(viewer._do_redo())

        self.assertEqual(undo_message, "Nothing to undo.")
        self.assertEqual(viewer.console_text.text, "Nothing to redo.")
        self.assertEqual(
            output.getvalue().splitlines(), ["Nothing to undo.", "Nothing to redo."]
        )
        self.assertEqual((viewer.position_history, viewer.redo_stack), ([], []))
        self.assert_original_state(viewer)
        viewer.update_selection_visual.assert_not_called()
        viewer.broadcast_event.assert_not_called()

    def hud_viewer(self, selected_node=1):
        viewer = make_viewer()
        viewer.meta_display_prop = None
        viewer.selected_node_idx = selected_node
        viewer.hud_displays = {"meta_display": FakeMetadataHud(viewer)}
        return viewer

    def test_the_snapshot_holds_just_the_name_of_the_displayed_property(self):
        viewer = self.hud_viewer()
        viewer.meta_display_prop = "Length"
        viewer._save_state()
        self.assertEqual(viewer.position_history[-1]["meta_display_prop"], "Length")

        viewer.meta_display_prop = None
        viewer._save_state()
        self.assertIsNone(viewer.position_history[-1]["meta_display_prop"])

    def test_undo_and_redo_restore_the_property_the_hud_displays(self):
        # What `spectrum` does after its colours: display the property.
        viewer = self.hud_viewer()
        hud = viewer.hud_displays["meta_display"]
        viewer._save_state()
        viewer.current_colors[1] = RED
        viewer.meta_display_prop = "Length"
        hud.show("Length: 20")

        self.assertTrue(quietly(viewer._do_undo))
        self.assertIsNone(viewer.meta_display_prop)
        self.assertFalse(hud.visible)

        self.assertTrue(quietly(viewer._do_redo))
        self.assertEqual(viewer.meta_display_prop, "Length")
        self.assertTrue(hud.visible)
        self.assertEqual(hud.text, "Length: 20")

    def test_undo_brings_back_the_property_a_later_command_replaced(self):
        viewer = self.hud_viewer()
        viewer.metadata["Mass"] = {"type": "number", "values": np.array([1, 2, 3, 4])}
        hud = viewer.hud_displays["meta_display"]
        viewer.meta_display_prop = "Length"
        viewer._save_state()
        viewer.meta_display_prop = "Mass"
        hud.show("Mass: 2")

        quietly(viewer._do_undo)
        self.assertEqual(viewer.meta_display_prop, "Length")
        self.assertEqual(hud.text, "Length: 20")

    def test_an_unchanged_property_is_redrawn_not_hidden(self):
        viewer = self.hud_viewer()
        hud = viewer.hud_displays["meta_display"]
        viewer.meta_display_prop = "Length"
        hud.show("Length: 20")
        viewer._save_state()
        viewer.metadata["Length"]["values"][1] = 99

        quietly(viewer._do_undo)
        self.assertEqual(viewer.meta_display_prop, "Length")
        self.assertTrue(hud.visible)
        self.assertEqual(hud.redrawn_for, [1])

    def test_a_viewer_without_the_hud_still_restores_the_property_name(self):
        viewer = make_viewer()
        viewer.meta_display_prop = None
        viewer._save_state()
        viewer.meta_display_prop = "Length"

        quietly(viewer._do_undo)
        self.assertIsNone(viewer.meta_display_prop)

    def test_a_state_saved_without_the_property_leaves_it_alone(self):
        viewer = self.hud_viewer()
        viewer.meta_display_prop = "Length"
        state = viewer._get_current_state()
        del state["meta_display_prop"]
        viewer._apply_state(state)
        self.assertEqual(viewer.meta_display_prop, "Length")

    def test_restored_state_drops_selected_nodes_it_hides(self):
        viewer = make_viewer()
        viewer.visible_mask[1] = False
        viewer._save_state()
        viewer.visible_mask[1] = True
        viewer.selected_indices = [0, 1, 3]

        self.assertTrue(quietly(viewer._do_undo))

        np.testing.assert_array_equal(viewer.visible_mask, [True, False, True, True])
        self.assertEqual(viewer.selected_indices, [0, 3])
        viewer.update_selection_visual.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["visible_mask"], [True, False, True, True])
        self.assertEqual(event["selected_indices"], [0, 3])


class UndoRedoStepTests(unittest.TestCase):
    """`undo N` and `redo N` run up to N steps and report how many ran; other arguments are refused."""

    def viewer_with_history(self, count):
        """A viewer holding COUNT saved states, each taken before a change; its size ends at 20 + COUNT - 1."""
        viewer = make_viewer()
        for step in range(count):
            viewer._save_state()
            viewer.current_sizes[0] = 20.0 + step
        return viewer

    def run_step_command(self, viewer, module, args):
        """Run undo or redo on VIEWER; return the mocks that recorded its success and its failure."""
        with mock.patch.object(Command_Engine, "command_succeeded") as succeeded, \
                mock.patch.object(Command_Engine, "command_failed") as failed, \
                redirect_stdout(io.StringIO()):
            module.run(viewer, args)
        return succeeded, failed

    def test_a_count_undoes_that_many_steps(self):
        viewer = self.viewer_with_history(3)

        succeeded, failed = self.run_step_command(viewer, undo_command, ["2"])

        self.assertEqual((len(viewer.position_history), len(viewer.redo_stack)), (1, 2))
        self.assertEqual(viewer.current_sizes[0], 20.0)
        self.assertEqual(str(succeeded.call_args.args[1]), "Undid 2 steps.")
        failed.assert_not_called()

    def test_a_count_stops_when_the_history_runs_out(self):
        viewer = self.viewer_with_history(2)

        succeeded, _ = self.run_step_command(viewer, undo_command, ["9"])

        self.assertEqual((viewer.position_history, len(viewer.redo_stack)), ([], 2))
        self.assertEqual(viewer.current_sizes[0], 10.0)
        self.assertEqual(str(succeeded.call_args.args[1]), "Undid 2 steps.")

    def test_one_step_is_reported_in_the_singular(self):
        viewer = self.viewer_with_history(2)

        succeeded, _ = self.run_step_command(viewer, undo_command, ["1"])

        self.assertEqual(str(succeeded.call_args.args[1]), "Undid 1 step.")
        self.assertEqual(len(viewer.position_history), 1)

    def test_a_count_of_ten_digits_or_more_undoes_all_the_history(self):
        viewer = self.viewer_with_history(2)

        succeeded, failed = self.run_step_command(viewer, undo_command, ["9" * 40])

        self.assertEqual(viewer.position_history, [])
        self.assertEqual(str(succeeded.call_args.args[1]), "Undid 2 steps.")
        failed.assert_not_called()

    def test_redo_count_reapplies_that_many_steps(self):
        viewer = self.viewer_with_history(3)
        self.run_step_command(viewer, undo_command, ["3"])
        self.assertEqual(viewer.current_sizes[0], 10.0)

        succeeded, failed = self.run_step_command(viewer, redo_command, ["2"])

        self.assertEqual((len(viewer.position_history), len(viewer.redo_stack)), (2, 1))
        self.assertEqual(viewer.current_sizes[0], 21.0)
        self.assertEqual(str(succeeded.call_args.args[1]), "Redid 2 steps.")
        failed.assert_not_called()

    def test_a_count_with_nothing_to_undo_or_redo_reports_that(self):
        viewer = make_viewer()

        undone, _ = self.run_step_command(viewer, undo_command, ["3"])
        redone, _ = self.run_step_command(viewer, redo_command, ["3"])

        self.assertEqual(str(undone.call_args.args[1]), "Nothing to undo.")
        self.assertEqual(str(redone.call_args.args[1]), "Nothing to redo.")
        self.assertEqual((viewer.position_history, viewer.redo_stack), ([], []))
        viewer.update_edges.assert_not_called()

    def test_without_a_count_one_step_is_reported_as_before(self):
        viewer = self.viewer_with_history(2)

        undone, _ = self.run_step_command(viewer, undo_command, [])
        redone, _ = self.run_step_command(viewer, redo_command, [])

        self.assertEqual(str(undone.call_args.args[1]), "Undo successful.")
        self.assertEqual(str(redone.call_args.args[1]), "Redo successful.")
        self.assertEqual(viewer.current_sizes[0], 21.0)

    def test_a_refused_count_changes_nothing(self):
        refused = (["0"], ["00"], ["-1"], ["+2"], ["1.5"], ["two"], ["2", "3"], ["3", "extra"])
        for name, module in (("undo", undo_command), ("redo", redo_command)):
            for args in refused:
                with self.subTest(command=name, args=args):
                    viewer = self.viewer_with_history(2)
                    if name == "redo":
                        self.run_step_command(viewer, undo_command, ["2"])
                    before = (len(viewer.position_history), len(viewer.redo_stack), viewer.current_sizes[0])
                    viewer.update_edges.reset_mock()

                    succeeded, failed = self.run_step_command(viewer, module, args)

                    failed.assert_called_once()
                    succeeded.assert_not_called()
                    self.assertEqual(
                        (len(viewer.position_history), len(viewer.redo_stack), viewer.current_sizes[0]), before
                    )
                    viewer.update_edges.assert_not_called()

    def test_a_refusal_names_the_argument_that_is_wrong(self):
        viewer = make_viewer()

        _, failed = self.run_step_command(viewer, undo_command, ["2", "extra"])
        self.assertIn("'extra' was not used", str(failed.call_args.args[1]))
        _, failed = self.run_step_command(viewer, redo_command, ["-1"])
        self.assertIn("not '-1'", str(failed.call_args.args[1]))


if __name__ == "__main__":
    unittest.main()
