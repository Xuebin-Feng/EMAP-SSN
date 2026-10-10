# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The undo history's shared snapshots (EMAPSSN_Viewer.MainViewer).

A full-state history entry holds read-only arrays, shares each unchanged one
with an earlier entry, and holds the groups as a _GroupLabelsSnapshot. These
tests check that undo and redo restore exactly what the history restored when
every entry held its own deep copies: a random run of commands, saves, undos
and redos is replayed on a viewer with the earlier helpers, copied below, and
the two must agree on every field after every step."""

import copy
import gc
import io
import os
import random
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

import Cache_Manifest as cache_manifest  # noqa: E402
import Command_Engine  # noqa: E402
import Metadata_Core  # noqa: E402
from EMAPSSN_Viewer import (  # noqa: E402
    MainViewer,
    UNDO_EXCLUDED_ATTRIBUTES,
    _GroupLabelsSnapshot,
)
from utilities.Localization import Message  # noqa: E402


class EarlierHistoryViewer(MainViewer):
    """MainViewer with the history helpers as they were before snapshots were
    shared: every entry deep-copies every field, groups as one set per node."""

    def _get_current_state(self):
        """Helper to package the entire visual and spatial state."""
        return {
            'pos': self.pos.copy() if hasattr(self, 'pos') else None,
            'visible_mask': self.visible_mask.copy() if hasattr(self, 'visible_mask') else None,
            'colors': self.current_colors.copy() if hasattr(self, 'current_colors') else None,
            'sizes': self.current_sizes.copy() if hasattr(self, 'current_sizes') else None,
            'shapes': self.current_shapes.copy() if hasattr(self, 'current_shapes') else None,
            'node_render_order': self.node_render_order.copy() if hasattr(self, 'node_render_order') else None,
            'clusters': self.cluster_labels.copy() if getattr(self, 'cluster_labels', None) is not None else None,
            'groups': [g.copy() for g in self.group_labels] if getattr(self, 'group_labels', None) is not None else None,
            'last_cluster_params': self.last_cluster_params if getattr(self, 'last_cluster_params', None) is not None else None,
            'metadata': {k: {'type': v['type'], 'values': v['values'].copy()} for k, v in self.metadata.items()} if getattr(self, 'metadata', None) else {},
            # Just the name of the property the metadata HUD displays, which `spectrum` and `meta show` set.
            'meta_display_prop': getattr(self, 'meta_display_prop', None),
            '_custom_data': self._get_custom_attributes_snapshot()
        }

    def _apply_state(self, state):
        """Helper to unpack a state dictionary and apply it to the viewer."""
        if state['pos'] is not None: self.pos = state['pos'].copy()
        if state['visible_mask'] is not None: self.visible_mask = state['visible_mask'].copy()
        if state['colors'] is not None: self.current_colors = state['colors'].copy()
        if state['sizes'] is not None: self.current_sizes = state['sizes'].copy()
        if state['shapes'] is not None: self.current_shapes = state['shapes'].copy()
        if state.get('node_render_order') is not None:
            self.node_render_order = cache_manifest.validate_node_render_order(
                state['node_render_order'], self.n_nodes
            ).copy()
        else:
            self.node_render_order = np.arange(self.n_nodes, dtype=np.int32)

        if state['clusters'] is not None:
            self.cluster_labels = state['clusters'].copy()
            self.last_cluster_params = state.get('last_cluster_params')
        else:
            self.cluster_labels = None
            self.last_cluster_params = None

        if state.get('groups') is not None:
            self.group_labels = [g.copy() for g in state['groups']]
        else:
            self.group_labels = [set() for _ in range(self.n_nodes)]

        if state.get('metadata') is not None:
            self.metadata = {k: {'type': v['type'], 'values': v['values'].copy()} for k, v in state['metadata'].items()}
        else:
            self.metadata = {}

        if '_custom_data' in state and state['_custom_data'] is not None:
            self._apply_custom_attributes_snapshot(state['_custom_data'])

        # The HUD follows the property the restored state displayed; the redraw
        # of its value comes with the undo or redo that called this.
        if 'meta_display_prop' in state and state['meta_display_prop'] != getattr(self, 'meta_display_prop', None):
            self._set_metadata_hud_property(state['meta_display_prop'])

        # Clean up any active selections if those nodes are now hidden in this restored state
        if hasattr(self, 'selected_indices'):
            self.selected_indices = [i for i in self.selected_indices if self.visible_mask[i]]

        self.update_selection_visual()
        self.update_edges()

    def _get_custom_attributes_snapshot(self):
        if not getattr(self, '_cacheable_attrs', None):
            return {}
        snapshot = {}
        for attr_name in self._cacheable_attrs - UNDO_EXCLUDED_ATTRIBUTES:
            val = getattr(self, attr_name, None)
            if isinstance(val, np.ndarray):
                snapshot[attr_name] = val.copy()
            else:
                snapshot[attr_name] = copy.deepcopy(val)
        return snapshot

    def _apply_custom_attributes_snapshot(self, snapshot):
        if not hasattr(self, '_cacheable_attrs'):
            self._cacheable_attrs = set()
        for attr_name, val in snapshot.items():
            if attr_name in UNDO_EXCLUDED_ATTRIBUTES:
                continue
            if isinstance(val, np.ndarray):
                setattr(self, attr_name, val.copy())
            else:
                setattr(self, attr_name, copy.deepcopy(val))
            self._cacheable_attrs.add(attr_name)

    def _save_state(self):
        """Saves current state to history and clears redo stack."""
        self._append_history_entry(self._get_current_state())

    def _do_undo(self):
        if len(self.position_history) > 0:
            state = self.position_history.pop()
            if state.get("_history_kind"):
                self.redo_stack.append(state)
                self._apply_metadata_history_entry(state, undo=True)
            else:
                self.redo_stack.append(self._get_current_state())
                self._apply_state(state)
            msg = Message("Undo successful.")
            changed = True
        else:
            msg = Message("Nothing to undo.")
            changed = False
        Command_Engine.show_status(self, msg)
        print(msg)
        if changed:
            self._redraw_metadata_hud()
            self.broadcast_metadata_state()
        return changed

    def _do_redo(self):
        if len(self.redo_stack) > 0:
            state = self.redo_stack.pop()
            if state.get("_history_kind"):
                self.position_history.append(state)
                self._apply_metadata_history_entry(state, undo=False)
            else:
                self.position_history.append(self._get_current_state())
                self._apply_state(state)
            msg = Message("Redo successful.")
            changed = True
        else:
            msg = Message("Nothing to redo.")
            changed = False
        Command_Engine.show_status(self, msg)
        print(msg)
        if changed:
            self._redraw_metadata_hud()
            self.broadcast_metadata_state()
        return changed


class FakeMetadataHud:
    """The metadata HUD's display: what it shows, and whether it is showing."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.visible = False
        self.text = ""

    def show(self, text):
        self.visible = True
        self.text = text

    def hide(self):
        self.visible = False
        self.text = ""

    def on_node_clicked(self, node_idx):
        prop = self.viewer.meta_display_prop
        self.show(f"{prop}: {self.viewer.metadata[prop]['values'][node_idx]}")


N = 24
SHAPES = ("disc", "star", "square", "triangle_up")
GROUP_NAMES = ("kinases", "A", "B", "grp 3", 7, 7.0, True, 1)
LIVE_ARRAYS = (
    "pos", "visible_mask", "current_colors", "current_sizes", "current_shapes",
    "node_render_order",
)


def make_viewer(viewer_class=MainViewer):
    """A viewer holding every field a full history state captures."""
    viewer = viewer_class.__new__(viewer_class)
    viewer.n_nodes = N
    viewer.full_headers = [f"n{i}" for i in range(N)]
    viewer.pos = np.arange(2 * N, dtype=np.float32).reshape(N, 2)
    viewer.visible_mask = np.ones(N, dtype=bool)
    viewer.current_colors = np.tile([0.0, 0.0, 1.0, 1.0], (N, 1)).astype(np.float32)
    viewer.current_sizes = np.full(N, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(N, "disc", dtype=object)
    viewer.node_render_order = np.arange(N, dtype=np.int32)
    viewer.cluster_labels = None
    viewer.last_cluster_params = None
    viewer.group_labels = [set() for _ in range(N)]
    length = np.arange(N, dtype=np.float64) * 10.0
    length[[2, 5]] = np.nan
    viewer.metadata = {
        "Length": {"type": "number", "values": length},
        "Organism": {
            "type": "text",
            "values": np.array([f"org{i % 5}" for i in range(N)], dtype=object),
        },
    }
    viewer._cacheable_attrs = {
        "sidebar_buttons_to_persist", "custom_scores", "custom_notes",
    }
    viewer.sidebar_buttons_to_persist = []
    viewer.custom_scores = np.linspace(0.0, 1.0, N)
    viewer.custom_notes = {"author": "x", "tags": ["a"]}
    viewer.selected_indices = []
    viewer.selected_node_idx = 1
    viewer.meta_display_prop = None
    viewer.hud_displays = {"meta_display": FakeMetadataHud(viewer)}
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.update_selection_visual = lambda: None
    viewer.update_edges = lambda: None
    viewer.broadcast_event = lambda event: None
    return viewer


def quietly(action, *args):
    with redirect_stdout(io.StringIO()):
        return action(*args)


def _is_nan(value):
    try:
        return value != value
    except Exception:
        return False


class SameStateMixin:
    """Exact comparison: types, dtypes, bytes, element types, key order."""

    def assert_same(self, new, old, path):
        self.assertIs(type(new), type(old), path)
        if isinstance(old, np.ndarray):
            self.assertEqual((new.dtype, new.shape), (old.dtype, old.shape), path)
            if old.dtype == object:
                # Each element's type and repr, which tells NaN and -0.0 apart.
                self.assertEqual(
                    [(type(x), repr(x)) for x in new.ravel().tolist()],
                    [(type(x), repr(x)) for x in old.ravel().tolist()],
                    path,
                )
            else:
                self.assertEqual(new.tobytes(), old.tobytes(), path)
        elif isinstance(old, (set, frozenset)):
            self.assertEqual(new, old, path)
            self.assertEqual(
                sorted((type(x).__name__, repr(x)) for x in new),
                sorted((type(x).__name__, repr(x)) for x in old),
                path,
            )
        elif isinstance(old, dict):
            self.assertEqual(list(new), list(old), path)
            for key in old:
                self.assert_same(new[key], old[key], f"{path}.{key}")
        elif isinstance(old, (list, tuple)):
            self.assertEqual(len(new), len(old), path)
            for index, (a, b) in enumerate(zip(new, old)):
                self.assert_same(a, b, f"{path}[{index}]")
        elif _is_nan(old):
            self.assertTrue(_is_nan(new), path)
        else:
            self.assertEqual(new, old, path)
            if isinstance(old, float) and old == 0.0:
                self.assertEqual(str(new), str(old), path)

    def assert_same_viewers(self, new, old, step):
        for name in LIVE_ARRAYS + (
            "cluster_labels", "last_cluster_params", "group_labels", "metadata",
            "meta_display_prop", "selected_indices", "sidebar_buttons_to_persist",
        ):
            self.assert_same(getattr(new, name), getattr(old, name), f"{step}: {name}")
        self.assertEqual(new._cacheable_attrs, old._cacheable_attrs, step)
        for name in sorted(old._cacheable_attrs):
            self.assert_same(getattr(new, name), getattr(old, name), f"{step}: {name}")
        new_hud, old_hud = new.hud_displays["meta_display"], old.hud_displays["meta_display"]
        self.assertEqual((new_hud.visible, new_hud.text), (old_hud.visible, old_hud.text), step)
        self.assertEqual(new.console_text.text, old.console_text.text, step)
        # Commands change the live arrays in place: none may be a read-only
        # history array.
        live = [getattr(new, name) for name in LIVE_ARRAYS]
        live += [entry["values"] for entry in new.metadata.values()]
        live += [getattr(new, name) for name in new._cacheable_attrs]
        if new.cluster_labels is not None:
            live.append(new.cluster_labels)
        for array in live:
            if isinstance(array, np.ndarray):
                self.assertTrue(array.flags.writeable, step)

    def assert_same_entry(self, new, old, path):
        if old.get("_history_kind"):
            self.assert_same(new, old, path)
            return
        self.assertEqual(list(new), list(old), path)
        for key in old:
            value = new[key]
            if key == "groups" and isinstance(value, _GroupLabelsSnapshot):
                value = value.to_list()
            self.assert_same(value, old[key], f"{path}.{key}")
        arrays = [new[key] for key in ("pos", "visible_mask", "colors", "sizes",
                                       "shapes", "node_render_order", "clusters")]
        arrays += [entry["values"] for entry in new["metadata"].values()]
        arrays += list(new["_custom_data"].values())
        for array in arrays:
            if isinstance(array, np.ndarray):
                self.assertFalse(array.flags.writeable, path)

    def assert_same_history(self, new, old, step, newest_only=False):
        """Compare both stacks' entries; with ``newest_only``, just the top ones."""
        for stack in ("position_history", "redo_stack"):
            new_stack, old_stack = getattr(new, stack), getattr(old, stack)
            self.assertEqual(len(new_stack), len(old_stack), f"{step}: {stack}")
            start = max(len(old_stack) - 1, 0) if newest_only else 0
            for index in range(start, len(old_stack)):
                self.assert_same_entry(
                    new_stack[index], old_stack[index], f"{step}: {stack}[{index}]"
                )


def random_change(rng, nprng):
    """One command's change, as a function to run on each viewer in turn.

    The function builds its own arrays and lists for each viewer, so the two
    viewers never hold the same mutable object."""
    kind = rng.choice((
        "colors", "colors", "sizes", "sizes_array", "shapes", "pos", "pos_array",
        "visible", "order", "order_swap", "clusters", "clusters_edit",
        "clusters_clear", "group_add", "group_add", "group_remove", "group_reset",
        "group_none", "meta_edit", "meta_text_edit", "meta_add", "meta_replace",
        "meta_reorder", "meta_drop_unrecorded", "custom_edit", "custom_array",
        "custom_notes", "custom_new", "sidebar", "hud", "selection",
    ))
    mask = nprng.random(N) < rng.random()
    index = rng.randrange(N)

    if kind == "colors":
        rgba = nprng.random(4).astype(np.float32)

        def change(v):
            v.current_colors[mask] = rgba
    elif kind == "sizes":
        size = rng.choice((0.0, -0.0, 5.0, 12.5, float("nan")))

        def change(v):
            v.current_sizes[mask] = size
    elif kind == "sizes_array":
        sizes = nprng.random(N).astype(rng.choice((np.float32, np.float64)))

        def change(v):
            v.current_sizes = sizes.copy()
    elif kind == "shapes":
        shape = rng.choice(SHAPES)

        def change(v):
            v.current_shapes[mask] = shape
    elif kind == "pos":
        xy = nprng.random(2).astype(np.float32)

        def change(v):
            v.pos[index] = xy
    elif kind == "pos_array":
        pos = nprng.random((N, 2)).astype(np.float32)

        def change(v):
            v.pos = pos.copy()
    elif kind == "visible":
        value = rng.random() < 0.5

        def change(v):
            v.visible_mask[mask] = value
    elif kind == "order":
        order = nprng.permutation(N).astype(np.int32)

        def change(v):
            v.node_render_order = order.copy()
    elif kind == "order_swap":
        other = rng.randrange(N)

        def change(v):
            order = v.node_render_order
            order[index], order[other] = order[other], order[index]
    elif kind == "clusters":
        labels = nprng.integers(-1, 4, N).astype(rng.choice((np.int32, np.int64)))
        params = ("leiden", rng.randrange(10))

        def change(v):
            v.cluster_labels = labels.copy()
            v.last_cluster_params = params
    elif kind == "clusters_edit":
        label = rng.randrange(-1, 6)

        def change(v):
            if v.cluster_labels is not None:
                v.cluster_labels[mask] = label
    elif kind == "clusters_clear":
        def change(v):
            v.cluster_labels = None
            v.last_cluster_params = None
    elif kind == "group_add":
        name = rng.choice(GROUP_NAMES)
        members = np.flatnonzero(mask).tolist()

        def change(v):
            if v.group_labels is None:
                v.group_labels = [set() for _ in range(N)]
            for member in members:
                v.group_labels[member].add(name)
    elif kind == "group_remove":
        name = rng.choice(GROUP_NAMES)

        def change(v):
            for groups in v.group_labels or ():
                groups.discard(name)
    elif kind == "group_reset":
        def change(v):
            v.group_labels = [set() for _ in range(N)]
    elif kind == "group_none":
        def change(v):
            v.group_labels = None
    elif kind == "meta_edit":
        value = rng.choice((float("nan"), 0.0, -0.0, 3.5, float(rng.randrange(100))))

        def change(v):
            entry = v.metadata.get("Length")
            if entry is not None:
                entry["values"][mask] = value
    elif kind == "meta_text_edit":
        text = rng.choice(("org1", "orgX", ""))

        def change(v):
            entry = v.metadata.get("Organism")
            if entry is not None:
                entry["values"][index] = text
    elif kind == "meta_add":
        name = rng.choice(("Mass", "pI", "Note"))
        values = nprng.random(N)
        values[nprng.random(N) < 0.3] = np.nan

        def change(v):
            v.metadata[name] = {"type": "number", "values": values.copy()}
    elif kind == "meta_replace":
        values = nprng.random(N) * 100

        def change(v):
            if "Length" in v.metadata:
                v.metadata["Length"] = {"type": "number", "values": values.copy()}
    elif kind == "meta_reorder":
        def change(v):
            v.metadata = dict(reversed(list(v.metadata.items())))
    elif kind == "meta_drop_unrecorded":
        name = rng.choice(("Mass", "pI", "Note", "Organism"))

        def change(v):
            v.metadata.pop(name, None)
    elif kind == "custom_edit":
        value = rng.choice((float("nan"), -0.0, 0.25))

        def change(v):
            if isinstance(v.custom_scores, np.ndarray):
                v.custom_scores[mask] = value
    elif kind == "custom_array":
        values = nprng.random(N)

        def change(v):
            v.custom_scores = values.copy()
    elif kind == "custom_notes":
        key = rng.choice(("author", "tags", "count"))
        value = rng.choice(("y", 3, None))

        def change(v):
            if key == "tags":
                v.custom_notes.setdefault("tags", []).append(value)
            else:
                v.custom_notes[key] = value
    elif kind == "custom_new":
        values = nprng.integers(0, 9, N)

        def change(v):
            v.custom_extra = values.copy()
            v._cacheable_attrs.add("custom_extra")
    elif kind == "sidebar":
        def change(v):
            v.sidebar_buttons_to_persist.append("meta")
    elif kind == "hud":
        prop = rng.choice((None, "Length", "Organism", "Mass"))

        def change(v):
            v._set_metadata_hud_property(prop)
    else:  # selection
        selected = sorted(rng.sample(range(N), rng.randrange(4)))

        def change(v):
            v.selected_indices = list(selected)
            v.selected_node_idx = index
    change.kind = kind
    return change


def compact_metadata_change(rng):
    """A metadata cell edit or column deletion, which records a compact entry."""
    if rng.random() < 0.7:
        column = rng.choice(("Length", "Organism", "Mass"))
        row = rng.randrange(N)
        number = rng.choice((float("nan"), 1.5, -0.0))
        text = rng.choice(("orgZ", "org2"))

        def change(v):
            # As the web table's cell edit does.
            entry = v.metadata.get(column)
            if entry is None:
                return
            after = number if entry["type"] == "number" else text
            Metadata_Core._record_metadata_cell_history(
                v, column, row, entry["values"][row], after
            )
            entry["values"][row] = after
    else:
        def change(v):
            if v.metadata:
                Metadata_Core.delete_metadata_columns(
                    v, [sorted(v.metadata)[0]], broadcast=False
                )
    return change


class SharedSnapshotsMatchEarlierHistoryTests(SameStateMixin, unittest.TestCase):
    def run_sequence(self, seed, steps):
        rng = random.Random(seed)
        nprng = np.random.default_rng(seed)
        new, old = make_viewer(), make_viewer(EarlierHistoryViewer)
        counts = dict.fromkeys(
            ("save", "undo", "redo", "compact", "capped", "redo_cleared",
             "redo_after_new_save"), 0)
        saved_since_redo_cleared = False

        def both(change):
            change(new)
            change(old)

        for step in range(steps):
            # A long run of commands first, so the 50-entry cap is passed.
            action = "save" if step < 60 else rng.choices(
                ("save", "compact", "undo", "redo", "change"), (5, 1, 3, 3, 1)
            )[0]
            label = f"seed {seed} step {step} {action}"
            if action in ("save", "compact"):
                if old.redo_stack:
                    counts["redo_cleared"] += 1
                    saved_since_redo_cleared = True
                if len(old.position_history) == 50:
                    counts["capped"] += 1
            if action == "save":
                new._save_state()
                old._save_state()
                for _ in range(rng.choice((1, 1, 2))):
                    both(random_change(rng, nprng))
            elif action == "compact":
                change = compact_metadata_change(rng)
                quietly(change, new)
                quietly(change, old)
            elif action == "change":
                both(random_change(rng, nprng))
            else:
                for _ in range(rng.choice((1, 1, 2, 3))):
                    method = "_do_undo" if action == "undo" else "_do_redo"
                    if action == "redo" and old.redo_stack and saved_since_redo_cleared:
                        counts["redo_after_new_save"] += 1
                    changed = quietly(getattr(new, method))
                    self.assertEqual(changed, quietly(getattr(old, method)), label)
                    self.assert_same_viewers(new, old, label)
                    self.assert_same_history(new, old, label, newest_only=True)
            if step % 20 == 0:
                self.assert_same_history(new, old, label)
            counts[action] = counts.get(action, 0) + 1
        self.assert_same_viewers(new, old, f"seed {seed} end")
        self.assert_same_history(new, old, f"seed {seed} end")
        return counts

    def test_random_runs_restore_what_the_earlier_history_restored(self):
        totals = {}
        for seed in range(6):
            with self.subTest(seed=seed):
                counts = self.run_sequence(seed, 400)
                for key, value in counts.items():
                    totals[key] = totals.get(key, 0) + value
        # The runs went past the cap, redid after new saves had cleared the
        # redo stack, and mixed in compact metadata entries.
        for key in ("save", "undo", "redo", "compact", "capped", "redo_cleared",
                    "redo_after_new_save"):
            self.assertGreater(totals.get(key, 0), 0, key)


class SnapshotMutationSafetyTests(SameStateMixin, unittest.TestCase):
    def change_everything_in_place(self, viewer):
        """What commands do: write into the live arrays and sets."""
        viewer.current_colors[1:3] = [1.0, 0.0, 0.0, 1.0]
        viewer.current_sizes[::2] = 30.0
        viewer.current_shapes[4] = "star"
        viewer.pos[0] = [9.0, 9.0]
        viewer.visible_mask[5] = False
        viewer.node_render_order[[0, 1]] = viewer.node_render_order[[1, 0]]
        viewer.cluster_labels[3] = 2
        viewer.group_labels[6].add("kinases")
        viewer.group_labels[0].discard("A")
        viewer.metadata["Length"]["values"][0] = -1.0
        viewer.metadata["Organism"]["values"][1] = "changed"
        viewer.custom_scores[2] = 99.0
        viewer.custom_notes["tags"].append("b")

    def snapshot_of_live(self, viewer):
        return {
            "arrays": {name: getattr(viewer, name).copy() for name in LIVE_ARRAYS},
            "clusters": viewer.cluster_labels.copy(),
            "groups": [set(g) for g in viewer.group_labels],
            "metadata": {k: e["values"].copy() for k, e in viewer.metadata.items()},
            "custom_scores": viewer.custom_scores.copy(),
            "custom_notes": copy.deepcopy(viewer.custom_notes),
        }

    def assert_live_equals(self, viewer, expected, label):
        self.assert_same(
            {name: getattr(viewer, name) for name in LIVE_ARRAYS},
            expected["arrays"], label)
        self.assert_same(viewer.cluster_labels, expected["clusters"], label)
        self.assert_same(viewer.group_labels, expected["groups"], label)
        self.assert_same(
            {k: e["values"] for k, e in viewer.metadata.items()},
            expected["metadata"], label)
        self.assert_same(viewer.custom_scores, expected["custom_scores"], label)
        self.assert_same(viewer.custom_notes, expected["custom_notes"], label)

    def viewer(self):
        viewer = make_viewer()
        viewer.cluster_labels = np.array([0, 1, 2, -1] * (N // 4), dtype=np.int64)
        viewer.group_labels[0].add("A")
        return viewer

    def test_changing_live_arrays_in_place_leaves_the_snapshot_alone(self):
        viewer = self.viewer()
        before = self.snapshot_of_live(viewer)
        viewer._save_state()
        self.change_everything_in_place(viewer)
        after = self.snapshot_of_live(viewer)

        quietly(viewer._do_undo)
        self.assert_live_equals(viewer, before, "undo")
        # The restored arrays are the viewer's own: changing them leaves the
        # entry redo restores alone.
        self.change_everything_in_place(viewer)
        quietly(viewer._do_redo)
        self.assert_live_equals(viewer, after, "redo")
        # Redo saved the state it replaced: `before` changed once, as `after`.
        # A second change (it swaps the render order back and adds a tag)
        # leaves that entry alone too.
        self.change_everything_in_place(viewer)
        quietly(viewer._do_undo)
        self.assert_live_equals(viewer, after, "undo after redo")

    def test_entries_that_share_arrays_each_restore_their_own_state(self):
        viewer = self.viewer()
        first = self.snapshot_of_live(viewer)
        viewer._save_state()
        viewer._save_state()  # Nothing changed: shares every array.
        a, b = viewer.position_history
        for key in ("pos", "visible_mask", "colors", "sizes", "shapes",
                    "node_render_order", "clusters", "groups"):
            self.assertIs(b[key], a[key], key)
        self.assertIs(b["metadata"]["Length"]["values"], a["metadata"]["Length"]["values"])
        self.assertIs(b["_custom_data"]["custom_scores"], a["_custom_data"]["custom_scores"])

        self.change_everything_in_place(viewer)
        changed = self.snapshot_of_live(viewer)
        quietly(viewer._do_undo)
        self.assert_live_equals(viewer, first, "first undo")
        self.change_everything_in_place(viewer)
        quietly(viewer._do_undo)
        self.assert_live_equals(viewer, first, "second undo")
        quietly(viewer._do_redo)
        self.assert_live_equals(viewer, changed, "redo")
        quietly(viewer._do_redo)
        self.assert_live_equals(viewer, changed, "second redo")
        quietly(viewer._do_undo)
        quietly(viewer._do_undo)
        self.assert_live_equals(viewer, first, "undo after redo")

    def test_history_arrays_are_read_only(self):
        viewer = self.viewer()
        viewer._save_state()
        state = viewer.position_history[-1]
        for array in (state["colors"], state["metadata"]["Length"]["values"],
                      state["_custom_data"]["custom_scores"], state["groups"].members[0]):
            with self.assertRaises(ValueError):
                array[0] = array[0]


class SnapshotSharingTests(SameStateMixin, unittest.TestCase):
    def test_a_snapshot_copies_only_what_changed_since_the_last_one(self):
        viewer = make_viewer()
        viewer.group_labels[0].add("A")
        viewer._save_state()
        viewer.current_colors[0] = [1.0, 0.0, 0.0, 1.0]
        viewer._save_state()
        first, second = viewer.position_history
        self.assertIsNot(second["colors"], first["colors"])
        for key in ("pos", "visible_mask", "sizes", "shapes", "node_render_order", "groups"):
            self.assertIs(second[key], first[key], key)
        for name in viewer.metadata:
            self.assertIs(second["metadata"][name]["values"], first["metadata"][name]["values"])

    def test_a_compact_entry_in_between_is_skipped_for_sharing(self):
        viewer = make_viewer()
        viewer._save_state()
        Metadata_Core._record_metadata_cell_history(viewer, "Length", 0, 0.0, 5.0)
        viewer.metadata["Length"]["values"][0] = 5.0
        viewer._save_state()
        first, _, third = viewer.position_history
        self.assertIs(third["colors"], first["colors"])
        self.assertIsNot(third["metadata"]["Length"]["values"], first["metadata"]["Length"]["values"])
        self.assertIs(third["metadata"]["Organism"]["values"], first["metadata"]["Organism"]["values"])

    def test_undo_and_redo_share_with_the_state_they_restore(self):
        viewer = make_viewer()
        viewer._save_state()
        viewer.current_sizes[0] = 50.0
        restored = viewer.position_history[-1]
        quietly(viewer._do_undo)
        captured = viewer.redo_stack[-1]
        self.assertIs(captured["colors"], restored["colors"])
        self.assertIsNot(captured["sizes"], restored["sizes"])
        quietly(viewer._do_redo)
        self.assertIs(viewer.position_history[-1]["colors"], captured["colors"])
        self.assertIsNot(viewer.position_history[-1]["sizes"], captured["sizes"])

    def test_nan_counts_as_unchanged_and_a_sign_of_zero_as_changed(self):
        viewer = make_viewer()
        viewer.metadata["Length"]["values"][0] = 0.0
        viewer._save_state()
        viewer._save_state()
        first, second = viewer.position_history
        self.assertIs(second["metadata"]["Length"]["values"], first["metadata"]["Length"]["values"])

        viewer.metadata["Length"]["values"][0] = -0.0
        viewer._save_state()
        self.assertIsNot(viewer.position_history[-1]["metadata"]["Length"]["values"],
                         second["metadata"]["Length"]["values"])
        viewer.metadata["Length"]["values"][0] = 1.0
        quietly(viewer._do_undo)
        self.assertEqual(str(viewer.metadata["Length"]["values"][0]), "-0.0")

    def test_an_equal_but_distinct_object_counts_as_a_change(self):
        viewer = make_viewer()
        viewer._save_state()
        viewer.current_shapes[0] = "".join(["di", "sc"])  # A new "disc".
        viewer._save_state()
        first, second = viewer.position_history
        self.assertIsNot(second["shapes"], first["shapes"])
        self.assertIs(second["shapes"][0], viewer.current_shapes[0])

    def test_an_array_from_a_hand_built_state_is_never_shared(self):
        # Only the read-only arrays the history made itself are shared.
        viewer = make_viewer()
        state = viewer._get_current_state()
        state["colors"] = viewer.current_colors.copy()  # Writeable.
        viewer.position_history.append(state)
        viewer._save_state()
        self.assertIsNot(viewer.position_history[-1]["colors"], state["colors"])
        self.assertIs(viewer.position_history[-1]["sizes"], state["sizes"])

    def test_groups_are_stored_per_name_and_restored_with_their_types(self):
        labels = [set() for _ in range(6)]
        labels[0].update({"A", 1})
        labels[1].add(True)
        labels[2].add(1.0)
        labels[3].update({"A", "B"})
        labels[5].add(np.str_("A"))
        snapshot = _GroupLabelsSnapshot.capture(labels)

        self.assertEqual(snapshot.length, 6)
        for indices in snapshot.members:
            self.assertEqual(indices.dtype, np.int32)
            self.assertTrue(np.all(np.diff(indices) > 0))
        restored = snapshot.to_list()
        self.assert_same(restored, labels, "groups")
        self.assertIsNot(restored[0], labels[0])
        self.assertIsNot(snapshot.to_list()[4], restored[4])

        self.assertIs(_GroupLabelsSnapshot.capture(labels, snapshot), snapshot)
        labels[4].add("B")
        changed = _GroupLabelsSnapshot.capture(labels, snapshot)
        self.assertIsNot(changed, snapshot)
        kept = dict(zip(snapshot.keys, snapshot.members))
        for key, indices in zip(changed.keys, changed.members):
            self.assertEqual(indices is kept[key], key != (str, "B"), key)
        self.assert_same(changed.to_list(), labels, "changed groups")

    def test_restoring_groups_leaves_the_garbage_collector_as_it_was(self):
        snapshot = _GroupLabelsSnapshot.capture([{"A"}, set()])
        self.assertTrue(gc.isenabled())
        snapshot.to_list()
        self.assertTrue(gc.isenabled())
        gc.disable()
        try:
            snapshot.to_list()
            self.assertFalse(gc.isenabled())
        finally:
            gc.enable()

    def test_groups_other_than_a_list_of_sets_are_copied_as_before(self):
        labels = [["A"], set()]
        self.assertEqual(_GroupLabelsSnapshot.capture(labels), [["A"], set()])
        self.assertEqual(_GroupLabelsSnapshot.capture([]).to_list(), [])

    def test_a_state_holding_groups_as_sets_still_restores(self):
        viewer = make_viewer()
        state = viewer._get_current_state()
        state["groups"] = [{"A"} if i == 3 else set() for i in range(N)]
        viewer._apply_state(state)
        self.assertEqual(viewer.group_labels[3], {"A"})
        self.assertIsNot(viewer.group_labels[3], state["groups"][3])
        self.assertEqual(sum(map(len, viewer.group_labels)), 1)


if __name__ == "__main__":
    unittest.main()
