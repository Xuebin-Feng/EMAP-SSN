# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tip panel in the Tools window (EMAPSSN_Tools.ToolsGUI)."""

import os
import pathlib
import re
import tempfile
import unittest
from unittest import mock

from tests.tip_panel_fixtures import (  # noqa: E402
    WithApplication,
    chips,
    images,
    plain_text,
)
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QLabel, QSpinBox  # noqa: E402

from desktop.Desktop_App import NAME_ITEM_ROLE  # noqa: E402
from desktop.Studio_Theme import status_color  # noqa: E402
from desktop.Tip_Panel import (  # noqa: E402
    FieldTip,
    SpacedTipLabel,
    Tip,
    tip_html,
    tip_plain,
)
from tests.consistency_report import mark_colour  # noqa: E402
from tests.tools_gui_fixtures import open_tools_window  # noqa: E402


class ToolsWindowTipTests(WithApplication):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    FREE_ROOM = 12 * 1024 ** 3

    def setUp(self):
        import EMAPSSN_Tools

        self.tools = EMAPSSN_Tools
        # The free room on a drive moves between two readings: the window's tooltips are made when it is built and
        # the tests read the live notes again later, so every reading in this class sees one fixed value.
        self.window = open_tools_window(
            self, show=False, free_bytes=mock.Mock(return_value=self.FREE_ROOM)
        )
        self.panel = self.window.tip_panel

    def widget(self, script, variable):
        for path, data in self.window.script_data.items():
            if os.path.basename(path) == script:
                return data["inputs"][variable]["widget"]
        raise KeyError((script, variable))

    def enter(self, widget):
        self.app.sendEvent(widget, QEvent(QEvent.Type.Enter))

    def shown(self):
        return self.panel.text()

    def test_every_tool_tip_is_laid_out(self):
        for script, tips in self.window.SCRIPT_TIPS.items():
            for key, tip in tips.items():
                self.assertIsInstance(tip, Tip, (script, key))
        # ...and every field's entry holds the field it describes, whether or not it has a tip of its own.
        for path, data in self.window.script_data.items():
            for variable, info in data["inputs"].items():
                entry = self.window.tip_db[info["widget"]]
                self.assertIsInstance(entry, FieldTip, (path, variable))
                self.assertEqual(entry.field.key, variable)

    def test_a_field_gets_its_tip_with_its_field_and_a_plain_tooltip(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        entry = self.window.tip_db[combo]
        self.assertIsInstance(entry, FieldTip)
        self.assertEqual((entry.field.key, entry.field.default, entry.field.widget), ("EXECUTION_MODE", "auto", combo))
        self.assertEqual(entry.field.title(), "Execution Mode")
        # The widget's accessible description is the tip as plain text.
        self.assertEqual(combo.toolTip(), tip_plain(entry))
        self.assertIn("Scalar: Processes one pairwise score matrix at a time.", combo.toolTip())
        batch = self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE")
        self.assertEqual(batch.toolTip(), tip_plain(self.window.tip_db[batch]))
        self.assertIn("Warning: Larger values improve throughput but require more RAM.", batch.toolTip())

    def test_on_a_cpu_only_computer_the_tooltip_does_not_list_the_tiled_option_the_dropdown_lacks(self):
        from utilities import Hardware_Acceleration

        real = Hardware_Acceleration.get_available_devices
        with mock.patch.object(Hardware_Acceleration, "get_available_devices",
                               lambda *a, **k: [d for d in real(*a, **k) if d.is_cpu]):
            window = open_tools_window(self, show=False, isolate=False)
        for path, data in window.script_data.items():
            if os.path.basename(path) == "Align_Similarity_Matrix.py":
                combo = data["inputs"]["EXECUTION_MODE"]["widget"]
        self.assertLess(combo.findData("tiled"), 0)
        self.assertEqual(combo.toolTip(), tip_plain(window.tip_db[combo]))
        self.assertNotIn("Tiled", combo.toolTip())

    def test_the_options_follow_the_dropdown(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        def listed():
            return [combo.itemText(combo.findData(value)) for value in ("auto", "scalar", "tiled")
                    if combo.findData(value) >= 0]

        # The device decides whether "tiled" is offered: try both.
        if combo.findData("tiled") < 0:
            combo.addItem("Tiled", "tiled")
        self.enter(combo)
        # The options as the dropdown names them, then the facts: the default and the key.
        self.assertEqual(chips(self.shown()), listed() + ["Auto", "EXECUTION_MODE"])
        self.assertIn("Tiled", listed())
        self.assertIn("memory-bounded embedding tiles", plain_text(self.shown()))
        combo.removeItem(combo.findData("tiled"))
        self.enter(combo)
        self.assertEqual(chips(self.shown()), listed() + ["Auto", "EXECUTION_MODE"])
        self.assertNotIn("Tiled", listed())
        self.assertNotIn("memory-bounded embedding tiles", plain_text(self.shown()))
        self.assertIn("Processes one pairwise score matrix at a time", plain_text(self.shown()))

    def test_the_precision_tip_has_its_caution_and_leaves_out_what_the_device_lacks(self):
        combo = self.widget("Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION")
        self.enter(combo)
        marks = [mark_colour(name) for name in images(self.shown())]
        self.assertEqual(marks, [status_color("warning")])
        self.assertIn("BF16 prints a low-precision warning", plain_text(self.shown()))
        if combo.findData("tf32") >= 0:
            combo.removeItem(combo.findData("tf32"))
        self.enter(combo)
        self.assertNotIn("TF32 (Nvidia GPU Only)", chips(self.shown()))

    def test_the_workers_tip_shows_this_computers_logical_cpus_and_its_range(self):
        spin_holder = self.widget("Align_Similarity_Matrix.py", "WORKERS")
        self.enter(spin_holder)
        text = re.sub(r"\s+", " ", plain_text(self.shown()))
        self.assertIn(f"This computer has {self.tools.MAX_CORES} logical CPU", text)
        self.assertIn(f"Range 1 – {self.tools.MAX_CORES}", text)
        self.assertIn("Default 12", text)
        self.assertIn("WORKERS", text)

    def test_the_memory_caution_is_a_warning_and_the_remark_an_info_note(self):
        self.enter(self.widget("Embedding_MSA.py", "WORKERS"))
        marks = [mark_colour(name) for name in images(self.shown())]
        self.assertEqual(marks, [status_color("warning"), status_color("subtle"), status_color("subtle")])
        self.assertIn("about 16 GB for 44,000 sequences", plain_text(self.shown()))

    def test_the_host_cache_tip_says_what_auto_picks_now(self):
        gib = self.tools.GIB
        control = self.widget("Align_Similarity_Matrix.py", "HOST_CACHE_GB")
        with mock.patch.object(self.tools, "resolve_host_cache_bytes", return_value=8 * gib) as budget, \
                mock.patch.object(self.tools, "system_memory_bytes", return_value=(64 * gib, 24 * gib)):
            self.enter(control)
        budget.assert_called_once_with("auto")
        text = re.sub(r"\s+", " ", plain_text(self.shown()))
        self.assertIn("AUTO would use 8.0 GiB now, with 24.0 GiB of memory available.", text)
        self.assertEqual(
            [chip_text for chip_text in chips(self.shown())][:2], ["AUTO ON", "AUTO OFF"],
        )
        self.assertIn(f"Range 0.0 – {self.tools.HOST_CACHE_MAX_GB:.1f} GiB", text)
        self.assertIn("Default auto", text)
        self.assertIn(f"up to {self.tools.HOST_CACHE_MAX_GB:g} GiB", text)

    WORKER_FIELDS = (
        ("Align_Similarity_Matrix.py", "WORKERS"),
        ("Align_Substitution_Matrix.py", "NUM_THREADS"),
        ("Embedding_MSA.py", "WORKERS"),
        ("Network_Injection.py", "WORKERS"),
        ("Embedding_SSEARCH.py", "WORKERS"),
    )
    DEVICE_FIELDS = (
        "Generate_Embeddings.py", "Align_Similarity_Matrix.py", "Embedding_MSA.py",
        "Network_Injection.py", "Embedding_SSEARCH.py",
    )

    def test_every_cpu_worker_field_says_how_many_logical_cpus_this_computer_has(self):
        for script, variable in self.WORKER_FIELDS:
            for cores in (self.tools.MAX_CORES, 3, 1):
                with self.subTest(script=script, variable=variable, cores=cores), \
                        mock.patch.object(self.tools, "MAX_CORES", cores):
                    self.enter(self.widget(script, variable))
                    text = re.sub(r"\s+", " ", plain_text(self.shown()))
                    self.assertIn(f"This computer has {cores} logical CPU", text)
                    # The count is worked out when the tip is shown, not when the window opened.
                    self.assertEqual(
                        text.count("This computer has"), 1,
                    )

    def test_the_host_cache_tip_of_both_tools_that_have_one_says_what_auto_picks_now(self):
        gib = self.tools.GIB
        for script in ("Align_Similarity_Matrix.py", "Network_Injection.py"):
            for budget, available in ((8, 24), (2, 5)):
                with self.subTest(script=script, budget=budget), \
                        mock.patch.object(self.tools, "resolve_host_cache_bytes", return_value=budget * gib), \
                        mock.patch.object(self.tools, "system_memory_bytes", return_value=(64 * gib, available * gib)):
                    self.enter(self.widget(script, "HOST_CACHE_GB"))
                    text = re.sub(r"\s+", " ", plain_text(self.shown()))
                    self.assertIn(
                        f"AUTO would use {budget:.1f} GiB now, with {available:.1f} GiB of memory available.", text
                    )
                    self.assertIn(f"{budget:.1f} GiB", chips(self.shown()))
                    self.assertIn("HOST_CACHE_GB", text)

    def test_every_device_field_lists_the_devices_found_by_name_and_says_nothing_without_one(self):
        for script in self.DEVICE_FIELDS:
            with self.subTest(script=script):
                combo = self.widget(script, "DEVICE_SELECTION")
                for index in reversed(range(combo.count())):
                    if combo.itemData(index, NAME_ITEM_ROLE):
                        combo.removeItem(index)
                self.enter(combo)
                self.assertNotIn("Detected devices", plain_text(self.shown()))
                for name, value in (("NVIDIA A100-SXM4-40GB", "cuda:0"), ("Apple M3 Max", "mps")):
                    combo.addItem(name, value)
                    combo.setItemData(combo.count() - 1, True, NAME_ITEM_ROLE)
                self.enter(combo)
                self.assertIn("Detected devices: NVIDIA A100-SXM4-40GB, Apple M3 Max", plain_text(self.shown()))
                self.assertEqual(
                    [name for name in chips(self.shown()) if name in ("NVIDIA A100-SXM4-40GB", "Apple M3 Max")],
                    ["NVIDIA A100-SXM4-40GB", "Apple M3 Max"],
                )

    def test_a_directory_tip_says_where_the_folder_is_and_two_of_them_how_much_room_is_free(self):
        gib = self.tools.GIB
        project = pathlib.Path(self.tools._PROJECT_ROOT)
        absolute = os.path.join(tempfile.gettempdir(), "tip_panel_probe")
        for key, typed, folder, with_room in (
            ("FASTA_DIR", "Input_Files/fasta", str(project / "Input_Files" / "fasta"), False),
            ("MSA_DIR", absolute, absolute, False),
            ("EMBED_DIR", absolute, absolute, False),
            ("NETWORK_DIR", absolute, absolute, False),
            ("REPORT_DIR", absolute, absolute, True),
            ("SETTING_EXPORT_DIR", "Exports", str(project / "Exports"), True),
        ):
            with self.subTest(key=key), mock.patch.object(self.tools, "free_bytes", return_value=7.5 * gib) as free:
                field = self.window.dir_inputs[key]
                field.setText(typed)
                self.enter(field)
                text = re.sub(r"\s+", " ", plain_text(self.shown()))
                self.assertIn("Resolves to", text)
                self.assertIn(
                    os.path.normcase(os.path.normpath(folder)),
                    [os.path.normcase(os.path.normpath(shown)) for shown in chips(self.shown())],
                )
                self.assertEqual("7.5 GiB free on its drive." in text, with_room)
                self.assertEqual(free.called, with_room)
                self.assertIn(key, text)
        # An empty field resolves to nothing: no note, and no disk is asked.
        with mock.patch.object(self.tools, "free_bytes") as free:
            self.window.dir_inputs["REPORT_DIR"].setText("   ")
            self.enter(self.window.dir_inputs["REPORT_DIR"])
            text = plain_text(self.shown())
            self.assertNotIn("Resolves to", text)
            self.assertNotIn("free on its drive", text)
            free.assert_not_called()
        # A drive that can't be read says nothing about room, but still names the folder.
        with mock.patch.object(self.tools, "free_bytes", return_value=None):
            self.window.dir_inputs["REPORT_DIR"].setText(absolute)
            self.enter(self.window.dir_inputs["REPORT_DIR"])
            text = plain_text(self.shown())
            self.assertIn("Resolves to", text)
            self.assertNotIn("free on its drive", text)

    def test_the_directory_tips_show_the_labels_title_and_the_key_and_the_default(self):
        for key, field in self.window.dir_inputs.items():
            with self.subTest(key=key):
                entry = self.window.tip_db[field]
                self.assertIsInstance(entry, FieldTip)
                self.assertEqual(entry.field.key, key)
                self.assertEqual(self.window.tip_db[field.parentWidget()], entry)
                self.assertEqual(field.parentWidget().toolTip(), tip_plain(entry))
                # The folders that get large files say how much room is free; the others say nothing of it.
                self.assertEqual(
                    "12.0 GiB free on its drive." in field.parentWidget().toolTip(),
                    key in ("REPORT_DIR", "STRUCTURE_DIR", "SETTING_EXPORT_DIR"),
                )
                self.enter(field)
                text = re.sub(r"\s+", " ", plain_text(self.shown()))
                self.assertTrue(text.startswith(entry.field.title()), text)
                self.assertIn("Default", text)
                self.assertIn(key, chips(self.shown()))

    def test_a_tool_setting_without_a_tip_of_its_own_still_shows_its_title_default_and_key(self):
        entry = self.tools.tool_tip_entry(None, "SOME_SETTING", QLabel("Some Setting:"), QSpinBox(), 4)
        self.assertIsInstance(entry, FieldTip)
        self.assertEqual(entry.tip.summary, "")
        self.assertEqual(
            re.sub(r"\s+", " ", plain_text(tip_html(entry))), "Some Setting Default 4 Range 0 – 99 Setting SOME_SETTING"
        )
        # A plain-text tip stays as it was.
        self.assertEqual(self.tools.tool_tip_entry("Plain.", "K", QLabel("L:"), QSpinBox()), "Plain.")

    def test_a_message_about_a_fields_state_is_a_note_on_its_tip_and_goes_with_the_state(self):
        from types import SimpleNamespace

        from Cache_Manifest import NetworkCompletenessInfo

        incomplete = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
        )
        complete = NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
        )
        network = self.widget("Embedding_MSA.py", "INPUT_NETWORK").combo
        switch = self.widget("Embedding_MSA.py", "SHOW_REGRESSION_PLOT")
        network_type = SimpleNamespace(network_type="embedding")
        with mock.patch.object(self.tools, "validate_network_schema", return_value=network_type), \
                mock.patch.object(self.tools, "inspect_network_completeness", return_value=incomplete) as inspect:
            network.addItem("sparse.h5")
            network.setCurrentText("sparse.h5")
            self.assertTrue(switch.isEnabled())
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Incomplete network: 10 sequences and 20/45 observed pairs", text)
            # The state's note joins the tip's own text; the tooltip is that tip as plain text.
            self.assertIn("Displays a diagnostic chart of the imputation fit for sparse networks.", text)
            self.assertEqual(switch.toolTip(), tip_plain(self.window.tip_db[switch]))
            self.assertEqual(switch.property("registeredTooltip"), switch.toolTip())
            self.assertEqual(
                [mark_colour(name) for name in images(self.shown())][-1], status_color("subtle"),
            )
            inspect.return_value = complete
            network.addItem("complete.h5")
            network.setCurrentText("complete.h5")
            self.assertFalse(switch.isEnabled())
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Complete network: 10 sequences", text)
            self.assertNotIn("Incomplete network", text)  # one state at a time, not a pile of notes
            self.assertEqual(text.count("Complete network"), 1)
            self.assertEqual(
                [mark_colour(name) for name in images(self.shown())][-1], status_color("warning"),
            )
        # A network that can't be read: a warning, and the base tip is what it was.
        with mock.patch.object(self.tools, "validate_network_schema", side_effect=ValueError("not a network")):
            network.addItem("broken.h5")
            network.setCurrentText("broken.h5")
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Unable to determine network type: not a network", text)
            self.assertEqual(text.count("Unable to determine network type"), 1)
        for combo in (self.widget("Embedding_MSA.py", "ALIGNMENT_SCORE"), self.widget("Embedding_MSA.py", "NORMALIZATION_MODE")):
            self.enter(combo)
            self.assertIn("Unable to determine network type", plain_text(self.shown()))
        # Back to a network that can be read, the earlier states' notes are gone, not kept under it.
        with mock.patch.object(self.tools, "validate_network_schema", return_value=network_type), \
                mock.patch.object(self.tools, "inspect_network_completeness", return_value=incomplete):
            network.setCurrentText("sparse.h5")
            self.enter(switch)
            text = re.sub(r"\s+", " ", plain_text(self.shown()))
            self.assertIn("Incomplete network", text)
            for message in ("Unable to determine", "Complete network:"):
                self.assertNotIn(message, text)

    def test_a_tooltip_changed_after_the_tip_was_registered_shows_instead_of_the_tip(self):
        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        self.assertEqual(combo.property("registeredTooltip"), tip_plain(self.window.tip_db[combo]))
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))
        combo.setToolTip("Something is wrong with this device.")
        self.enter(combo)
        self.assertEqual(self.shown(), SpacedTipLabel("Something is wrong with this device.").text())
        # Cleared (as a widget with no message has it), the field's own tip shows again.
        combo.setToolTip("")
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))
        # Back to what was registered: the same.
        combo.setToolTip(combo.property("registeredTooltip"))
        self.enter(combo)
        self.assertIn("Execution Mode", plain_text(self.shown()))

    def test_hovering_a_child_widget_of_a_field_shows_the_same_tip(self):
        control = self.widget("Align_Similarity_Matrix.py", "HOST_CACHE_GB")
        self.enter(control.spinbox)
        self.assertIn("Host Cache", plain_text(self.shown()))
        self.enter(self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE"))
        self.assertNotIn("Host Cache", plain_text(self.shown()))

    def test_native_tooltips_still_route_to_the_panel_and_the_popup_is_suppressed(self):
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QHelpEvent

        combo = self.widget("Align_Similarity_Matrix.py", "EXECUTION_MODE")
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))
        self.assertTrue(self.window.eventFilter(combo, event))
        self.assertIn("Execution Mode", plain_text(self.shown()))

    def test_the_tip_area_keeps_its_height_whatever_a_tip_holds_and_the_window_around_it_stays_put(self):
        window, panel = self.window, self.panel
        window.resize(1000, 700)
        window.show()
        self.app.processEvents()
        self.assertEqual(panel.height(), panel.preferred_height())
        held = (panel.height(), window.left_split.sizes(), window.tabs.height(), window.minimumSizeHint())
        seen = {}
        for widget, entry in window.tip_db.items():
            if isinstance(entry, FieldTip):
                seen.setdefault(id(entry), entry)
        self.assertGreater(len(seen), 50)
        shows = [lambda entry=entry: panel.setTip(entry) for entry in seen.values()]
        shows += [lambda: panel.setStatus("danger", "Failed.\n" * 30), lambda: panel.setText("Plain\n" * 30)]
        scrolled = 0
        for show in shows:
            show()
            self.app.processEvents()
            needed, available = panel.document().size().height(), panel.viewport().height()
            self.assertEqual(panel.verticalScrollBar().isVisible(), needed > available)
            scrolled += panel.verticalScrollBar().isVisible()
            self.assertFalse(panel.horizontalScrollBar().isVisible())
            self.assertEqual(
                (panel.height(), window.left_split.sizes(), window.tabs.height(), window.minimumSizeHint()), held,
            )
        # What does not fit scrolls (the longest are an options table with notes), and not everything does.
        self.assertGreater(scrolled, 0)
        self.assertLess(scrolled, len(shows))
        # Tools hovers by Enter and has no lock: a message gives way to the next field at once.
        panel.setStatus("info", "message")
        self.enter(self.widget("Align_Similarity_Matrix.py", "BATCH_SIZE"))
        self.assertIn("Batch Size", plain_text(self.shown()))


if __name__ == "__main__":
    unittest.main()
