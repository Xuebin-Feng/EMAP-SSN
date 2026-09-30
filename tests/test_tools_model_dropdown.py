"""Tools GUI: a saved model no plugin provides stays visible and blocks running."""
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class ToolsModelDropdownTests(unittest.TestCase):
    def open_tools(self, saved_model):
        """Open the Tools window with MODEL_NAME saved in a scratch project root."""
        from PySide6.QtWidgets import QApplication, QTextBrowser
        import EMAPSSN_Tools

        app = QApplication.instance() or QApplication([])
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        settings = pathlib.Path(root.name, "tools_settings.json")
        settings.write_text(
            json.dumps({"Generate_Embeddings.py": {"MODEL_NAME": saved_model}}),
            encoding="utf-8",
        )
        for patcher in (
            mock.patch.object(EMAPSSN_Tools, "_PROJECT_ROOT", root.name),
            mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        window = EMAPSSN_Tools.ToolsGUI()

        def close():
            window.close()
            window.deleteLater()
            app.processEvents()

        self.addCleanup(close)
        path, data = next(
            (key, value) for key, value in window.script_data.items()
            if pathlib.Path(key).name == "Generate_Embeddings.py"
        )
        return EMAPSSN_Tools, window, path, data["inputs"]["MODEL_NAME"]["widget"], settings

    def test_removed_saved_model_stays_visible_and_blocks_running(self):
        tools, window, path, combo, settings = self.open_tools("esm2_t36_3b")
        self.assertEqual(combo.currentData(), "esm2_t36_3b")
        self.assertEqual(combo.currentText(), "Unavailable saved model [esm2_t36_3b]")
        before = settings.read_text(encoding="utf-8")

        with mock.patch.object(tools.QMessageBox, "critical") as critical:
            window.save_and_run(path)

        critical.assert_called_once()
        self.assertIn("'esm2_t36_3b' is no longer supported", critical.call_args.args[2])
        self.assertEqual(settings.read_text(encoding="utf-8"), before)

    def test_supported_saved_model_is_selected_normally(self):
        _, _, _, combo, _ = self.open_tools("esm2_t6_8m")
        self.assertEqual(combo.currentData(), "esm2_t6_8m")
        self.assertEqual(combo.findText("Unavailable saved model [esm2_t6_8m]"), -1)
        self.assertFalse(combo.currentText().startswith("Unavailable"))


if __name__ == "__main__":
    unittest.main()
