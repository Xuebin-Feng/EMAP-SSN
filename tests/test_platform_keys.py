# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""utilities/Platform_Keys.py: key names in the form the platform's users know (macOS reads Ctrl as Cmd)."""
import sys
import unittest
from pathlib import Path
from unittest import mock

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities.Platform_Keys import platform_key_text  # noqa: E402


class PlatformKeyTextTests(unittest.TestCase):
    def test_other_platforms_get_the_text_unchanged(self):
        text = "Undo (Ctrl+Z), Redo (Ctrl+Y), [Scroll + Ctrl] Rotate"
        for platform in ("win32", "linux", "freebsd"):
            self.assertEqual(platform_key_text(text, platform), text)

    def test_the_default_is_this_systems_platform_when_called(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(platform_key_text("Ctrl+Z"), "Cmd+Z")
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(platform_key_text("Ctrl+Z"), "Ctrl+Z")

    def test_a_mac_reads_ctrl_as_cmd_and_ctrl_y_as_the_mac_redo_key(self):
        self.assertEqual(platform_key_text("Undo (Ctrl+Z)", "darwin"), "Undo (Cmd+Z)")
        self.assertEqual(platform_key_text("Redo (Ctrl+Y)", "darwin"), "Redo (Cmd+Shift+Z)")
        self.assertEqual(platform_key_text("[RightClick + Drag + Ctrl] Rotate | [Scroll + Ctrl] Turn", "darwin"),
                         "[RightClick + Drag + Cmd] Rotate | [Scroll + Cmd] Turn")
        self.assertEqual(platform_key_text("Ctrl+Z and Ctrl+Y, then Ctrl+C", "darwin"),
                         "Cmd+Z and Cmd+Shift+Z, then Cmd+C")

    def test_ctrl_inside_a_longer_word_stays(self):
        for word in ("Control+Y", "Ctrlx", "xCtrl", "Ctrl2", "Ctrl_Y", "ctrl+y", "CTRL+Y"):
            self.assertEqual(platform_key_text(word, "darwin"), word)
        self.assertEqual(platform_key_text("Ctrl+YY Ctrl+Y2", "darwin"), "Cmd+YY Cmd+Y2")

    def test_the_chinese_texts_keep_ctrl_in_latin_letters_and_read_the_same(self):
        self.assertEqual(platform_key_text("撤销（Ctrl+Z）", "darwin"), "撤销（Cmd+Z）")
        self.assertEqual(platform_key_text("重做（Ctrl+Y）", "darwin"), "重做（Cmd+Shift+Z）")
        self.assertEqual(platform_key_text("按Ctrl键并滚动，或用 Ctrl+V 粘贴", "darwin"), "按Cmd键并滚动，或用 Cmd+V 粘贴")
        self.assertEqual(platform_key_text("重做（Ctrl+Y）", "linux"), "重做（Ctrl+Y）")


if __name__ == "__main__":
    unittest.main()
