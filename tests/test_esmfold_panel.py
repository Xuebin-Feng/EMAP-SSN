# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Fold View page's resizable side panel (esmfold_shell.js, esmfold.html).

Beside the canvas (a window wider than 1040 px) the panel is 340 px wide, and a handle on its left edge makes it wider,
up to half the window and 640 px: by dragging it, with the arrow keys on it, or reset with a double click or Home. The
canvas takes the room the panel leaves, and the width is kept in the browser, but not in the saved session. The drawer of a
narrower window and the strip of a phone are as they were. Like the other page tests these open esmfold.html in Qt WebEngine
over the real server and Mol*; tests/test_esmfold_shell_webgl.py checks that a real canvas follows the panel.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.esmfold_long_names import USER_NAME
from tests.test_esmfold_colors import A, B
from tests.test_esmfold_layout import OVERFLOW, LongNameTestCase

from PySide6.QtCore import QPoint, Qt  # noqa: E402  (after the harness has set the engine's environment)
from PySide6.QtTest import QTest  # noqa: E402

KEY = "ssn_fold_panel_width"
# What the panel, its handle and the canvas are doing, and the keys of the handle, as a person would press them.
PANEL = """
window.panelState = function () {
    const side = document.getElementById('side'), handle = document.getElementById('side-resizer');
    const card = document.querySelector('.stage-card').getBoundingClientRect(), sideBox = side.getBoundingClientRect();
    const grip = handle.getBoundingClientRect(), viewer = document.getElementById('viewer').getBoundingClientRect();
    const stored = (() => { try { return localStorage.getItem('ssn_fold_panel_width'); } catch (error) { return 'unreadable'; } })();
    return {width: sideBox.width, left: sideBox.left, right: sideBox.right, shown: getComputedStyle(handle).display !== 'none',
            handleLeft: grip.left, handleTop: grip.top, handleHeight: grip.height, cardWidth: card.width, cardRight: card.right,
            viewerWidth: viewer.width, viewerHeight: viewer.height, app: document.getElementById('app').clientWidth,
            now: Number(handle.getAttribute('aria-valuenow')), min: Number(handle.getAttribute('aria-valuemin')),
            max: Number(handle.getAttribute('aria-valuemax')), focused: document.activeElement === handle,
            resizing: document.body.dataset.resizing || null, stored, variable: document.getElementById('app').style.getPropertyValue('--side-width'),
            canvasCalls: window.__resizeCalls};
};
window.press = function (key, options = {}) {
    document.getElementById('side-resizer').dispatchEvent(new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true, ...options}));
};
window.__resizeCalls = 0;
(() => {
    const handleResize = viewerInstance.handleResize.bind(viewerInstance);
    viewerInstance.handleResize = () => { window.__resizeCalls++; return handleResize(); };
})();
"""


class PanelTestCase(LongNameTestCase):
    def open_two(self):
        self.write_with(A, "information_content")
        self.write_with(B, "information_content", shift=(3.0, 1.0, 0.0), wobble=0.4)
        self.open_scene(A, B)
        self.js(PANEL)
        self.js(OVERFLOW)

    def at_width(self, window):
        self.view.resize(window, 800)
        self.addCleanup(self.view.resize, 1300, 800)
        self.settle(300)

    def state(self):
        return self.script("return panelState()")

    def press(self, key, **options):
        self.script("press(%s, %s); return 1" % (json.dumps(key), json.dumps(options)))
        self.settle(80)
        return self.state()

    def handle_point(self, offset=0):
        """The middle of the handle (the panel's left edge) in the view's own pixels, `offset` further right."""
        state = self.state()
        return QPoint(round(state["left"]) + offset, round(state["handleTop"] + state["handleHeight"] / 2))

    def drag(self, to, grab=0):
        """Take hold of the edge `grab` pixels inside the panel, and let go with the pointer at x = `to`, with a real mouse."""
        target = self.view.focusProxy() or self.view
        QTest.mouseMove(target, self.handle_point(grab))
        QTest.mousePress(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, self.handle_point(grab))
        self.pump(0.1)
        midway = self.state()
        for x in range(self.handle_point(grab).x(), to, -40 if to < self.handle_point(grab).x() else 40):
            QTest.mouseMove(target, QPoint(x, self.handle_point(grab).y()))
            self.pump(0.02)
        QTest.mouseMove(target, QPoint(to, self.handle_point(grab).y()))
        self.pump(0.2)
        during = self.state()
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(to, self.handle_point(grab).y()))
        self.pump(0.2)
        return midway, during, self.state()


class HandleTests(PanelTestCase):
    def test_the_handle_is_a_separator_on_the_panels_edge_that_a_keyboard_can_reach(self):
        self.open_two()
        state = self.state()
        handle = self.script("""
            const h = document.getElementById('side-resizer');
            return {role: h.getAttribute('role'), orientation: h.getAttribute('aria-orientation'), controls: h.getAttribute('aria-controls'),
                    label: h.getAttribute('aria-label'), title: h.title, tabindex: h.tabIndex,
                    owner: document.getElementById(h.getAttribute('aria-controls')).tagName};""")
        self.assertEqual((handle["role"], handle["orientation"], handle["controls"], handle["owner"]), ("separator", "vertical", "side", "ASIDE"))
        self.assertEqual((handle["label"], handle["tabindex"]), ("Resize the panel", 0))
        self.assertTrue(handle["title"])
        # At 1300 px: the panel is at its least and can go to the cap, which half of the window is more than.
        self.assertEqual((state["width"], state["now"], state["min"], state["max"]), (340, 340, 340, 640))
        # On the panel's edge, as tall as the workspace, between the canvas and the panel.
        self.assertTrue(state["shown"])
        self.assertAlmostEqual(state["handleLeft"], state["left"], delta=1)
        self.assertGreater(state["handleHeight"], 300)
        self.assertLessEqual(state["cardRight"], state["left"])
        # Tab reaches it, between the canvas and the panel's cards.
        self.script("document.getElementById('seq-toggle').focus(); return 1")
        self.assertEqual(self.script("return [...document.querySelectorAll('[tabindex=\"0\"]')].includes(document.getElementById('side-resizer'))"), True)

    def test_the_limits_follow_the_window_half_of_it_and_never_more_than_640(self):
        self.open_two()
        for window, maximum in ((1300, 640), (1280, 640), (1200, 600), (1100, 550), (1041, 520)):
            with self.subTest(window=window):
                self.at_width(window)
                state = self.state()
                self.assertEqual((state["app"], state["min"], state["max"]), (window, 340, maximum))
        # The least is the width the panel always had.
        self.assertEqual(self.state()["now"], 340)

    def test_the_panel_never_passes_its_limits_whatever_asks_for_a_width(self):
        # esmfold.html clamps the width the page asks for to the window's limits, so a width set from anywhere (a script, a
        # saved value, a window that has since got smaller) cannot make the panel too narrow or take the window.
        self.open_two()
        for window, asked, shown in ((1300, 100, 340), (1300, 900, 640), (1300, 500, 500), (1100, 900, 550), (1100, 450, 450), (1041, 640, 520)):
            with self.subTest(window=window, asked=asked):
                self.at_width(window)
                self.script("document.getElementById('app').style.setProperty('--side-width', '%dpx'); return 1" % asked)
                self.settle(100)
                # (Half of 1041 px is 520.5 px; the handle's own limits are whole pixels.)
                self.assertAlmostEqual(self.state()["width"], shown, delta=0.6)

    def test_the_arrow_keys_home_and_end_move_the_edge_and_stop_at_the_limits(self):
        self.open_two()
        self.script("document.getElementById('side-resizer').focus(); return 1")
        self.assertTrue(self.state()["focused"])
        # The panel is on the right, so the edge moving left (Left) makes it wider.
        self.assertEqual(self.press("ArrowLeft")["width"], 360)
        self.assertEqual(self.press("ArrowLeft")["width"], 380)
        self.assertEqual(self.press("ArrowRight")["width"], 360)
        self.assertEqual(self.press("ArrowLeft", shiftKey=True)["width"], 460)
        after = self.press("ArrowLeft", shiftKey=True)
        self.assertEqual((after["width"], after["now"]), (560, 560))
        self.assertEqual(self.press("ArrowLeft", shiftKey=True)["width"], 640)
        # Not past the widest, nor under the least.
        self.assertEqual(self.press("ArrowLeft", shiftKey=True)["width"], 640)
        self.assertEqual(self.press("Home")["width"], 340)
        self.assertEqual(self.press("ArrowRight", shiftKey=True)["width"], 340)
        self.assertEqual(self.press("ArrowRight")["width"], 340)
        self.assertEqual(self.press("End")["width"], 640)
        self.assertEqual(self.press("Home")["width"], 340)
        # The keys it uses are taken (the page does not scroll with them), and a key it does not use is left alone (Tab must go on).
        taken = self.script("""
            const result = {};
            for (const key of ['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Tab', 'ArrowUp', 'a']) {
                const event = new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true});
                document.getElementById('side-resizer').dispatchEvent(event);
                result[key] = event.defaultPrevented;
            }
            return result;""")
        self.assertEqual(taken, {"ArrowLeft": True, "ArrowRight": True, "Home": True, "End": True, "Tab": False, "ArrowUp": False, "a": False})
        self.assertEqual(self.console_errors(), [])

    def test_a_double_click_gives_the_panel_its_width_back(self):
        self.open_two()
        self.press("ArrowLeft", shiftKey=True)
        self.assertEqual(self.state()["width"], 440)
        self.script("document.getElementById('side-resizer').dispatchEvent(new MouseEvent('dblclick', {bubbles: true})); return 1")
        self.settle(100)
        state = self.state()
        self.assertEqual((state["width"], state["now"], state["stored"], state["variable"]), (340, 340, None, ""))

    def test_a_drag_with_a_real_mouse_moves_the_edge_with_the_pointer_and_the_canvas_gives_way(self):
        self.open_two()
        before = self.state()
        # Take hold of the edge and pull it 150 px to the left, 3 px inside the panel.
        midway, during, after = self.drag(to=round(before["left"]) + 3 - 150, grab=3)
        self.assertEqual(midway["resizing"], "1")
        self.assertTrue(midway["focused"])
        self.assertEqual(during["width"], 490)
        # The edge is under the pointer all along; the canvas has what the panel left, and nothing else moved.
        self.assertAlmostEqual(during["left"], before["left"] - 150, delta=1)
        self.assertAlmostEqual(during["cardWidth"], before["cardWidth"] - 150, delta=1)
        self.assertAlmostEqual(during["viewerWidth"], before["viewerWidth"] - 150, delta=1)
        self.assertEqual(during["viewerHeight"], before["viewerHeight"])
        self.assertEqual(after["resizing"], None)
        self.assertEqual((after["width"], after["now"], after["stored"]), (490, 490, "490"))
        # Mol* is told, in step with the drag and at its end.
        self.assertGreater(after["canvasCalls"], 0)
        # Further, to the window's own limit (the cap at this width), and back past the least.
        _, _, wide = self.drag(to=10, grab=3)
        self.assertEqual((wide["width"], wide["stored"]), (640, "640"))
        _, _, narrow = self.drag(to=1290, grab=3)
        self.assertEqual((narrow["width"], narrow["stored"]), (340, None))
        self.assertEqual(self.console_errors(), [])

    def test_a_drag_that_loses_the_pointer_ends_there(self):
        self.open_two()
        started = self.script("""
            const handle = document.getElementById('side-resizer');
            const down = new PointerEvent('pointerdown', {pointerId: 41, button: 0, clientX: handle.getBoundingClientRect().left + 2, clientY: 300, bubbles: true, cancelable: true});
            handle.dispatchEvent(down);
            return document.body.dataset.resizing || null;""")
        self.assertEqual(started, "1")
        self.script("""
            const handle = document.getElementById('side-resizer');
            handle.dispatchEvent(new PointerEvent('pointermove', {pointerId: 41, clientX: handle.getBoundingClientRect().left - 98, clientY: 300, bubbles: true}));
            return 1;""")
        self.assertEqual(self.state()["width"], 440)
        # Another pointer's moves are not the drag's; and when the pointer is lost the drag is over, with the width as it got to.
        self.script("""
            const handle = document.getElementById('side-resizer');
            handle.dispatchEvent(new PointerEvent('pointermove', {pointerId: 42, clientX: 0, clientY: 300, bubbles: true}));
            return 1;""")
        self.assertEqual(self.state()["width"], 440)
        self.script("document.getElementById('side-resizer').dispatchEvent(new PointerEvent('lostpointercapture', {pointerId: 41, bubbles: true})); return 1")
        state = self.state()
        self.assertEqual((state["resizing"], state["stored"], state["width"]), (None, "440", 440))
        # A pointer the system takes away (a touch that becomes a scroll) ends it too.
        for ending in ("pointercancel", "pointerup"):
            self.script("""
                const handle = document.getElementById('side-resizer');
                handle.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 44, button: 0, clientX: handle.getBoundingClientRect().left, clientY: 300, bubbles: true, cancelable: true}));
                return 1;""")
            self.assertEqual(self.state()["resizing"], "1")
            self.script("document.getElementById('side-resizer').dispatchEvent(new PointerEvent(%s, {pointerId: 44, bubbles: true})); return 1" % json.dumps(ending))
            self.assertIsNone(self.state()["resizing"], ending)
        # A right button does not take hold.
        self.script("""
            const handle = document.getElementById('side-resizer');
            handle.dispatchEvent(new PointerEvent('pointerdown', {pointerId: 43, button: 2, clientX: 5, clientY: 300, bubbles: true, cancelable: true}));
            return 1;""")
        self.assertIsNone(self.state()["resizing"])

    def test_a_single_real_click_on_the_handle_takes_the_focus_and_changes_nothing(self):
        # (A double click is tested with the event the browser sends for it: the offscreen engine does not deliver QTest's whole.)
        self.open_two()
        self.press("End")
        target = self.view.focusProxy() or self.view
        QTest.mouseClick(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, self.handle_point(2))
        self.pump(0.3)
        state = self.state()
        self.assertEqual((state["width"], state["stored"], state["resizing"], state["focused"]), (640, "640", None, True))

    def test_a_smaller_window_takes_the_panel_in_with_it_and_the_width_comes_back_with_the_window(self):
        self.open_two()
        self.press("End")
        self.assertEqual(self.state()["width"], 640)
        self.at_width(1100)
        small = self.state()
        self.assertEqual((small["width"], small["now"], small["max"]), (550, 550, 550))
        # The canvas is still there with room to draw in.
        self.assertGreater(small["viewerWidth"], 300)
        self.at_width(1300)
        self.assertEqual(self.state()["width"], 640)


class RememberTests(PanelTestCase):
    def test_the_width_is_kept_in_the_browser_and_comes_back_on_the_next_visit(self):
        self.open_two()
        self.press("ArrowLeft", shiftKey=True)
        self.press("ArrowLeft")
        self.assertEqual(self.state()["stored"], "460")
        self.open_page()
        self.js(PANEL)
        after = self.state()
        self.assertEqual((after["width"], after["now"], after["stored"]), (460, 460, "460"))
        # Home gives it back and forgets it.
        self.press("Home")
        self.assertIsNone(self.state()["stored"])
        self.open_page()
        self.js(PANEL)
        self.assertEqual(self.state()["width"], 340)

    def test_a_width_nobody_could_have_made_is_ignored(self):
        self.open_two()
        for stored in ("garbage", "100", "340", "-50", "99999", "400.5", ""):
            with self.subTest(stored=stored):
                self.script("localStorage.setItem(%s, %s); return 1" % (json.dumps(KEY), json.dumps(stored)))
                self.open_page()
                self.js(PANEL)
                self.assertEqual(self.state()["width"], 340)
        self.script("localStorage.setItem(%s, '641'); return 1" % json.dumps(KEY))
        self.open_page()
        self.js(PANEL)
        self.assertEqual(self.state()["width"], 340)
        self.script("localStorage.setItem(%s, '640'); return 1" % json.dumps(KEY))
        self.open_page()
        self.js(PANEL)
        self.assertEqual(self.state()["width"], 640)

    def test_a_width_the_window_has_no_room_for_is_shown_as_far_as_it_goes(self):
        self.open_two()
        self.script("localStorage.setItem(%s, '600'); return 1" % json.dumps(KEY))
        self.at_width(1100)
        self.open_page()
        self.js(PANEL)
        self.assertEqual(self.state()["width"], 550)
        self.at_width(1300)
        self.assertEqual(self.state()["width"], 600)

    def test_a_browser_that_will_not_keep_it_still_gets_a_working_handle(self):
        self.open_two()
        self.script("""
            Storage.prototype.setItem = () => { throw new Error('storage is full'); };
            Storage.prototype.removeItem = () => { throw new Error('storage is full'); };
            Storage.prototype.getItem = () => { throw new Error('storage is blocked'); };
            return 1;""")
        self.assertEqual(self.press("ArrowLeft", shiftKey=True)["width"], 440)
        self.assertEqual(self.press("Home")["width"], 340)
        self.script("document.getElementById('side-resizer').dispatchEvent(new MouseEvent('dblclick', {bubbles: true})); return 1")
        self.assertEqual(self.console_errors(), [])

    def test_a_browser_that_will_not_say_what_it_kept_still_opens_a_working_panel(self):
        # Storage that throws when the page reads it, from the moment the page is made (before its own scripts run).
        from PySide6.QtWebEngineCore import QWebEngineScript

        script = QWebEngineScript()
        script.setName("blocked-storage")
        script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setSourceCode("Storage.prototype.getItem = () => { throw new Error('storage is blocked'); };")
        scripts = self.view.page().scripts()
        scripts.insert(script)
        self.addCleanup(scripts.remove, script)
        self.open_two()
        self.assertEqual(self.script("try { localStorage.getItem('x'); return 'readable'; } catch (error) { return 'blocked'; }"), "blocked")
        self.assertEqual(self.state()["width"], 340)
        self.assertEqual(self.press("ArrowLeft", shiftKey=True)["width"], 440)
        self.assertEqual(self.console_errors(), [])

    def test_the_width_is_no_part_of_the_session_the_page_saves(self):
        self.open_two()
        self.settle(1800)
        del self.saved[:]
        self.press("End")
        self.settle(1800)
        # Nothing changed in the scene, so nothing was saved; and the session as it would be saved says nothing of the panel.
        self.assertEqual(self.saved, [])
        text = json.dumps(self.script("return takeSnapshot()"))
        for word in ("side-width", "fold_panel", "panelWidth", "640px"):
            self.assertNotIn(word, text)


class LayoutTests(PanelTestCase):
    def test_the_drawer_and_the_strip_are_as_they_were_whatever_width_the_panel_was_given(self):
        self.open_two()
        self.press("End")
        for window, width, shown in ((1040, 340, False), (900, 340, False), (701, 340, False), (600, None, False)):
            with self.subTest(window=window):
                self.at_width(window)
                self.script("setPanel(true); return 1")
                self.settle(300)
                state = self.state()
                self.assertEqual(state["shown"], shown)
                if width is not None:
                    # A drawer: 340 px (or what the window allows), over the canvas, which keeps the whole window.
                    self.assertEqual(state["width"], width)
                    self.assertAlmostEqual(state["cardRight"], state["app"] - 10, delta=1)
                else:
                    # A strip: as wide as the window.
                    self.assertAlmostEqual(state["width"], state["app"], delta=1)
        # Back beside the canvas, the width they had is there again.
        self.at_width(1300)
        self.assertEqual(self.state()["width"], 640)

    def test_the_handle_is_out_of_the_tab_order_where_it_is_not_shown_and_with_the_advanced_view(self):
        self.open_two()
        self.at_width(1040)
        self.assertFalse(self.state()["shown"])
        self.assertFalse(self.script("document.getElementById('side-resizer').focus(); return document.activeElement === document.getElementById('side-resizer')"))
        self.at_width(1300)
        self.script("setAdvanced(true); return 1")
        self.settle(200)
        self.assertFalse(self.state()["shown"])
        self.script("setAdvanced(false); return 1")
        self.settle(200)
        state = self.state()
        self.assertTrue(state["shown"])
        self.assertEqual((state["width"], state["now"]), (340, 340))
        # The window got smaller while the panel was away: the handle says what the panel is when it comes back.
        self.press("End")
        self.script("setAdvanced(true); return 1")
        self.at_width(1100)
        self.script("setAdvanced(false); return 1")
        self.settle(200)
        state = self.state()
        self.assertEqual((state["width"], state["now"], state["max"]), (550, 550, 550))

    def test_without_mol_star_the_panel_and_its_handle_are_gone_together(self):
        self.open_two()
        self.assertTrue(self.state()["shown"])
        self.script("failStart(new Error('no WebGL, no Mol*')); return 1")
        self.settle(200)
        self.assertFalse(self.state()["shown"])
        self.assertEqual(self.script("return getComputedStyle(document.getElementById('side')).display"), "none")

    def test_the_cards_use_the_room_a_wider_panel_gives_them(self):
        self.scene(USER_NAME, B[:-4], layer="information_content")
        self.js(PANEL)
        narrow = self.script("return document.getElementById('color-body').clientWidth")
        self.press("ArrowLeft", shiftKey=True)
        self.press("ArrowLeft", shiftKey=True)
        for tab in ("structures", "color", "residue"):
            with self.subTest(tab=tab):
                self.assertEqual(self.overflowing("#side"), [])
        # The Color by card's body is as much wider as the panel is, and no wider than that once the handle is let go.
        self.assertEqual(self.script("return document.getElementById('color-body').clientWidth"), narrow + 200)
        self.press("Home")
        self.assertEqual(self.script("return document.getElementById('color-body').clientWidth"), narrow)

    def test_the_help_dialog_has_a_line_for_the_handle_in_each_table(self):
        self.open_two()
        rows = self.script("""
            const tables = [...document.querySelectorAll('#help-modal table')];
            return tables.map(table => [...table.rows].map(row => [...row.cells].map(cell => cell.textContent.trim())));""")
        mouse, keyboard = rows[0], rows[1]
        self.assertIn(["Resize the panel", "Drag the handle on its left edge; double-click it to restore the width"], mouse)
        self.assertIn(["Left Right Home", "On the handle of the panel: make it wider or narrower, or restore its width"], keyboard)
        for width in (1300, 600):
            self.at_width(width)
            self.script("document.getElementById('help-modal').classList.add('open'); return 1")
            self.assertEqual(self.overflowing("#help-modal .modal"), [])
            self.script("document.getElementById('help-modal').classList.remove('open'); return 1")

    def test_the_handle_and_its_help_come_in_simplified_chinese(self):
        from desktop.Desktop_App import install_translations

        install_translations(self.app, "zh_CN")
        self.addCleanup(install_translations, self.app, None)
        self.open_two()
        label = self.script("const h = document.getElementById('side-resizer'); return [h.getAttribute('aria-label'), h.title]")
        self.assertNotEqual(label[0], "Resize the panel")
        self.assertNotEqual(label[1], "Drag to resize the panel. Double-click to restore its width.")
        self.assertTrue(all(any("一" <= character <= "鿿" for character in text) for text in label), label)
        rows = self.script("return [...document.querySelectorAll('#help-modal tr')].map(row => row.textContent.trim()).slice(-8)")
        self.assertTrue(any("一" <= character <= "鿿" for character in " ".join(rows)))
        self.assertNotIn("Resize the panel", " ".join(rows))


if __name__ == "__main__":
    unittest.main()
