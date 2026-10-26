"""The zoom command (commands/zoom.py): the zoom, a percentage of a view
ZOOM_100_PERCENT_WIDTH units wide, it accepts and the camera rectangle it sets,
and what it leaves alone when it refuses a zoom. The Viewer's zoom keeps to
limits, by the command, the Zoom box, the wheel and a pinch alike: at most
200000% (a view 0.5 wide), and at least the zoom that shows every node, hidden
ones too, as the Viewer opens on them, rounded down to two significant figures;
that lowest zoom follows the nodes as they move. The wheel zooms smoothly
(desktop/Viewer_Zoom.py): VisPy's amount a notch, played out over a few frames."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import zoom as zoom_command
from tests.command_fixtures import reported_outcomes


ORIGINAL_RECT = SimpleNamespace(pos=(0.0, 0.0), width=100.0, height=50.0)


class FakeCamera:
    """Holds a rectangle (left, bottom, width, height) as the vispy camera does.

    With fail_next set, the next assignment stores its rectangle and then raises,
    as the real camera does when its transform cannot be built.
    """

    def __init__(self, rect=(0.0, 0.0, 100.0, 50.0), fail_next=False):
        self._rect = rect
        self.fail_next = fail_next

    @property
    def rect(self):
        left, bottom, width, height = self._rect
        return SimpleNamespace(pos=(left, bottom), width=width, height=height)

    @rect.setter
    def rect(self, value):
        self._rect = tuple(value)
        if self.fail_next:
            self.fail_next = False
            raise np.linalg.LinAlgError("Singular matrix")


def zoom_viewer(canvas_size=(200, 100), camera=None):
    """A 100 x 50 camera view centered on (50, 25)."""
    return SimpleNamespace(
        canvas=SimpleNamespace(size=canvas_size),
        view=SimpleNamespace(camera=camera if camera is not None else FakeCamera()),
        _hud_timer=SimpleNamespace(start=mock.Mock()),
        console_text=SimpleNamespace(text=""),
    )


class HundredPercentIs1000:
    """Mixin: the literal numbers of a test class assume 100% is a view 1000 units wide.

    So the highest zoom is 200000%, and a network 1089 units across opens at 92%.
    The tests then hold whatever the shipped ZOOM_100_PERCENT_WIDTH is, which
    ShippedZoomTests checks; changing that one number changes no other test.
    """

    def setUp(self):
        super().setUp()
        patcher = mock.patch.multiple(zoom_command, ZOOM_100_PERCENT_WIDTH=1000.0, MAX_ZOOM_PERCENT=200000.0)
        patcher.start()
        self.addCleanup(patcher.stop)


class ZoomCommandTests(HundredPercentIs1000, unittest.TestCase):
    def run_zoom(self, *words, canvas_size=(200, 100), camera=None):
        viewer = zoom_viewer(canvas_size, camera)
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            zoom_command.run(viewer, list(words))
        return viewer, succeeded, failed

    def assert_refused(self, viewer, succeeded, failed, message):
        self.assertEqual(viewer.view.camera.rect, ORIGINAL_RECT)
        viewer._hud_timer.start.assert_not_called()
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn(message, failed.call_args.args[1])

    def test_a_hundred_percent_is_a_fixed_width_of_layout_units(self):
        # One named number says what 100% is; the width of a zoom and the zoom of a width follow from it.
        self.assertEqual(zoom_command.width_at_zoom(100.0), 1000.0)
        self.assertEqual(zoom_command.width_at_zoom(200.0), 500.0)
        self.assertEqual(zoom_command.zoom_percent(2000.0), 50.0)
        with mock.patch.object(zoom_command, "ZOOM_100_PERCENT_WIDTH", 300.0):
            self.assertEqual(zoom_command.width_at_zoom(100.0), 300.0)
            self.assertEqual(zoom_command.zoom_percent(150.0), 200.0)

    def test_the_zoom_must_be_a_positive_finite_number(self):
        # float() parses all of these; none of them gives a usable view.
        for zoom in ("0", "-5", "nan", "inf", "-inf", "1e400", "0%", "-5%"):
            with self.subTest(zoom=zoom):
                self.assert_refused(
                    *self.run_zoom(zoom), "Zoom must be a positive, finite percentage."
                )

    def test_a_zoom_that_is_not_a_number_is_refused(self):
        # A % sign comes once, after the number; nothing else is a number here.
        message = "Zoom must be a number, with an optional % sign, as in zoom 200 or zoom 200%."
        for words in (("wide",), ("200%%",), ("%",), ("%200",), ("20", "0"), ("200", "%", "%"),
                      ("200", "300"), ("200", "wide"), ("1_0",), ("٢٠٠",), ("--5",)):
            with self.subTest(words=words):
                self.assert_refused(*self.run_zoom(*words), message)

    def test_the_percent_sign_is_optional_and_may_stand_apart(self):
        for words in (("200",), ("200%",), ("200", "%"), (" 200 % ",), ("2e2",), ("200.0%",)):
            with self.subTest(words=words):
                viewer, succeeded, failed = self.run_zoom(*words)
                failed.assert_not_called()
                self.assertEqual(viewer.view.camera.rect.width, 500.0)
                self.assertEqual(succeeded.call_args.args[1], "Zoom set to 200%")

    def test_canvas_without_area_is_refused_without_dividing_by_zero(self):
        for canvas_size in ((200, 0), (0, 100)):
            with self.subTest(canvas_size=canvas_size):
                self.assert_refused(
                    *self.run_zoom("200", canvas_size=canvas_size), "The canvas has no visible area"
                )

    def test_a_zoom_whose_width_overflows_the_view_range_is_refused(self):
        # 1e-303% is 1e308 units wide: finite, but a 1:3 canvas makes the view 3e308 high, which overflows.
        self.assert_refused(
            *self.run_zoom("1e-303", canvas_size=(100, 300)), "Zoom is too low for the current view."
        )
        # A zoom so low that its width is not even a float.
        self.assert_refused(*self.run_zoom("1e-320"), "Zoom is too low for the current view.")

    def test_zooms_beyond_the_float32_scale_range_are_refused(self):
        # A view 1e50 wide once made the camera's matrix singular after its rectangle had changed.
        for zoom in ("1e-47", "1e-305"):
            with self.subTest(zoom=zoom):
                self.assert_refused(*self.run_zoom(zoom), "Zoom is too low for the current view.")

    def test_a_zoom_too_high_for_the_view_centre_is_refused(self):
        # Centred at (50, 25), a view under about 1e-3 units wide is not drawn to within a pixel.
        self.assert_refused(*self.run_zoom("1e19"), "Zoom is too high to draw accurately")

    def test_a_high_zoom_the_centre_can_hold_is_set_exactly(self):
        # This viewer has no nodes, so no zoom limits (ZoomLimitTests) apply: 0.01 units wide.
        viewer, succeeded, failed = self.run_zoom("10000000")

        self.assertEqual(viewer.view.camera.rect.width, 0.01)
        self.assertEqual(viewer.view.camera.rect.height, 0.01 / 2.0)
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Zoom set to 10000000%")

    def test_zoom_sets_the_range_around_the_current_center(self):
        viewer, succeeded, failed = self.run_zoom("200")

        # 200% is 500 wide, on a 2:1 canvas 250 high, centered on (50, 25), with no margin added.
        self.assertEqual(
            viewer.view.camera.rect,
            SimpleNamespace(pos=(-200.0, -100.0), width=500.0, height=250.0),
        )
        viewer._hud_timer.start.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Zoom set to 200%")

    def test_the_success_report_gives_the_zoom_as_typed(self):
        for text, shown in (("50", "50"), ("33.5%", "33.5"), ("0.25", "0.25"), ("1234567", "1234567"),
                            ("0.00001", "0.00001")):
            with self.subTest(text=text):
                viewer = zoom_viewer()
                with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
                    zoom_command.run(viewer, [text])
                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], f"Zoom set to {shown}%")

    def test_help_explains_the_percentage_and_its_limits(self):
        viewer = zoom_viewer()
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()) as printed:
            zoom_command.run(viewer, [])
        failed.assert_not_called()
        text = printed.getvalue()
        self.assertIn("Usage: zoom <percent>", text)
        self.assertIn("100% is a view 1000 layout units wide", text)
        self.assertIn("200000%", text)
        self.assertIn("zoom 200 ", text)
        self.assertIn("zoom 50% ", text)
        self.assertNotIn("width", text.split("Examples:")[1])

    def test_a_camera_that_fails_to_apply_the_view_gets_its_previous_view_back(self):
        viewer, succeeded, failed = self.run_zoom("200", camera=FakeCamera(fail_next=True))

        self.assertEqual(viewer.view.camera.rect, ORIGINAL_RECT)
        viewer._hud_timer.start.assert_not_called()
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("Zoom could not be applied: Singular matrix", failed.call_args.args[1])


class ShippedZoomTests(unittest.TestCase):
    """What the shipped ZOOM_100_PERCENT_WIDTH makes of the Viewer, and the texts that state it.

    Nothing here names the number: it is read from commands.zoom, so changing it
    changes only what the texts and the constant itself say (the line in
    commands/zoom.py, the metadata, the agent prompt and the two docs pages).
    """

    def test_the_highest_zoom_is_the_narrowest_view_and_follows_the_constant(self):
        width = zoom_command.ZOOM_100_PERCENT_WIDTH
        self.assertEqual(zoom_command.MAX_ZOOM_PERCENT, 100.0 * width / zoom_command.MIN_VIEW_WIDTH)
        self.assertEqual(zoom_command.width_at_zoom(100.0), width)
        self.assertEqual(zoom_command.width_at_zoom(zoom_command.MAX_ZOOM_PERCENT), zoom_command.MIN_VIEW_WIDTH)
        viewer = network_viewer()
        lowest, highest = zoom_command.zoom_limits(viewer)
        self.assertEqual(highest, zoom_command.MAX_ZOOM_PERCENT)
        # The whole 1089-wide network still fits at the lowest zoom, which is its zoom to two figures.
        self.assertGreaterEqual(zoom_command.view_width_limits(viewer)[1], 1089.0)
        self.assertEqual(lowest, zoom_command.round_down(100.0 * width / 1089.0))

    def test_a_layout_a_few_hundred_units_across_opens_near_a_hundred_percent(self):
        # What the constant is for: the layouts the Viewer opens on (the 4K-13K node caches are
        # 80-240 units across, 110-290 with the margins) open at about 70% to 170% on a 3:2 canvas.
        for extent in (110.0, 200.0, 290.0):
            positions = [[0.0, 0.0], [extent, extent]]
            viewer = network_viewer(positions)
            left, bottom, opening_width, height = zoom_command.opening_view(positions)
            visible = max(opening_width, height * 1200 / 800)
            with self.subTest(extent=extent):
                opens_at = zoom_command.zoom_percent(visible)
                self.assertTrue(40 <= opens_at <= 250, opens_at)
                self.assertAlmostEqual(opens_at, zoom_command.zoom_limits(viewer)[0], delta=opens_at * 0.1)

    def test_agent_texts_state_the_same_hundred_percent(self):
        # The command's metadata and the agent's prompt say what 100% is as the constant does.
        from desktop.Command_Metadata import COMMAND_METADATA

        width = f"{zoom_command.ZOOM_100_PERCENT_WIDTH:g}"
        highest = f"{zoom_command.MAX_ZOOM_PERCENT:g}%"
        description = next(
            entry["description"] for entry in COMMAND_METADATA["zoom"]["arguments"] if entry["name"] == "percent"
        )
        self.assertIn(f"100% is a view {width} layout units wide", description)
        self.assertIn(highest, description)
        prompt = os.path.join(SRC_DIR, "resources", "agent", "system_prompt.md")
        with open(prompt, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIn(f"100% is a view {width} viewer-coordinate units wide", text)
        self.assertIn(f"up to {highest}", text)

    def test_the_docs_state_the_same_hundred_percent_and_highest_zoom(self):
        # The zoom card and the Quick Start's Zoom bullet, in English and in Chinese, give
        # both numbers; only the numbers are checked, not the wording around them.
        import re

        width = f"{zoom_command.ZOOM_100_PERCENT_WIDTH:g}"
        highest = f"{zoom_command.MAX_ZOOM_PERCENT:g}%"
        docs = (os.path.join(PROJECT_ROOT, "docs"),
                os.path.join(SRC_DIR, "resources", "languages", "zh_CN", "docs"))
        for folder in docs:
            with open(os.path.join(folder, "list_of_commands.html"), encoding="utf-8") as handle:
                page = handle.read()
            card = page.split('id="cmd-zoom"', 1)[1].split("</section>", 1)[0]
            card = re.sub(r"<[^>]+>", "", card)
            with open(os.path.join(folder, "quickstart.html"), encoding="utf-8") as handle:
                quickstart = re.sub(r"<[^>]+>", "", handle.read())
            with self.subTest(folder=folder):
                self.assertRegex(card, r"100%%.{0,30}?\b%s\b" % width)
                self.assertIn(highest, card)
                self.assertRegex(quickstart, r"100%%.{0,30}?\b%s\b" % width)


# A 900 x 300 network, a node at each corner of that box and one hidden between.
NETWORK = np.array([[0.0, 0.0], [900.0, 0.0], [0.0, 300.0], [900.0, 300.0], [450.0, 150.0]])


def network_viewer(positions=NETWORK, canvas_size=(1200, 800), camera=None):
    """A viewer showing positions on a canvas, its camera on the view the Viewer opens with."""
    camera = camera or FakeCamera(zoom_command.opening_view(positions))
    viewer = zoom_viewer(canvas_size, camera)
    viewer.pos = np.array(positions, dtype=float)
    viewer.visible_mask = np.ones(len(positions), dtype=bool)
    return viewer


class ZoomLimitTests(HundredPercentIs1000, unittest.TestCase):
    def test_the_viewer_opens_on_the_view_set_range_used_to_draw(self):
        # The network, 5% of its extent each side (at least 10 units), then the
        # camera's set_range margin of 5% of that range each side.
        width, height = 900 + 2 * 45, 300 + 2 * 15
        expected = (-45 - width * 0.05, -15 - height * 0.05, width * 1.1, height * 1.1)
        for got, want in zip(zoom_command.opening_view(NETWORK), expected):
            self.assertAlmostEqual(got, want)
        # A single node still gets 10 units each side.
        self.assertEqual(zoom_command.opening_view([[5.0, 5.0]]), (-6.0, -6.0, 22.0, 22.0))

    def test_significant_figures_are_rounded_down(self):
        for value, expected in ((23.26, 23.0), (1.87, 1.8), (91.827, 91.0), (99.99, 99.0), (10.0, 10.0),
                                (432.9, 430.0), (4545.4, 4500.0), (0.826, 0.82), (0.0123, 0.012),
                                (40.0, 40.0), (39.99999999999999, 40.0)):
            with self.subTest(value=value):
                self.assertEqual(zoom_command.round_down(value), expected)
        # Three figures, and the float nearest the decimal, as typing it gives.
        self.assertEqual(zoom_command.round_down(1.2999, 3), 1.29)
        self.assertEqual(zoom_command.round_down(0.29999, 2), 0.29)

    def test_the_lowest_zoom_shows_the_whole_network_rounded_down(self):
        # 1089 wide, 363 high: on a 3:2 canvas the width decides, 91.8%, which is 91%.
        self.assertEqual(zoom_command.zoom_limits(network_viewer()), (91.0, 200000.0))
        # The view at that zoom is 1098.9 wide, so the whole network still fits.
        self.assertAlmostEqual(zoom_command.view_width_limits(network_viewer())[1], 1000.0 * 100 / 91)
        self.assertGreater(zoom_command.view_width_limits(network_viewer())[1], 1089.0)
        self.assertEqual(zoom_command.view_width_limits(network_viewer())[0], 0.5)
        # A tall network on a wide canvas: its height decides, at the canvas's shape (1210 x 2 wide).
        tall = [[0.0, 0.0], [100.0, 1000.0]]
        self.assertEqual(zoom_command.zoom_limits(network_viewer(tall, (1200, 600)))[0], 41.0)
        # Exactly a round percentage stays that percentage, though floating point leaves this
        # one 2500.000000000001 wide, 39.99999999999999%.
        exact = [[0.0, 0.0], [2066.1157024793397, 0.0]]
        self.assertGreater(zoom_command.opening_view(exact)[2], 2500.0)
        self.assertLess(zoom_command.zoom_percent(zoom_command.opening_view(exact)[2]), 40.0)
        self.assertEqual(zoom_command.zoom_limits(network_viewer(exact))[0], 40.0)
        # One node: 22 units, but 33 across a 3:2 canvas, which is 3030%.
        self.assertEqual(zoom_command.zoom_limits(network_viewer([[5.0, 5.0]]))[0], 3000.0)
        # A very large network: under 1%, to two figures.
        wide = [[0.0, 0.0], [100000.0, 0.0]]
        self.assertEqual(zoom_command.zoom_limits(network_viewer(wide))[0], 0.82)

    def test_hidden_nodes_count_and_the_lowest_zoom_follows_moved_nodes(self):
        # A hidden node far out still sets the lowest zoom: 2420 wide, 41.3%.
        far = network_viewer(np.vstack([NETWORK, [[2000.0, 0.0]]]))
        far.visible_mask[-1] = False
        self.assertEqual(zoom_command.zoom_limits(far)[0], 41.0)
        viewer = network_viewer()
        viewer.pos[1] = [2000.0, 0.0]  # dragged further out: 2420 wide
        self.assertEqual(zoom_command.zoom_limits(viewer)[0], 41.0)
        viewer.pos[1] = viewer.pos[3] = [100.0, 0.0]  # two dragged in: 544.5 wide
        self.assertEqual(zoom_command.zoom_limits(viewer)[0], 180.0)

    def test_without_nodes_there_is_no_lowest_zoom(self):
        viewer = zoom_viewer()
        self.assertEqual(zoom_command.zoom_limits(viewer), (None, 200000.0))
        self.assertEqual(zoom_command.view_width_limits(viewer), (0.5, None))
        viewer.pos = np.empty((0, 2))
        self.assertEqual(zoom_command.zoom_limits(viewer), (None, 200000.0))
        self.assertEqual(zoom_command.view_width_limits(viewer), (0.5, None))

    def test_the_zoom_goes_by_the_width_the_camera_shows(self):
        # A camera asked for a tall view shows a wider one, at the canvas's shape (its real
        # rectangle); the zoom is that of what shows, not of what was asked for.
        camera = FakeCamera((0.0, 0.0, 100.0, 400.0))
        self.assertEqual(zoom_command.visible_width(camera), 100.0)
        camera._real_rect = SimpleNamespace(width=300.0)
        self.assertEqual(zoom_command.visible_width(camera), 300.0)
        viewer = network_viewer(camera=camera)
        self.assertAlmostEqual(zoom_command.current_zoom(viewer), 100.0 * 1000.0 / 300.0)
        # A view with no width has no zoom to show.
        camera._real_rect = SimpleNamespace(width=0.0)
        self.assertIsNone(zoom_command.current_zoom(viewer))

    def test_a_zoom_stops_at_a_limit_and_goes_no_further_past_one(self):
        viewer = network_viewer(camera=FakeCamera((0.0, 0.0, 1000.0, 666.0)))
        factor = zoom_command.limited_zoom_factor
        widest = 1000.0 * 100 / 91
        self.assertAlmostEqual(factor(viewer, 2.0), widest / 1000.0)  # out to 91%, no further
        self.assertEqual(factor(viewer, 1.05), 1.05)            # within: as asked
        self.assertAlmostEqual(factor(viewer, 1e-5), 0.0005)    # in to 200000%, no further
        self.assertEqual(factor(viewer, [0.5, 0.5]), 0.5)       # the camera's pair form
        viewer.view.camera.rect = (0.0, 0.0, 1500.0, 1000.0)    # past the lowest zoom, nodes moved in
        self.assertEqual(factor(viewer, 1.2), 1.0)
        self.assertEqual(factor(viewer, 0.9), 0.9)
        viewer.view.camera.rect = (0.0, 0.0, 0.4, 0.3)          # past the highest zoom
        self.assertEqual(factor(viewer, 0.5), 1.0)
        self.assertEqual(factor(viewer, 1.5), 1.5)

    def test_the_command_refuses_a_zoom_outside_the_limits_and_takes_each_limit(self):
        for zoom, refused in (("90.9", True), ("200000.1", True), ("91", False), ("91%", False),
                              ("200000", False), ("200000 %", False), ("100", False)):
            viewer = network_viewer()
            opened = viewer.view.camera.rect
            with self.subTest(zoom=zoom), reported_outcomes() as (succeeded, failed), \
                    redirect_stdout(io.StringIO()):
                zoom_command.run(viewer, zoom.split())
                if refused:
                    self.assertEqual(viewer.view.camera.rect, opened)
                    succeeded.assert_not_called()
                    self.assertEqual(
                        str(failed.call_args.args[1]),
                        "Error: Zoom must be between 91% and 200000%, "
                        "the lowest being the zoom that shows every node.",
                    )
                else:
                    failed.assert_not_called()
                    percent = float(zoom.strip("% "))
                    self.assertEqual(viewer.view.camera.rect.width, 1000.0 * 100 / percent)

    def test_the_limit_message_gives_the_limits_in_percent_to_the_figures_they_have(self):
        for nodes, zoom, expected in (([[0.0, 0.0], [100000.0, 0.0]], "0.5", "0.82% and 200000%"),
                                      ([[5.0, 5.0]], "100", "3000% and 200000%")):
            viewer = network_viewer(nodes)
            with self.subTest(expected=expected), reported_outcomes() as (succeeded, failed),                     redirect_stdout(io.StringIO()):
                zoom_command.run(viewer, [zoom])
                self.assertIn(expected, str(failed.call_args.args[1]))

    def test_the_zoom_box_zooms_through_the_commands_checks(self):
        # set_zoom is what the zoom command and the Zoom box share: it zooms and
        # returns the zoom, or raises the command's error and leaves the view,
        # without reporting anything itself.
        viewer = network_viewer()
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(zoom_command.set_zoom(viewer, 200.0), 200.0)
            self.assertEqual(viewer.view.camera.rect.width, 500.0)
            viewer._hud_timer.start.assert_called_once_with()
            for zoom in (90.0, 200001.0, float("nan"), -5.0, 0.0):
                with self.subTest(zoom=zoom), self.assertRaises(ValueError) as caught:
                    zoom_command.set_zoom(viewer, zoom)
                self.assertIsInstance(caught.exception.args[0], zoom_command.Message)
                self.assertEqual(viewer.view.camera.rect.width, 500.0)
        succeeded.assert_not_called()
        failed.assert_not_called()
        self.assertEqual(printed.getvalue(), "")

    def test_a_limit_a_float_digit_off_is_still_the_limit(self):
        # The view at the lowest zoom, taken back to a percentage, may be a float digit
        # under it (as 41 and 43 are); the Zoom box sends that and the command takes it.
        off = [p for p in range(10, 100) if zoom_command.zoom_percent(zoom_command.width_at_zoom(float(p))) < p]
        self.assertTrue(off)
        for lowest in off:
            viewer = network_viewer()
            percent = zoom_command.zoom_percent(zoom_command.width_at_zoom(float(lowest)))
            with self.subTest(lowest=lowest), mock.patch.object(
                zoom_command, "zoom_limits", return_value=(float(lowest), 200000.0)
            ):
                self.assertEqual(zoom_command.set_zoom(viewer, percent), percent)
                self.assertGreaterEqual(viewer.view.camera.rect.width, 1000.0)


class FakeClock:
    """A clock a test moves on itself."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeTimer:
    """A QTimer's start, stop and timeout, run frame by frame by the test."""

    def __init__(self):
        self.active = False
        self.timeout = SimpleNamespace(connect=self._connect)
        self._slot = None

    def _connect(self, slot):
        self._slot = slot

    def isActive(self):
        return self.active

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def fire(self):
        self._slot()


def run_frames(timer, clock, camera, frame_s=0.015, limit=200):
    """Fire the timer every frame_s until it stops; the view's width after each frame."""
    widths = []
    while timer.active and len(widths) < limit:
        clock.now += frame_s
        timer.fire()
        widths.append(camera.rect.width)
    return widths


class ScaledCamera:
    """A camera whose zoom scales its rectangle about center, as PanZoomCamera's does.

    limit, if set, is the widest it goes: zoom cuts the factor there and
    returns the factor applied, as the Viewer's limited zoom does.
    """

    def __init__(self, width=400.0, limit=None):
        self.rect = SimpleNamespace(pos=(0.0, 0.0), width=width, height=width)
        self.limit = limit
        self.centers = []

    def zoom(self, factor, center):
        if self.limit is not None:
            factor = min(factor, self.limit / self.rect.width)
        left, bottom = self.rect.pos
        width = self.rect.width * factor
        self.rect = SimpleNamespace(pos=(center[0] - (center[0] - left) * factor,
                                         center[1] - (center[1] - bottom) * factor),
                                    width=width, height=width)
        self.centers.append(center)
        return factor


class SmoothWheelZoomTests(unittest.TestCase):
    """desktop.Viewer_Zoom: a wheel notch's zoom, played out over a few frames."""

    def make(self, **camera):
        from desktop.Viewer_Zoom import SmoothWheelZoom

        self.camera, self.clock, self.timer = ScaledCamera(**camera), FakeClock(), FakeTimer()
        self.steps = []
        return SmoothWheelZoom(self.camera, self.camera.zoom, lambda: self.steps.append(True),
                               clock=self.clock, timer=self.timer)

    def test_a_notch_eases_out_to_its_whole_zoom(self):
        from desktop.Viewer_Zoom import WHEEL_ZOOM_TAU_S

        zoom = self.make()
        zoom.add(1.25, (40.0, 40.0))
        self.assertTrue(self.timer.active)
        self.assertEqual(self.camera.rect.width, 400.0)
        widths = run_frames(self.timer, self.clock, self.camera)
        self.assertFalse(self.timer.active)
        self.assertAlmostEqual(widths[-1], 500.0, places=9)
        # Each frame zooms less than the one before, and most of it comes early;
        # the last also takes the little left, under 0.15% of the view.
        changes = np.diff([400.0] + widths)
        self.assertTrue(np.all(changes > 0) and np.all(np.diff(changes[:-1]) < 0), changes)
        self.assertLess(changes[-1], 0.0015 * widths[-1])
        frames_to_most = next(index for index, width in enumerate(widths) if width >= 400.0 + 0.63 * 100.0)
        self.assertAlmostEqual((frames_to_most + 1) * 0.015, WHEEL_ZOOM_TAU_S, delta=0.03)
        # Done within half a second, the HUD told every frame, about the cursor every frame.
        self.assertLessEqual(len(widths) * 0.015, 0.5)
        self.assertEqual(len(self.steps), len(widths))
        self.assertEqual(set(self.camera.centers), {(40.0, 40.0)})

    def test_notches_add_up_and_a_notch_back_takes_one_away(self):
        zoom = self.make()
        zoom.add(1.25, (0.0, 0.0))
        self.clock.now += 0.015
        self.timer.fire()
        zoom.add(1.25, (0.0, 0.0))
        zoom.add(1.25, (0.0, 0.0))
        zoom.add(0.8, (0.0, 0.0))
        self.assertAlmostEqual(run_frames(self.timer, self.clock, self.camera)[-1], 400.0 * 1.25 ** 2, places=9)

    def test_a_slow_frame_catches_up(self):
        zoom = self.make()
        zoom.add(2.0, (0.0, 0.0))
        self.clock.now += 1.0
        self.timer.fire()
        self.assertAlmostEqual(self.camera.rect.width, 800.0, places=6)
        self.assertFalse(self.timer.active)

    def test_it_stops_at_a_limit(self):
        zoom = self.make(limit=450.0)
        zoom.add(1.25, (0.0, 0.0))
        widths = run_frames(self.timer, self.clock, self.camera)
        self.assertAlmostEqual(widths[-1], 450.0, places=9)
        self.assertFalse(self.timer.active)
        self.assertLess(len(widths), 20)

    def test_a_view_moved_by_anything_else_ends_it_there(self):
        zoom = self.make()
        zoom.add(1.25, (0.0, 0.0))
        self.clock.now += 0.015
        self.timer.fire()
        # As a pan, the zoom command, the Zoom box or a print would.
        self.camera.rect = SimpleNamespace(pos=(7.0, 7.0), width=300.0, height=300.0)
        self.clock.now += 0.015
        self.timer.fire()
        self.assertFalse(self.timer.active)
        self.assertEqual(self.camera.rect.width, 300.0)
        # A notch after that zooms from the new view, without the old notch's rest.
        zoom.add(1.25, (0.0, 0.0))
        self.assertAlmostEqual(run_frames(self.timer, self.clock, self.camera)[-1], 375.0, places=9)

    def test_nothing_to_zoom_starts_nothing(self):
        zoom = self.make()
        for factor in (1.0, 0.0, -2.0, float("nan"), float("inf")):
            zoom.add(factor, (0.0, 0.0))
        self.assertFalse(self.timer.active)
        zoom.add(1.25, (0.0, 0.0))
        zoom.stop()
        self.assertFalse(self.timer.active)
        self.clock.now += 0.015
        self.timer.fire()
        self.assertEqual(self.camera.rect.width, 400.0)


class ViewerWheelZoomTests(HundredPercentIs1000, unittest.TestCase):
    """The Viewer's camera: a wheel or pinch zoom goes through camera.zoom, which keeps to the limits."""

    def test_the_cameras_wheel_and_pinch_zoom_through_its_zoom_method(self):
        import inspect

        from vispy.scene.cameras import PanZoomCamera

        source = inspect.getsource(PanZoomCamera.viewbox_mouse_event)
        for event in ("'mouse_wheel'", "'gesture_zoom'"):
            branch = source.split(event, 1)[1].split("elif", 1)[0]
            self.assertIn("self.zoom(", branch, event)
        # The Viewer's smooth wheel zooms by VisPy's amount a notch, about the same point.
        wheel = source.split("'mouse_wheel'", 1)[1].split("elif", 1)[0]
        self.assertIn("center = self._scene_transform.imap(event.pos)", wheel)
        self.assertIn("(1 + self.zoom_factor)**(-event.delta[1] * 30)", wheel)

    def test_a_wheel_notch_zooms_by_vispys_amount_over_a_few_frames(self):
        from vispy.scene.cameras import PanZoomCamera

        import EMAPSSN_Viewer
        from desktop.Viewer_Zoom import SmoothWheelZoom

        camera = PanZoomCamera(aspect=1)
        viewer = network_viewer(camera=camera)
        camera.rect = (0.0, 0.0, 400.0, 400.0)
        camera._scene_transform = SimpleNamespace(imap=lambda pos: np.array([pos[0], pos[1], 0.0, 1.0]))
        EMAPSSN_Viewer.MainViewer._limit_camera_zoom(viewer)
        clock, timer = FakeClock(), FakeTimer()
        viewer._wheel_zoom = SmoothWheelZoom(camera, camera.zoom, clock=clock, timer=timer)
        viewer._original_camera_mouse_event = mock.Mock()
        viewer._note_view_motion = mock.Mock()
        wheel = SimpleNamespace(type="mouse_wheel", handled=False, pos=(100.0, 300.0), delta=(0.0, -1.0),
                                button=None, buttons=[])

        EMAPSSN_Viewer.MainViewer._camera_mouse_event(viewer, wheel)
        self.assertTrue(wheel.handled)
        viewer._note_view_motion.assert_called_once_with()  # fewer edges while it zooms
        viewer._original_camera_mouse_event.assert_not_called()
        self.assertEqual(camera.rect.width, 400.0)  # Nothing yet: it plays out over the frames.
        widths = run_frames(timer, clock, camera)
        notch = (1 + camera.zoom_factor) ** 30
        self.assertAlmostEqual(widths[-1], 400.0 * notch, places=6)
        self.assertGreaterEqual(len(widths), 5)
        # The point under the cursor stays where it is.
        left, bottom = camera.rect.left, camera.rect.bottom
        self.assertAlmostEqual((100.0 - left) / camera.rect.width, 100.0 / 400.0)
        self.assertAlmostEqual((300.0 - bottom) / camera.rect.height, 300.0 / 400.0)
        # A notch past the lowest zoom stops there, in fewer frames than a whole notch takes.
        camera.rect = (0.0, 0.0, 1000.0, 1000.0)
        EMAPSSN_Viewer.MainViewer._camera_mouse_event(viewer, SimpleNamespace(**{**vars(wheel), "handled": False}))
        at_limit = run_frames(timer, clock, camera)
        self.assertAlmostEqual(at_limit[-1], 1000.0 * 100 / 91)
        self.assertLess(len(at_limit), len(widths) - 2)
        # The right button rotates dragged nodes with the wheel; the camera never sees it.
        right = SimpleNamespace(type="mouse_wheel", handled=False, pos=(0.0, 0.0), delta=(0.0, 1.0),
                                button=None, buttons=[2])
        EMAPSSN_Viewer.MainViewer._camera_mouse_event(viewer, right)
        self.assertFalse(timer.active)
        # Other events still go to the camera.
        drag = SimpleNamespace(type="mouse_move", handled=False, button=1, buttons=[1])
        EMAPSSN_Viewer.MainViewer._camera_mouse_event(viewer, drag)
        viewer._original_camera_mouse_event.assert_called_once_with(drag)
        # Low Resource Mode's No Zoom Animation zooms at once, as the camera itself does.
        viewer._original_camera_mouse_event.reset_mock()
        low = SimpleNamespace(**{**vars(wheel), "handled": False})
        with mock.patch.object(EMAPSSN_Viewer.cfg, "LOW_RESOURCE_ANIMATIONS", True):
            EMAPSSN_Viewer.MainViewer._camera_mouse_event(viewer, low)
        viewer._original_camera_mouse_event.assert_called_once_with(low)
        self.assertFalse(timer.active)

    def test_the_viewer_zooms_its_wheel_smoothly(self):
        import ast

        import EMAPSSN_Viewer

        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    and node.name == "MainViewer").body
        init = next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assigned = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Assign)]
        self.assertIn("self.view.camera.viewbox_mouse_event = self._camera_mouse_event", assigned)
        # Made after the zoom is limited, so it zooms through the limits.
        lines = {ast.unparse(node): node.lineno for node in ast.walk(init) if isinstance(node, (ast.Assign, ast.Expr))}
        smooth = next(line for text, line in lines.items() if text.startswith("self._wheel_zoom = SmoothWheelZoom("))
        self.assertLess(lines["self._limit_camera_zoom()"], smooth)

    def test_the_viewer_opens_on_every_node_and_limits_its_cameras_zoom(self):
        import ast

        import EMAPSSN_Viewer

        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    and node.name == "MainViewer").body
        init = next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        calls = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Call)]
        self.assertIn("self._limit_camera_zoom()", calls)
        assigned = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Assign)]
        self.assertIn("self.view.camera.rect = opening_view(self.pos)", assigned)
        self.assertFalse([call for call in calls if "set_range" in call])

    def test_a_real_camera_reports_the_zoom_of_what_shows_and_zooms_to_an_exact_percentage(self):
        from vispy import scene

        # A tall request on a wide box: the camera widens it, so what shows is wider than the
        # rectangle it was asked for, and the zoom is that of what shows.
        box = scene.widgets.ViewBox()
        box.size = (800, 400)
        camera = scene.PanZoomCamera(aspect=1)
        box.camera = camera
        camera.rect = (0.0, 0.0, 100.0, 100.0)
        self.assertEqual((camera.rect.width, camera._real_rect.width), (100.0, 200.0))
        viewer = network_viewer([[0.0, 0.0], [100.0, 100.0]], (800, 400), camera)
        self.assertEqual(zoom_command.current_zoom(viewer), 500.0)
        # The window's shape changes what shows, and so the zoom.
        box.size = (400, 400)
        viewer.canvas.size = (400, 400)
        self.assertEqual(zoom_command.current_zoom(viewer), 1000.0)
        # `zoom 800` shows exactly an 800% view, and the camera then reports it.
        with redirect_stdout(io.StringIO()):
            self.assertEqual(zoom_command.set_zoom(viewer, 800), 800)
        self.assertAlmostEqual(camera._real_rect.width, 125.0)
        self.assertAlmostEqual(zoom_command.current_zoom(viewer), 800.0)
        # The wheel stops at the lowest zoom for this shape, and the highest.
        import EMAPSSN_Viewer

        EMAPSSN_Viewer.MainViewer._limit_camera_zoom(viewer)
        camera.zoom(1e9)
        self.assertAlmostEqual(zoom_command.current_zoom(viewer), zoom_command.zoom_limits(viewer)[0])
        camera.zoom(1e-9)
        self.assertAlmostEqual(zoom_command.current_zoom(viewer), 200000.0)

    def test_a_wheel_zoom_stops_at_the_lowest_and_highest_zoom(self):
        from vispy.scene.cameras import PanZoomCamera

        import EMAPSSN_Viewer

        camera = PanZoomCamera(aspect=1)
        viewer = network_viewer(camera=camera)
        camera.rect = zoom_command.opening_view(NETWORK)
        EMAPSSN_Viewer.MainViewer._limit_camera_zoom(viewer)
        for _ in range(20):
            camera.zoom(1.5, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 1000.0 * 100 / 91)
        # It says what it applied, which the smooth wheel stops on: here, nothing.
        self.assertEqual(camera.zoom(1.5, (450.0, 150.0)), 1.0)
        camera.zoom(0.5, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 1000.0 * 50 / 91)
        for _ in range(40):
            camera.zoom(0.25, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 0.5)
        # Dragging a node out lets the view widen with it.
        viewer.pos[1] = [3000.0, 0.0]  # 3630 wide: 27.5%, so 27%
        for _ in range(30):
            camera.zoom(2.0, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 1000.0 * 100 / 27)


class ResizingCanvas:
    """A canvas that sizes its view when it is resized, as VisPy's SceneCanvas does.

    The view is sized by the first callback connected to the resize event, as
    SceneCanvas connects its own handler when it is made; the camera then
    works out the rectangle it shows for the view's new shape.
    """

    def __init__(self, view, size):
        from vispy.util.event import EmitterGroup, Event

        self.view = view
        self.size = size
        self.events = EmitterGroup(source=self, resize=Event)
        self.events.resize.connect(self.size_the_view)

    def size_the_view(self, event):
        self.view.size = self.size

    def resize(self, size):
        self.size = size
        self.events.resize()

    def update(self):
        pass


class ZoomBoxResizeTests(HundredPercentIs1000, unittest.TestCase):
    """The Zoom box shows the zoom of the picture after any resize of the window.

    VisPy calls the most recently connected callback first, so the Viewer's
    resize handler ran before the canvas had sized the view: the box read the
    camera as the resize before had left it, a window behind. Only the box was
    late; the camera's rectangle never moved. The Viewer now connects last.
    """

    def make_viewer(self, position):
        from vispy import scene

        import EMAPSSN_Viewer

        view = scene.widgets.ViewBox()
        view.size = (1400, 900)
        view.camera = scene.PanZoomCamera(aspect=1)
        # 200% on a 1400 x 900 view: 500 units wide, as tall as that view's shape makes it.
        view.camera.rect = (0.0, 0.0, 500.0, 500.0 * 900 / 1400)
        canvas = ResizingCanvas(view, (1400, 900))
        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        viewer.canvas, viewer.view = canvas, view
        viewer.hud_layout = {"status_x_offset": 10.0, "status_line_spacing": 34.0}
        viewer.hud_displays = {}
        viewer.pos = NETWORK.copy()
        viewer.visible_mask = np.ones(len(NETWORK), dtype=bool)
        viewer.hidden_text = SimpleNamespace(color=None, text="", pos=None)
        viewer.selected_node_idx = None
        viewer.tooltip = None
        viewer.zoom_box = SimpleNamespace(spin_box=mock.Mock())
        viewer.update_console_background = mock.Mock()
        viewer._refresh_session_card = mock.Mock()
        viewer.reposition_expand_btn = mock.Mock()
        canvas.events.resize.connect(viewer.on_resize, position=position)
        return viewer

    def box(self, viewer):
        """The zoom the box was last told to show."""
        return viewer.zoom_box.spin_box.show_zoom.call_args.args[0]

    def test_the_box_matches_the_picture_after_every_resize_and_the_camera_never_moves(self):
        viewer = self.make_viewer("last")
        camera = viewer.view.camera
        rectangle = (tuple(camera.rect.pos), camera.rect.width, camera.rect.height)
        # A shorter window shows more width at the same height of picture, so a smaller zoom, and
        # the same window again shows the zoom it had: none of it left behind by the earlier sizes.
        for height, zoom in ((900, 200.0), (600, 200.0 * 600 / 900), (900, 200.0), (300, 200.0 * 300 / 900),
                             (900, 200.0), (40, 200.0 * 40 / 900), (900, 200.0), (1, 200.0 / 900), (900, 200.0)):
            with self.subTest(height=height):
                viewer.canvas.resize((1400, height))
                self.assertAlmostEqual(self.box(viewer), zoom_command.current_zoom(viewer))
                self.assertAlmostEqual(self.box(viewer), zoom, places=6)
                self.assertEqual((tuple(camera.rect.pos), camera.rect.width, camera.rect.height), rectangle)

    def test_a_width_only_resize_shows_the_zoom_of_the_width_it_gives(self):
        viewer = self.make_viewer("last")
        # Narrower, the view keeps its width and shows more height; wider, it shows more width.
        for width, zoom in ((563, 200.0), (2000, 200.0 * 1400 / 2000), (1400, 200.0)):
            viewer.canvas.resize((width, 900))
            self.assertAlmostEqual(self.box(viewer), zoom, places=6)
            self.assertAlmostEqual(self.box(viewer), zoom_command.current_zoom(viewer))

    def test_the_box_is_a_resize_behind_when_connected_first_as_vispy_does_by_default(self):
        # The premise: the default position is first, ahead of the canvas's own handler.
        viewer = self.make_viewer("first")
        viewer.canvas.resize((1400, 600))
        self.assertAlmostEqual(self.box(viewer), 200.0)  # what the window before showed
        self.assertAlmostEqual(zoom_command.current_zoom(viewer), 200.0 * 600 / 900)

    def test_the_viewer_connects_its_resize_handler_last(self):
        import ast

        import EMAPSSN_Viewer

        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    and node.name == "MainViewer").body
        init = next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        calls = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Call)]
        self.assertIn("self.canvas.events.resize.connect(self.on_resize, position='last')", calls)


if __name__ == "__main__":
    unittest.main()
