"""Print command SVG export tests (src/commands/print.py): layered SVG export,
edge opacity bands under Auto Edge Opacity, chunked streaming, outcomes, node
stacking order, marker shapes, frame/viewBox, zoom scaling, outline/edge width
models, selection outlines, and OpenGL PNG-vs-SVG parity."""

import io
import json
import os
import re
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.held_file_fixtures import transient_hold
from tests.print_command_fixtures import (
    ElementTreeLines,
    SRC_DIR,
    elements,
    export_svg,
    make_viewer,
    node_fills,
    number,
    numbered_colors,
    print_command,
    setUpModule,
    square_network,
    tearDownModule,
    view_at,
    vispy_outline_coverage,
)


class PrintSvgExportTests(unittest.TestCase):
    def test_svg_edges_follow_the_on_screen_similarity_threshold(self):
        viewer = square_network()

        lines = elements(export_svg(viewer), "line")

        # Scores 0.9 and 0.95 pass the 0.5 slider; 0.2 and 0.1 are hidden on screen.
        self.assertEqual(len(lines), 2)

    def test_svg_edges_match_shared_render_filter(self):
        viewer = square_network(current_slider_threshold=0.15)
        expected, _ = print_command.edge_stages(viewer, SimpleNamespace(UMAP_MODE=False))

        lines = elements(export_svg(viewer), "line")

        self.assertEqual(len(lines), len(expected))
        self.assertEqual(len(lines), 3)

    def test_svg_in_umap_mode_draws_only_edges_of_selected_nodes(self):
        viewer = square_network(current_slider_threshold=0.0, selected_indices=[0])

        lines = elements(export_svg(viewer, UMAP_MODE=True), "line")

        # Node 0 touches edges (0, 1) and (3, 0), matching the UMAP-mode display.
        self.assertEqual(len(lines), 2)

    def test_svg_uses_configured_edge_and_node_boundary_colors(self):
        svg = export_svg(
            square_network(),
            EDGE_COLOR="#ff0000",
            NODE_BOUNDARY_COLOR="#00ff00",
            EDGE_ALPHA=0.4,
        )

        lines = elements(svg, "line")
        circles = elements(svg, "circle")
        self.assertTrue(lines)
        self.assertTrue(all('stroke="rgb(255,0,0)" stroke-opacity="0.400"' in line for line in lines))
        self.assertEqual(len(circles), 4)
        self.assertTrue(all('stroke="rgb(0,255,0)"' in circle for circle in circles))


class PrintSvgOpacityBandsTests(unittest.TestCase):
    """With Edge Opacity's Auto switch on, the SVG's edges go out in opacity bands.

    The screen draws each edge at an alpha that follows its score (15 % at the
    lowest score drawn, 85 % at the highest); the SVG groups the edges in
    SVG_OPACITY_BANDS evenly spaced opacities, most opaque first, each in a
    <g> with its stroke-opacity (and on every line too, so a reader that
    ignores the group's still has it). With Auto off, or in a layout that
    ignores it, the file is the one it always was.
    """

    SVG = "{http://www.w3.org/2000/svg}"
    SCORES = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5]

    @classmethod
    def network(cls, scores=None, threshold=0.3, by_score=True, **overrides):
        from desktop.Viewer_Edges import ViewerEdges
        scores = np.array(cls.SCORES if scores is None else scores)
        viewer = square_network(
            edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=np.int32)[:len(scores)],
            edge_scores=scores, current_slider_threshold=threshold, **overrides)
        layer = ViewerEdges(None, viewer.edges, scores, viewer.pos, (0, 0, 0, 0.2), 1.0,
                            use_position_texture=True)
        layer.set_opacity_by_score(by_score)
        viewer.edge_layer = layer
        return viewer

    @staticmethod
    def band_opacity(alpha, bands=128):
        """The opacity of the band an edge of this alpha belongs in, worked out here on its own."""
        step = round((alpha - 0.15) / 0.7 * (bands - 1))
        return 0.15 + 0.7 * step / (bands - 1)

    def groups(self, svg):
        import xml.etree.ElementTree as ElementTree
        root = ElementTree.fromstring(svg)
        edges = next(group for group in root.iter(self.SVG + "g") if group.get("id") == "edges")
        return edges, edges.findall(self.SVG + "g")

    @staticmethod
    def coordinates(line):
        return tuple(line.get(name) for name in ("x1", "y1", "x2", "y2"))

    def test_the_edges_are_in_bands_most_opaque_first_with_each_edges_own_opacity(self):
        viewer = self.network()
        svg = export_svg(viewer, AUTO_EDGE_ALPHA=True)
        plain = ElementTreeLines.parse(export_svg(viewer, AUTO_EDGE_ALPHA=False))
        edges, groups = self.groups(svg)
        # The scores kept by the 0.3 threshold, in loaded order: 0.4, 0.6, 0.8, 1.0, 0.5 (0.2 is below it).
        kept = [score for score in self.SCORES if score >= 0.3]
        alphas = [0.15 + 0.7 * (score - 0.4) / 0.6 for score in kept]
        wanted = [f"{self.band_opacity(alpha):.3f}" for alpha in alphas]
        self.assertEqual(len(plain), len(kept))
        by_edge = {}
        opacities = []
        for number, group in enumerate(groups, start=1):
            self.assertEqual(group.get("id"), f"edges-band-{number}")
            opacities.append(group.get("stroke-opacity"))
            lines = group.findall(self.SVG + "line")
            self.assertTrue(lines)
            for line in lines:
                # Every line states its own opacity, the group's.
                self.assertEqual(line.get("stroke-opacity"), group.get("stroke-opacity"))
                self.assertEqual(line.get("stroke"), "rgb(0,0,0)")
                by_edge[self.coordinates(line)] = group.get("stroke-opacity")
        self.assertEqual(len(by_edge), len(kept))
        for coordinates, expected in zip(plain, wanted):
            self.assertEqual(by_edge[coordinates], expected)
        # Most opaque first: the edges go out in descending score, as the screen draws them.
        self.assertEqual(opacities, sorted(set(wanted), reverse=True))
        self.assertEqual(opacities[0], "0.850")
        self.assertEqual(opacities[-1], "0.150")
        # The groups hold the edges and nothing else; the nodes layer is as it was.
        self.assertEqual(len(edges), len(groups))
        self.assertEqual(len(elements(svg, "circle")), 4)

    def test_the_file_stays_plain_svg(self):
        import xml.etree.ElementTree as ElementTree
        svg = export_svg(self.network(), AUTO_EDGE_ALPHA=True)
        root = ElementTree.fromstring(svg)  # well formed
        self.assertEqual(root.tag, self.SVG + "svg")
        _, groups = self.groups(svg)
        for group in groups:
            self.assertEqual(sorted(group.attrib), ["id", "name", "stroke-opacity"])
        # Only the elements and attributes the file always had, and the groups.
        tags = {element.tag.removeprefix(self.SVG) for element in root.iter()}
        self.assertEqual(tags, {"svg", "rect", "g", "line", "circle"})
        lines = elements(svg, "line")
        for line in lines:
            self.assertEqual(re.findall(r'(\w[\w-]*)="', line),
                             ["x1", "y1", "x2", "y2", "stroke", "stroke-opacity", "stroke-width"])

    def test_equal_scores_are_one_band_at_85_percent(self):
        for scores in ([0.7, 0.7, 0.7], [0.9, 0.2]):
            with self.subTest(scores=scores):
                threshold = 0.5 if len(scores) == 2 else 0.3
                svg = export_svg(self.network(scores, threshold=threshold), AUTO_EDGE_ALPHA=True)
                _, groups = self.groups(svg)
                self.assertEqual([group.get("stroke-opacity") for group in groups], ["0.850"])
                self.assertEqual(len(groups[0].findall(self.SVG + "line")), len(
                    [score for score in scores if score >= threshold]))

    def test_the_threshold_decides_which_edge_is_the_faintest(self):
        low = self.groups(export_svg(self.network(threshold=0.3), AUTO_EDGE_ALPHA=True))[1]
        high = self.groups(export_svg(self.network(threshold=0.7), AUTO_EDGE_ALPHA=True))[1]
        for groups in (low, high):
            self.assertEqual(groups[0].get("stroke-opacity"), "0.850")
            self.assertEqual(groups[-1].get("stroke-opacity"), "0.150")
        self.assertEqual(sum(len(group) for group in low), 5)
        self.assertEqual(sum(len(group) for group in high), 2)

    def test_the_file_is_the_same_whatever_the_chunk_size(self):
        viewer = self.network()
        whole = export_svg(viewer, AUTO_EDGE_ALPHA=True)
        for chunk in (1, 2, 4):
            with self.subTest(chunk=chunk), mock.patch.object(print_command, "SVG_CHUNK_SIZE", chunk):
                self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), whole)

    def test_auto_off_is_the_file_it_always_was(self):
        # Byte for byte what the code before the switch wrote for this network.
        expected_edges = (
            '  <g id="edges" name="Edges">\n'
            '    <line x1="26.250" y1="36.250" x2="36.250" y2="36.250" stroke="rgb(0,0,0)" '
            'stroke-opacity="0.200" stroke-width="1.000" />\n'
            '    <line x1="36.250" y1="26.250" x2="26.250" y2="26.250" stroke="rgb(0,0,0)" '
            'stroke-opacity="0.200" stroke-width="1.000" />\n'
            '  </g>\n')
        bare = export_svg(square_network())
        self.assertIn(expected_edges, bare)
        for configuration in ({}, {"AUTO_EDGE_ALPHA": False}):
            with self.subTest(configuration=configuration):
                viewer = self.network(scores=[0.90, 0.20, 0.95, 0.10], threshold=0.5)
                self.assertEqual(export_svg(viewer, **configuration), bare)
        self.assertNotIn("edges-band", bare)

    def test_a_layout_that_ignores_auto_writes_the_plain_file(self):
        # UMAP and PaCMAP draw every edge at the slider's alpha.
        viewer = self.network(threshold=0.0, selected_indices=[0])
        plain = export_svg(viewer, AUTO_EDGE_ALPHA=False, UMAP_MODE=True)
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True, UMAP_MODE=True), plain)
        self.assertNotIn("edges-band", plain)
        # So does an edge layer that does not draw by score (the coordinate fallback, no scores).
        viewer = self.network(by_score=False)
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), export_svg(viewer, AUTO_EDGE_ALPHA=False))
        viewer = square_network()
        self.assertEqual(export_svg(viewer, AUTO_EDGE_ALPHA=True), export_svg(viewer, AUTO_EDGE_ALPHA=False))

    def test_the_scores_line_up_with_the_edges_the_screen_keeps(self):
        rng = np.random.default_rng(8)
        visible = rng.random(20) > 0.2
        viewer = SimpleNamespace(
            edges=rng.integers(0, 20, (200, 2)).astype(np.int32), edge_scores=rng.random(200).astype(np.float32),
            visible_mask=visible, n_nodes=20, current_slider_threshold=0.4, selected_indices=[])
        configuration = SimpleNamespace(UMAP_MODE=False)
        active, _ = print_command.edge_stages(viewer, configuration)
        scores = print_command.active_edge_scores(viewer, configuration)
        keep = (viewer.edge_scores >= 0.4) & visible[viewer.edges[:, 0]] & visible[viewer.edges[:, 1]]
        np.testing.assert_array_equal(active, viewer.edges[keep])
        np.testing.assert_array_equal(scores, viewer.edge_scores[keep])

    def test_at_least_32_bands_and_each_edge_is_within_half_a_step_of_its_alpha(self):
        self.assertEqual(print_command.SVG_OPACITY_BANDS, 128)
        self.assertGreaterEqual(print_command.SVG_OPACITY_BANDS, 32)
        rng = np.random.default_rng(3)
        alphas = np.concatenate(([0.15, 0.85], rng.uniform(0.15, 0.85, 5000))).astype(np.float32)
        order, bands = print_command._svg_edge_bands(alphas)
        self.assertEqual(sorted(order.tolist()), list(range(len(alphas))))
        self.assertEqual(bands[0][1], 0)
        self.assertEqual(bands[-1][2], len(alphas))
        opacities = [band[0] for band in bands]
        self.assertEqual(opacities, sorted(opacities, reverse=True))
        step = 0.7 / 127
        worst = 0.0
        for opacity, start, stop in bands:
            members = order[start:stop]
            self.assertTrue((np.diff(members) > 0).all())  # a band keeps the edges' own order
            worst = max(worst, float(np.abs(alphas[members] - opacity).max()))
        self.assertLessEqual(worst, step / 2 + 1e-6)
        # Half a step is under one 8-bit level for a contrast of 255: 0.0028 * 255 < 1.
        self.assertLess(step / 2 * 255, 1.0)
        self.assertEqual(len(bands), len({round(opacity, 9) for opacity in opacities}))
        # No edges, no bands.
        empty_order, empty = print_command._svg_edge_bands(np.empty(0, dtype=np.float32))
        self.assertEqual((len(empty_order), empty), (0, []))


class PrintSvgStreamingTests(unittest.TestCase):
    """The SVG is written a chunk at a time into a file that replaces the target when done."""

    @staticmethod
    def mixed_network(count=23):
        rng = np.random.default_rng(5)
        shapes = ["disc", "ring", "square", "^", "v", "D", "*", "+", "x", "|", "-", ">", "p", "P", "blob"]
        return square_network(
            visible_mask=rng.random(count) < 0.9,
            pos=(rng.random((count, 2)) * 50).astype(np.float32),
            current_colors=rng.random((count, 4)).astype(np.float32),
            current_sizes=rng.choice([4.0, 10.0, 17.5], count).astype(np.float32),
            current_shapes=np.array([shapes[i % len(shapes)] for i in range(count)], dtype=object),
            edges=rng.integers(0, count, (60, 2)).astype(np.int32),
            edge_scores=rng.random(60),
            current_slider_threshold=0.3,
        )

    def test_the_file_is_the_same_whatever_the_chunk_size(self):
        viewer = self.mixed_network()
        whole = export_svg(viewer)
        self.assertGreater(len(elements(whole, "line")), 10)
        for chunk in (1, 2, 7):
            with self.subTest(chunk=chunk), mock.patch.object(print_command, "SVG_CHUNK_SIZE", chunk):
                self.assertEqual(export_svg(viewer), whole)

    def test_a_failure_while_writing_keeps_the_earlier_file_and_leaves_nothing_behind(self):
        viewer = self.mixed_network()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("earlier picture")

            def broken_nodes(*arguments):
                yield "    <circle />\n"
                raise RuntimeError("disk full")

            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    mock.patch.object(print_command, "_svg_node_chunks", broken_nodes), \
                    redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, "disk full"):
                print_command._write_svg(print_command._svg_snapshot(viewer), path)

            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "earlier picture")

    def test_a_virus_scan_holding_the_finished_file_for_a_moment_does_not_fail_the_picture(self):
        viewer = self.mixed_network()
        whole = export_svg(viewer)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    transient_hold(2, winerror=5) as (attempts, pauses), \
                    redirect_stdout(io.StringIO()):
                print_command._write_svg(print_command._svg_snapshot(viewer), path)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), whole)

    def test_a_hold_that_never_ends_fails_the_picture_saying_so_and_leaves_nothing_behind(self):
        viewer = self.mixed_network()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "network.svg")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("earlier picture")
            with mock.patch.multiple(print_command.cfg, create=True, UMAP_MODE=False), \
                    transient_hold(10 ** 6), redirect_stdout(io.StringIO()), \
                    self.assertRaises(PermissionError) as raised:
                print_command._write_svg(print_command._svg_snapshot(viewer), path)

            self.assertIn("another program kept the file in use", str(raised.exception))
            self.assertEqual(os.listdir(directory), ["network.svg"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "earlier picture")


class PrintSvgOutcomeTests(unittest.TestCase):
    """What `print <name> svg` writes, reports and opens."""

    def run_svg_print(self, visible_mask):
        viewer = square_network(visible_mask=np.array(visible_mask))
        for name, value in vars(make_viewer()).items():
            if name not in ("canvas", "view"):
                setattr(viewer, name, value)
        viewer.canvas.update = mock.Mock()
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", save_dir
        ), mock.patch.object(
            print_command.app, "process_events"
        ), redirect_stdout(io.StringIO()):
            print_command.run(viewer, ["network", "svg"])
            written = os.listdir(save_dir)
        return viewer, engine, viewer.background_job_scheduler, written, save_dir

    def test_svg_with_every_node_hidden_fails_and_writes_nothing(self):
        viewer, engine, scheduler, written, _ = self.run_svg_print([False] * 4)

        self.assertEqual(written, [])
        engine.command_failed.assert_called_once()
        self.assertEqual(str(engine.command_failed.call_args.args[1]), "Error: No visible nodes to export.")
        engine.command_succeeded.assert_not_called()
        self.assertEqual(scheduler.jobs, [])
        self.assertEqual(viewer.console_text.text, "Error: No visible nodes to export.")
        self.assertTrue(viewer.zoom_text.visible)

    def test_svg_with_visible_nodes_reports_the_written_file(self):
        viewer, engine, scheduler, written, save_dir = self.run_svg_print([True] * 4)

        self.assertEqual(written, ["network.svg"])
        engine.command_failed.assert_not_called()
        engine.command_succeeded.assert_called_once()
        # The job is print's, for the file, and its report opens the save folder.
        [job] = scheduler.jobs
        self.assertEqual(job.command_name, "print")
        self.assertEqual(job.output_path, os.path.join(save_dir, "network.svg"))
        self.assertEqual(scheduler.reports[0]["reveal_directory"], save_dir)
        # The format is named without the extension's dot ("Saved .SVG" before).
        self.assertEqual(scheduler.report, "Saved SVG: network.svg")


class PrintSvgNodeOrderTests(unittest.TestCase):
    """The SVG stacks nodes as the screen does: the later covers the earlier."""

    def test_nodes_are_written_in_the_order_the_viewer_draws_them(self):
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_node_render_order=lambda: np.array([2, 0, 3, 1], dtype=np.int32),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [25, 5, 35, 15])

    def test_hidden_nodes_stay_out_in_the_viewers_order(self):
        # The viewer's order lists visible nodes only.
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_mask=np.array([False, True, False, True]),
            visible_node_render_order=lambda: np.array([3, 1], dtype=np.int32),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [35, 15])

    def test_a_viewer_without_a_render_order_keeps_index_order(self):
        viewer = square_network(
            current_colors=numbered_colors(),
            visible_mask=np.array([True, False, True, True]),
        )

        self.assertEqual(node_fills(export_svg(viewer)), [5, 25, 35])

    def test_promoted_and_selected_nodes_are_written_last_like_on_screen(self):
        import EMAPSSN_Config
        from EMAPSSN_Viewer import MainViewer

        def real_viewer(**overrides):
            viewer = MainViewer.__new__(MainViewer)
            for name, value in vars(
                square_network(current_colors=numbered_colors(), **overrides)
            ).items():
                setattr(viewer, name, value)
            viewer.n_nodes = 4
            viewer.node_render_order = np.arange(4, dtype=np.int32)
            viewer.selected_node_idx = None
            viewer.left_click_highlight_indices = None
            return viewer

        # What the color command does to the nodes it recolours.
        promoted = real_viewer()
        promoted.promote_nodes([1])
        self.assertEqual(node_fills(export_svg(promoted)), [5, 25, 35, 15])

        # Node 0 is selected, so its neighbours 1 and 3 are raised below it.
        with mock.patch.multiple(
            EMAPSSN_Config,
            CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
        ):
            selected = real_viewer(selected_indices=[0])
            self.assertEqual(node_fills(export_svg(selected)), [25, 15, 35, 5])


class PrintSvgShapeTests(unittest.TestCase):
    ALIASES = {
        "o": "disc", "circle": "disc", "s": "square", "^": "triangle_up",
        "triangle": "triangle_up", "v": "triangle_down", "D": "diamond",
        "*": "star", "+": "cross", "|": "vbar", "-": "hbar", "_": "hbar",
        ">": "arrow", "->": "tailed_arrow", "p": "clobber",
        "P": "cross_lines", "++": "cross_lines",
    }

    @staticmethod
    def with_shape(shape):
        return export_svg(square_network(current_shapes=np.array([shape] * 4, dtype=object)))

    def test_the_table_covers_every_vispy_marker_alias(self):
        from vispy.visuals.markers import symbol_aliases

        for alias, canonical in symbol_aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(print_command._canonical_shape(alias), canonical)

    def test_an_alias_is_drawn_exactly_like_its_canonical_name(self):
        for alias, canonical in self.ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(self.with_shape(alias), self.with_shape(canonical))

    def test_bars_crosses_and_rings_are_filled_and_outlined_like_every_shape(self):
        # VisPy fills them in the node's colour and outlines them; the SVG drew
        # them as bare strokes in the node's colour.
        for alias, tag in (("+", "polygon"), ("x", "polygon"), ("|", "polygon"), ("-", "polygon"),
                           ("ring", "path")):
            with self.subTest(alias=alias):
                nodes = self.with_shape(alias).split('<g id="nodes"')[1]
                drawn = elements(nodes, tag)
                self.assertEqual(len(drawn), 4)
                for element in drawn:
                    self.assertIn('fill="rgb(63,127,255)"', element)
                    self.assertIn('stroke="rgb(0,0,0)"', element)

    def test_a_ring_is_the_band_from_half_the_radius_to_the_radius(self):
        nodes = self.with_shape("ring").split('<g id="nodes"')[1]
        ring = elements(nodes, "path")[0]
        self.assertIn('fill-rule="evenodd"', ring)
        # 10-pixel nodes at one unit per pixel: circles of radius 5 and 2.5.
        radii = {float(value) for value in re.findall(r"A ([\d.]+) ", ring)}
        self.assertEqual(radii, {5.0, 2.5})

    def test_cross_lines_is_two_lines_in_the_outline_colour_alone(self):
        # VisPy's cross_lines has no area: only its outline shows.
        nodes = self.with_shape("P").split('<g id="nodes"')[1]
        drawn = elements(nodes, "path")
        self.assertEqual(len(drawn), 4)
        for element in drawn:
            self.assertIn('fill="none"', element)
            self.assertIn('stroke="rgb(0,0,0)"', element)
            self.assertIn('stroke-linecap="round"', element)

    def test_shapes_have_vispys_proportions(self):
        polygons = dict(print_command._SVG_POLYGONS)
        # An equilateral triangle with sides of the node's size, centred on its box.
        xs, ys = zip(*polygons["triangle_up"])
        self.assertEqual((min(xs), max(xs)), (-1.0, 1.0))
        self.assertAlmostEqual(min(ys), -np.sqrt(3) / 2)
        self.assertAlmostEqual(max(ys), np.sqrt(3) / 2)
        # A regular star whose top and bottom points are as far from the centre.
        star = np.array(polygons["star"])
        self.assertAlmostEqual(-star[:, 1].min(), star[:, 1].max(), delta=0.01)
        inner = np.hypot(star[1::2, 0], star[1::2, 1] - 0.1)
        np.testing.assert_allclose(inner, np.sin(np.pi / 10) / np.sin(3 * np.pi / 10))
        # Bars two thirds of the radius wide.
        self.assertEqual(sorted({x for x, _ in polygons["vbar"]}), [-1 / 3, 1 / 3])
        # Clobber's outline turns where its circles cross, as far out as each other.
        radius, crossings = print_command._unit_clobber()
        self.assertEqual(radius, 0.64)
        distances = [np.hypot(x, y - 0.25 * 0.72 / np.sqrt(3)) for x, y in crossings]
        np.testing.assert_allclose(distances, distances[0])

    def test_an_unknown_shape_is_still_drawn_as_a_circle(self):
        self.assertEqual(self.with_shape("blob"), self.with_shape("disc"))
        self.assertIsNone(print_command._canonical_shape(None))


class PrintSvgFrameTests(unittest.TestCase):
    """The view box frames the network as print's PNG does: what is drawn, and
    a border of 20 pixels of the PNG, which are device pixels.

    square_network's four nodes lie on a 10-unit square at one unit per
    logical pixel. A node of radius r with a 0.5-pixel outline is drawn to
    r + 0.25 + 1 from its centre: half its outline, then VisPy's 1-pixel
    antialias band.
    """

    @staticmethod
    def view_box(svg):
        match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        return float(match.group(1)), float(match.group(2))

    def test_the_view_box_is_the_drawing_and_a_twenty_pixel_border(self):
        # 10 between the centres, 5 + 1.25 drawn past them, 20 of border.
        svg = export_svg(square_network())
        self.assertEqual(self.view_box(svg), (62.5, 62.5))
        # The first node's centre is 6.25 + 20 in from the left, and as far
        # down from the top (y points down in the SVG).
        circle = elements(svg, "circle")[0]
        self.assertEqual((number(circle, "cx"), number(circle, "cy")), (26.25, 36.25))

    def test_the_border_is_twenty_pixels_of_the_png_at_the_display_scale(self):
        # At twice the display scale, 20 device pixels are 10 logical ones,
        # and the outline (1 device pixel) and its band reach 0.5 + 0.5.
        sharp = square_network()
        sharp.canvas.pixel_scale = 2.0
        self.assertEqual(self.view_box(export_svg(sharp)), (41.5, 41.5))

    def test_a_ring_reaches_as_far_as_a_disc(self):
        rings = square_network(current_shapes=np.array(["ring"] * 4, dtype=object))
        self.assertEqual(self.view_box(export_svg(rings)), self.view_box(export_svg(square_network())))

    def test_each_shape_is_framed_by_its_own_reach(self):
        # A bar of 30-pixel nodes reaches 15 across but only 5 up and down;
        # its square corners move the outline straight out.
        bars = square_network(
            current_sizes=np.full(4, 30.0), current_shapes=np.array(["hbar"] * 4, dtype=object),
        )
        self.assertEqual(self.view_box(export_svg(bars)), (82.5, 62.5))

    def test_a_sharp_corner_reaches_a_mitre_further(self):
        # A diamond's corners are right angles: the outline and its band go
        # sqrt(2) times as far along their bisector as a side moves.
        diamonds = square_network(current_shapes=np.array(["diamond"] * 4, dtype=object))
        reach = 5.0 + 1.25 * np.sqrt(2)
        width, height = self.view_box(export_svg(diamonds))
        self.assertAlmostEqual(width, 10 + 2 * reach + 40, places=3)
        self.assertAlmostEqual(height, width)

    def test_nodes_without_size_are_framed_by_their_edges(self):
        # GL widens an edge across its main direction only: the two
        # horizontal edges shown (scores 0.9 and 0.95) by 1.5 up and down.
        bare = square_network(current_sizes=np.zeros(4))
        self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (50.0, 53.0))
        # The edges are measured a chunk at a time, every chunk counting.
        with mock.patch.object(print_command, "SVG_CHUNK_SIZE", 1):
            self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (50.0, 53.0))
        bare.current_slider_threshold = 0.0
        with mock.patch.object(print_command, "SVG_CHUNK_SIZE", 1):
            self.assertEqual(self.view_box(export_svg(bare, EDGE_WIDTH=3.0)), (53.0, 53.0))

    def test_a_wide_network_gets_the_same_border_on_every_side(self):
        viewer = square_network(
            pos=np.array([[0.0, 0.0], [400.0, 0.0], [400.0, 200.0], [0.0, 200.0]])
        )

        self.assertEqual(self.view_box(export_svg(viewer)), (452.5, 252.5))

    def test_large_nodes_at_the_edge_fit_inside_the_view_box(self):
        svg = export_svg(square_network(current_sizes=np.full(4, 40.0)))

        width, height = self.view_box(svg)
        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
            cx, cy, r = (
                float(re.search(rf'{name}="([\d.]+)"', circle).group(1))
                for name in ("cx", "cy", "r")
            )
            self.assertGreaterEqual(cx - r, 0.0)
            self.assertLessEqual(cx + r, width)
            self.assertGreaterEqual(cy - r, 0.0)
            self.assertLessEqual(cy + r, height)

    def test_the_frame_follows_the_zoom_in_scene_units(self):
        # At half a scene unit per pixel, 40-pixel nodes reach 20 + 1.25
        # pixels past their centres, 10.625 units, and the border is 10 units.
        svg = export_svg(square_network(view=view_at(0.5), current_sizes=np.full(4, 40.0)))

        self.assertEqual(self.view_box(svg), (51.25, 51.25))


class PrintSvgScaleTests(unittest.TestCase):
    """Sizes set in screen pixels are written in scene units at the zoom shown.

    The view below shows 160 scene units across its 800 pixels, so one pixel
    is 0.2 units and a 12-pixel node has a radius of 6 pixels, or 1.2 units.
    """

    UNITS_PER_PIXEL = 0.2

    def network(self, **overrides):
        overrides.setdefault("view", view_at(self.UNITS_PER_PIXEL))
        overrides.setdefault("current_sizes", np.full(4, 12.0))
        return square_network(**overrides)

    def test_node_radius_is_half_its_size_in_pixels_in_scene_units(self):
        circles = elements(export_svg(self.network()), "circle")

        self.assertEqual(len(circles), 4)
        for circle in circles:
            self.assertAlmostEqual(number(circle, "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_every_shape_is_scaled_by_the_same_factor(self):
        squares = export_svg(
            self.network(current_shapes=np.array(["square"] * 4, dtype=object))
        )
        crosses = export_svg(
            self.network(current_shapes=np.array(["cross"] * 4, dtype=object))
        )

        drawn_squares = elements(squares, "rect")[1:]  # the first is the background
        self.assertEqual(len(drawn_squares), 4)
        for square in drawn_squares:
            self.assertAlmostEqual(number(square, "width"), 12 * self.UNITS_PER_PIXEL)
        # A cross reaches the radius, with bars a third of it each side of its centre.
        drawn_crosses = elements(crosses, "polygon")
        self.assertEqual(len(drawn_crosses), 4)
        for cross in drawn_crosses:
            points = re.search(r'points="([^"]+)"', cross).group(1).split()
            xs = [float(point.split(",")[0]) for point in points]
            offsets = sorted({round(x - (max(xs) + min(xs)) / 2, 3) for x in xs})
            self.assertEqual(offsets, [-1.2, -0.4, 0.4, 1.2])

    def test_edge_and_node_outline_widths_are_converted_too(self):
        svg = export_svg(self.network(), EDGE_WIDTH=1.5)

        lines = elements(svg, "line")
        self.assertEqual(len(lines), 2)
        # GL draws a 1.5-pixel edge 2 pixels wide.
        for line in lines:
            self.assertAlmostEqual(number(line, "stroke-width"), 2 * self.UNITS_PER_PIXEL)
        # The 0.5-pixel outline of every filled node, as wide as VisPy draws it.
        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
            self.assertAlmostEqual(
                number(circle, "stroke-width"), 0.9818 * self.UNITS_PER_PIXEL, delta=0.001,
            )

    def test_node_outline_follows_the_boundary_width_setting(self):
        # Outlines of 2 pixels and more look 0.23 pixels narrower than set.
        for setting in (2.0, 4.0):
            with self.subTest(setting=setting):
                svg = export_svg(self.network(), NODE_BOUNDARY_WIDTH=setting)
                circles = elements(svg, "circle")
                self.assertEqual(len(circles), 4)
                for circle in circles:
                    self.assertAlmostEqual(
                        number(circle, "stroke-width"), (setting - 0.2317) * self.UNITS_PER_PIXEL,
                        delta=0.001,
                    )

    def test_doubling_the_zoom_halves_the_sizes_relative_to_the_scene(self):
        def drawn(units_per_pixel):
            svg = export_svg(self.network(view=view_at(units_per_pixel)), EDGE_WIDTH=1.5)
            return (
                elements(svg, "circle")[0],
                elements(svg, "line")[0],
            )

        circle, line = drawn(0.2)
        zoomed_circle, zoomed_line = drawn(0.1)

        self.assertAlmostEqual(number(zoomed_circle, "r"), number(circle, "r") / 2)
        self.assertAlmostEqual(
            number(zoomed_circle, "stroke-width"), number(circle, "stroke-width") / 2
        )
        self.assertAlmostEqual(
            number(zoomed_line, "stroke-width"), number(line, "stroke-width") / 2
        )
        # The network itself keeps its scene coordinates: only the frame,
        # which follows what is drawn, moves it as a whole.
        shift = number(zoomed_circle, "cx") - number(circle, "cx")
        self.assertAlmostEqual(number(zoomed_circle, "cy") - number(circle, "cy"), shift, places=3)
        for name in ("x1", "y1", "x2", "y2"):
            self.assertAlmostEqual(number(zoomed_line, name) - number(line, name), shift, places=3)

    def test_a_node_is_drawn_as_wide_next_to_its_neighbour_as_on_screen(self):
        # Nodes 0 and 1 are 10 scene units apart: 50 pixels on screen at 0.2 per pixel.
        circles = elements(export_svg(self.network()), "circle")
        centres = [number(circle, "cx") for circle in circles[:2]]

        svg_ratio = 2 * number(circles[0], "r") / abs(centres[1] - centres[0])

        self.assertAlmostEqual(svg_ratio, 12 / 50)

    def test_the_view_scale_comes_from_the_camera_real_rect(self):
        # The camera widens the rect it was asked for to the view's aspect
        # ratio; the widened one is what lies across the view's pixels.
        view = view_at(self.UNITS_PER_PIXEL)
        view.camera.rect = SimpleNamespace(width=100.0, height=100.0)

        circles = elements(export_svg(self.network(view=view)), "circle")

        self.assertAlmostEqual(number(circles[0], "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_a_camera_without_a_real_rect_uses_its_rect(self):
        view = view_at(self.UNITS_PER_PIXEL)
        del view.camera._real_rect

        circles = elements(export_svg(self.network(view=view)), "circle")

        self.assertAlmostEqual(number(circles[0], "r"), 12 / 2 * self.UNITS_PER_PIXEL)

    def test_a_high_dpi_display_draws_what_its_png_shows(self):
        # Sizes are logical pixels, which VisPy scales to the display itself,
        # and the view is measured in them, so node sizes stay. Outlines and
        # edges are drawn in device pixels: a 0.5-pixel outline is 1 device
        # pixel there, which looks 1.74 device pixels wide, and a 1.3-pixel
        # edge is 2.6, which GL draws 3 device pixels wide.
        def drawn(pixel_scale):
            viewer = self.network()
            viewer.canvas.pixel_scale = pixel_scale
            svg = export_svg(viewer, EDGE_WIDTH=1.3)
            return elements(svg, "circle")[0], elements(svg, "line")[0]

        circle, line = drawn(1.0)
        sharp_circle, sharp_line = drawn(2.0)

        self.assertEqual(number(sharp_circle, "r"), number(circle, "r"))
        self.assertAlmostEqual(number(circle, "stroke-width"), 0.9818 * 0.2, delta=0.001)
        self.assertAlmostEqual(number(sharp_circle, "stroke-width"), 1.7423 / 2 * 0.2, delta=0.001)
        self.assertAlmostEqual(number(line, "stroke-width"), 1.0 * 0.2)
        self.assertAlmostEqual(number(sharp_line, "stroke-width"), 1.5 * 0.2)

    def test_a_view_without_area_leaves_sizes_as_they_are(self):
        svg = export_svg(self.network(view=view_at(self.UNITS_PER_PIXEL, size=(0, 0))))

        circles = elements(svg, "circle")
        self.assertEqual(len(circles), 4)
        for circle in circles:
            self.assertAlmostEqual(number(circle, "r"), 6.0)

    def test_the_scale_matches_the_real_camera_mapping_in_any_view_shape(self):
        # What VisPy itself maps from the scene to the view's pixels, in a view
        # wider than the requested range, taller than it, and after a zoom.
        from vispy import scene

        view = scene.ViewBox()
        view.camera = scene.PanZoomCamera(aspect=1)
        view.camera.set_range(x=(0, 100), y=(0, 100))
        for size, zoom in (((800, 600), 1.0), ((400, 900), 1.0), ((800, 600), 0.5)):
            with self.subTest(size=size, zoom=zoom):
                view.size = size
                view.camera.zoom(zoom)
                transform = view.camera.transform
                pixels_per_unit_x, pixels_per_unit_y = abs(transform.scale[0]), abs(transform.scale[1])

                scale = print_command._scene_units_per_pixel(SimpleNamespace(view=view))

                self.assertAlmostEqual(scale * pixels_per_unit_x, 1.0, places=5)
                self.assertAlmostEqual(scale * pixels_per_unit_y, 1.0, places=5)


class PrintSvgWidthModelTests(unittest.TestCase):
    """Outline and edge widths as the screen draws them, in device pixels."""

    def test_an_outline_is_as_wide_as_the_colour_vispy_spreads_across_it(self):
        step = 0.0005
        distances = np.arange(-12.0, 12.0, step) + step / 2
        for antialias in (1.0, 0.5, 2.0):
            for width in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0):
                for filled in (True, False):
                    with self.subTest(antialias=antialias, width=width, filled=filled):
                        if filled:
                            spread = sum(vispy_outline_coverage(r, width, antialias) for r in distances) * step
                        else:
                            # cross_lines: a line with background on both sides.
                            spread = 2 * step * sum(
                                vispy_outline_coverage(r, width, antialias, filled=False)
                                for r in distances[distances > 0]
                            )
                        self.assertAlmostEqual(
                            print_command._outline_ink_px(width, antialias, open_lines=not filled),
                            spread, delta=2e-3,
                        )

    def test_a_thin_outline_looks_about_one_and_three_quarter_pixels_wide(self):
        self.assertAlmostEqual(print_command._outline_ink_px(0.5), 0.9818, places=4)
        self.assertAlmostEqual(print_command._outline_ink_px(1.0), 1.7424, places=4)
        self.assertAlmostEqual(print_command._outline_ink_px(5.0), 5.0 - 0.2317, places=4)

    def test_no_outline_is_drawn_without_width_and_a_hard_one_without_antialiasing(self):
        self.assertEqual(print_command._outline_ink_px(0.0), 0.0)
        self.assertEqual(print_command._outline_ink_px(float("nan")), 0.0)
        # Without the fading band, an outline under a pixel is that much fainter.
        self.assertEqual(print_command._outline_ink_px(0.5, antialias=0.0), 0.25)
        self.assertEqual(print_command._outline_ink_px(3.0, antialias=0.0), 3.0)

    def test_gl_draws_edges_a_whole_number_of_device_pixels_wide(self):
        for width, scale, drawn in (
            (0.5, 1.0, 1.0), (1.3, 1.0, 1.0), (1.5, 1.0, 2.0), (2.5, 1.0, 2.0),
            (0.25, 2.0, 1.0), (1.3, 2.0, 3.0), (1.0, 1.5, 2.0), (3.0, 1.5, 4.0),
            (0.0, 2.0, 0.0),
        ):
            with self.subTest(width=width, scale=scale):
                self.assertEqual(print_command._gl_line_px(width, scale), drawn)

    def test_the_display_scale_and_band_come_from_the_viewer(self):
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(pixel_scale=1.5), markers=SimpleNamespace(antialias=0.5),
        )
        self.assertEqual(print_command._display_scale(viewer), 1.5)
        self.assertEqual(print_command._marker_antialias(viewer), 0.5)
        # A viewer that cannot say draws as VisPy does by default.
        for unknown in (SimpleNamespace(), SimpleNamespace(canvas=SimpleNamespace(pixel_scale=0))):
            self.assertEqual(print_command._display_scale(unknown), 1.0)
            self.assertEqual(print_command._marker_antialias(unknown), 1.0)


class PrintSvgOutlineTests(unittest.TestCase):
    """Each node's outline is the one the screen draws it with."""

    @staticmethod
    def outlined_network(**overrides):
        colors = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
        colors[2] = (1.0, 0.0, 0.0, 1.0)
        widths = np.array([0.5, 0.5, 2.0, 0.5])
        return square_network(
            current_colors=numbered_colors(),
            node_outlines=lambda: (colors, widths),
            **overrides,
        )

    def test_a_node_drawn_with_its_own_outline_keeps_it(self):
        circles = elements(export_svg(self.outlined_network()), "circle")

        self.assertEqual(len(circles), 4)
        for index, circle in enumerate(circles):
            with self.subTest(node=index):
                if index == 2:
                    self.assertIn('stroke="rgb(255,0,0)"', circle)
                    self.assertAlmostEqual(number(circle, "stroke-width"), 1.768, delta=0.001)
                else:
                    self.assertIn('stroke="rgb(0,0,0)"', circle)
                    self.assertAlmostEqual(number(circle, "stroke-width"), 0.982, delta=0.001)

    def test_outlines_follow_the_drawing_order(self):
        viewer = self.outlined_network(
            visible_node_render_order=lambda: np.array([2, 0, 3, 1], dtype=np.int32),
        )
        svg = export_svg(viewer)

        self.assertEqual(node_fills(svg), [25, 5, 35, 15])
        self.assertIn('stroke="rgb(255,0,0)"', elements(svg, "circle")[0])

    def test_selected_nodes_and_their_neighbours_keep_their_borders(self):
        import EMAPSSN_Config
        from EMAPSSN_Viewer import MainViewer

        viewer = MainViewer.__new__(MainViewer)
        for name, value in vars(square_network(current_colors=numbered_colors())).items():
            setattr(viewer, name, value)
        viewer.n_nodes = 4
        viewer.node_render_order = np.arange(4, dtype=np.int32)
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        viewer.selected_indices = [0]
        with mock.patch.multiple(
            EMAPSSN_Config,
            CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black", HOVER_COLOR="blue",
        ):
            svg = export_svg(viewer)

        # Node 0 is selected and drawn last, after its neighbours 1 and 3.
        self.assertEqual(node_fills(svg), [25, 15, 35, 5])
        strokes = [re.search(r'stroke="(rgb\([\d,]+\))"', circle).group(1) for circle in elements(svg, "circle")]
        self.assertEqual(strokes, ["rgb(0,0,0)", "rgb(255,0,0)", "rgb(255,0,0)", "rgb(0,0,255)"])
        widths = [number(circle, "stroke-width") for circle in elements(svg, "circle")]
        self.assertEqual(widths, [0.982, 1.768, 1.768, 1.768])


# Draws scenes with the Viewer's own visuals on a real canvas, prints each as a
# PNG and as an SVG with print itself, draws the SVG with QtSvg at the PNG's
# pixels per scene unit, and reports how the two compare.
_PNG_SVG_SCRIPT = textwrap.dedent(
    r"""
    import importlib.util, json, os, shutil, sys, tempfile, types
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np
    SRC = sys.argv[1]
    sys.path.insert(0, SRC)
    from vispy import app, scene
    app.use_app("pyside6")
    import matplotlib.image as mpimg
    from PySide6 import QtCore, QtGui, QtSvg
    from desktop.Viewer_Edges import ViewerEdges
    import desktop.Viewer_Zoom, utilities.Output_Names, Viewer_Visual_State

    SAVE_DIR = tempfile.mkdtemp(prefix="print_png_svg_")
    engine = types.ModuleType("Command_Engine")
    for name in ("command_artifact", "command_succeeded", "command_failed", "show_status"):
        setattr(engine, name, mock.Mock())
    config = types.ModuleType("EMAPSSN_Config")
    config.SEQUENCE_SET = "test"
    config.resolve_directory_path = lambda value: SAVE_DIR
    spec = importlib.util.spec_from_file_location("print_under_test", os.path.join(SRC, "commands", "print.py"))
    print_command = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"Command_Engine": engine, "EMAPSSN_Config": config}):
        spec.loader.exec_module(print_command)

    canvas = scene.SceneCanvas(size=(400, 300), bgcolor="white", show=False)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, 200), y=(0, 150), margin=0)
    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_gl_state("translucent", depth_test=False)
    if (canvas.render()[..., 0] < 250).sum() != 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    DPR = float(canvas.pixel_scale)
    # Device pixels per scene unit, from the view rather than from print.
    PPU = canvas.physical_size[0] / view.camera.rect.width

    class Scheduler:
        def is_output_path_reserved(self, path):
            return False

        def enqueue(self, command_name, description, payload, worker, output_path, allow_overwrite=False):
            worker(payload)

    layer = None

    def show(pos, sizes, shapes, outline_colors, outline_widths, edges=(), edge_width=1.0,
             color=(1.0, 1.0, 1.0, 1.0), scores=None, auto=False):
        global layer
        pos = np.asarray(pos, dtype=np.float32)
        count = len(pos)
        colors = np.tile(np.asarray(color, dtype=np.float32), (count, 1))
        outline_colors = np.asarray(outline_colors, dtype=np.float32)
        outline_widths = np.asarray(outline_widths, dtype=np.float32)
        markers.set_data(pos=pos, face_color=colors, edge_color=outline_colors,
                         size=np.asarray(sizes, np.float32), edge_width=outline_widths, symbol=list(shapes))
        if layer is not None:
            layer.visual.parent = None
            layer = None
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        if len(edges):
            layer = ViewerEdges(view.scene, edges, scores, pos, (0.0, 0.0, 0.0, 1.0), edge_width)
            layer.set_opacity_by_score(auto)
            layer.visible = layer.show_threshold(pos, None, 0.0) > 0
            markers.order = 1
        config.EDGE_COLOR, config.EDGE_ALPHA, config.EDGE_WIDTH = "#000000", 1.0, edge_width
        config.UMAP_MODE = False
        config.AUTO_EDGE_ALPHA = auto
        return SimpleNamespace(
            canvas=canvas, view=view, markers=markers, n_nodes=count,
            visible_mask=np.ones(count, dtype=bool), pos=pos, current_colors=colors,
            current_sizes=np.asarray(sizes, dtype=np.float32), current_shapes=np.array(shapes, dtype=object),
            edges=edges, edge_scores=np.ones(len(edges)) if scores is None else np.asarray(scores),
            current_slider_threshold=0.0, selected_indices=[], edge_layer=layer,
            node_outlines=lambda: (outline_colors, outline_widths), background_job_scheduler=Scheduler(),
        )

    def raster(path):
        renderer = QtSvg.QSvgRenderer(path)
        box = renderer.viewBoxF()
        width, height = int(round(box.width() * PPU)), int(round(box.height() * PPU))
        image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QtCore.QRectF(0, 0, width, height))
        painter.end()
        pixels = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
        return pixels[:, :width, :3].astype(np.float64) / 255.0

    def prints(viewer, name):
        # The PNG and the SVG print makes, the SVG drawn at the PNG's scale: coverage 0..1.
        print_command.run(viewer, [name])
        print_command.run(viewer, [name + "_svg", "svg"])
        png = mpimg.imread(os.path.join(SAVE_DIR, name + ".png"))[..., :3].astype(np.float64)
        svg = raster(os.path.join(SAVE_DIR, name + "_svg.svg"))
        return 1.0 - png.mean(axis=2), 1.0 - svg.mean(axis=2)

    def margins(cover):
        mask = cover > print_command.PNG_BACKGROUND_TOLERANCE
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        return [int(cols[0]), int(rows[0]), cover.shape[1] - 1 - int(cols[-1]), cover.shape[0] - 1 - int(rows[-1])]

    def frame(png, svg):
        return {"png_shape": list(png.shape), "svg_shape": list(svg.shape),
                "png_margins": margins(png), "svg_margins": margins(svg)}

    results = {"dpr": DPR}

    # White discs with outlines of several widths, one a selection's (red,
    # 2 pixels): all the colour on the page is outline.
    widths = [0.5, 0.25, 1.0, 2.0, 0.5]
    outline_colors = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    outline_colors[3] = (1.0, 0.0, 0.0, 1.0)
    pos = [[20.0 + 40.0 * k, 75.0 + (8.0 if k % 2 else 0.0)] for k in range(5)]
    png, svg = prints(show(pos, [24.0] * 5, ["disc"] * 5, outline_colors, widths), "outlines")
    inks = []
    for cover in (png, svg):
        left, top = margins(cover)[:2]
        half, per_node = int(18.0 * DPR), []
        for x, y in pos:
            cx = left + (x - pos[0][0]) * PPU + 12.0 * DPR
            cy = top + (max(p[1] for p in pos) - y) * PPU + 12.0 * DPR
            window = cover[int(cy) - half:int(cy) + half + 1, int(cx) - half:int(cx) + half + 1]
            per_node.append(float(window.sum() / (2.0 * np.pi * 12.0 * DPR) / DPR))
        inks.append(per_node)
    results["outlines"] = dict(frame(png, svg), png_ink=inks[0], svg_ink=inks[1])

    # Long straight edges at odd places between nodes of no size.
    pos = [[10.0, 20.0], [190.0, 20.0], [10.3, 40.0], [190.3, 40.0],
           [30.0, 60.0], [30.0, 140.0], [60.2, 60.0], [60.2, 140.0]]
    png, svg = prints(show(pos, [0.0] * 8, ["disc"] * 8, np.tile([0, 0, 0, 1.0], (8, 1)), [0.5] * 8,
                           [[0, 1], [2, 3], [4, 5], [6, 7]], 1.3), "edges")
    length = (2 * 180.0 + 2 * 80.0) * PPU * DPR
    results["edges"] = dict(frame(png, svg), png_ink=float(png.sum() / length), svg_ink=float(svg.sum() / length))

    # Edges whose alpha follows their score (Auto): the PNG has them drawn by the shader, the SVG
    # writes them in opacity bands; each edge is as dark on both. Scores 0.1 to 1: 15 % to 85 %.
    pos = [[10.0, 20.0], [190.0, 20.0], [10.0, 50.0], [190.0, 50.0],
           [10.0, 80.0], [190.0, 80.0], [10.0, 110.0], [190.0, 110.0]]
    scores = [0.1, 0.4, 0.7, 1.0]
    png, svg = prints(show(pos, [0.0] * 8, ["disc"] * 8, np.tile([0, 0, 0, 1.0], (8, 1)), [0.5] * 8,
                           [[0, 1], [2, 3], [4, 5], [6, 7]], 1.3, scores=scores, auto=True), "auto_edges")

    def edge_inks(cover):
        # The ink of each edge: the coverage of the rows it lies in, one run of rows to an edge.
        rows = np.flatnonzero(cover.max(axis=1) > 0.02)
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        return [float(cover[run[0]:run[-1] + 1].sum()) for run in runs]

    results["auto_edges"] = dict(frame(png, svg), png_inks=edge_inks(png), svg_inks=edge_inks(svg))

    # One node of each marker, filled and outlined.
    shapes = {}
    for shape in ("disc", "square", "triangle_up", "triangle_down", "diamond", "star", "cross", "x",
                  "vbar", "hbar", "arrow", "tailed_arrow", "clobber", "cross_lines", "ring"):
        png, svg = prints(show([[100.0, 75.0]], [60.0], [shape], [[0.0, 0.0, 0.0, 1.0]], [1.0],
                               color=(0.2, 0.4, 0.9, 1.0)), "shape_" + shape)
        height, width = min(png.shape[0], svg.shape[0]), min(png.shape[1], svg.shape[1])
        a, b = png[:height, :width] > 0.25, svg[:height, :width] > 0.25
        # Overlap at the best shift of up to 3 pixels: the frames may differ by that.
        overlap = max(
            (a & np.roll(b, (dy, dx), axis=(0, 1))).sum() / max((a | np.roll(b, (dy, dx), axis=(0, 1))).sum(), 1)
            for dy in range(-3, 4) for dx in range(-3, 4)
        )
        shapes[shape] = dict(png_shape=list(png.shape), svg_shape=list(svg.shape),
                             overlap=float(overlap), ink_ratio=float(svg.sum() / max(png.sum(), 1e-9)))
    results["shapes"] = shapes
    shutil.rmtree(SAVE_DIR, ignore_errors=True)
    print("RESULTS " + json.dumps(results), flush=True)
    """
)


class PrintSvgMatchesPngTests(unittest.TestCase):
    """An SVG of the view looks like the PNG of it, on a 100 % and a 200 % screen.

    A real OpenGL canvas draws each scene as the Viewer does; print makes a
    PNG and an SVG of it; the SVG is drawn at the PNG's pixels per scene
    unit. Qt's high-DPI scaling is switched off and QT_SCALE_FACTOR sets the
    display scale, so the test runs at both on any screen.
    """

    @classmethod
    def setUpClass(cls):
        cls.results = {}

    def compare(self, scale):
        if scale in self.results:
            return self.results[scale]
        from tests.gl_window_environment import native_display_scale, run_gl_script

        output = run_gl_script(self, _PNG_SVG_SCRIPT, args=(SRC_DIR,), scale=scale, timeout=300)
        lines = [line for line in output.splitlines() if line.startswith("RESULTS ")]
        self.assertTrue(lines, output[-3000:])
        results = json.loads(lines[-1][len("RESULTS "):])
        expected_dpr = scale * native_display_scale()
        if results["dpr"] != expected_dpr:
            self.skipTest(f"Qt drew at a display scale of {results['dpr']}, not {expected_dpr}")
        self.results[scale] = results
        return results

    def assert_frames_match(self, scene):
        # The PNG's trim is whole pixels and its sharp points' faintest tips
        # may miss a pixel centre; anything else is a different frame.
        for png, svg in zip(scene["png_shape"], scene["svg_shape"]):
            self.assertLessEqual(abs(png - svg), 4, scene)

    def test_the_svg_is_framed_like_the_png(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                results = self.compare(scale)
                for name in ("outlines", "edges"):
                    scene = results[name]
                    self.assertEqual(scene["png_margins"], [20, 20, 20, 20], name)
                    self.assert_frames_match(scene)
                    for margin in scene["svg_margins"]:
                        self.assertLessEqual(abs(margin - 20), 2, (name, scene))
                for shape, scene in results["shapes"].items():
                    with self.subTest(shape=shape):
                        self.assert_frames_match(scene)

    def test_outlines_are_as_wide_as_on_the_png(self):
        # Outlines of 0.5, 0.25, 1 and 2 pixels; the 2-pixel one is a
        # selection's, which the SVG copies from the viewer.
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                scene = self.compare(scale)["outlines"]
                for node, (png, svg) in enumerate(zip(scene["png_ink"], scene["svg_ink"])):
                    self.assertAlmostEqual(svg / png, 1.0, delta=0.08, msg=(node, scene))

    def test_edges_are_as_wide_as_on_the_png(self):
        # 1.3-pixel edges: round(1.3 * dpr) device pixels (1 at 100 %, 3 at 200 %, 5 at 400 %).
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                results = self.compare(scale)
                dpr = results["dpr"]
                drawn = round(1.3 * dpr) / dpr
                scene = results["edges"]
                self.assertAlmostEqual(scene["png_ink"], drawn, delta=0.05, msg=scene)
                self.assertAlmostEqual(scene["svg_ink"] / scene["png_ink"], 1.0, delta=0.05, msg=scene)

    def test_edges_whose_alpha_follows_the_score_are_as_dark_as_on_the_png(self):
        # Four edges of scores 0.1, 0.4, 0.7 and 1 (listed from the top row down): opacities 85, 62, 38 and 15 %.
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                scene = self.compare(scale)["auto_edges"]
                png, svg = scene["png_inks"], scene["svg_inks"]
                self.assertEqual((len(png), len(svg)), (4, 4), scene)
                # The PNG really draws them by score, the top row (score 1) darkest: 85 % to 15 %.
                self.assertEqual(png, sorted(png, reverse=True), scene)
                self.assertAlmostEqual(png[0] / png[3], 0.85 / 0.15, delta=0.6, msg=scene)
                for edge, (on_png, on_svg) in enumerate(zip(png, svg)):
                    self.assertAlmostEqual(on_svg / on_png, 1.0, delta=0.06, msg=(edge, scene))

    def test_every_shape_is_drawn_as_on_the_png(self):
        for scale in (1, 2):
            for shape, scene in self.compare(scale)["shapes"].items():
                with self.subTest(display_scale=scale, shape=shape):
                    self.assertAlmostEqual(scene["ink_ratio"], 1.0, delta=0.1, msg=scene)
                    # cross_lines is a line a pixel or two wide, whose overlap
                    # says little; its colour above says how wide it is.
                    if shape != "cross_lines":
                        self.assertGreaterEqual(scene["overlap"], 0.9, scene)


if __name__ == "__main__":
    unittest.main()
