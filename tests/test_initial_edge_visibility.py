import ast
import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
VIEWER_PATH = PROJECT_ROOT / "src" / "emapssn_viewer.py"


def _main_viewer_init():
    tree = ast.parse(VIEWER_PATH.read_text(encoding="utf-8"))
    main_viewer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainViewer"
    )
    return next(
        node
        for node in main_viewer.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def _self_call_lines(function, method_name):
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
    )


class InitialEdgeVisibilityTests(unittest.TestCase):
    def test_cached_visibility_is_applied_before_the_window_is_shown(self):
        initializer = _main_viewer_init()
        draw_line = _self_call_lines(initializer, "draw_network")[0]
        update_lines = _self_call_lines(initializer, "update_edges")
        show_line = next(
            node.lineno
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "show_window_in_front"
        )
        threshold_line = next(
            node.lineno
            for node in ast.walk(initializer)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "current_slider_threshold"
                for target in node.targets
            )
        )

        startup_updates = [
            line
            for line in update_lines
            if threshold_line < line < show_line
        ]
        self.assertTrue(startup_updates)
        self.assertLess(draw_line, threshold_line)


if __name__ == "__main__":
    unittest.main()
