"""The Viewer canvas's DPI (desktop/Viewer_Dpi.py): VisPy's number when usable, else 96."""
import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_Dpi import FALLBACK_DPI, canvas_dpi  # noqa: E402


def _raising(exception):
    def get_dpi(raise_error=True):
        raise exception
    return get_dpi


class CanvasDpiTests(unittest.TestCase):
    def call(self, get_dpi, configured=None):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            return canvas_dpi(get_dpi, configured), out.getvalue()

    def test_usable_value_passes_through_silently(self):
        value, text = self.call(lambda raise_error=True: 109.25)
        self.assertEqual(value, 109.25)
        self.assertEqual(text, "")

    def test_failures_and_unusable_values_fall_back(self):
        for get_dpi in (_raising(ZeroDivisionError()), _raising(OSError()),
                        lambda raise_error=True: 0, lambda raise_error=True: -3.0,
                        lambda raise_error=True: math.inf, lambda raise_error=True: math.nan,
                        lambda raise_error=True: None, lambda raise_error=True: "x"):
            value, text = self.call(get_dpi)
            self.assertEqual(value, FALLBACK_DPI)
            self.assertEqual(text.count("\n"), 1)
            self.assertIn("no usable size", text)

    def test_configured_dpi_is_left_to_vispy(self):
        value, text = self.call(_raising(ZeroDivisionError()), configured=120)
        self.assertIsNone(value)
        self.assertEqual(text, "")

    def test_real_get_dpi_gives_a_number(self):
        value, _ = self.call(None)
        self.assertTrue(value is None or (math.isfinite(float(value)) and value > 0))


if __name__ == "__main__":
    unittest.main()
