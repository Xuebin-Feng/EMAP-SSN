# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration profile values (EMAPSSN_Config ConfigGUI): how
_normalize_profile_data coerces and validates a saved or custom profile, and
how _custom_profile_data falls back to the defaults. Both read only their
arguments and the window's _custom_settings, so the tests call them on the
loaded class with a stand-in window instead of building one."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from tests.config_gui_loader import load_config_namespace


class ProfileNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            namespace = load_config_namespace()
        cls.gui = namespace["ConfigGUI"]
        cls.defaults = {
            tab_id: spec["defaults"]
            for tab_id, spec in namespace["TAB_PROFILE_SPECS"].items()
        }

    def normalize(self, tab_id, raw_data, **options):
        return self.gui._normalize_profile_data(None, tab_id, raw_data, **options)

    def assert_rejected(self, cases):
        for tab_id, raw_data, message in cases:
            with self.subTest(tab_id=tab_id, raw_data=raw_data):
                with self.assertRaises(ValueError) as raised:
                    self.normalize(tab_id, raw_data)
                self.assertEqual(str(raised.exception), message)

    def test_text_values_are_coerced_to_each_settings_type(self):
        visual = self.normalize(
            "visual_effects",
            {"NODE_SIZE": "12.0", "LOW_RESOURCE_MODE": "yes", "HOVER_COLOR": "#00ff00"},
        )
        self.assertEqual(visual["NODE_SIZE"], 12)
        self.assertIs(type(visual["NODE_SIZE"]), int)
        self.assertIs(visual["LOW_RESOURCE_MODE"], True)
        self.assertEqual(visual["HOVER_COLOR"], "#00ff00")
        # Keys the profile leaves out keep their defaults.
        self.assertEqual(visual["EDGE_WIDTH"], self.defaults["visual_effects"]["EDGE_WIDTH"])

        for text, expected in (("0", False), (" F ", False), ("no", False), ("t", True)):
            with self.subTest(LOW_RESOURCE_MODE=text):
                self.assertIs(
                    self.normalize("visual_effects", {"LOW_RESOURCE_MODE": text})[
                        "LOW_RESOURCE_MODE"
                    ],
                    expected,
                )

        physics = self.normalize(
            "simulation_physics", {"DT": "0.01", "PACKING_GEOMETRY": "Circle"}
        )
        self.assertEqual(physics["DT"], 0.01)
        self.assertEqual(physics["PACKING_GEOMETRY"], "Circle")

    def test_optional_numbers_accept_none_in_any_spelling(self):
        for value in (" none ", "None", "", None):
            with self.subTest(TOP_EDGE_PERCENT=value):
                self.assertIsNone(
                    self.normalize("inputs_outputs", {"TOP_EDGE_PERCENT": value})[
                        "TOP_EDGE_PERCENT"
                    ]
                )
        self.assertEqual(
            self.normalize("inputs_outputs", {"TOP_EDGE_PERCENT": "2.5"})["TOP_EDGE_PERCENT"],
            2.5,
        )

    def test_wrongly_typed_values_are_rejected_with_the_key(self):
        self.assert_rejected([
            ("visual_effects", {"NODE_SIZE": 10.5}, "invalid value for NODE_SIZE: expected an integer"),
            ("visual_effects", {"NODE_SIZE": True}, "invalid value for NODE_SIZE: expected an integer"),
            ("visual_effects", {"LOW_RESOURCE_MODE": "maybe"},
             "invalid value for LOW_RESOURCE_MODE: expected true or false"),
            ("visual_effects", {"HOVER_COLOR": 5}, "invalid value for HOVER_COLOR: expected text"),
            ("visual_effects", {"HOVER_COLOR": "notacolor"},
             "invalid color value for HOVER_COLOR: notacolor"),
            ("simulation_physics", {"DT": True}, "invalid value for DT: expected a number"),
            ("simulation_physics", {"DT": "fast"},
             "invalid value for DT: could not convert string to float: 'fast'"),
        ])

    def test_values_outside_their_domain_are_rejected(self):
        self.assert_rejected([
            ("simulation_physics", {"DT": "nan"}, "invalid value for DT: expected a finite number"),
            ("simulation_physics", {"DT": "inf"}, "invalid value for DT: expected a finite number"),
            ("simulation_physics", {"PACKING_GEOMETRY": "Hexagon"},
             "invalid value for PACKING_GEOMETRY; expected one of: Circle, Square"),
            ("visual_effects", {"NODE_SIZE": 25}, "invalid value for NODE_SIZE; expected 1 to 20"),
            ("inputs_outputs", {"TOP_EDGE_PERCENT": 150},
             "invalid value for TOP_EDGE_PERCENT; expected 0.0 to 100.0"),
            ("inputs_outputs", {"ALIGNMENT_SCORE": ""},
             "invalid value for ALIGNMENT_SCORE; expected one of: global, local"),
            # The default NORM_MODE is alignment_length.
            ("inputs_outputs", {"ALIGNMENT_SCORE": "local"},
             "NORM_MODE alignment_length is unavailable for local alignment scores"),
        ])

    def test_unknown_keys_are_rejected_unless_retired_or_allowed(self):
        self.assert_rejected([
            ("visual_effects", {"BOGUS": 1, "ALSO": 2}, "unknown setting key(s): ALSO, BOGUS"),
            ("visual_effects", [], "the JSON root must be an object"),
        ])
        self.assertEqual(
            self.normalize("visual_effects", {"BOGUS": 1}, allow_extra=True),
            self.defaults["visual_effects"],
        )
        # Settings retired from these tabs are dropped quietly.
        physics = self.normalize("simulation_physics", {"PHYSICS_ENGINE": "legacy"})
        self.assertNotIn("PHYSICS_ENGINE", physics)
        directories = self.normalize("directories", {"LOGO_DIR": "logos"})
        self.assertNotIn("LOGO_DIR", directories)

    def test_custom_settings_fall_back_to_defaults_when_invalid(self):
        window = SimpleNamespace(
            _custom_settings={"NODE_SIZE": "12", "LOW_RESOURCE_MODE": "yes", "DT": "x"}
        )
        window._normalize_profile_data = (
            lambda tab_id, raw_data: self.gui._normalize_profile_data(window, tab_id, raw_data)
        )
        # Settings of other tabs (DT) are not part of this tab's profile.
        visual = self.gui._custom_profile_data(window, "visual_effects")
        self.assertEqual(visual["NODE_SIZE"], 12)
        self.assertIs(visual["LOW_RESOURCE_MODE"], True)

        window._custom_settings["NODE_SIZE"] = 10.5
        with redirect_stdout(io.StringIO()) as output:
            visual = self.gui._custom_profile_data(window, "visual_effects")
        self.assertEqual(visual, self.defaults["visual_effects"])
        self.assertIsNot(visual, self.defaults["visual_effects"])
        self.assertEqual(
            output.getvalue(),
            "Invalid custom settings for visual_effects; using defaults: "
            "invalid value for NODE_SIZE: expected an integer\n",
        )


if __name__ == "__main__":
    unittest.main()
