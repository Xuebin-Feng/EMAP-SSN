# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared hardware-detection report fixture for the GPU and installer tests.

Not collected by unittest: it defines no TestCase.
"""


def detected_report(**overrides):
    """Return a minimal Detect_GPU-style report for a CPU-only Windows host."""
    report = {
        "platform": "windows",
        "os_version": "10.0.26100",
        "windows_build": 26100,
        "controllers": [],
        "processors": [],
        "nvidia_devices": [],
        "vendor": "CPU",
        "backend": "cpu",
        "gfx_target": None,
        "reason": "test",
    }
    report.update(overrides)
    return report
