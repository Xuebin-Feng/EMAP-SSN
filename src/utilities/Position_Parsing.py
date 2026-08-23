# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared lexical rules for displayed alignment-position arguments."""

import re


POSITION_MAGNITUDE_PATTERN = r"\d+(?:\.\d+)?"
NONNEGATIVE_POSITION_PATTERN = rf"\+?{POSITION_MAGNITUDE_PATTERN}"
DISPLAYED_POSITION_ATOM_PATTERN = (
    rf"(?:{NONNEGATIVE_POSITION_PATTERN}|\(-{POSITION_MAGNITUDE_PATTERN}\))"
)

_NONNEGATIVE_POSITION_RE = re.compile(rf"^{NONNEGATIVE_POSITION_PATTERN}$")
_PARENTHESIZED_NEGATIVE_POSITION_RE = re.compile(
    rf"^\(-({POSITION_MAGNITUDE_PATTERN})\)$"
)
_BARE_NEGATIVE_POSITION_RE = re.compile(
    rf"(?<![\w.()])(-{POSITION_MAGNITUDE_PATTERN})(?![\d.])"
)


def reject_bare_negative_positions(value):
    """Reject the first bare negative position found in a position argument."""
    text = str(value).strip()
    match = _BARE_NEGATIVE_POSITION_RE.search(text)
    if match:
        position = match.group(1)
        raise ValueError(
            f"Negative position '{position}' must be written as '({position})'. "
            "Parentheses are required around negative positions."
        )


def normalize_displayed_position_atom(value, *, allow_end=False):
    """Return the alignment-label form of one user-facing position atom."""
    text = str(value).strip()
    upper_text = text.upper()
    if allow_end and upper_text in {"E", "END"}:
        return upper_text

    reject_bare_negative_positions(text)

    negative_match = _PARENTHESIZED_NEGATIVE_POSITION_RE.fullmatch(text)
    if negative_match:
        return f"-{negative_match.group(1)}"
    if _NONNEGATIVE_POSITION_RE.fullmatch(text):
        return text.removeprefix('+')

    expected = "a non-negative integer or insertion label"
    if allow_end:
        expected += ", E, or END"
    raise ValueError(
        f"Invalid position label '{value}'; expected {expected}, or a negative "
        "position enclosed in parentheses."
    )


def format_alignment_offset_display(alignment, configured_offset):
    """Format the active offset or mark the configured offset as inactive."""
    if alignment is not None and getattr(alignment, "has_reference", False):
        return str(getattr(alignment, "offset", 0))
    return f"{configured_offset} (inactive)"


def sort_alignment_labels(labels):
    """Sort integer and insertion-style alignment labels numerically."""

    def label_key(label):
        try:
            parts = str(label).split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return major, minor
        except (TypeError, ValueError):
            return 0, 0

    return sorted(labels, key=label_key)
