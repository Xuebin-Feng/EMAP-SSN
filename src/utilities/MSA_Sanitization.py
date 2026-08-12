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

"""Position-preserving sanitization helpers for multiple-sequence alignments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
import sys

import numpy as np

from utilities.FASTA_Sanitization import VALID_RESIDUE_CODES, sanitize_header


GAP_CODES = frozenset({"-", "."})
VALID_RESIDUES = frozenset(VALID_RESIDUE_CODES)

# Codes 1-21 intentionally retain the legacy sparse-MSA representation.
# Previously aliased residues now receive distinct codes in newly encoded data.
AA_TO_INT = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "B": 22,
    "Z": 23,
    "J": 24,
    "U": 25,
    "O": 26,
}
INT_TO_AA = {code: residue for residue, code in AA_TO_INT.items()}


class MSAValidationError(ValueError):
    """Raised when an alignment is structurally unsafe or ambiguous to load."""


@dataclass
class MSASanitizationStats:
    """Compact accounting for safe, position-preserving MSA repairs."""

    headers_modified: int = 0
    residues_uppercased: int = 0
    formatting_whitespace_removed: int = 0
    gap_symbols_normalized: int = 0
    illegal_residues_replaced: int = 0
    sparse_codes_canonicalized: int = 0
    sparse_entries_removed: int = 0
    invalid_symbols: Counter = field(default_factory=Counter)

    @property
    def changed(self):
        return any(
            (
                self.headers_modified,
                self.residues_uppercased,
                self.formatting_whitespace_removed,
                self.gap_symbols_normalized,
                self.illegal_residues_replaced,
                self.sparse_codes_canonicalized,
                self.sparse_entries_removed,
            )
        )


def sanitize_msa_header(header, stats=None):
    """Apply the shared header policy without renaming or merging MSA rows."""
    text = str(header)
    safe_header, modified = sanitize_header(text)
    if stats is not None and modified:
        stats.headers_modified += 1
    if not safe_header:
        raise MSAValidationError("MSA headers must not be empty after sanitization.")
    return safe_header


def sanitize_aligned_sequence(sequence, stats=None):
    """Sanitize one aligned sequence while retaining every biological column."""
    stats = stats if stats is not None else MSASanitizationStats()
    output = []
    for char in str(sequence):
        if char.isspace():
            stats.formatting_whitespace_removed += 1
            continue

        upper = char.upper()
        if len(upper) == 1 and upper in VALID_RESIDUES:
            if char != upper:
                stats.residues_uppercased += 1
            output.append(upper)
        elif char == "-":
            output.append("-")
        elif char == ".":
            stats.gap_symbols_normalized += 1
            output.append("-")
        else:
            stats.illegal_residues_replaced += 1
            stats.invalid_symbols[char] += 1
            output.append("X")
    return "".join(output)


def sanitize_msa_headers(headers, stats=None):
    """Sanitize headers and reject duplicates or cleanup-induced collisions."""
    stats = stats if stats is not None else MSASanitizationStats()
    raw_seen = set()
    clean_seen = set()
    clean_headers = []

    for row_idx, header in enumerate(headers, start=1):
        raw_header = str(header)
        if not raw_header.strip():
            raise MSAValidationError(f"MSA row {row_idx} has an empty header.")
        if raw_header in raw_seen:
            raise MSAValidationError(f"Duplicate MSA header: '{raw_header}'.")
        raw_seen.add(raw_header)

        clean_header = sanitize_msa_header(raw_header, stats)
        if clean_header in clean_seen:
            raise MSAValidationError(
                f"MSA header sanitization creates a duplicate header: "
                f"'{clean_header}'."
            )
        clean_seen.add(clean_header)
        clean_headers.append(clean_header)

    return clean_headers


def load_sanitized_msa_fasta(file_path):
    """Strictly parse and sanitize an aligned FASTA without modifying the file."""
    source_path = os.fspath(file_path)
    if not os.path.isfile(source_path):
        raise MSAValidationError(f"MSA FASTA file not found: {source_path}")

    raw_headers = []
    raw_sequences = []
    current_header = None
    current_sequence = []

    try:
        with open(source_path, "r", encoding="utf-8-sig", errors="strict") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                if line.startswith(">"):
                    if current_header is not None:
                        if not current_sequence:
                            raise MSAValidationError(
                                f"MSA record '{current_header}' has no sequence data."
                            )
                        raw_headers.append(current_header)
                        raw_sequences.append("".join(current_sequence))
                    current_header = line[1:]
                    current_sequence = []
                    if not current_header.strip():
                        raise MSAValidationError(
                            f"MSA FASTA line {line_number} has an empty header."
                        )
                else:
                    if current_header is None:
                        raise MSAValidationError(
                            f"MSA FASTA line {line_number} contains sequence data "
                            "before the first header."
                        )
                    current_sequence.append(line)
    except UnicodeDecodeError as exc:
        raise MSAValidationError(f"MSA FASTA is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise MSAValidationError(f"Unable to read MSA FASTA: {exc}") from exc

    if current_header is not None:
        if not current_sequence:
            raise MSAValidationError(
                f"MSA record '{current_header}' has no sequence data."
            )
        raw_headers.append(current_header)
        raw_sequences.append("".join(current_sequence))

    if not raw_headers:
        raise MSAValidationError("MSA FASTA contains no records.")

    stats = MSASanitizationStats()
    headers = sanitize_msa_headers(raw_headers, stats)
    sequences = [sanitize_aligned_sequence(sequence, stats) for sequence in raw_sequences]
    alignment_length = len(sequences[0])
    if alignment_length == 0:
        raise MSAValidationError("MSA sequences must not be empty after sanitization.")

    unequal = [
        (headers[idx], len(sequence))
        for idx, sequence in enumerate(sequences)
        if len(sequence) != alignment_length
    ]
    if unequal:
        examples = ", ".join(
            f"'{header}' ({length})" for header, length in unequal[:5]
        )
        raise MSAValidationError(
            f"MSA sequences must have equal aligned lengths; expected "
            f"{alignment_length}, found {examples}."
        )
    return headers, sequences, stats


def parse_int_to_aa_mapping(raw_mapping):
    """Validate and normalize a decoded sparse-HDF5 residue mapping."""
    if not isinstance(raw_mapping, dict):
        raise MSAValidationError("HDF5 int_to_aa must decode to a JSON object.")

    mapping = {}
    for raw_code, symbol in raw_mapping.items():
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise MSAValidationError(
                f"HDF5 int_to_aa contains a non-integer code: {raw_code!r}."
            ) from exc
        if str(code) != str(raw_code).strip():
            raise MSAValidationError(
                f"HDF5 int_to_aa contains a non-canonical code: {raw_code!r}."
            )
        if code in mapping:
            raise MSAValidationError(f"Duplicate HDF5 residue code: {code}.")
        if not isinstance(symbol, str):
            raise MSAValidationError(
                f"HDF5 residue code {code} must map to a string."
            )
        mapping[code] = symbol
    return mapping


def canonicalize_sparse_values(values, int_to_aa, stats=None):
    """Map arbitrary sparse residue codes onto the canonical MSA code set."""
    stats = stats if stats is not None else MSASanitizationStats()
    source = np.asarray(values)
    canonical = np.empty(source.shape, dtype=np.uint8)
    keep = np.ones(source.shape, dtype=bool)

    for raw_code in np.unique(source):
        code = int(raw_code)
        mask = source == raw_code
        count = int(np.count_nonzero(mask))
        symbol = int_to_aa.get(code)

        if code == 0 or symbol in GAP_CODES:
            keep[mask] = False
            canonical[mask] = 0
            stats.sparse_entries_removed += count
            if code != 0 or symbol == ".":
                stats.gap_symbols_normalized += count
            continue

        upper = symbol.upper() if isinstance(symbol, str) else ""
        if len(upper) == 1 and upper in VALID_RESIDUES:
            target_code = AA_TO_INT[upper]
            if symbol != upper:
                stats.residues_uppercased += count
        else:
            target_code = AA_TO_INT["X"]
            stats.illegal_residues_replaced += count
            stats.invalid_symbols[
                symbol if isinstance(symbol, str) and symbol else f"code:{code}"
            ] += count

        canonical[mask] = target_code
        if code != target_code:
            stats.sparse_codes_canonicalized += count

    return canonical, keep


def print_msa_sanitization_result(
    stats,
    source_path,
    *,
    output_path=None,
    source_modified=False,
):
    """Print one compact warning when MSA sanitization changed loaded content."""
    if stats is None or not stats.changed:
        return False

    lines = [
        "WARNING: MSA sanitization was applied",
        f"  Source: {os.path.normpath(os.fspath(source_path))}",
    ]
    fields = (
        ("headers_modified", "Headers modified"),
        ("residues_uppercased", "Residues uppercased"),
        ("formatting_whitespace_removed", "Formatting whitespace removed"),
        ("gap_symbols_normalized", "Gap symbols normalized"),
        ("illegal_residues_replaced", "Illegal residues replaced with X"),
        ("sparse_codes_canonicalized", "Sparse codes canonicalized"),
        ("sparse_entries_removed", "Sparse gap/zero entries removed"),
    )
    for key, label in fields:
        value = getattr(stats, key)
        if value:
            lines.append(f"  {label}: {value}")
    if stats.invalid_symbols:
        examples = ", ".join(
            f"{symbol!r} ({count})"
            for symbol, count in stats.invalid_symbols.most_common(10)
        )
        lines.append(f"  Invalid symbol examples: {examples}")
    if output_path is not None:
        lines.append(
            f"  Sanitized sparse alignment written to: "
            f"{os.path.normpath(os.fspath(output_path))}"
        )
    elif not source_modified:
        lines.append("  Source file was not modified; sanitization is in memory only.")

    message = "\n".join(lines)
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        print(f"\033[93m{message}\033[0m")
    else:
        print(message)
    return True
