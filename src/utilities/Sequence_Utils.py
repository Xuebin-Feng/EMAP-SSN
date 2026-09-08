# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated sequence sanitization, FASTA cleaning, MSA validation, and position parsing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import os
import re
import sys
import tempfile

# =====================================================================
# 1. Lexical Rules for Alignment-Position Arguments
# =====================================================================

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


# =====================================================================
# 2. FASTA Sanitization, Deduplication & Normalization
# =====================================================================

VALID_RESIDUE_CODES = "ACDEFGHIKLMNPQRSTVWYBZJXUO"
RESIDUE_BOUNDARY_PATTERN = re.compile(
    rf"[{VALID_RESIDUE_CODES}].*[{VALID_RESIDUE_CODES}]|[{VALID_RESIDUE_CODES}]"
)
INVALID_RESIDUE_PATTERN = re.compile(rf"[^{VALID_RESIDUE_CODES}]")


def read_fasta(file_path):
    """Read a FASTA file into ordered header and sequence lists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    headers = []
    sequences = []
    current_header = None
    current_sequence = []

    with open(file_path, "r", encoding="utf-8-sig") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_sequence))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)

        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_sequence))

    return headers, sequences


def write_fasta_atomic(file_path, headers, sequences):
    """Atomically write canonical FASTA records as UTF-8 with LF newlines."""
    if len(headers) != len(sequences):
        raise ValueError("FASTA header and sequence counts do not match.")

    output_path = os.path.abspath(file_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            for header, sequence in zip(headers, sequences):
                temporary_file.write(f">{header}\n{sequence}\n")

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return output_path


def sanitize_header(header):
    """Apply the canonical SSN header character and whitespace rules."""
    safe_header = header.translate(str.maketrans("[]{}", "()()"))
    safe_header = re.sub(r'[?*"#%@$/\\]', "_", safe_header)
    safe_header = re.sub(r"\s+", " ", safe_header).strip()
    safe_header = safe_header.replace(" ", "_")
    safe_header = re.sub(r"_+", "_", safe_header)
    return safe_header, safe_header != header


def simplify_node_label(header):
    """Return a compact accession or first-token label for a FASTA header."""
    match = re.search(
        r"\b([A-Z]{2}_\d+(?:\.\d+)?|[A-Z]{3}\d{5,7}(?:\.\d+)?)\b",
        header,
    )
    if match:
        return match.group(1)

    marker = "|gb|"
    if marker in header:
        try:
            return header.split(marker)[1].split("|")[0]
        except IndexError:
            pass

    return header.split()[0] if header else ""


def sanitize_sequence(seq):
    """Uppercase a sequence, trim terminal artifacts, and mask internal ones."""
    upper_seq = seq.upper()
    match = RESIDUE_BOUNDARY_PATTERN.search(upper_seq)

    if not match:
        return "", upper_seq, []

    core_seq = match.group(0)
    stripped_chars = upper_seq[:match.start()] + upper_seq[match.end():]
    invalid_internal_chars = INVALID_RESIDUE_PATTERN.findall(core_seq)
    final_seq = INVALID_RESIDUE_PATTERN.sub("X", core_seq)
    return final_seq, stripped_chars, invalid_internal_chars


def select_preferred_header(current_headers):
    """Select the longest header and report true duplicate-header counts."""
    header_counts = Counter(current_headers)
    unique_headers = list(header_counts)
    duplicate_headers = {
        header for header, count in header_counts.items() if count > 1
    }
    duplicates_count = sum(count - 1 for count in header_counts.values())

    if len(unique_headers) > 1:
        best_header = sorted(
            unique_headers,
            key=lambda value: (-len(value), value),
        )[0]
        discarded_headers = sorted(
            (header for header in unique_headers if header != best_header),
            key=lambda value: (-len(value), value),
        )
    else:
        best_header = unique_headers[0]
        discarded_headers = []

    return best_header, discarded_headers, duplicate_headers, duplicates_count


def allocate_unique_headers(header_to_seqs):
    """Allocate globally unique output headers without replacing existing names."""
    reserved_headers = set(header_to_seqs)
    used_headers = set()
    assigned_by_header = {}

    for header, unique_seqs in header_to_seqs.items():
        if len(unique_seqs) == 1:
            assigned_headers = [header]
        else:
            assigned_headers = []
            suffix = 1
            separator = "" if header.endswith("_") else "_"
            for _ in unique_seqs:
                candidate = f"{header}{separator}{suffix}"
                while candidate in reserved_headers or candidate in used_headers:
                    suffix += 1
                    candidate = f"{header}{separator}{suffix}"
                assigned_headers.append(candidate)
                suffix += 1

        for assigned_header in assigned_headers:
            if assigned_header in used_headers:
                raise RuntimeError(
                    f"Unable to allocate a unique FASTA header for '{header}'."
                )
            used_headers.add(assigned_header)

        assigned_by_header[header] = assigned_headers

    return assigned_by_header


def sanitize_fasta_records(headers, sequences):
    """Sanitize and deduplicate FASTA records without header or length filtering."""
    if len(headers) != len(sequences):
        raise ValueError("FASTA header and sequence counts do not match.")

    stats = {
        "original_records": len(headers),
        "final_records": 0,
        "headers_modified": 0,
        "sequences_modified": 0,
        "empty_sequences_removed": 0,
        "exact_duplicates_removed": 0,
        "different_headers_merged": 0,
        "headers_renamed": 0,
        "changed": False,
    }

    seq_to_headers = {}
    for header, seq in zip(headers, sequences):
        safe_header, header_modified = sanitize_header(header)
        cleaned_seq, _, _ = sanitize_sequence(seq)

        if header_modified:
            stats["headers_modified"] += 1
        if cleaned_seq != seq:
            stats["sequences_modified"] += 1

        if cleaned_seq:
            seq_to_headers.setdefault(cleaned_seq, []).append(safe_header)
        else:
            stats["empty_sequences_removed"] += 1

    header_to_seqs = {}
    for seq, current_headers in seq_to_headers.items():
        (
            best_header,
            discarded_headers,
            _,
            duplicates_count,
        ) = select_preferred_header(current_headers)

        stats["exact_duplicates_removed"] += duplicates_count
        stats["different_headers_merged"] += len(discarded_headers)
        header_to_seqs.setdefault(best_header, []).append(seq)

    assigned_headers_by_base = allocate_unique_headers(header_to_seqs)
    clean_headers = []
    clean_sequences = []

    for header, unique_seqs in header_to_seqs.items():
        assigned_headers = assigned_headers_by_base[header]
        if len(unique_seqs) > 1:
            stats["headers_renamed"] += len(unique_seqs)

        clean_headers.extend(assigned_headers)
        clean_sequences.extend(unique_seqs)

    if len(clean_headers) != len(set(clean_headers)):
        raise RuntimeError("FASTA sanitization produced duplicate output headers.")

    stats["final_records"] = len(clean_headers)
    stats["changed"] = any(
        stats[key]
        for key in (
            "headers_modified",
            "sequences_modified",
            "empty_sequences_removed",
            "exact_duplicates_removed",
            "different_headers_merged",
            "headers_renamed",
        )
    ) or stats["final_records"] != stats["original_records"]

    return clean_headers, clean_sequences, stats


def print_sanitization_result(stats):
    """Print a compact result only when sanitization changed the FASTA records."""
    if not stats["changed"]:
        return False

    print("\nFASTA sanitization result:")
    print(f"  Original records:          {stats['original_records']}")
    print(f"  Final records:             {stats['final_records']}")

    labels = (
        ("headers_modified", "Headers modified"),
        ("sequences_modified", "Sequences modified"),
        ("empty_sequences_removed", "Empty sequences removed"),
        ("exact_duplicates_removed", "Exact duplicates removed"),
        ("different_headers_merged", "Different headers merged"),
        ("headers_renamed", "Headers renamed"),
    )
    for key, label in labels:
        if stats[key]:
            print(f"  {label + ':':<27} {stats[key]}")

    return True


def load_sanitized_fasta(file_path, *, report=True):
    """Read, sanitize, optionally report, and return one FASTA record set."""
    headers, sequences = read_fasta(file_path)
    clean_headers, clean_sequences, stats = sanitize_fasta_records(
        headers,
        sequences,
    )
    if report:
        print_sanitization_result(stats)
    return clean_headers, clean_sequences, stats


# =====================================================================
# 3. Position-Preserving Multiple Sequence Alignment (MSA) Sanitization
# =====================================================================

GAP_CODES = frozenset({"-", "."})
VALID_RESIDUES = frozenset(VALID_RESIDUE_CODES)

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
    import numpy as np

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


__all__ = [
    "POSITION_MAGNITUDE_PATTERN",
    "NONNEGATIVE_POSITION_PATTERN",
    "DISPLAYED_POSITION_ATOM_PATTERN",
    "reject_bare_negative_positions",
    "normalize_displayed_position_atom",
    "format_alignment_offset_display",
    "sort_alignment_labels",
    "VALID_RESIDUE_CODES",
    "RESIDUE_BOUNDARY_PATTERN",
    "INVALID_RESIDUE_PATTERN",
    "read_fasta",
    "write_fasta_atomic",
    "sanitize_header",
    "simplify_node_label",
    "sanitize_sequence",
    "select_preferred_header",
    "allocate_unique_headers",
    "sanitize_fasta_records",
    "print_sanitization_result",
    "load_sanitized_fasta",
    "GAP_CODES",
    "VALID_RESIDUES",
    "AA_TO_INT",
    "INT_TO_AA",
    "MSAValidationError",
    "MSASanitizationStats",
    "sanitize_msa_header",
    "sanitize_aligned_sequence",
    "sanitize_msa_headers",
    "load_sanitized_msa_fasta",
    "parse_int_to_aa_mapping",
    "canonicalize_sparse_values",
    "print_msa_sanitization_result",
]
