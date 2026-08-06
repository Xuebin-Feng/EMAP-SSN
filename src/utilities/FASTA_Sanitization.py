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

"""Shared FASTA sanitization helpers for sequence-processing utilities."""

from collections import Counter
import os
import re


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


def sanitize_header(header):
    """Apply the canonical SSN header character and whitespace rules."""
    safe_header = header.translate(str.maketrans("[]{}", "()()"))
    safe_header = re.sub(r'[?*"#%@$/\\]', "_", safe_header)
    safe_header = re.sub(r"_+", "_", safe_header)
    safe_header = re.sub(r"\s+", " ", safe_header).strip()
    return safe_header, safe_header != header


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
    """
    Sanitize and deduplicate FASTA records without header or length filtering.

    Returns sanitized headers, sanitized sequences, and a compact statistics mapping.
    """
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


def load_sanitized_fasta(file_path):
    """Read, sanitize, optionally report, and return one FASTA record set."""
    headers, sequences = read_fasta(file_path)
    clean_headers, clean_sequences, stats = sanitize_fasta_records(
        headers,
        sequences,
    )
    print_sanitization_result(stats)
    return clean_headers, clean_sequences, stats

