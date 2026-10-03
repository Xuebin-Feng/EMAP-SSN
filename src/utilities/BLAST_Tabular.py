# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Strict external BLAST tabular parsing and HDF5 network publication."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
import os
import re
import sys

import h5py
import numpy as np
from tqdm import tqdm

from utilities.Sequence_Utils import sanitize_header


SUPPORTED_LAYOUTS = {
    "standard_outfmt6",
    "outfmt7_fields",
    "custom_columns",
}
SUBJECT_FIELD_NAMES = (
    "subject title",
    "subject id",
    "subject acc.ver",
    "subject acc.",
    "subject accession.version",
    "subject accession",
)
EVALUE_FIELD_NAMES = ("evalue", "expect value")
SCORE_TRANSFORM = "-log10(E + 1e-300)"

DIAMOND_PROGRAM = "DIAMOND"
DIAMOND_DEFAULT_MAX_TARGET_SEQS = 25
BLAST_DEFAULT_MAX_TARGET_SEQS = 500
SEARCH_HEADER_SCAN_LINES = 1000
# A per-query target limit is reported only when at least this many queries
# sit at the largest target count while more queries than that report them,
# and they make up at least this fraction of the queries at that count.
CAP_EVIDENCE_MIN_QUERIES = 3
CAP_EVIDENCE_MIN_FRACTION = 0.1
_BLAST_PROGRAM_LINE = re.compile(r"^#\s*((?:T|PSI|RPS|DELTA)?BLAST[A-Z]*)\s+(\d\S*)")
_DIAMOND_PROGRAM_LINE = re.compile(
    r"^#\s*DIAMOND\s+v?(\d[0-9A-Za-z.+_-]*?)\.?(?:\s|$)"
)
_DIAMOND_COMMANDS = {
    "makedb", "blastp", "blastx", "cluster", "linclust", "realign", "recluster",
    "reassign", "view", "merge-daa", "help", "version", "getseq", "dbinfo",
    "test", "makeidx", "greedy-vertex-cover",
}
FIRST_WORD_ADVICE = (
    "BLAST+ qseqid/sseqid and DIAMOND report only the first word of each FASTA "
    "header, while the manifest uses complete headers. Remove descriptions from "
    "the FASTA headers before searching, or report complete titles: BLAST+ "
    '-outfmt "7 qseqid stitle evalue" with BLAST_LAYOUT outfmt7_fields, or '
    "DIAMOND --outfmt 6 qtitle stitle evalue with BLAST_LAYOUT custom_columns "
    "(columns 1, 2 and 3)."
)


class BlastParseError(ValueError):
    """Raised when external BLAST or FASTA input violates the import contract."""


@dataclass(frozen=True)
class HeaderManifest:
    headers: tuple[str, ...]
    sequences: tuple[str, ...]
    index_by_header: dict[str, int]
    modifications: tuple[tuple[str, str], ...]
    # Sanitized first word -> complete headers, for headers that contain spaces.
    first_word_headers: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHeader:
    """Search provenance declared by a tabular file's leading comment lines."""

    program: str = "Unknown"
    version: str = "Unknown"
    invocation: str = "Unknown"

    @property
    def network_tag(self):
        """Program tag used in the imported network's filename."""
        return DIAMOND_PROGRAM if self.program == DIAMOND_PROGRAM else "BLAST"


@dataclass(frozen=True)
class SearchCompleteness:
    queries_observed: int
    max_targets_per_query: int
    queries_at_max_targets: int
    query_rows_grouped: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ParseSummary:
    output_path: str
    fasta_header_count: int
    fasta_headers_sanitized: int
    blast_header_count: int
    blast_headers_sanitized: int
    data_rows: int
    self_rows: int
    unique_edges: int
    search: SearchHeader = SearchHeader()
    queries_observed: int = 0
    max_targets_per_query: int = 0
    queries_at_max_targets: int = 0
    query_rows_grouped: bool = True
    warnings: tuple[str, ...] = ()


class HeaderSanitizationTracker:
    """Track distinct BLAST headers and reject sanitization collisions."""

    def __init__(self, manifest: HeaderManifest):
        self._manifest = manifest
        self.raw_to_sanitized: dict[str, str] = {}
        self.sanitized_to_raw: dict[str, str] = {}
        self.modifications: list[tuple[str, str]] = []

    def observe(self, raw_header: str, line_number: int) -> str:
        if raw_header in self.raw_to_sanitized:
            return self.raw_to_sanitized[raw_header]

        clean_header, modified = sanitize_header(raw_header)
        if not clean_header:
            raise BlastParseError(
                f"BLAST line {line_number}: header {raw_header!r} is empty after "
                "sanitization."
            )

        previous_raw = self.sanitized_to_raw.get(clean_header)
        if previous_raw is not None and previous_raw != raw_header:
            raise BlastParseError(
                f"BLAST line {line_number}: distinct headers {previous_raw!r} and "
                f"{raw_header!r} both sanitize to {clean_header!r}."
            )
        if clean_header not in self._manifest.index_by_header:
            raise BlastParseError(
                f"BLAST line {line_number}: sanitized header {clean_header!r} "
                f"(from {raw_header!r}) is not present in the FASTA manifest."
                + _first_word_hint(self._manifest, clean_header)
            )

        self.raw_to_sanitized[raw_header] = clean_header
        self.sanitized_to_raw[clean_header] = raw_header
        if modified:
            self.modifications.append((raw_header, clean_header))
        return clean_header

    @property
    def distinct_count(self) -> int:
        return len(self.raw_to_sanitized)


def _first_word_hint(manifest, clean_header):
    """Explain an unmatched header that looks like a truncated FASTA header."""
    matches = manifest.first_word_headers.get(clean_header)
    if matches:
        others = f" and {len(matches) - 1} other(s)" if len(matches) > 1 else ""
        return (
            f" It matches the first word of the FASTA header {matches[0]!r}"
            f"{others}. {FIRST_WORD_ADVICE}"
        )
    if manifest.first_word_headers:
        count = sum(len(headers) for headers in manifest.first_word_headers.values())
        return f" {count} FASTA header(s) contain spaces. {FIRST_WORD_ADVICE}"
    return ""


class QueryHitCounter:
    """Count distinct targets per query and the queries reporting each target.

    BLAST+ and DIAMOND write each query's rows contiguously, so one set of the
    current query's targets is enough. A query that reappears after another
    query marks the rows as ungrouped, and target-limit checks are skipped.
    The parse loop calls ``start_query`` whenever the row's query differs from
    ``query`` and adds each target to ``targets``, keeping per-row work to a
    set insertion.
    """

    def __init__(self, record_count):
        self.targets_per_query = [0] * record_count
        self.queries_per_target = [0] * record_count
        self.observed = bytearray(record_count)
        self.grouped = True
        self.query = None
        self.targets = set()

    def start_query(self, query):
        if query == self.query:
            return
        self.finish()
        if self.observed[query]:
            self.grouped = False
        self.observed[query] = 1
        self.query = query

    def finish(self):
        if self.query is None:
            return
        self.targets_per_query[self.query] += len(self.targets)
        for target in self.targets:
            self.queries_per_target[target] += 1
        self.targets.clear()
        self.query = None


def parse_search_comment(line):
    """Return the search provenance one comment line declares, if any.

    Recognizes BLAST+ program lines (``# BLASTP 2.17.0+``) and the lines that
    DIAMOND ``--header verbose`` writes (``# DIAMOND v2.1.23.`` and
    ``# Invocation: ...``).
    """
    match = _DIAMOND_PROGRAM_LINE.match(line)
    if match:
        return {"program": DIAMOND_PROGRAM, "version": match.group(1)}
    match = _BLAST_PROGRAM_LINE.match(line)
    if match:
        return {"program": match.group(1), "version": match.group(2)}
    if line.startswith("# Invocation:"):
        invocation = line.split(":", 1)[1].strip()
        return {"invocation": invocation} if invocation else {}
    return {}


def read_search_header(blast_path, max_lines=SEARCH_HEADER_SCAN_LINES):
    """Read the search program, version and command from leading comments.

    Only comments before the first data row are examined: DIAMOND writes its
    ``--header verbose`` lines there and BLAST+ outfmt 7 starts with its
    program line. The first declaration of each value wins. A missing or
    undecodable file yields an unknown header; the parser reports those
    problems itself.
    """
    declared = {}
    try:
        with open(blast_path, "rb") as blast_file:
            for line_number, raw_line in enumerate(blast_file, 1):
                if line_number > max_lines:
                    break
                line = raw_line.decode(
                    "utf-8-sig" if line_number == 1 else "utf-8"
                ).rstrip("\r\n")
                if not line:
                    continue
                if not line.startswith("#"):
                    break
                for key, value in parse_search_comment(line).items():
                    declared.setdefault(key, value)
    except (OSError, UnicodeError):
        pass
    return SearchHeader(**declared)


def diamond_target_limit(invocation):
    """Return ``(max_target_seqs, explicit, top)`` from a DIAMOND command line.

    ``max_target_seqs`` is None unless the command is ``blastp`` or ``blastx``
    (``view`` re-reads an archive whose limits were set earlier); 0 means
    unlimited. ``top`` is the ``--top`` percentage, which overrides the limit.
    DIAMOND takes ``-k N``, ``-kN`` and ``-k=N``, and the last value wins.
    """
    tokens = str(invocation).split()
    command = next((token for token in tokens if token in _DIAMOND_COMMANDS), None)
    if command not in {"blastp", "blastx"}:
        return None, False, None
    limit, explicit, top = DIAMOND_DEFAULT_MAX_TARGET_SEQS, False, None
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token == "--top":
            top = following or "?"
            continue
        if token in {"-k", "--max-target-seqs"}:
            value = following
        elif token.startswith("-k") and not token.startswith("--"):
            value = token[2:].lstrip("=")
        else:
            continue
        try:
            limit, explicit = int(value), True
        except ValueError:
            continue
    return limit, explicit, top


def _cap_hint(limit):
    if limit == DIAMOND_DEFAULT_MAX_TARGET_SEQS:
        return (
            " DIAMOND reports at most 25 targets per query unless it is run "
            "with -k 0."
        )
    if limit == BLAST_DEFAULT_MAX_TARGET_SEQS:
        return (
            " BLAST+ reports at most 500 targets per query unless "
            "-max_target_seqs is raised to at least the number of sequences."
        )
    return (
        " Rerun without a per-query target limit (DIAMOND -k 0, or BLAST+ "
        "-max_target_seqs of at least the number of sequences)."
    )


def assess_search_completeness(counter, headers, self_rows, search):
    """Summarize query coverage and warn about truncated all-vs-all searches."""
    record_count = len(headers)
    observed = [index for index in range(record_count) if counter.observed[index]]
    counts = [counter.targets_per_query[index] for index in observed]
    max_targets = max(counts, default=0)
    at_max = [
        index for index in observed
        if max_targets and counter.targets_per_query[index] == max_targets
    ]
    warnings = []

    limit, explicit, top = (None, False, None)
    if search.program == DIAMOND_PROGRAM:
        limit, explicit, top = diamond_target_limit(search.invocation)
    if top is not None:
        warnings.append(
            f"DIAMOND was run with --top {top}, which keeps only hits within "
            f"{top}% of each query's best score, so weaker homologs are missing "
            "from this network. For an all-vs-all network, rerun DIAMOND with "
            "-k 0 and without --top."
        )
    elif limit is not None:
        reached = sum(1 for count in counts if count == limit) if limit else 0
        if 0 < limit < record_count and reached:
            default = "" if explicit else " (its default)"
            warnings.append(
                f"DIAMOND was run with --max-target-seqs {limit}{default} on "
                f"{record_count} sequences, and {reached} queries reached that "
                f"limit, so hits beyond each of their top {limit} targets are "
                "missing. Rerun DIAMOND with -k 0 for an all-vs-all network."
            )
    elif counter.grouped and at_max:
        over_reported = sum(
            1 for index in at_max
            if counter.queries_per_target[index] > max_targets
        )
        if (
            over_reported >= CAP_EVIDENCE_MIN_QUERIES
            and over_reported >= CAP_EVIDENCE_MIN_FRACTION * len(at_max)
        ):
            warnings.append(
                f"The search appears to be limited to {max_targets} target "
                f"sequences per query: {len(at_max)} queries report exactly "
                f"{max_targets} targets, the most of any query, and "
                f"{over_reported} of them are themselves reported as a hit by "
                f"more than {max_targets} queries. Hits beyond each query's top "
                f"{max_targets} targets are probably missing."
                + _cap_hint(max_targets)
            )

    missing = [index for index in range(record_count) if not counter.observed[index]]
    if self_rows and missing:
        examples = ", ".join(repr(headers[index]) for index in missing[:3])
        warnings.append(
            f"{len(missing)} of {record_count} FASTA records never appear as a "
            f"query (for example {examples}). Other queries report self hits, so "
            "the search may not have covered the whole FASTA or may have stopped "
            "early; very short or low-complexity sequences can also lack a self "
            "hit."
        )
    return SearchCompleteness(
        queries_observed=len(observed),
        max_targets_per_query=max_targets,
        queries_at_max_targets=len(at_max),
        query_rows_grouped=counter.grouped,
        warnings=tuple(warnings),
    )


def _console_safe(value) -> str:
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, "backslashreplace").decode(encoding, "replace")
    except LookupError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def print_header_sanitization_summary(source, total, modifications, max_examples=20):
    """Always report whether FASTA or BLAST headers were changed."""
    modifications = list(modifications)
    suffix = " distinct headers" if source == "BLAST" else ""
    print(
        f"{source} headers sanitized: {len(modifications)} of {total}{suffix}"
    )
    if not modifications:
        print(f"No {source} headers required sanitization.")
        return

    for original, sanitized in modifications[:max_examples]:
        print(
            "  "
            + _console_safe(repr(original))
            + " -> "
            + _console_safe(repr(sanitized))
        )
    omitted = len(modifications) - max_examples
    if omitted > 0:
        print(f"  ... {omitted} additional sanitized header(s) omitted.")


def load_header_manifest(fasta_path) -> HeaderManifest:
    """Read FASTA records, sanitizing only full headers and preserving order."""
    headers: list[str] = []
    sequences: list[str] = []
    current_header = None
    current_sequence: list[str] = []

    try:
        fasta_file = open(fasta_path, "r", encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise BlastParseError(f"Unable to read FASTA file: {error}") from error

    try:
        for line_number, line in enumerate(fasta_file, 1):
            line = line.rstrip("\r\n")
            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_sequence))
                current_header = line[1:]
                current_sequence = []
            elif line.strip():
                if current_header is None:
                    raise BlastParseError(
                        f"FASTA line {line_number}: sequence data appears before "
                        "the first header."
                    )
                current_sequence.append(line.strip())
        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_sequence))
    except UnicodeError as error:
        raise BlastParseError(f"FASTA input is not valid UTF-8: {error}") from error
    finally:
        fasta_file.close()

    if not headers:
        raise BlastParseError("The FASTA manifest contains no records.")

    raw_seen: dict[str, int] = {}
    sanitized_seen: dict[str, str] = {}
    sanitized_headers: list[str] = []
    modifications: list[tuple[str, str]] = []
    first_word_headers: dict[str, list[str]] = {}
    for record_number, (raw_header, sequence) in enumerate(
        zip(headers, sequences), 1
    ):
        if raw_header in raw_seen:
            raise BlastParseError(
                f"Duplicate raw FASTA header {raw_header!r} at records "
                f"{raw_seen[raw_header]} and {record_number}."
            )
        raw_seen[raw_header] = record_number
        if not sequence:
            raise BlastParseError(
                f"FASTA record {record_number} ({raw_header!r}) has no sequence."
            )

        clean_header, modified = sanitize_header(raw_header)
        if not clean_header:
            raise BlastParseError(
                f"FASTA record {record_number} header {raw_header!r} is empty "
                "after sanitization."
            )
        previous_raw = sanitized_seen.get(clean_header)
        if previous_raw is not None and previous_raw != raw_header:
            raise BlastParseError(
                f"Distinct FASTA headers {previous_raw!r} and {raw_header!r} both "
                f"sanitize to {clean_header!r}."
            )
        sanitized_seen[clean_header] = raw_header
        sanitized_headers.append(clean_header)
        if modified:
            modifications.append((raw_header, clean_header))
        words = raw_header.split()
        if len(words) > 1:
            first_word, _ = sanitize_header(words[0])
            first_word_headers.setdefault(first_word, []).append(clean_header)

    return HeaderManifest(
        headers=tuple(sanitized_headers),
        sequences=tuple(sequences),
        index_by_header={
            header: index for index, header in enumerate(sanitized_headers)
        },
        modifications=tuple(modifications),
        first_word_headers={
            word: tuple(full_headers)
            for word, full_headers in first_word_headers.items()
        },
    )


def calculate_file_sha256(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(file_path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_manifest_sha256(headers, sequences):
    digest = hashlib.sha256()
    for header, sequence in zip(headers, sequences):
        for value in (header, sequence):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _bounded_slices(length, max_items):
    max_items = int(max_items)
    if max_items <= 0:
        raise BlastParseError("BATCH_SIZE must be a positive integer.")
    for start in range(0, int(length), max_items):
        yield start, min(start + max_items, int(length))


def _sort_and_reduce_edge_buffer(sources, targets, scores):
    arr_i = np.asarray(sources, dtype=np.uint32)
    arr_j = np.asarray(targets, dtype=np.uint32)
    arr_score = np.asarray(scores, dtype=np.float32)
    if not len(arr_i):
        return arr_i, arr_j, arr_score

    order = np.lexsort((arr_j, arr_i))
    arr_i = arr_i[order]
    arr_j = arr_j[order]
    arr_score = arr_score[order]
    pair_starts = np.empty(len(arr_i), dtype=bool)
    pair_starts[0] = True
    pair_starts[1:] = (arr_i[1:] != arr_i[:-1]) | (arr_j[1:] != arr_j[:-1])
    starts = np.flatnonzero(pair_starts)
    return (
        arr_i[starts],
        arr_j[starts],
        np.maximum.reduceat(arr_score, starts),
    )


def _write_sorted_edge_run(runs_group, run_index, sources, targets, scores):
    arr_i, arr_j, arr_score = _sort_and_reduce_edge_buffer(
        sources, targets, scores
    )
    run = runs_group.create_group(f"run_{run_index:08d}")
    run.create_dataset("i", data=arr_i, dtype=np.uint32)
    run.create_dataset("j", data=arr_j, dtype=np.uint32)
    run.create_dataset("score", data=arr_score, dtype=np.float32)


def _iter_sorted_edges(container, read_size):
    for start, end in _bounded_slices(len(container["i"]), read_size):
        for source, target, score in zip(
            container["i"][start:end],
            container["j"][start:end],
            container["score"][start:end],
        ):
            yield int(source), int(target), float(score)


def _append_edge_buffer(output, sources, targets, scores):
    if not sources:
        return
    old_size = len(output["i"])
    new_size = old_size + len(sources)
    for name in ("i", "j", "score"):
        output[name].resize((new_size,))
    output["i"][old_size:new_size] = sources
    output["j"][old_size:new_size] = targets
    output["score"][old_size:new_size] = scores


def _merge_sorted_runs(runs_group, output, batch_size):
    iterators = [
        _iter_sorted_edges(runs_group[name], batch_size)
        for name in sorted(runs_group)
    ]
    merged = heapq.merge(*iterators, key=lambda edge: (edge[0], edge[1]))
    sources: list[int] = []
    targets: list[int] = []
    scores: list[float] = []
    current_pair = None
    best_score = None
    edge_count = 0

    for source, target, score in merged:
        pair = (source, target)
        if pair == current_pair:
            best_score = max(best_score, score)
            continue
        if current_pair is not None:
            sources.append(current_pair[0])
            targets.append(current_pair[1])
            scores.append(best_score)
            edge_count += 1
            if len(sources) >= batch_size:
                _append_edge_buffer(output, sources, targets, scores)
                sources.clear()
                targets.clear()
                scores.clear()
        current_pair = pair
        best_score = score

    if current_pair is not None:
        sources.append(current_pair[0])
        targets.append(current_pair[1])
        scores.append(best_score)
        edge_count += 1
    _append_edge_buffer(output, sources, targets, scores)
    return edge_count


def _field_index(fields, supported_names, label, line_number):
    for supported_name in supported_names:
        matches = [
            index for index, field in enumerate(fields) if field == supported_name
        ]
        if len(matches) > 1:
            raise BlastParseError(
                f"BLAST line {line_number}: outfmt-7 fields repeat the supported "
                f"{label} field {supported_name!r}."
            )
        if matches:
            return matches[0]
    raise BlastParseError(
        f"BLAST line {line_number}: outfmt-7 fields do not contain a supported "
        f"{label} field."
    )


def _decode_blast_line(raw_line, line_number):
    encoding = "utf-8-sig" if line_number == 1 else "utf-8"
    try:
        return raw_line.decode(encoding).rstrip("\r\n")
    except UnicodeDecodeError as error:
        raise BlastParseError(
            f"BLAST line {line_number}: input is not valid UTF-8 ({error})."
        ) from error


def _parse_numeric_evalue(raw_value, line_number):
    try:
        value = float(raw_value)
    except ValueError as error:
        raise BlastParseError(
            f"BLAST line {line_number}: invalid E-value {raw_value!r}."
        ) from error
    if value < 0 or not math.isfinite(value):
        raise BlastParseError(
            f"BLAST line {line_number}: E-value must be finite and non-negative; "
            f"received {raw_value!r}."
        )
    return value


def _check_declared_evalue_column(line, evalue_index, line_number):
    """Reject an E-value column that a # Fields line declares as another field."""
    fields = [name.strip() for name in line.split(":", 1)[1].split(",")]
    if evalue_index >= len(fields):
        problem = (
            f"declares only {len(fields)} columns, so E-value column "
            f"{evalue_index + 1} does not exist"
        )
    elif fields[evalue_index].casefold() not in EVALUE_FIELD_NAMES:
        problem = (
            f"declares column {evalue_index + 1} as {fields[evalue_index]!r}, "
            "not an E-value"
        )
    else:
        return
    raise BlastParseError(
        f"BLAST line {line_number}: # Fields {problem}. Check BLAST_LAYOUT and "
        "EVALUE_COLUMN."
    )


def _reject_column_name_row(layout, columns, evalue_index, line_number):
    """Explain a row whose E-value column holds a column name, not a value."""
    name = columns[evalue_index].strip()
    if layout != "outfmt7_fields" and name.casefold() in EVALUE_FIELD_NAMES:
        raise BlastParseError(
            f"BLAST line {line_number}: this row names its columns ({name!r} in "
            "the E-value column) instead of holding data, as DIAMOND "
            "'--header simple' writes. Rerun DIAMOND with '--header verbose', "
            "whose comment lines are recorded as provenance, or without --header."
        ) from None


def _parse_blast_to_runs(
    blast_path,
    layout,
    custom_columns,
    manifest,
    tracker,
    counter,
    runs_group,
    batch_size,
    show_progress,
):
    sources: list[int] = []
    targets: list[int] = []
    scores: list[float] = []
    data_rows = 0
    self_rows = 0
    run_index = 0
    expected_custom_width = None
    outfmt7_fields = None
    outfmt7_subject_index = None
    outfmt7_evalue_index = None
    current_query = None
    current_block_has_fields = False
    blast_program = "Unknown"
    blast_version = "Unknown"
    blast_database = "Unknown"

    query_index, subject_index, evalue_index = custom_columns
    if layout == "standard_outfmt6":
        query_index, subject_index, evalue_index = 0, 1, 10
    add_target = counter.targets.add

    file_size = os.path.getsize(blast_path)
    progress = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Processing",
        disable=not show_progress,
    )
    try:
        with open(blast_path, "rb") as blast_file:
            for line_number, raw_line in enumerate(blast_file, 1):
                progress.update(len(raw_line))
                line = _decode_blast_line(raw_line, line_number)
                if not line:
                    continue

                if line.startswith("#"):
                    if layout != "outfmt7_fields":
                        if line.startswith("# Fields:"):
                            _check_declared_evalue_column(
                                line, evalue_index, line_number
                            )
                        continue
                    if line.startswith("# Query:"):
                        raw_query = line.split(":", 1)[1].strip()
                        current_query = tracker.observe(raw_query, line_number)
                        counter.start_query(manifest.index_by_header[current_query])
                        current_block_has_fields = False
                    elif line.startswith("# Fields:"):
                        fields = tuple(
                            field.strip().casefold()
                            for field in line.split(":", 1)[1].split(",")
                        )
                        if outfmt7_fields is not None and fields != outfmt7_fields:
                            raise BlastParseError(
                                f"BLAST line {line_number}: outfmt-7 field schema "
                                "differs from an earlier query block."
                            )
                        outfmt7_fields = fields
                        outfmt7_subject_index = _field_index(
                            fields, SUBJECT_FIELD_NAMES, "subject", line_number
                        )
                        outfmt7_evalue_index = _field_index(
                            fields, EVALUE_FIELD_NAMES, "E-value", line_number
                        )
                        current_block_has_fields = True
                    elif line.startswith("# BLAST") and blast_program == "Unknown":
                        parts = line[2:].strip().split(maxsplit=1)
                        blast_program = parts[0] if parts else "Unknown"
                        blast_version = parts[1] if len(parts) > 1 else "Unknown"
                    elif line.startswith("# Database:") and blast_database == "Unknown":
                        blast_database = line.split(":", 1)[1].strip() or "Unknown"
                    continue

                columns = line.split("\t")
                data_rows += 1
                if layout == "standard_outfmt6":
                    if len(columns) != 12:
                        raise BlastParseError(
                            f"BLAST line {line_number}: standard outfmt 6 requires "
                            f"exactly 12 tab-delimited fields; found {len(columns)}."
                        )
                    raw_query = columns[query_index]
                    raw_subject = columns[subject_index]
                elif layout == "custom_columns":
                    if expected_custom_width is None:
                        expected_custom_width = len(columns)
                    elif len(columns) != expected_custom_width:
                        raise BlastParseError(
                            f"BLAST line {line_number}: custom row has {len(columns)} "
                            f"fields; expected {expected_custom_width}."
                        )
                    if len(columns) <= max(query_index, subject_index, evalue_index):
                        raise BlastParseError(
                            f"BLAST line {line_number}: custom column mapping exceeds "
                            f"the {len(columns)} available fields."
                        )
                    raw_query = columns[query_index]
                    raw_subject = columns[subject_index]
                else:
                    if current_query is None:
                        raise BlastParseError(
                            f"BLAST line {line_number}: outfmt-7 data appears before "
                            "a # Query: declaration."
                        )
                    if not current_block_has_fields:
                        raise BlastParseError(
                            f"BLAST line {line_number}: outfmt-7 data appears before "
                            "a # Fields: declaration."
                        )
                    if len(columns) != len(outfmt7_fields):
                        raise BlastParseError(
                            f"BLAST line {line_number}: row has {len(columns)} fields "
                            f"but # Fields declares {len(outfmt7_fields)}."
                        )
                    raw_query = None
                    raw_subject = columns[outfmt7_subject_index]
                    evalue_index = outfmt7_evalue_index

                try:
                    raw_evalue = _parse_numeric_evalue(
                        columns[evalue_index], line_number
                    )
                except BlastParseError:
                    _reject_column_name_row(
                        layout, columns, evalue_index, line_number
                    )
                    raise
                query_header = (
                    current_query
                    if layout == "outfmt7_fields"
                    else tracker.observe(raw_query, line_number)
                )
                subject_header = tracker.observe(raw_subject, line_number)
                source = manifest.index_by_header[query_header]
                target = manifest.index_by_header[subject_header]
                if source != counter.query:
                    counter.start_query(source)
                add_target(target)
                if source == target:
                    self_rows += 1
                    continue

                sources.append(min(source, target))
                targets.append(max(source, target))
                scores.append(-math.log10(raw_evalue + 1e-300))
                if len(sources) >= batch_size:
                    _write_sorted_edge_run(
                        runs_group, run_index, sources, targets, scores
                    )
                    run_index += 1
                    sources.clear()
                    targets.clear()
                    scores.clear()
    finally:
        progress.close()

    counter.finish()
    if sources:
        _write_sorted_edge_run(runs_group, run_index, sources, targets, scores)

    resolved_columns = {
        "query_column_1based": 0 if layout == "outfmt7_fields" else query_index + 1,
        "subject_column_1based": (
            (0 if outfmt7_subject_index is None else outfmt7_subject_index + 1)
            if layout == "outfmt7_fields"
            else subject_index + 1
        ),
        "evalue_column_1based": (
            (0 if outfmt7_evalue_index is None else outfmt7_evalue_index + 1)
            if layout == "outfmt7_fields"
            else evalue_index + 1
        ),
    }
    provenance = {
        "blast_program": blast_program,
        "blast_version": blast_version,
        "blast_database": blast_database,
        "blast_fields": (
            json.dumps(outfmt7_fields)
            if outfmt7_fields is not None
            else "Unknown"
        ),
    }
    return data_rows, self_rows, resolved_columns, provenance


def validate_final_output(
    output_path,
    expected_headers,
    expected_edges,
    read_size,
    expected_attributes=None,
):
    """Validate the completed HDF5 network before atomic publication."""
    try:
        with h5py.File(output_path, "r") as network:
            required = ("headers", "i", "j", "score")
            missing = [name for name in required if name not in network]
            if missing:
                return False, "missing dataset(s): " + ", ".join(missing)
            if any(network[name].ndim != 1 for name in required):
                return False, "network datasets must be one-dimensional"
            headers = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in network["headers"][:]
            ]
            if headers != list(expected_headers):
                return False, "headers dataset does not match the FASTA manifest"
            edge_count = len(network["i"])
            if len(network["j"]) != edge_count or len(network["score"]) != edge_count:
                return False, "edge dataset lengths do not match"
            if edge_count != expected_edges:
                return False, "edge count does not match the merged edge total"
            if np.dtype(network["i"].dtype) not in {
                np.dtype(np.uint16),
                np.dtype(np.uint32),
            }:
                return False, f"invalid i dtype: {network['i'].dtype}"
            if network["j"].dtype != network["i"].dtype:
                return False, "i and j dtypes do not match"
            if np.dtype(network["score"].dtype) != np.dtype(np.float32):
                return False, f"invalid score dtype: {network['score'].dtype}"

            required_attributes = (
                "model_name",
                "matrix",
                "score_transform",
                "source_blast_filename",
                "source_blast_sha256",
                "source_fasta_filename",
                "source_fasta_sha256",
                "manifest_sha256",
                "blast_layout",
                "query_column_1based",
                "subject_column_1based",
                "evalue_column_1based",
                "fasta_header_count",
                "fasta_headers_sanitized",
                "blast_header_count",
                "blast_headers_sanitized",
                "data_rows",
                "self_rows",
                "unique_edges",
                "blast_program",
                "blast_version",
                "blast_database",
                "blast_fields",
                "search_program",
                "search_version",
                "search_invocation",
                "queries_observed",
                "max_targets_per_query",
                "queries_at_max_targets",
                "import_warnings",
            )
            missing_attributes = [
                name for name in required_attributes if name not in network.attrs
            ]
            if missing_attributes:
                return False, "missing provenance attribute(s): " + ", ".join(
                    missing_attributes
                )
            if network.attrs["model_name"] != "BLAST":
                return False, "model_name must be 'BLAST'"
            if network.attrs["score_transform"] != SCORE_TRANSFORM:
                return False, "score_transform does not match the parser transform"
            if network.attrs["blast_layout"] not in SUPPORTED_LAYOUTS:
                return False, "blast_layout is not supported"
            if int(network.attrs["fasta_header_count"]) != len(expected_headers):
                return False, "fasta_header_count does not match /headers"
            if int(network.attrs["unique_edges"]) != expected_edges:
                return False, "unique_edges does not match the edge datasets"
            for count_name, total_name in (
                ("fasta_headers_sanitized", "fasta_header_count"),
                ("blast_headers_sanitized", "blast_header_count"),
                ("queries_observed", "fasta_header_count"),
            ):
                count = int(network.attrs[count_name])
                total = int(network.attrs[total_name])
                if count < 0 or count > total:
                    return False, f"{count_name} is outside its valid range"
            for hash_name in (
                "source_blast_sha256",
                "source_fasta_sha256",
                "manifest_sha256",
            ):
                hash_value = str(network.attrs[hash_name])
                if len(hash_value) != 64 or any(
                    character not in "0123456789abcdef" for character in hash_value
                ):
                    return False, f"{hash_name} is not a lowercase SHA-256 digest"
            for column_name in (
                "query_column_1based",
                "subject_column_1based",
                "evalue_column_1based",
            ):
                if int(network.attrs[column_name]) < 0:
                    return False, f"{column_name} cannot be negative"
            if expected_attributes is not None:
                for name, expected_value in expected_attributes.items():
                    if network.attrs[name] != expected_value:
                        return False, f"provenance attribute {name!r} changed"

            previous_pair = None
            for start, end in _bounded_slices(edge_count, read_size):
                arr_i = network["i"][start:end]
                arr_j = network["j"][start:end]
                arr_score = network["score"][start:end]
                if np.any(arr_i >= len(expected_headers)) or np.any(
                    arr_j >= len(expected_headers)
                ):
                    return False, "edge index is outside the header array"
                if np.any(arr_i >= arr_j):
                    return False, "edge pair does not satisfy i < j"
                if not np.isfinite(arr_score).all():
                    return False, "score dataset contains a non-finite value"
                for source, target in zip(arr_i, arr_j):
                    pair = (int(source), int(target))
                    if previous_pair is not None and pair <= previous_pair:
                        return False, "edge pairs are not strictly sorted and unique"
                    previous_pair = pair
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        return False, f"unable to read final output: {error}"
    return True, ""


def build_blast_network(
    blast_path,
    fasta_path,
    output_path,
    *,
    layout="standard_outfmt6",
    query_column=1,
    subject_column=2,
    evalue_column=11,
    matrix="Imported",
    batch_size=1000000,
    show_progress=True,
):
    """Build, validate, and atomically publish one external BLAST network.

    BLAST+ and DIAMOND output are both accepted. Program, version and command
    are read from the file's leading comment lines when present, and the
    per-query hit counts are checked for a truncated all-vs-all search.
    """
    blast_path = os.path.abspath(os.path.normpath(os.fspath(blast_path)))
    fasta_path = os.path.abspath(os.path.normpath(os.fspath(fasta_path)))
    output_path = os.path.abspath(os.path.normpath(os.fspath(output_path)))
    if layout not in SUPPORTED_LAYOUTS:
        raise BlastParseError(
            f"BLAST_LAYOUT must be one of {sorted(SUPPORTED_LAYOUTS)}; "
            f"received {layout!r}."
        )
    if not os.path.isfile(blast_path):
        raise BlastParseError(f"BLAST tabular file was not found: {blast_path}")
    if not os.path.isfile(fasta_path):
        raise BlastParseError(f"FASTA manifest was not found: {fasta_path}")
    try:
        batch_size = int(batch_size)
    except (TypeError, ValueError) as error:
        raise BlastParseError("BATCH_SIZE must be a positive integer.") from error
    if batch_size <= 0:
        raise BlastParseError("BATCH_SIZE must be a positive integer.")

    try:
        custom_columns = tuple(
            int(value) - 1 for value in (query_column, subject_column, evalue_column)
        )
    except (TypeError, ValueError) as error:
        raise BlastParseError(
            "Custom BLAST columns must be positive integers."
        ) from error
    if layout == "custom_columns":
        if any(index < 0 for index in custom_columns):
            raise BlastParseError("Custom BLAST columns must be positive integers.")
        if len(set(custom_columns)) != 3:
            raise BlastParseError("Custom BLAST columns must be distinct.")

    search = read_search_header(blast_path)
    manifest = load_header_manifest(fasta_path)
    if len(manifest.headers) > np.iinfo(np.uint32).max:
        raise BlastParseError("The FASTA manifest exceeds uint32 index capacity.")
    print_header_sanitization_summary(
        "FASTA", len(manifest.headers), manifest.modifications
    )
    tracker = HeaderSanitizationTracker(manifest)
    counter = QueryHitCounter(len(manifest.headers))

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    partial_path = output_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    data_rows = 0
    self_rows = 0
    edge_count = 0
    blast_summary_printed = False
    try:
        index_dtype = np.uint16 if len(manifest.headers) <= 65535 else np.uint32
        hdf5_chunk = max(1, min(batch_size, 65536))
        with h5py.File(partial_path, "w") as output:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            output.create_dataset(
                "headers",
                data=np.asarray(manifest.headers, dtype=object),
                dtype=string_dtype,
            )
            output.create_dataset(
                "i",
                shape=(0,),
                maxshape=(None,),
                chunks=(hdf5_chunk,),
                dtype=index_dtype,
            )
            output.create_dataset(
                "j",
                shape=(0,),
                maxshape=(None,),
                chunks=(hdf5_chunk,),
                dtype=index_dtype,
            )
            output.create_dataset(
                "score",
                shape=(0,),
                maxshape=(None,),
                chunks=(hdf5_chunk,),
                dtype=np.float32,
            )
            runs_group = output.create_group("_sorted_runs")
            try:
                (
                    data_rows,
                    self_rows,
                    resolved_columns,
                    provenance,
                ) = _parse_blast_to_runs(
                    blast_path,
                    layout,
                    custom_columns,
                    manifest,
                    tracker,
                    counter,
                    runs_group,
                    batch_size,
                    show_progress,
                )
            finally:
                print_header_sanitization_summary(
                    "BLAST", tracker.distinct_count, tracker.modifications
                )
                blast_summary_printed = True

            completeness = assess_search_completeness(
                counter, manifest.headers, self_rows, search
            )
            edge_count = _merge_sorted_runs(runs_group, output, batch_size)
            del output["_sorted_runs"]
            attrs = {
                "model_name": "BLAST",
                "matrix": str(matrix).strip() or "Unknown",
                "score_transform": SCORE_TRANSFORM,
                "source_blast_filename": os.path.basename(blast_path),
                "source_blast_sha256": calculate_file_sha256(blast_path),
                "source_fasta_filename": os.path.basename(fasta_path),
                "source_fasta_sha256": calculate_file_sha256(fasta_path),
                "manifest_sha256": calculate_manifest_sha256(
                    manifest.headers, manifest.sequences
                ),
                "blast_layout": layout,
                "fasta_header_count": len(manifest.headers),
                "fasta_headers_sanitized": len(manifest.modifications),
                "blast_header_count": tracker.distinct_count,
                "blast_headers_sanitized": len(tracker.modifications),
                "data_rows": data_rows,
                "self_rows": self_rows,
                "unique_edges": edge_count,
                **resolved_columns,
                **provenance,
                "search_program": search.program,
                "search_version": search.version,
                "search_invocation": search.invocation,
                "queries_observed": completeness.queries_observed,
                "max_targets_per_query": completeness.max_targets_per_query,
                "queries_at_max_targets": completeness.queries_at_max_targets,
                "import_warnings": json.dumps(list(completeness.warnings)),
            }
            for name, value in attrs.items():
                output.attrs[name] = value
            output.flush()

        valid, reason = validate_final_output(
            partial_path,
            manifest.headers,
            edge_count,
            batch_size,
            expected_attributes=attrs,
        )
        if not valid:
            raise RuntimeError(f"Final output validation failed: {reason}")
        os.replace(partial_path, output_path)
    except Exception:
        if not blast_summary_printed:
            print_header_sanitization_summary(
                "BLAST", tracker.distinct_count, tracker.modifications
            )
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

    return ParseSummary(
        output_path=output_path,
        fasta_header_count=len(manifest.headers),
        fasta_headers_sanitized=len(manifest.modifications),
        blast_header_count=tracker.distinct_count,
        blast_headers_sanitized=len(tracker.modifications),
        data_rows=data_rows,
        self_rows=self_rows,
        unique_edges=edge_count,
        search=search,
        queries_observed=completeness.queries_observed,
        max_targets_per_query=completeness.max_targets_per_query,
        queries_at_max_targets=completeness.queries_at_max_targets,
        query_rows_grouped=completeness.query_rows_grouped,
        warnings=completeness.warnings,
    )
