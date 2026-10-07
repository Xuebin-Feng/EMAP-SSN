"""Real alignments for tests: sparse loaders, FASTA files and Alignment_Manager.

Alignment_Manager always holds a sparse loader, so fixtures that stand in for
the viewer's alignment load their rows through the production FASTA loader
instead of passing Bio alignments or lists of records.
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager  # noqa: E402


def sparse_alignment(records):
    """Load (header, aligned sequence) pairs as an in-memory sparse alignment.

    Headers pass through the loader's canonical sanitization, so use headers
    without whitespace when a test compares them with network headers.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "alignment.fasta")
        with open(path, "w", encoding="utf-8") as handle:
            for header, sequence in records:
                handle.write(f">{header}\n{sequence}\n")
        with redirect_stdout(io.StringIO()):
            return Alignment_Manager.InMemorySparseLoader(path)


def write_fasta(path, records):
    """Write (header, sequence) pairs as a FASTA file."""
    with open(path, "w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n{sequence}\n")


def load_manager(path, headers, reference=None):
    """Load an MSA the way the Viewer does, for the given network headers."""
    with redirect_stdout(io.StringIO()):
        return Alignment_Manager.Alignment_Manager(
            path,
            full_headers=headers,
            active_reference=reference,
        )

