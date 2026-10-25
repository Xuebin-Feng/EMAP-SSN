# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's optional protein language model embedding, matched to its nodes.

The embedding files the Tools write (Generate_Embeddings.py and the tools that
extract, inject or crop them) hold one per-residue matrix per sequence, keyed
by its sanitized header, and run to many gigabytes. So the Viewer reads only
their manifest when it loads, as Alignment_Manager reads an MSA: the headers,
sequences, model and width. A command reads the per-residue rows of the nodes
it needs, or their pooled vectors, when it runs.

A node has an embedding when the file holds its exact header with the same
sanitized sequence. A node whose sequence differs (an embedding of the
full-length sequences used with a cropped FASTA, say) has none: its rows would
not line up with its residues. As with the MSA, partial and zero coverage are
accepted with a warning, and only a file that breaks the embedding-file
interface is refused.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from utilities.Localization import Message
from utilities.Terminal_Colors import colours_enabled

POOLING_METHODS = ("mean", "max")


def _print_red_warning(message):
    """Print a bright-red warning to terminals and plain text to redirected logs."""
    if colours_enabled():
        print(f"\033[91m{message}\033[0m")
    else:
        print(message)


def _source_signature(path):
    """The file's (size, modification time), or None when it cannot be read."""
    try:
        status = os.stat(path)
    except (OSError, TypeError, ValueError):
        return None
    return (status.st_size, status.st_mtime_ns)


def _no_file(path):
    return not path or str(path).strip() == "" or str(path).strip().lower() == "none"


def _check_pooling_method(method):
    if method not in POOLING_METHODS:
        raise ValueError(
            f"Unknown pooling method '{method}'; expected one of: "
            + ", ".join(POOLING_METHODS)
        )


def read_embedding_summary(path):
    """Return (manifest, width) of the embedding file at PATH.

    The manifest is utilities.HDF5_Storage.EmbeddingManifest: the headers and
    sanitized sequences in file order, the model name and the saving mode. The
    width is read from one sequence's matrix, so no per-residue data is read.
    Raises ValueError (or OSError for an unreadable file) when the file breaks
    the interface or its generation is incomplete.
    """
    import h5py
    from utilities.HDF5_Storage import read_embedding_manifest

    with h5py.File(path, "r") as handle:
        manifest = read_embedding_manifest(
            handle, require_complete=True, validate_embeddings=False
        )
        width = None
        if manifest.headers:
            group = handle["embeddings"]
            if not isinstance(group, h5py.Group):
                raise ValueError("'/embeddings' must be an HDF5 group.")
            first = group.get(manifest.headers[0])
            if not isinstance(first, h5py.Dataset) or first.ndim != 2 or first.shape[1] <= 0:
                raise ValueError(
                    f"Embedding database has no usable matrix for '{manifest.headers[0]}'."
                )
            width = int(first.shape[1])
    return manifest, width


@dataclass(frozen=True)
class EmbeddingSource:
    """What a reader needs to read node embeddings, safe to hand to another thread.

    NODE_HEADERS and NODE_SEQUENCES hold, for each Viewer node, the header of
    its matrix in the file and its stored sequence, or None when the node has
    no embedding. Every read first checks the file is the one that was loaded.
    """

    path: str
    signature: tuple
    model_name: str
    saving_mode: str
    width: int
    node_headers: tuple
    node_sequences: tuple

    def check_unchanged(self):
        if _source_signature(self.path) != self.signature:
            raise ValueError(Message(
                "The embedding file changed after the Viewer loaded it: {path}. "
                "Reopen the Viewer to use the new file.",
                path=self.path,
            ))

    @contextmanager
    def open(self):
        """Open the file for reading, after checking it has not changed."""
        import h5py

        self.check_unchanged()
        with h5py.File(self.path, "r") as handle:
            yield handle

    def residues(self, node_index, handle=None):
        """The node's per-residue embedding as a float32 (length, width) array."""
        if handle is None:
            with self.open() as opened:
                return self.residues(node_index, opened)
        from utilities.HDF5_Storage import validate_embedding_array

        header = self.node_headers[node_index]
        if header is None:
            raise ValueError(Message(
                "Node {index} has no embedding in {file}.",
                index=node_index, file=os.path.basename(self.path),
            ))
        dataset = handle["embeddings"][header]
        validate_embedding_array(
            dataset, self.node_sequences[node_index], self.saving_mode,
            feature_dimension=self.width, header=header,
        )
        return np.asarray(dataset[()], dtype=np.float32)

    def pooled(self, node_indices, method="mean", handle=None):
        """One float32 vector per node: the mean or maximum over its residues."""
        _check_pooling_method(method)
        if handle is None:
            with self.open() as opened:
                return self.pooled(node_indices, method, opened)
        indices = [int(index) for index in node_indices]
        vectors = np.empty((len(indices), self.width), dtype=np.float32)
        for row, index in enumerate(indices):
            residues = self.residues(index, handle)
            if method == "mean":
                vectors[row] = residues.mean(axis=0)
            else:
                vectors[row] = residues.max(axis=0)
        return vectors


class Embedding_Manager:
    def __init__(self, embed_file, full_headers=None, sequences=None, reuse=None):
        """Read EMBED_FILE's manifest and match it to the nodes in FULL_HEADERS.

        SEQUENCES maps a node's header, or its first word, to its sanitized
        FASTA sequence (the Viewer's sequences_map). A node it has no sequence
        for is matched by header alone. REUSE is a manager loaded earlier, or
        None; when it read this same file, unchanged on disk, its manifest is
        taken as it is.
        """
        self.embed_file = embed_file
        # The reader of the loaded file, or None when no embedding is loaded.
        self.source = None
        # Why the file failed to load, as a Message, when it did.
        self.load_failure = None
        self.model_name = None
        self.saving_mode = None
        self.width = None
        self.sequence_count = 0
        self.network_headers = list(full_headers) if full_headers is not None else []
        self.matched_headers = []
        self.missing_headers = []
        self.mismatched_headers = []
        self.viewer_to_emb = np.full(len(self.network_headers), -1, dtype=int)
        self.embedded_node_mask = np.zeros(len(self.network_headers), dtype=bool)
        self._manifest = None
        self._source_signature = None
        self._pooled = {method: {} for method in POOLING_METHODS}

        if _no_file(embed_file):
            print("An embedding file is not selected and will not be loaded.")
            return

        if self._can_reuse_manifest_of(reuse):
            print(f"--- Reusing the loaded embedding manifest; the file is unchanged: {embed_file} ---")
            self._manifest, self.width = reuse._manifest, reuse.width
            self._source_signature = reuse._source_signature
        else:
            print(f"--- Reading the embedding manifest: {embed_file} ---")
            # Taken before reading, so a file that changes during the read
            # fails the first read of its rows rather than being trusted.
            self._source_signature = _source_signature(embed_file)
            try:
                self._manifest, self.width = read_embedding_summary(embed_file)
            except (OSError, ValueError, KeyError) as error:
                self.load_failure = Message("Embedding file rejected: {error}", error=error)
                _print_red_warning(f"ERROR: {self.load_failure}")
                return
            if self._source_signature is None:
                self.load_failure = Message(
                    "Embedding file could not be read: {path}", path=embed_file
                )
                _print_red_warning(f"ERROR: {self.load_failure}")
                self._manifest = None
                return

        self.model_name = self._manifest.model_name
        self.saving_mode = self._manifest.saving_mode
        self.sequence_count = len(self._manifest.headers)
        node_headers, node_sequences = self._match_nodes(sequences or {})
        self.source = EmbeddingSource(
            path=os.fspath(embed_file),
            signature=self._source_signature,
            model_name=self.model_name,
            saving_mode=self.saving_mode,
            width=self.width or 0,
            node_headers=tuple(node_headers),
            node_sequences=tuple(node_sequences),
        )
        print(
            f"Embedding Ready. Model: {self.model_name}, width {self.width}, "
            f"{self.saving_mode}; {len(self.matched_headers)}/{len(self.network_headers)} "
            "network nodes embedded."
        )
        if self.missing_headers or self.mismatched_headers:
            self._warn_incomplete_coverage()

    def _can_reuse_manifest_of(self, previous):
        """Whether PREVIOUS read this same file, unchanged since."""
        if previous is None or getattr(previous, "_manifest", None) is None:
            return False
        signature = getattr(previous, "_source_signature", None)
        return (
            signature is not None
            and getattr(previous, "embed_file", None) == self.embed_file
            and _source_signature(self.embed_file) == signature
        )

    def _match_nodes(self, sequences):
        """Map each node to its matrix by exact header and identical sequence."""
        row_of = {header: row for row, header in enumerate(self._manifest.headers)}
        stored = self._manifest.sequences
        node_headers = [None] * len(self.network_headers)
        node_sequences = [None] * len(self.network_headers)
        for node, header in enumerate(self.network_headers):
            row = row_of.get(header)
            if row is None:
                self.missing_headers.append(header)
                continue
            parts = header.split()
            expected = sequences.get(header)
            if expected is None and parts:
                expected = sequences.get(parts[0])
            if expected is not None and expected != stored[row]:
                self.mismatched_headers.append(header)
                continue
            self.viewer_to_emb[node] = row
            self.embedded_node_mask[node] = True
            self.matched_headers.append(header)
            node_headers[node] = header
            node_sequences[node] = stored[row]
        return node_headers, node_sequences

    def _warn_incomplete_coverage(self):
        total = len(self.network_headers)
        embedded = len(self.matched_headers)
        coverage_pct = (100.0 * embedded / total) if total else 100.0
        lines = [
            "WARNING: Incomplete embedding coverage",
            f"Embedded network nodes: {embedded}/{total} ({coverage_pct:.1f}%)",
            f"Excluded from embedding-dependent analyses: {total - embedded}",
        ]
        for label, headers in (
            ("Headers missing from the embedding file", self.missing_headers),
            ("Headers whose stored sequence differs from the FASTA", self.mismatched_headers),
        ):
            if not headers:
                continue
            shown = headers[:10]
            lines.append(f"{label} (showing {len(shown)} of {len(headers)}):")
            lines.extend(f"  - {header}" for header in shown)
            if len(headers) > len(shown):
                lines.append(f"  ... and {len(headers) - len(shown)} more.")
        _print_red_warning("\n".join(lines))

    def describe(self, network_path=None):
        """What the embedding command reports after the coverage, as a list of Messages.

        The nodes whose sequence differs from the file's, when there are any;
        the model with its width and saving mode; and, as the Config's
        Consistency Check notes it, that the network at NETWORK_PATH was built
        with another model. The Viewer accepts such an embedding, and
        embedding commands use the embedding's model. Both viewers' commands
        say the same through this, the desktop's translated and VR's in English.
        """
        notes = []
        if self.mismatched_headers:
            notes.append(Message(
                "%n node(s) not embedded: the file holds a different sequence.",
                n=len(self.mismatched_headers),
            ))
        if self.width is None:
            notes.append(Message(
                "Embedding model: {model} ({saving_mode}).",
                model=self.model_name, saving_mode=self.saving_mode,
            ))
        else:
            notes.append(Message(
                "Embedding model: {model} (width {width}, {saving_mode}).",
                model=self.model_name, width=self.width, saving_mode=self.saving_mode,
            ))
        if network_path:
            from Cache_Manifest import read_network_metadata

            try:
                network = read_network_metadata(network_path)
            except (OSError, ValueError, TypeError):
                network = None
            if (
                network is not None
                and network.network_type == "alignment"
                and network.model_name != self.model_name
            ):
                notes.append(Message(
                    "NOTE: The network was built with {network}, not {model}. The "
                    "Viewer accepts this; embedding commands use the embedding's model.",
                    network=network.model_name, model=self.model_name,
                ))
        return notes

    def residues(self, node_index):
        """The node's per-residue embedding as a float32 (length, width) array."""
        return self._require_source().residues(node_index)

    def pooled(self, node_indices, method="mean"):
        """One float32 vector per node, its residues' mean or maximum.

        Vectors read once are kept for the rest of the session, so asking
        again for the same nodes reads nothing.
        """
        source = self._require_source()
        _check_pooling_method(method)
        indices = [int(index) for index in node_indices]
        cache = self._pooled[method]
        unread = [index for index in dict.fromkeys(indices) if index not in cache]
        if unread:
            for index, vector in zip(unread, source.pooled(unread, method)):
                cache[index] = vector
        vectors = np.empty((len(indices), source.width), dtype=np.float32)
        for row, index in enumerate(indices):
            vectors[row] = cache[index]
        return vectors

    def _require_source(self):
        if self.source is None:
            raise ValueError(Message("No embedding is loaded."))
        return self.source
