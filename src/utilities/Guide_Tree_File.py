# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""Where the embedding MSA saves its guide tree, how its labels are quoted, and how the file is read back.

The tree of an alignment ``<dir>/<name>_alignment.fasta`` is
``<dir>/Guide_Trees/<name>_guide_tree.nex``. The tool that writes the tree and
anything that looks for it go through guide_tree_path_for_alignment, so both
agree on the place. read_guide_tree reads such a file back: the writer's quoting
(newick_label, nexus_string) is inverted here, so a header comes back as it went in.
This module imports no torch, numba or Qt, so a reader can load it cheaply.
"""

import math
import os
import re
from dataclasses import dataclass

GUIDE_TREE_SUBFOLDER = "Guide_Trees"

_ALIGNMENT_SUFFIXES = ("_sparse.h5", ".h5", ".fasta")
_ALIGNMENT_STEM_END = "_alignment"


def guide_tree_path_for_alignment(msa_path):
    """The guide tree's path for an alignment file, or None when its name does not follow the tool's.

    The name must end in ``_alignment`` followed by ``.fasta``, ``.h5`` or
    ``_sparse.h5`` (the sparse file the converter writes beside a FASTA). The
    folder keeps its spelling, relative or absolute. A renamed or external
    alignment, or an empty path, has no tree.
    """
    if not msa_path or not str(msa_path).strip():
        return None
    folder, filename = os.path.split(os.fspath(msa_path))
    lowered = filename.lower()
    for suffix in _ALIGNMENT_SUFFIXES:
        if lowered.endswith(suffix):
            stem = filename[: len(filename) - len(suffix)]
            break
    else:
        return None
    if not stem.lower().endswith(_ALIGNMENT_STEM_END):
        return None
    name = stem[: len(stem) - len(_ALIGNMENT_STEM_END)]
    if not name:
        return None
    return os.path.join(folder, GUIDE_TREE_SUBFOLDER, f"{name}_guide_tree.nex")


_NEWICK_PLAIN_LABEL = re.compile(r"[A-Za-z0-9.|]+\Z")


def newick_label(text):
    """Quote a label unless it holds only letters, digits, dots and bars.

    NEXUS, like the Newick standard, reads an unquoted underscore as a space
    and splits words at - + * = / < >, so quoting keeps a header as it is.
    """
    if _NEWICK_PLAIN_LABEL.match(text):
        return text
    return nexus_string(text)


def nexus_string(text):
    """Single-quote a NEXUS word, doubling any quote inside it."""
    return "'" + str(text).replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Reading a guide tree
# ---------------------------------------------------------------------------

# The most a tree file may be, hold and nest. A tree of 44,000 leaves is about 5 MB; these keep a file that
# is not one of ours from taking the Viewer's time or memory. The leaf limit also bounds the internal nodes
# (twice as many nodes in all), so a flood of "(" is refused before it fills memory.
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_LEAVES = 250_000
MAX_LABEL_CHARACTERS = 65_536


class GuideTreeError(ValueError):
    """The file is not a guide tree this reader can use; the message says why."""


@dataclass(frozen=True, eq=False)
class GuideTree:
    """A guide tree as parallel arrays over its nodes.

    Nodes are numbered in post-order, so every child comes before its parent,
    the root is the last node (``parent[-1] == -1``), a node's subtree is the
    block of nodes that ends at it, and the leaves, taken in node order, run
    left to right as the file has them. ``labels[k]`` names the k-th leaf.

    parent, length and support are numpy arrays. parent: int32 per node. length: float32 branch length to the parent (0 for
    the root and for a node the file gives none; has_lengths says whether any
    was given). support: int8 per node, the percentage an internal node's label
    gives (0 to 100) or -1. rooted: True for ``[&R]``, False for ``[&U]``,
    None when the file says neither. notes: the EMAPSSN block, name to value.
    """

    parent: object
    length: object
    support: object
    labels: list
    name: str
    rooted: object
    has_lengths: bool
    notes: dict

    @property
    def node_count(self):
        return len(self.parent)

    @property
    def leaf_count(self):
        return len(self.labels)

    def leaf_nodes(self):
        """The node number of each leaf, in the order of ``labels``."""
        import numpy as np

        has_child = np.zeros(self.node_count, dtype=bool)
        has_child[self.parent[self.parent >= 0]] = True
        return np.flatnonzero(~has_child).astype(np.int32)


# One token of the text: white space, a quoted word, the start of a comment, punctuation, or a word. A quote
# that never closes, or a stray "]", matches none of them, and the reader says so.
_TOKEN = re.compile(
    r"(\s+)"
    r"|('[^']*(?:''[^']*)*')"
    r"|(\[)"
    r"|([(),:;=])"
    r"|([^\s()\[\],:;=']+)"
)
_BRACKET = re.compile(r"[\[\]]")


def _tokens(text):
    """The tokens of a NEXUS text as (kind, value): "w" a word, "q" a quoted word (unquoted, its doubled
    quotes single), "p" one of ( ) , : ; =, and "c" a metacomment such as ``&R``. Other comments are skipped.

    A comment may hold comments. Nothing recurses, and a text is read once.
    """
    match = _TOKEN.match
    find_bracket = _BRACKET.search
    position, end = 0, len(text)
    while position < end:
        found = match(text, position)
        if found is None:
            raise GuideTreeError(
                f"Unreadable text at character {position + 1}: a quote that never closes, or a stray bracket."
            )
        kind = found.lastindex
        position = found.end()
        if kind == 1:
            continue
        if kind == 5:
            yield "w", found.group(5)
        elif kind == 4:
            yield "p", found.group(4)
        elif kind == 2:
            yield "q", found.group(2)[1:-1].replace("''", "'")
        else:
            start, depth = position, 1
            while depth:
                bracket = find_bracket(text, position)
                if bracket is None:
                    raise GuideTreeError("A comment in [ ] never closes.")
                depth += 1 if bracket.group() == "[" else -1
                position = bracket.end()
            comment = text[start:position - 1].strip()
            if comment.startswith("&"):
                yield "c", comment


def _read_command(tokens, keep=8):
    """The first ``keep`` tokens of the command that the next token begins, reading to its ";"."""
    found = []
    for token in tokens:
        if token == ("p", ";"):
            break
        if len(found) < keep:
            found.append(token)
    return found


# What the next token of a Newick tree may be.
_NODE, _LEAF, _CLOSED, _NAMED, _LENGTHED = range(5)


def _read_newick(tokens, max_leaves):
    """The tree of a TREE command's Newick text, to its ";": (parent, length, support, labels, rooted, has_lengths).

    Iterative: the open nodes are a list, not a call stack. A leaf takes a node number when it is read and an
    internal node when its ")" closes it, which is post-order. An unquoted label reads its underscores as spaces,
    as NEXUS does; the writer quotes every label that has one.
    """
    max_nodes = 2 * max_leaves
    parent, length, support, labels = [], [], [], []
    open_nodes = []
    root = last = None
    state = _NODE
    started = False
    rooted = None
    has_lengths = False

    def new_node():
        parent.append(-1)
        length.append(0.0)
        support.append(-1)
        if len(parent) > max_nodes:
            raise GuideTreeError(f"The tree has more than {max_nodes:,} nodes.")
        return len(parent) - 1

    for kind, value in tokens:
        if kind == "c":
            if not started:
                flag = value.replace(" ", "").upper()
                rooted = True if flag == "&R" else False if flag == "&U" else rooted
            continue
        started = True
        if kind == "p":
            if value == "(":
                if state != _NODE:
                    raise GuideTreeError("A ( where no node can start.")
                open_nodes.append([])
                if len(open_nodes) > max_nodes:
                    raise GuideTreeError(f"The tree has more than {max_nodes:,} nested nodes.")
            elif value == ":":
                if state not in (_LEAF, _CLOSED, _NAMED):
                    raise GuideTreeError("A branch length where no node ends, or a leaf with no name.")
                for kind2, number in tokens:
                    if kind2 != "c":
                        break
                else:
                    kind2, number = "p", ";"
                try:
                    if kind2 != "w":
                        raise ValueError(number)
                    branch = float(number)
                except ValueError:
                    raise GuideTreeError(f"A branch length is not a number: {number!r}.") from None
                if not math.isfinite(branch):
                    raise GuideTreeError(f"A branch length is not finite: {number!r}.")
                length[last] = branch
                has_lengths = True
                state = _LENGTHED
            elif value in (",", ")", ";"):
                if state == _NODE:
                    raise GuideTreeError("A node is missing before " + repr(value) + ".")
                if value == ",":
                    if not open_nodes:
                        raise GuideTreeError("A , outside any parentheses.")
                    state = _NODE
                elif value == ")":
                    if not open_nodes:
                        raise GuideTreeError("A ) with no ( before it.")
                    children = open_nodes.pop()
                    node = new_node()
                    for child in children:
                        parent[child] = node
                    if open_nodes:
                        open_nodes[-1].append(node)
                    else:
                        root = node
                    last, state = node, _CLOSED
                else:
                    if open_nodes:
                        raise GuideTreeError("The tree ends before every ( is closed.")
                    break
            else:
                raise GuideTreeError("An = inside the tree.")
        else:
            if state == _NODE:
                label = value.replace("_", " ") if kind == "w" else value
                if len(label) > MAX_LABEL_CHARACTERS:
                    raise GuideTreeError(f"A leaf's name is longer than {MAX_LABEL_CHARACTERS:,} characters.")
                if len(labels) >= max_leaves:
                    raise GuideTreeError(f"The tree has more than {max_leaves:,} leaves.")
                node = new_node()
                labels.append(label)
                if open_nodes:
                    open_nodes[-1].append(node)
                else:
                    root = node
                last, state = node, _LEAF
            elif state == _CLOSED:
                try:
                    percent = int(value)
                except ValueError:
                    percent = -1
                support[last] = percent if 0 <= percent <= 100 else -1
                state = _NAMED
            else:
                raise GuideTreeError(f"A label ({value[:40]!r}) where none can be.")
    else:
        raise GuideTreeError("The tree does not end with ;.")
    if root is None:
        raise GuideTreeError("The tree has no leaves.")
    return parent, length, support, labels, rooted, has_lengths


def _read_tree_command(tokens, max_leaves):
    """The tree of ``TREE name = [&R] (...);``, after the word TREE: (name, parent, length, support, labels, rooted, has_lengths)."""
    name = []
    for kind, value in tokens:
        if kind == "p" and value == "=":
            break
        if kind == "p":
            raise GuideTreeError("A TREE command must read TREE name = tree;")
        if kind in ("w", "q") and value != "*":
            name.append(value)
    else:
        raise GuideTreeError("A TREE command has no tree.")
    return (" ".join(name), *_read_newick(tokens, max_leaves))


def parse_guide_tree(text, *, max_leaves=MAX_LEAVES):
    """The guide tree of a NEXUS text: its first TREE command, and the EMAPSSN block's notes.

    Raises GuideTreeError for a text that is not NEXUS, holds no tree, has a TRANSLATE table (the
    writer makes none, and a table would rename the labels) or is not a well-formed Newick tree.
    """
    tokens = _tokens(text)
    first = next(tokens, None)
    if first is None or first[0] != "w" or first[1].upper() != "#NEXUS":
        raise GuideTreeError("The file is not NEXUS: it does not begin with #NEXUS.")
    notes, block, tree = {}, None, None
    for kind, value in tokens:
        if kind == "p" and value == ";":
            continue
        word = value.upper() if kind == "w" else ""
        if word == "BEGIN":
            named = _read_command(tokens, 2)
            block = named[0][1].upper() if named and named[0][0] == "w" else ""
        elif word in ("END", "ENDBLOCK"):
            _read_command(tokens, 0)
            block = None
        elif block == "TREES" and word in ("TREE", "UTREE"):
            if tree is None:
                tree = _read_tree_command(tokens, max_leaves)
            else:
                _read_command(tokens, 0)
        elif block == "TREES" and word == "TRANSLATE":
            raise GuideTreeError("The TREES block has a TRANSLATE table, which this reader does not take.")
        elif block == "EMAPSSN" and kind in ("w", "q"):
            command = _read_command(tokens, 2)
            if len(command) == 2 and command[0] == ("p", "=") and command[1][0] in ("w", "q"):
                notes.setdefault(value, command[1][1])
        else:
            _read_command(tokens, 0)
    if tree is None:
        raise GuideTreeError("The file holds no tree.")
    import numpy as np

    name, parent, length, support, labels, rooted, has_lengths = tree
    return GuideTree(
        parent=np.asarray(parent, dtype=np.int32),
        length=np.asarray(length, dtype=np.float32),
        support=np.asarray(support, dtype=np.int8),
        labels=labels,
        name=name,
        rooted=rooted,
        has_lengths=has_lengths,
        notes=notes,
    )


def read_guide_tree(path, *, max_bytes=MAX_FILE_BYTES, max_leaves=MAX_LEAVES):
    """Read the guide tree file at ``path``.

    OSError when the file cannot be opened or read (FileNotFoundError when it is not there);
    GuideTreeError when it is too large, not UTF-8 text or not a guide tree (parse_guide_tree).
    """
    with open(path, "rb") as handle:
        if os.fstat(handle.fileno()).st_size > max_bytes:
            raise GuideTreeError(f"The tree file is larger than {max_bytes // (1024 * 1024)} MB.")
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise GuideTreeError(f"The tree file is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise GuideTreeError("The tree file is not UTF-8 text.") from None
    return parse_guide_tree(text, max_leaves=max_leaves)
