# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the Fold View page's per-residue layers (web_ui/residue_layers.py).

A small helix drawn atom by atom with a pLDDT in its B-factors, and the
residue data file esmfold writes beside a PDB, written by hand in the
format's fixed shape: a few numbers for each residue, with the alignment column
each came from. Nothing here folds anything or reads a real cache.

Not collected by unittest; import with ``from tests.esmfold_layer_fixtures import ...``.
"""

import json
import math
import os

THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
    "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
    "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
# Two proteins of one family: the same helix with a few other residues.
FIRST_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SECOND_SEQUENCE = "MKSAYVAKQRQLSFVKSHYSRQMEERLGLIEVQ"
MSA_FILE = "fixture_alignment.fasta"


def helix_pdb(sequence, plddt, *, chain="A", shift=(0.0, 0.0, 0.0), wobble=0.0, first=1):
    """The PDB text of an alpha helix of ``sequence``, with ``plddt`` (0 to 100) as each residue's B-factor.

    Backbone atoms and a C-beta for every residue but glycine, numbered from
    ``first``. ``shift`` moves the whole helix and ``wobble`` bends its axis a
    little, so a second helix is a slightly different copy of the first.
    """
    trace = []
    for index in range(len(sequence)):
        angle = math.radians(100 * index)
        trace.append((2.3 * math.cos(angle) + wobble * math.sin(index * 0.7), 2.3 * math.sin(angle), 1.5 * index))

    def towards(origin, target, fraction):
        return tuple(a + (b - a) * fraction for a, b in zip(origin, target))

    lines = []
    serial = 1

    def atom(name, element, residue, index, point, b_factor):
        nonlocal serial
        x, y, z = (value + offset for value, offset in zip(point, shift))
        lines.append(
            f"ATOM  {serial:5d} {name:<4s} {residue} {chain}{first + index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}          {element:>2s}"
        )
        serial += 1

    for index, letter in enumerate(sequence):
        residue, b_factor = THREE_LETTER[letter], plddt[index]
        ahead = trace[index + 1] if index + 1 < len(sequence) else towards(trace[index], trace[index - 1], -1)
        behind = trace[index - 1] if index else towards(trace[index], trace[index + 1], -1)
        outward = (trace[index][0], trace[index][1], 0.0)
        length = math.hypot(outward[0], outward[1])
        outward = (outward[0] / length, outward[1] / length, 0.0)
        carbonyl = towards(trace[index], ahead, 0.3)
        atom("N", "N", residue, index, towards(trace[index], behind, 0.3), b_factor)
        atom("CA", "C", residue, index, trace[index], b_factor)
        atom("C", "C", residue, index, carbonyl, b_factor)
        atom("O", "O", residue, index, tuple(a + 1.2 * b for a, b in zip(carbonyl, outward)), b_factor)
        if letter != "G":
            atom("CB", "C", residue, index, tuple(a + 1.5 * b for a, b in zip(trace[index], outward)), b_factor)
    return "\n".join(lines + ["TER", "END"]) + "\n"


def plddt_values(length, offset=0):
    """pLDDT values (40 to 94) that differ from residue to residue, so a colouring can be checked."""
    return [40 + (index * 53 + offset) % 55 for index in range(length)]


def information_content(length, offset=0):
    """A fixture's information content (%) for each residue, with none for residue 4 (a gappy column)."""
    return [None if index == 3 else round((index * 7 + offset) % 101 + 0.5, 1) for index in range(length)]


def same_residue_pct(length, offset=0):
    """A fixture's residue agreement (%) for each residue."""
    return [float((index * 13 + offset) % 101) for index in range(length)]


def residue_data(structure_name, sequence, *, header=None, chain="A", first=1, offset=0, properties=None,
                 column_start=100, sequences=12):
    """The residue data of ``structure_name`` as the dict esmfold writes it.

    The default properties are the two the writer starts with: information
    content and residue agreement, in file order. ``properties`` replaces them
    with {id: property dict}, whose "values" may be built with
    ``residue_property``.
    """
    length = len(sequence)
    columns = [column_start + 2 * index for index in range(length)]
    columns[1] = None
    source = {"kind": "msa", "file": MSA_FILE, "alignment_id": "fixture", "columns": columns, "sequences": sequences}
    if properties is None:
        properties = {
            "information_content": residue_property(
                "Information Content", "%", [0, 100], information_content(length, offset), source,
                "How far the alignment column of each residue is from a column of all twenty amino acids at random.",
            ),
            "same_residue_pct": residue_property(
                "Residue Agreement", "%", [0, 100], same_residue_pct(length, offset), source,
                "Share of the aligned sequences with a residue in this column that have the same amino acid as this protein.",
            ),
        }
    return {
        "format": "emapssn-residue-data",
        "version": 1,
        "structure": structure_name,
        "header": header or structure_name[:-len(".pdb")],
        "sequence": sequence,
        "residue_numbering": {"chain": chain, "first": first},
        "properties": properties,
    }


def residue_property(label, unit, value_range, values, source=None, description=""):
    """One property of a residue data file."""
    entry = {"label": label, "unit": unit, "range": list(value_range), "description": description, "values": values}
    if source is not None:
        entry["source"] = source
    return entry


def write_structure(directory, name, sequence, *, plddt=None, sidecar=True, sidecar_changes=None, **options):
    """Write ``name`` (a .pdb file name) and, unless ``sidecar`` is false, its residue data file.

    ``options`` shape the helix (shift, wobble) and the residue data (offset,
    properties); ``sidecar_changes`` updates the residue data dict before it is
    written, to make a damaged file. Returns the path of the PDB.
    """
    helix = {key: options.pop(key) for key in ("chain", "shift", "wobble", "first") if key in options}
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(helix_pdb(sequence, plddt or plddt_values(len(sequence)), **helix))
    if sidecar:
        data = residue_data(name, sequence, **{**helix_numbering(helix), **options})
        data.update(sidecar_changes or {})
        write_sidecar(directory, name, data)
    return path


def helix_numbering(helix):
    return {key: helix[key] for key in ("chain", "first") if key in helix}


def write_sidecar(directory, structure_name, data):
    """Write ``data`` as the residue data file of ``structure_name``; returns its path."""
    path = os.path.join(directory, structure_name[:-len(".pdb")] + ".residues.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle)
    return path
