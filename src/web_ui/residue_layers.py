# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Per-residue layers for the Fold View page, from a structure's residue data.

Next to each predicted structure, ``<name>.pdb``, esmfold writes
``<name>.residues.json``: numbers for each residue, such as the conservation of
its alignment column, written when the structure was folded. This module turns
that file into what the page needs to colour the structure by one of those
numbers:

* layer_listing(): the layers offered for a structure, each with its label,
  unit, range, source, legend and the address of its annotation. The
  confidence layer, pLDDT, which every structure has in its B-factors, is
  always first. It is the colouring the page starts with.
* layer_annotation(): one layer as a MolViewSpec annotation, a list of rows
  Mol*'s MolViewSpec extension reads: a residue selector in the PDB's own chain
  and residue numbering (auth_asym_id, auth_seq_id), a colour and a tooltip.
  The page attaches it to the structure it has already loaded, so the PDB, its
  pLDDT and the superposition stay as they are.

Every property of the file is a layer: nothing here names conservation. How a
property is coloured is the LAYER_STYLES entry for its id, or DEFAULT_SCALE;
supporting another property (hydrophobicity, an embedding-derived score) is
the writer adding it to the file, and at most one entry here.

A file is shown on a structure only when it describes that structure: its
name is the structure's, and its sequence and residue numbers are the ones the
PDB holds, which are read from the PDB itself. An older file of the same
protein is fine (its source line names the alignment it came from); a file of
another protein, left beside a structure folded again under the same name
without data, is refused with the reason.

The module imports no Qt and reads no Viewer state, so the web server's
request threads can call it. Texts it makes for the page (the tooltip of a
residue, a layer's source, the reason a structure has no layers) are Messages,
shown in the Viewer's language; a property's own label, unit and description
are shown as the file has them.
"""

import json
import math
import os
import re
from urllib.parse import quote

from utilities.Localization import Message, display_text

ROUTE_PREFIX = "/api/esmfold/layers/"
SIDECAR_SUFFIX = ".residues.json"
SIDECAR_FORMAT = "emapssn-residue-data"
SIDECAR_VERSION = 1
# A structure of a few thousand residues has a file of a few hundred kilobytes.
MAX_SIDECAR_BYTES = 16 * 1024 * 1024
PLDDT_ID = "plddt"
_PROPERTY_ID = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
BACKGROUND_COLOR = "#dcdcdc"

# Colour scales as evenly spaced stops from the low end of a property's range
# to its high end, taken from ColorBrewer. The sequential ones start well
# above white, so that the lowest residues still show against Mol*'s pale
# background.
COLOR_SCALES = {
    "purples": ("#dadaeb", "#bcbddc", "#9e9ac8", "#6a51a3", "#3f007d"),
    "greens": ("#c7e9c0", "#a1d99b", "#74c476", "#238b45", "#00441b"),
    "reds": ("#fcbba1", "#fc9272", "#fb6a4a", "#cb181d", "#67000d"),
    "blue-red": ("#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"),
}
DEFAULT_SCALE = "greens"
# The scale of each property, by the id its file gives it. A property without
# an entry takes DEFAULT_SCALE, so a new property needs no entry to be offered;
# an entry only gives it a scale of its own.
LAYER_STYLES = {
    "information_content": {"scale": "purples"},
    "same_residue_pct": {"scale": "reds"},
}

# The colours of Mol*'s own plddt-confidence theme, which the page applies.
PLDDT_BANDS = (
    ("#0053d6", Message("Very high (pLDDT > 90)")),
    ("#65cbf3", Message("Confident (pLDDT 70 to 90)")),
    ("#ffdb13", Message("Low (pLDDT 50 to 70)")),
    ("#ff7d45", Message("Very low (pLDDT < 50)")),
)


class ResidueDataError(ValueError):
    """The residue data of a structure cannot be used; the message says why, translated."""

    def __init__(self, message):
        super().__init__(display_text(message))


def sidecar_name(structure_name):
    """The residue data file's name for the PDB file ``structure_name``, or None."""
    if not isinstance(structure_name, str) or not structure_name.lower().endswith(".pdb"):
        return None
    return structure_name[:-len(".pdb")] + SIDECAR_SUFFIX


def _number(value):
    """value as a finite float, or None for null, NaN, infinity and anything not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _integer(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value, default=""):
    return value if isinstance(value, str) else default


class ResidueProperty:
    """One property of the residue data: a number for each residue, or None."""

    def __init__(self, property_id, label, unit, value_range, description, values, source):
        self.id = property_id
        self.label = label
        self.unit = unit
        self.range = value_range
        self.description = description
        self.values = values
        self.source = source

    @property
    def covered(self):
        return sum(value is not None for value in self.values)

    def columns(self):
        """The 0-based alignment column of each residue, or None when the file gives none."""
        columns = self.source.get("columns")
        if isinstance(columns, list) and len(columns) == len(self.values):
            return [_integer(column) for column in columns]
        return None


class ResidueData:
    """A structure's residue data: where its numbers start, and the usable properties."""

    def __init__(self, structure_name, chain, first, properties, token):
        self.structure_name = structure_name
        self.chain = chain
        self.first = first
        self.properties = properties
        # Changes whenever the file does, so a page can tell its copy is old.
        self.token = token


def _read_property(property_id, entry):
    """The ResidueProperty an entry of the file's "properties" describes, or None when unusable."""
    if not _PROPERTY_ID.fullmatch(property_id) or property_id == PLDDT_ID or not isinstance(entry, dict):
        return None
    raw = entry.get("values")
    if not isinstance(raw, list):
        return None
    values = [_number(value) for value in raw]
    present = [value for value in values if value is not None]
    if not present:
        return None
    value_range = entry.get("range")
    low = high = None
    if isinstance(value_range, list) and len(value_range) == 2:
        low, high = _number(value_range[0]), _number(value_range[1])
    if low is None or high is None or not low < high:
        low, high = min(present), max(present)
        if low == high:
            low, high = low - 0.5, high + 0.5
    source = entry.get("source")
    return ResidueProperty(
        property_id,
        _text(entry.get("label"), property_id) or property_id,
        _text(entry.get("unit")),
        (low, high),
        _text(entry.get("description")),
        values,
        source if isinstance(source, dict) else {},
    )


# The three-letter names of the twenty amino acids, and selenomethionine, which stands for M.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}
STANDARD_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _structure_chains(path):
    """The residues of the PDB at ``path`` by chain: {chain: ((number, residue name), ...)}, in file order.

    A residue's number includes its insertion code. It is read at each
    request, not kept: a file rewritten within the file system's clock tick
    and to the same size (another protein of the same length) would pass for
    the old one, and reading a few thousand lines takes milliseconds.
    """
    chains = {}
    last = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            residue = (line[21], line[22:27].strip())
            if residue != last:
                chains.setdefault(line[21], []).append((residue[1], line[17:20].strip()))
                last = residue
    return {chain: tuple(residues) for chain, residues in chains.items()}


def _check_structure(structure_path, chain, first, sequence):
    """Raise ResidueDataError unless the PDB holds the protein ``sequence``, numbered from ``first`` in ``chain``.

    The PDB's residues of the chain must be as many as the sequence's, with
    those numbers, and each standard residue must be the sequence's letter at
    its place. A residue the PDB names otherwise (UNK, SEC) and a letter that
    is not a standard amino acid (X) are not compared.
    """
    different = ResidueDataError(
        Message("The per-residue data file was made for a different protein than this structure holds.")
    )
    try:
        residues = _structure_chains(structure_path).get(chain, ())
    except OSError:
        raise ResidueDataError(Message("The structure file cannot be read for its residues.")) from None
    if len(residues) != len(sequence):
        raise different
    for (number, name), letter in zip(residues, sequence.upper()):
        expected = THREE_TO_ONE.get(name)
        if expected is not None and letter in STANDARD_LETTERS and letter != expected:
            raise different
    try:
        numbers = [int(number) for number, _ in residues]
    except ValueError:
        numbers = None
    if numbers != list(range(first, first + len(sequence))):
        raise ResidueDataError(
            Message("The residues of this structure are not numbered as its per-residue data file expects.")
        )


def read_residue_data(structure_name, structure_path, sidecar_path):
    """The ResidueData in the file ``sidecar_path`` for the structure ``structure_name``, the PDB at ``structure_path``.

    Raises ResidueDataError, with a translated reason, when the file is
    missing, or is not residue data for this structure: its name is not the
    structure's, or its sequence or numbering is not what the PDB holds. A
    property that is not usable (no numbers, or numbers for another count of
    residues than its sequence) is left out, and the others stay.
    """
    try:
        status = os.stat(sidecar_path)
    except OSError:
        raise ResidueDataError(Message("No per-residue data was saved with this structure.")) from None
    if status.st_size > MAX_SIDECAR_BYTES:
        raise ResidueDataError(Message("The per-residue data file of this structure is too large."))
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        raise ResidueDataError(Message("The per-residue data file of this structure cannot be read.")) from None
    if not isinstance(document, dict) or document.get("format") != SIDECAR_FORMAT:
        raise ResidueDataError(Message("The per-residue data file of this structure is not in a known format."))
    if _integer(document.get("version")) != SIDECAR_VERSION:
        raise ResidueDataError(Message("The per-residue data file of this structure has an unsupported version."))
    if document.get("structure") != structure_name:
        raise ResidueDataError(Message("The per-residue data file belongs to another structure."))
    numbering = document.get("residue_numbering")
    chain = numbering.get("chain") if isinstance(numbering, dict) else None
    first = _integer(numbering.get("first")) if isinstance(numbering, dict) else None
    if not isinstance(chain, str) or not chain.strip() or first is None:
        raise ResidueDataError(Message("The per-residue data file does not say how the residues are numbered."))
    sequence = document.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ResidueDataError(Message("The per-residue data file does not hold the sequence it was made for."))
    _check_structure(structure_path, chain, first, sequence)
    properties = {}
    entries = document.get("properties")
    for property_id, entry in (entries.items() if isinstance(entries, dict) else ()):
        layer = _read_property(property_id, entry)
        if layer is None or len(sequence) != len(layer.values):
            continue
        properties[property_id] = layer
    structure_status = os.stat(structure_path)
    token = f"{status.st_mtime_ns}-{status.st_size}-{structure_status.st_mtime_ns}"
    return ResidueData(structure_name, chain, first, properties, token)


def _hex_to_rgb(color):
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


def scale_color(scale, fraction):
    """The colour ("#rrggbb") of the named scale at ``fraction`` (0 to 1) of its length."""
    stops = COLOR_SCALES[scale]
    fraction = min(1.0, max(0.0, fraction))
    position = fraction * (len(stops) - 1)
    lower = min(int(position), len(stops) - 2)
    weight = position - lower
    low, high = _hex_to_rgb(stops[lower]), _hex_to_rgb(stops[lower + 1])
    return "#" + "".join(f"{round(a + (b - a) * weight):02x}" for a, b in zip(low, high))


def _scale_of(property_id):
    scale = LAYER_STYLES.get(property_id, {}).get("scale", DEFAULT_SCALE)
    return scale if scale in COLOR_SCALES else DEFAULT_SCALE


def property_color(layer, value):
    """The colour of ``value`` on the layer's scale across its range."""
    low, high = layer.range
    return scale_color(_scale_of(layer.id), (value - low) / (high - low))


def _format_value(value):
    return f"{round(value, 2):g}"


def _value_text(layer, value):
    text = _format_value(value)
    return f"{text} {layer.unit}" if layer.unit else text


def residue_tooltip(layer, value, column):
    """The tooltip of a residue: the layer's label and value, and its alignment column if known."""
    if column is None:
        return f"{layer.label} {_value_text(layer, value)}"
    return display_text(Message(
        "{label} {value} (MSA column {column})",
        label=layer.label, value=_value_text(layer, value), column=column + 1,
    ))


def layer_annotation(data, property_id):
    """The MolViewSpec annotation rows of one property, or None when it has none."""
    layer = data.properties.get(property_id)
    if layer is None:
        return None
    columns = layer.columns() or [None] * len(layer.values)
    return [
        {
            "auth_asym_id": data.chain,
            "auth_seq_id": data.first + index,
            "color": property_color(layer, value),
            "value": value,
            "tooltip": residue_tooltip(layer, value, columns[index]),
        }
        for index, value in enumerate(layer.values)
        if value is not None
    ]


def _source_text(layer):
    name = _text(layer.source.get("file"))
    if not name:
        return ""
    count = _integer(layer.source.get("sequences"))
    if count is not None and count > 0:
        return display_text(Message("From {file}, computed when folded, using %n sequence(s)", n=count, file=name))
    return display_text(Message("From {file}, computed when folded", file=name))


def _layer_entry(structure_name, data, layer):
    low, high = layer.range
    scale = COLOR_SCALES[_scale_of(layer.id)]
    return {
        "id": layer.id,
        "label": layer.label,
        "unit": layer.unit,
        "description": layer.description,
        "source": _source_text(layer),
        "range": [low, high],
        "legend": {
            "kind": "gradient",
            "min": low,
            "max": high,
            "stops": [
                [low + (high - low) * index / (len(scale) - 1), scale_color(_scale_of(layer.id), index / (len(scale) - 1))]
                for index in range(len(scale))
            ],
        },
        "residues": len(layer.values),
        "covered": layer.covered,
        "annotation": f"{ROUTE_PREFIX}{quote(structure_name, safe='')}/{quote(layer.id, safe='')}?v={data.token}",
    }


def plddt_layer():
    """The confidence layer, which every predicted structure has, in the page's own words."""
    return {
        "id": PLDDT_ID,
        "label": "pLDDT",
        "unit": "",
        "description": display_text(Message(
            "The model's confidence in each residue, stored in the B-factor column of the PDB file."
        )),
        "source": "",
        "legend": {
            "kind": "bands",
            "bands": [{"color": color, "label": display_text(label)} for color, label in PLDDT_BANDS],
        },
    }


def layer_texts():
    """The words the layer controls show besides the layers' own, in the Viewer's language."""
    return {
        "color_by": display_text(Message("Color by")),
        "no_data": display_text(Message("No data")),
    }


def layer_listing(structure_name, structure_path, sidecar_path):
    """What the page needs to offer layers for ``structure_name``, the PDB at ``structure_path``.

    The result always lists pLDDT first. "unavailable" says why no other
    layer is offered, and is empty when at least one is.
    """
    layers = [plddt_layer()]
    unavailable = ""
    try:
        data = read_residue_data(structure_name, structure_path, sidecar_path)
    except ResidueDataError as error:
        unavailable = str(error)
    else:
        layers += [_layer_entry(structure_name, data, layer) for layer in data.properties.values()]
        if not data.properties:
            unavailable = display_text(Message("The per-residue data of this structure holds no usable property."))
    return {
        "structure": structure_name,
        "texts": layer_texts(),
        "background": BACKGROUND_COLOR,
        "layers": layers,
        "unavailable": unavailable,
    }
