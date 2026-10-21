# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Sessions the Fold View page saved with the Mol* it was redesigned around, to keep restoring.

A saved session is a promise to the person who made it, so every Mol* the page
is moved to has to open the ones the page saved before. These were made by
tests/esmfold_session_maker.py with Mol* 5.10.1 (the page of the redesign,
main aa4c1eb) and are frozen in tests/esmfold_saved_session_data.py. There is a
scene for each thing the page can do to a session:

    pair    two folded structures, the second fitted onto the first; the scene
            colored by a layer of the residue data, and one structure's row on
            another layer
    styled  a changed representation and a changed opacity on both structures
    opened  a folded structure and a structure opened from a file, whose
            coordinates the session carries
    marked  a measured distance and a label, and a pinned residue

Each is kept twice, as the page gave it away: the session the page posts to the
Viewer after every change (``session``, what the Viewer stores and hands back
when the page opens) and the file the Session menu downloads (``molj``, Mol*'s
own .molj text). ``expected`` is what each scene looked like (SUMMARY_JS) when
the page that made them restored them.

Not collected by unittest; import with ``from tests.esmfold_saved_sessions import ...``.
"""

import base64
import json
import zlib

from tests import esmfold_layer_fixtures as fixtures
from tests.esmfold_saved_session_data import MOLSTAR, PACKED
from tests.esmfold_session_fixture import write_session_structures

SCENES = ("pair", "styled", "opened", "marked")
# The file a structure was opened from, by the opened scene (and the marked one's tools' targets).
OPENED_NAME = "mine.pdb"
FIRST, SECOND = "protein_a.pdb", "protein_b.pdb"


def opened_structure_pdb():
    """The text of the file the opened scene opened: the first protein, moved and bent a little, with its pLDDT as B-factors."""
    return fixtures.helix_pdb(fixtures.FIRST_SEQUENCE, fixtures.plddt_values(33), shift=(1.0, 2.0, 0.0), wobble=0.3)


def write_saved_structures(directory):
    """The two folded structures the scenes name, written into ``directory`` with their residue data."""
    write_session_structures(directory)


def saved_scene(name):
    """{"session": dict, "molj": str, "expected": dict} of one scene (see SCENES)."""
    return json.loads(zlib.decompress(base64.b64decode("".join(PACKED[name]))))


def pack(scene):
    """What the maker stores of a scene: its JSON, deflated, as base64 lines."""
    text = json.dumps(scene, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
    packed = base64.b64encode(zlib.compress(text, 9)).decode("ascii")
    return [packed[start:start + 100] for start in range(0, len(packed), 100)]


# What a restored scene shows, as plain JSON, read through the page's own functions (esmfold_*.js) and Mol*'s state.
# Needs the harness' HELPERS (colors, fit, cardNow) and loads after the page has finished restoring.
SUMMARY_JS = """
window.sessionSummary = async function () {
    const card = await cardNow();
    const themes = reprThemes();
    const structures = loadedStructures().map((structure, index) => {
        const posed = structure.transform ? structure.transform.cell.transform.params.transform.params.data : null;
        return {
            file: structureFile(structure) || null, name: structureName(structure), folded: isFolded(structure),
            coloring: coloringOf(structure), layer: activeLayerId(structure),
            themes: structure.components.flatMap(component => component.representations.map(
                representation => representation.cell.transform.params.colorTheme.name)),
            representation: representationType(structure), opacity: structureOpacity(structure),
            representations: structure.components.reduce((n, component) => n + component.representations.length, 0),
            matrix: posed ? Array.from(posed) : null,
            rmsd: index && posed ? Math.round(1e4 * fit(index).rmsd) / 1e4 : null,
            colors: colors(index, themes[index].name, themes[index].params)
        };
    });
    const measured = plugin.managers.structure.measurement.state;
    const kinds = {};
    for (const cell of plugin.state.data.cells.values()) {
        const id = cell.transform.transformer.id.replace('ms-plugin.', '');
        kinds[id] = (kinds[id] || 0) + 1;
    }
    return {
        structures, mode: colorMode, card: {selected: card.selected, chips: card.chips.map(chip => chip[0]), legend: card.legendTitle},
        measurements: {distances: measured.distances.length, labels: measured.labels.length},
        focus: {nodes: focusNodes().length, current: !!plugin.managers.structure.focus.current,
                marked: plugin.managers.structure.selection.stats.elementCount, pinned: pinnedResidue ? pinnedResidue.number : null},
        kinds
    };
};
"""
