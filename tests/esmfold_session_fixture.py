# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A Mol* session as the Fold View page saved it before the page was redesigned.

Sessions the page saved (molstar_session.json in each layout cache folder) must
keep loading, so this one is frozen: it was made by esmfold.html at 0bcb3e1, the
last page that drew Mol*'s own interface, over two structures of
tests/esmfold_layer_fixtures.py (protein_a.pdb, and protein_b.pdb shifted and
bent a little), each coloured by pLDDT, the second one superposed on the first.
It has no camera, because the page that made it ran without WebGL.

Not collected by unittest; import with ``from tests.esmfold_session_fixture import ...``.
"""

import base64
import json
import zlib

from tests import esmfold_layer_fixtures as fixtures

FIRST = "protein_a.pdb"
SECOND = "protein_b.pdb"

_PACKED = (
    "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"
)


def session_before_the_redesign():
    """The saved session: the dict a page gives plugin.state.setSnapshot."""
    return json.loads(zlib.decompress(base64.b64decode(_PACKED)))


def write_session_structures(directory):
    """Write the two structures the session names into ``directory``, as they were when it was saved."""
    fixtures.write_structure(directory, FIRST, fixtures.FIRST_SEQUENCE)
    fixtures.write_structure(
        directory, SECOND, fixtures.SECOND_SEQUENCE, shift=(14.0, 6.0, -3.0), wobble=0.3, offset=20,
        plddt=fixtures.plddt_values(33, 17),
    )
