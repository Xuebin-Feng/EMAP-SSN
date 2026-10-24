# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Long names for the Fold View page's layout tests: the kind a person's own files have.

A protein from a database comes with a header such as
"KAK4446774.1_RlpA-like_double-psi_beta-barrel-protein_domain-containing_protein-containing_protein_Podospora_aff._communis_PSN243_esm3-large-2024-03",
and the alignment it was folded with, a file name of the same kind. Neither has a space to break a line at.
These names and the helper that writes a structure with them are shared by the page tests that lay the text out.

Not collected by unittest; import with ``from tests.esmfold_long_names import ...``.
"""

from tests import esmfold_layer_fixtures as fixtures

# From a person's report: a structure name of 148 characters, and the MSA file name beside it.
USER_NAME = (
    "KAK4446774.1_RlpA-like_double-psi_beta-barrel-protein_domain-containing_protein-containing_protein_"
    "Podospora_aff._communis_PSN243_esm3-large-2024-03"
)
USER_MSA = "Fungal_Expansins_Loosenins_Simplified_(trimmed)_mafft_einsi_alignment_2024-03-18.fasta"
SECOND_NAME = "XP_007399765.1_hypothetical_protein_PHACADRAFT_222222_Phanerochaete_carnosa_HHB-10118-sp_esm3-large-2024-03"
# More than 120 characters each, with nothing to break at but hyphens, underscores and dots.
NAME_120 = (
    "A0A0B4YRD9_expansin-like_protein_EXLX1_Bacillus_subtilis_subsp._subtilis_str._168_cellulose-binding_domain_"
    "containing_protein_v2"
)
MSA_120 = (
    "Fungal_Expansins_Loosenins_Simplified_(trimmed)_mafft_einsi_alignment_with_gaps_removed_at_90pct_"
    "and_duplicates_collapsed_2024-03-18_final_v7.fasta"
)
# The name of a file a person opens (an experimental structure), with the spaces and brackets a file name may have.
OPENED_NAME = "Fungal Expansin (crystal structure) 1.8 A resolution, chain A, PDB 6XYZ refined 2024-03_final_version_2"
# Nearly 150 characters with no hyphen, dot or space: only underscores, which are no place to break a line at, between its words.
UNDERSCORE_NAME = (
    "serine_protease_inhibitor_kazal_type_domain_containing_protein_putative_uncharacterized_protein_isoform_x1_"
    "from_Podospora_aff_communis_strain_psn243"
)
# A log path of the kind a failed fold reports: no spaces either.
LONG_LOG = (
    "C:\\Users\\Someone_With_A_Long_Name\\AppData\\Local\\EMAP-SSN\\Cache_Files\\esmfold_logs\\"
    "KAK4446774.1_RlpA-like_double-psi_beta-barrel-protein_esm3-large-2024-03.fold.log"
)


def write_long(directory, name, sequence, msa, *, label_a="Information Content", label_b="Residue Agreement",
               sequences=3925, **options):
    """Write the structure ``name`` (without .pdb) with the two usual layers, whose source is the MSA file ``msa``.

    ``label_a`` and ``label_b`` name the layers; ``options`` shape the helix as fixtures.write_structure does.
    """
    length = len(sequence)
    source = {
        "kind": "msa", "file": msa, "alignment_id": "sha256:" + "ab" * 32,
        "columns": [100 + 2 * index for index in range(length)], "sequences": sequences,
    }
    properties = {
        "information_content": fixtures.residue_property(
            label_a, "%", [0, 100], fixtures.information_content(length),
            {**source, "measure": "logo bits", "max_bits": 4.3219},
            "How far the alignment column of each residue is from a column of all twenty amino acids at random."),
        "same_residue_pct": fixtures.residue_property(
            label_b, "%", [0, 100], fixtures.same_residue_pct(length), source,
            "Share of the aligned sequences with a residue in this column that have the same amino acid as this protein."),
    }
    return fixtures.write_structure(directory, name + ".pdb", sequence, sidecar=True, properties=properties, **options)
