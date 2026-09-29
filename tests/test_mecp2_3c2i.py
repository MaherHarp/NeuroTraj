"""Regression test on a real structure: PDB 3C2I, human MeCP2 MBD bound to
methylated BDNF promoter DNA (Ho et al., Mol. Cell 2008). PDB data are CC0.

Published recognition mode: Arg111 and Arg133 contact the 5-methyl groups of the
two 5mC bases of the symmetrically methylated CpG, one on each strand.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from neurodna import Complex, IncompleteSelectionError

PDB = Path(__file__).parent / "data" / "3C2I.pdb"
DNA = "chainID B C and not resname HOH"


@pytest.fixture(scope="module")
def mecp2() -> Complex:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MDAnalysis notes about PDB header records
        return Complex.load(PDB, protein="protein", dna=DNA)


def test_nucleic_keyword_would_drop_both_methylcytosines():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(IncompleteSelectionError, match=r"B:5CM8, C:5CM33"):
            Complex.load(PDB, protein="protein", dna="nucleic")


def test_strands_and_symmetric_mcpg(mecp2):
    strands = {s.residues[0].chain: s.annotated_sequence for s in mecp2.dna_strands()}
    assert strands == {"B": "TCTGGAA[5mC]GGAATTCTTCTA", "C": "ATAGAAGAATTC[5mC]GTTCCAG"}
    steps = mecp2.cpg_steps()
    assert list(zip(steps.c_chain, steps.c_resnum)) == [("B", 8), ("C", 33)]
    assert set(steps.c_modification) == {"5mC"}


def test_arg111_and_arg133_read_the_methyl_groups(mecp2):
    m = mecp2.contacts(cutoff=4.0, target="modification_markers")
    pairs = dict(zip(m.protein_label, m.dna_label))
    assert pairs["A:ARG111"] == "B:5CM8"
    assert pairs["A:ARG133"] == "C:5CM33"
    closest = m.set_index("protein_label").min_distance_A
    assert closest["A:ARG111"] == pytest.approx(3.57, abs=0.01)
    assert closest["A:ARG133"] == pytest.approx(3.43, abs=0.01)


def test_crystal_cell_is_not_used_as_periodic_box(mecp2):
    # 3C2I's CRYST1 is space group C 1 2 1: a crystal lattice, not an MD box.
    assert "C 1 2 1" in mecp2.periodic_policy
    df = mecp2.contacts()  # default: minimum heavy-atom distance <= 4.5 Å
    assert df.attrs["minimum_image"] is False and df.attrs["cutoff_A"] == 4.5
    assert {"A:ARG111", "A:ARG133"} <= set(df.protein_label)


def test_heavy_atoms_of_the_real_structure(mecp2):
    # Crystal structure without hydrogens: every selected atom is heavy, including
    # selenium in selenomethionine (MSE).
    assert len(mecp2.protein_heavy) == len(mecp2.protein)
    assert "SE" in set(mecp2.protein_heavy.names)
