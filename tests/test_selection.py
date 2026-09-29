"""Selection validation: empty, invalid, overlapping and incomplete selections."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import DNA, PROTEIN, synthetic_atoms

from neurodna import (
    Complex,
    EmptySelectionError,
    IncompleteSelectionError,
    OverlappingSelectionError,
    SelectionError,
    UnsupportedResidueError,
)


def test_valid_selections_load(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    assert len(cx.protein_residue_keys) == 4
    assert [k.label() for k in cx.dna_residue_keys] == ["C:DC1", "C:5CM2", "C:DG3"]


@pytest.mark.parametrize("which", ["protein", "dna"])
def test_empty_selection_is_rejected(synthetic_pdb, which):
    kwargs = {"protein": PROTEIN, "dna": DNA, which: "chainID Z"}
    with pytest.raises(EmptySelectionError, match="matched no atoms"):
        Complex.load(synthetic_pdb, **kwargs)


def test_blank_selection_is_rejected(synthetic_pdb):
    with pytest.raises(SelectionError, match="non-empty"):
        Complex.load(synthetic_pdb, protein="  ", dna=DNA)


def test_invalid_selection_syntax_is_wrapped(synthetic_pdb):
    with pytest.raises(SelectionError, match="invalid protein selection"):
        Complex.load(synthetic_pdb, protein="resname (", dna=DNA)


def test_overlapping_selections_are_rejected(synthetic_pdb):
    with pytest.raises(OverlappingSelectionError, match=r"C:DC1.*C:5CM2.*C:DG3"):
        Complex.load(synthetic_pdb, protein="all", dna=DNA)


def test_partial_residue_overlap_is_rejected(synthetic_pdb):
    # Disjoint atoms, but residue C:DC1 is split between the two selections.
    with pytest.raises(OverlappingSelectionError, match="C:DC1"):
        Complex.load(synthetic_pdb, protein="protein or (chainID C and resnum 1 and name P)",
                     dna="chainID C and not (resnum 1 and name P)")


def test_nucleic_keyword_does_not_silently_drop_methylated_cytosine(synthetic_pdb):
    # MDAnalysis' 'nucleic' keyword does not match 5CM. neurodna must notice.
    with pytest.raises(IncompleteSelectionError, match="C:5CM2"):
        Complex.load(synthetic_pdb, protein=PROTEIN, dna="nucleic")


def test_partial_dna_can_be_allowed_explicitly(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna="nucleic", allow_partial_dna=True)
    assert [k.resname for k in cx.dna_residue_keys] == ["DC", "DG"]


def test_unknown_protein_residue_is_reported(write_pdb):
    atoms = synthetic_atoms() + [
        replace(a, resname="XYZ", resnum=99) for a in synthetic_atoms()[:1]
    ]
    path = write_pdb(atoms)
    with pytest.raises(UnsupportedResidueError, match=r"XYZ.*A:XYZ99") as info:
        Complex.load(path, protein="chainID A", dna=DNA)
    assert [k.label() for k in info.value.residues] == ["A:XYZ99"]
