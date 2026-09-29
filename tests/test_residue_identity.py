"""Residue identity: duplicate numbers across chains, insertion codes, ambiguity."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import DNA, PROTEIN, Atom, synthetic_atoms

from neurodna import AmbiguousResidueError, Complex, ResidueKey


def test_same_residue_number_in_two_chains_stays_distinct(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    arg_keys = [k for k in cx.protein_residue_keys if k.resname == "ARG"]
    assert arg_keys == [
        ResidueKey(segid="A", chain="A", resnum=1, icode="", resname="ARG"),
        ResidueKey(segid="B", chain="B", resnum=1, icode="", resname="ARG"),
    ]


def test_contacts_attribute_duplicate_numbers_to_the_right_chain(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    df = cx.contacts(cutoff=4.0)
    got = {(r.protein_label, r.dna_label): round(r.min_distance_A, 3) for r in df.itertuples()}
    # Both arginines are residue 1; only the chain tells them apart.
    assert got == {("A:ARG1", "C:5CM2"): 3.0, ("B:ARG1", "C:DC1"): 3.5}
    assert set(df["protein_resnum"]) == {1}
    assert set(df["protein_chain"]) == {"A", "B"}


def test_insertion_codes_are_preserved(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    table = cx.protein_residues()
    res52 = table[table["resnum"] == 52]
    assert list(zip(res52["resname"], res52["icode"])) == [("GLY", ""), ("SER", "A")]
    assert list(res52["label"]) == ["A:GLY52", "A:SER52A"]


def test_repeated_residue_identifier_in_one_chain_is_ambiguous(write_pdb):
    # A:ARG1 appears twice (non-contiguously): residue numbers are not unique.
    extra = [replace(a, x=a.x + 40) for a in synthetic_atoms() if (a.chain, a.resnum) == ("A", 1)]
    atoms = synthetic_atoms()
    atoms[5:5] = extra  # after A:GLY52 / A:SER52A
    path = write_pdb(atoms)
    with pytest.raises(AmbiguousResidueError, match="more than once") as info:
        Complex.load(path, protein=PROTEIN, dna=DNA)
    assert {k.label() for k in info.value.residues} == {"A:ARG1"}


def test_chains_merged_by_shared_segid_are_reported(write_pdb):
    # With an explicit segid, MDAnalysis groups consecutive atoms by (segid, resnum), so
    # adjacent A:ARG1 and B:ARG1 become ONE residue. That must be reported, not hidden.
    atoms = [replace(a, segid="PROT") if a.chain in "AB" else a for a in synthetic_atoms()]
    arg_b = [a for a in atoms if (a.chain, a.resnum) == ("B", 1)]
    atoms = [a for a in atoms if a not in arg_b]
    atoms[3:3] = arg_b  # directly after A:ARG1
    path = write_pdb(atoms)
    with pytest.raises(AmbiguousResidueError, match=r"several chains \['A', 'B'\]"):
        Complex.load(path, protein=PROTEIN, dna=DNA)


def test_segid_and_chain_are_both_kept(write_pdb):
    atoms = [replace(a, segid="DNA1") if a.chain == "C" else a for a in synthetic_atoms()]
    path = write_pdb(atoms)
    cx = Complex.load(path, protein=PROTEIN, dna="segid DNA1")
    assert cx.dna_residue_keys[0] == ResidueKey("DNA1", "C", 1, "", "DC")
    assert cx.dna_residue_keys[0].label() == "DNA1/C:DC1"


def test_alternate_locations_are_ambiguous_until_one_is_chosen(write_pdb):
    atoms = synthetic_atoms()
    ca = next(a for a in atoms if (a.chain, a.resnum, a.name) == ("B", 1, "CA"))
    i = atoms.index(ca)
    atoms[i:i + 1] = [replace(ca, altloc="A"), replace(ca, altloc="B", x=ca.x + 0.4)]
    path = write_pdb(atoms)
    with pytest.raises(AmbiguousResidueError, match="alternate locations.*B:ARG1"):
        Complex.load(path, protein=PROTEIN, dna=DNA)
    cx = Complex.load(path, protein="protein and not altloc B", dna=DNA)
    assert len(cx.protein) == sum(a.chain != "C" for a in atoms) - 1  # altloc B dropped


def test_residue_key_label_and_ordering():
    k1 = ResidueKey("", "A", 52, "", "GLY")
    k2 = ResidueKey("", "A", 52, "A", "SER")
    assert k1.label() == "A:GLY52" and str(k2) == "A:SER52A"
    assert k1 != k2 and k1.location != k2.location
    assert sorted([k2, k1]) == [k1, k2]


def test_pdb_fixture_uses_standard_columns():
    # Guard the fixture itself: the PDB writer must put names in the standard columns.
    line = Atom("C5A", "5CM", "C", 2, 8.0, 6.0, element="C").pdb_line(1)
    assert line[12:16] == " C5A" and line[17:20] == "5CM" and line[21] == "C"
    assert int(line[22:26]) == 2 and line[76:78] == " C"
