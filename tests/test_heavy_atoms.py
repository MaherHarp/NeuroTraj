"""Heavy-atom identification and its use in contacts (SYNTHETIC fixtures)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import DNA, PROTEIN, Atom, synthetic_atoms

from neurodna import AmbiguousElementError, Complex
from neurodna.elements import classify_by_name, classify_heavy_atoms

# MDAnalysis warns about the deliberately blank element columns in these fixtures.
expected_element_warnings = pytest.mark.filterwarnings(
    "ignore:.*[Ee]lement.*:UserWarning", "ignore:Unknown masses:PendingDeprecationWarning"
)


def with_hydrogen(atoms: list[Atom], hydrogen: Atom) -> list[Atom]:
    """Insert ``hydrogen`` right after the last atom of its residue (keeps residues contiguous)."""
    last = max(i for i, a in enumerate(atoms) if (a.chain, a.resnum) == (hydrogen.chain, hydrogen.resnum))
    return atoms[: last + 1] + [hydrogen] + atoms[last + 1 :]


def test_hydrogens_never_make_contacts(write_pdb):
    # SYNTHETIC: a hydrogen on B:ARG1 sits 2.0 Å from C:DG3 N1, whose nearest heavy atom
    # on B:ARG1 is > 10 Å away. The contact must not appear.
    h = Atom("HH11", "ARG", "B", 1, 14.0, 6.0, element="H")  # DG3 N1 is at (14, 4, 0)
    cx = Complex.load(write_pdb(with_hydrogen(synthetic_atoms(), h)), protein=PROTEIN, dna=DNA)
    assert "HH11" in set(cx.protein.names) and "HH11" not in set(cx.protein_heavy.names)
    pairs = set(zip(cx.contacts(cutoff=4.5).protein_label, cx.contacts(cutoff=4.5).dna_label))
    assert ("B:ARG1", "C:DG3") not in pairs
    table = cx.protein_residues().set_index("label")
    assert (table.loc["B:ARG1", "n_atoms"], table.loc["B:ARG1", "n_heavy_atoms"]) == (4, 3)


@expected_element_warnings
def test_blank_element_falls_back_to_standard_name(write_pdb):
    # SYNTHETIC: blank element columns everywhere; names alone identify heavy atoms.
    atoms = [replace(a, element=" ") for a in synthetic_atoms()]
    path = write_pdb(atoms)
    cx = Complex.load(path, protein=PROTEIN, dna=DNA)
    assert len(cx.dna_heavy) == len(cx.dna)
    assert cx.contacts(cutoff=4.0).min_distance_A.round(3).tolist() == [3.0, 3.5]


def test_element_contradicting_name_is_rejected(write_pdb):
    # SYNTHETIC: a cysteine 'HG' (hydrogen) whose element column says Hg (mercury).
    atoms = synthetic_atoms() + [
        Atom("SG", "CYS", "A", 70, 40.0, 40.0, element="S"),
        Atom("HG", "CYS", "A", 70, 41.3, 40.0, element="HG"),
    ]
    with pytest.raises(AmbiguousElementError, match=r"A:CYS70 HG: element 'Hg' contradicts") as info:
        Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)
    assert len(info.value.atoms) == 1


@expected_element_warnings
def test_unclassifiable_atom_without_element_is_rejected(write_pdb):
    # SYNTHETIC: a virtual-site-like atom 'MW' with no element in a protein residue.
    atoms = synthetic_atoms()
    atoms = with_hydrogen(atoms, Atom("MW", "ARG", "A", 1, 8.0, 20.0, element=" "))
    with pytest.raises(AmbiguousElementError, match="A:ARG1 MW: no element"):
        Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("CA", "heavy"), ("C5A", "heavy"), ("OP1", "heavy"), ("SE", "heavy"), ("O3'", "heavy"),
     ("H", "hydrogen"), ("HH11", "hydrogen"), ("1HG2", "hydrogen"), ("H5''", "hydrogen"),
     ("MW", None), ("ZN", None), ("", None)],
)
def test_classify_by_name(name, expected):
    assert classify_by_name(name) == expected


def test_classify_heavy_atoms_cases():
    names = ["CA", "HG", "HG", "SE", "D1", "CB", "LP1"]
    elements = ["C", "H", "Hg", "Se", "D", "Xx", ""]
    heavy, problems = classify_heavy_atoms(names, elements)
    assert heavy[[0, 3]].all() and not heavy[[1, 4]].any()  # D (deuterium) is a hydrogen
    assert [i for i, _ in problems] == [2, 5, 6]  # Hg vs HG, invalid 'Xx', unclassifiable LP1
