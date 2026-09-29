"""Contact geometry against hand-computed distances, plus the plotting smoke test."""

from __future__ import annotations

import matplotlib
import pytest
from conftest import DNA, PROTEIN

from neurodna import Complex, dna_moiety, protein_moiety

matplotlib.use("Agg")


@pytest.fixture
def cx(synthetic_pdb):
    return Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)


def test_residue_contacts_report_closest_atoms(cx):
    df = cx.contacts(cutoff=4.5).set_index(["protein_label", "dna_label"])
    a = df.loc[("A:ARG1", "C:5CM2")]
    assert (a.protein_atom, a.dna_atom, a.n_atom_pairs) == ("NH1", "C5A", 2)  # NH1 3.0, CZ 4.3
    assert a.min_distance_A == pytest.approx(3.0)
    assert a.dna_modification == "5mC"
    b = df.loc[("B:ARG1", "C:DC1")]
    assert (b.protein_atom, b.dna_atom, b.n_atom_pairs) == ("NH1", "P", 1)
    assert df.attrs["units"] == {"distance": "angstrom", "time": "ps"}
    assert "geometric" in df.attrs["contact_definition"]


def test_cutoff_is_inclusive_and_monotonic(cx):
    assert len(cx.contacts(cutoff=2.9)) == 0
    assert len(cx.contacts(cutoff=3.0)) == 1  # exactly 3.0 Å is a contact
    assert len(cx.contacts(cutoff=3.5)) == 2
    with pytest.raises(ValueError, match="positive"):
        cx.contacts(cutoff=0)


def test_atom_level_contacts_classify_moieties(cx):
    df = cx.contacts(cutoff=4.75, level="atom")  # B:ARG1 CZ–P is 4.8 Å
    b = df[df.protein_label == "B:ARG1"]
    assert list(zip(b.dna_atom, b.dna_moiety, b.distance_A.round(2))) == [
        ("P", "phosphate", 3.5),
        ("OP1", "phosphate", 4.7),
    ]
    assert set(df.protein_moiety) == {"sidechain"}


def test_moiety_helpers():
    assert dna_moiety("OP2") == dna_moiety("O1P") == dna_moiety("O3'") == "phosphate"
    assert dna_moiety("C1'") == dna_moiety("C1*") == dna_moiety("H2''") == "sugar"
    assert dna_moiety("N7") == dna_moiety("C5A") == "base"
    assert protein_moiety("CA") == protein_moiety("O") == "backbone"
    assert protein_moiety("NH1") == "sidechain"


def test_empty_contact_table_keeps_columns(cx):
    df = cx.contacts(cutoff=1.0)
    assert df.empty
    assert {"protein_label", "dna_label", "min_distance_A"} <= set(df.columns)


def test_plot_contact_occupancy_smoke(write_pdb):
    from conftest import moved, synthetic_atoms

    from neurodna.plotting import plot_contact_occupancy

    atoms = synthetic_atoms()
    path = write_pdb([atoms, moved(atoms, "B", 1, -10.0)])
    freq = Complex.load(path, protein=PROTEIN, dna=DNA).ensemble_contact_frequency()
    ax = plot_contact_occupancy(freq)
    assert len(ax.patches) == 2
    assert "models" in ax.get_xlabel()
    assert ax.get_title() == "Geometric protein–DNA contacts (min. heavy-atom distance ≤ 4.5 Å)"
