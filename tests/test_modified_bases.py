"""Modified nucleotides: registry, identity checks, sequences and marker contacts."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from conftest import DNA, PROTEIN, synthetic_atoms

from neurodna import (
    AmbiguousResidueError,
    Complex,
    ResidueSpec,
    UnsupportedResidueError,
    default_dna_registry,
)


def test_methylated_cytosine_is_identified(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    table = cx.dna_residues().set_index("label")
    assert table.loc["C:5CM2", "parent"] == "DC"
    assert table.loc["C:5CM2", "one_letter"] == "C"
    assert table.loc["C:5CM2", "modification"] == "5mC"
    assert pd.isna(table.loc["C:DC1", "modification"])  # missing = unmodified
    assert table.loc["C:5CM2", "n_atoms"] == 6


def test_strand_sequence_keeps_the_modification(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    (strand,) = cx.dna_strands()
    assert strand.sequence == "CCG"
    assert strand.annotated_sequence == "C[5mC]G"
    assert not strand.circular


def test_mcpg_step_is_found(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    steps = cx.cpg_steps()
    assert len(steps) == 1
    step = steps.iloc[0]
    assert (step.c_resname, step.c_resnum, step.g_resname, step.g_resnum) == ("5CM", 2, "DG", 3)
    assert step.c_modification == "5mC"


def test_strands_follow_covalent_linkage_not_file_order(write_pdb):
    # Write the residues in reverse order; linkage geometry still defines 5'->3'.
    atoms = synthetic_atoms()
    protein = [a for a in atoms if a.chain != "C"]
    dna = [a for a in atoms if a.chain == "C"]
    dna_reversed = sorted(dna, key=lambda a: -a.resnum)
    cx = Complex.load(write_pdb(protein + dna_reversed), protein=PROTEIN, dna=DNA)
    (strand,) = cx.dna_strands()
    assert strand.annotated_sequence == "C[5mC]G"


def test_marker_atom_contacts_isolate_the_methyl_group(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    df = cx.contacts(cutoff=4.5, target="modification_markers", level="atom")
    assert list(zip(df.protein_label, df.protein_atom, df.dna_atom, df.distance_A.round(3))) == [
        ("A:ARG1", "NH1", "C5A", 3.0),
        ("A:ARG1", "CZ", "C5A", 4.3),
    ]
    assert set(df.dna_modification) == {"5mC"}
    assert set(df.dna_moiety) == {"base"}


def test_unregistered_modified_base_is_reported_not_dropped(write_pdb):
    atoms = [replace(a, resname="5HC") if a.resname == "5CM" else a for a in synthetic_atoms()]
    with pytest.raises(UnsupportedResidueError, match=r"\['5HC'\].*C:5HC2") as info:
        Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)
    assert [k.resname for k in info.value.residues] == ["5HC"]


def test_modified_base_can_be_registered(write_pdb):
    atoms = [replace(a, resname="5HC") if a.resname == "5CM" else a for a in synthetic_atoms()]
    registry = default_dna_registry()
    registry.register(
        ResidueSpec("5HC", parent="DC", one_letter="C", modification="5hmC", marker_atoms=("C5A",))
    )
    cx = Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA, dna_registry=registry)
    assert cx.dna_strands()[0].annotated_sequence == "C[5hmC]G"
    assert "5HC" not in default_dna_registry()  # defaults are fresh copies


def test_modified_base_without_marker_atom_is_ambiguous(write_pdb):
    atoms = [a for a in synthetic_atoms() if a.name != "C5A"]
    with pytest.raises(AmbiguousResidueError, match=r"C:5CM2 lacks \['C5A'\]"):
        Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)


def test_marker_atoms_can_be_renamed_for_other_force_fields(write_pdb):
    atoms = [replace(a, name="C5M") if a.name == "C5A" else a for a in synthetic_atoms()]
    registry = default_dna_registry()
    spec = ResidueSpec("5CM", parent="DC", one_letter="C", modification="5mC", marker_atoms=("C5M",))
    with pytest.raises(ValueError, match="replace=True"):
        registry.register(spec)
    registry.register(spec, replace=True)
    cx = Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA, dna_registry=registry)
    df = cx.contacts(cutoff=4.0, target="modification_markers")
    assert list(df.dna_atom) == ["C5M"]


def test_rna_in_dna_selection_is_rejected(write_pdb):
    atoms = synthetic_atoms()
    c1 = next(a for a in atoms if a.resname == "DG" and a.name == "C1'")
    atoms.append(replace(c1, name="O2'", y=c1.y + 1.4, element="O"))
    with pytest.raises(UnsupportedResidueError, match="look like RNA.*C:DG3"):
        Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)


def test_modified_spec_requires_marker_atoms():
    with pytest.raises(ValueError, match="marker atom"):
        ResidueSpec("XMC", parent="DC", one_letter="C", modification="5mC")
