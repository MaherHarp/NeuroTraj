"""PDB header parsing: the real 3C2I header plus a SYNTHETIC REMARK 470 snippet."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurodna.pdbheader import read_pdb_header

REAL = Path(__file__).parent / "data" / "3C2I.pdb"


def test_3c2i_identity_and_revisions():
    h = read_pdb_header(REAL)
    assert h.idcode == "3C2I" and h.experiment == "X-RAY DIFFRACTION" and h.resolution_A == 2.5
    assert "METHYL-CPG BINDING DOMAIN OF HUMAN MECP2" in h.title and "BDNF" in h.title
    assert h.revisions[0] == (5, "20-NOV-24", "REMARK")
    assert [c.chains for c in h.compounds] == [("A",), ("B",), ("C",)]
    assert h.compound_for_chain("A").fragment == "UNP RESIDUES 77-167, HUMAN MECP2 MBD DOMAIN"


def test_3c2i_sequence_mapping_and_modifications():
    h = read_pdb_header(REAL)
    unp = h.dbrefs[0]
    assert (unp.chain, unp.accession, unp.seq_begin, unp.seq_end, unp.db_begin) == (
        "A", "P51608", 77, 167, 77)
    engineered = [r for r in h.seqadv if "ENGINEERED" in r.detail]
    assert [(r.label(), r.detail) for r in engineered] == [
        ("A:MSE140", "ENGINEERED MUTATION (UNP P51608 ALA140)")]
    assert {r.label() for r in h.modres} == {"A:MSE94", "A:MSE140", "B:5CM8", "C:5CM33"}
    assert h.hetnam["5CM"] == "5-METHYL-2'-DEOXY-CYTIDINE-5'-MONOPHOSPHATE"


def test_3c2i_missing_residues_atoms_and_assembly():
    h = read_pdb_header(REAL)
    missing = [r.resnum for r in h.missing_residues]
    assert missing == list(range(77, 91)) + list(range(163, 174))
    assert h.missing_atoms == ()
    (assembly,) = h.assemblies
    assert assembly.chains == ("A", "B", "C") and assembly.author_unit == "TRIMERIC"
    assert assembly.identity_only


def test_synthetic_missing_atoms_and_multiple_operators(tmp_path):
    # SYNTHETIC header text, not from a real entry.
    text = "\n".join([
        "HEADER    SYNTHETIC TEST                          01-JAN-00   9ZZZ              ",
        "REMARK 350 BIOMOLECULE: 1                                                       ",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, B                                  ",
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000            ",
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000            ",
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000            ",
        "REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000       10.00000            ",
        "REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000            ",
        "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000            ",
        "REMARK 470                                                                      ",
        "REMARK 470 MISSING ATOM                                                         ",
        "REMARK 470   M RES CSSEQI  ATOMS                                                ",
        "REMARK 470     LYS A  97    CG   CD   CE   NZ                                   ",
        "REMARK 470     ARG A 102A   NH1  NH2                                            ",
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  ",
    ]) + "\n"
    path = tmp_path / "synthetic_header.pdb"
    path.write_text(text)
    h = read_pdb_header(path)
    assert [(r.label(), r.atoms) for r in h.missing_atoms] == [
        ("A:LYS97", ("CG", "CD", "CE", "NZ")), ("A:ARG102A", ("NH1", "NH2"))]
    (assembly,) = h.assemblies
    assert len(assembly.operators) == 2 and not assembly.identity_only
    np.testing.assert_allclose(assembly.operators[1][:, 3], [10.0, 0.0, 0.0])
