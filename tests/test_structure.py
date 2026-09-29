"""Backbone RMSD, RMSF, superposition and whole-molecule checks (SYNTHETIC fixtures).

The protein here is a SYNTHETIC 4-residue poly-Ala zig-zag (conftest.synthetic_peptide)
with a far-away synthetic nucleotide as the DNA partner.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import coords, far_nucleotide, memory_universe, random_rotation, synthetic_peptide
from MDAnalysis.analysis.rms import rmsd as mda_rmsd

from neurodna import (
    BACKBONE_SELECTION,
    Complex,
    DynamicsUnavailableError,
    EmptySelectionError,
    FrameKind,
    SelectionError,
    WrappedStructureError,
)
from neurodna.structure import kabsch, rmsd, superpose

PROTEIN = "protein"
DNA = "chainID D"
TRAJ = FrameKind.TRAJECTORY


def system(write_pdb, chains=("A",), drop=()):
    """SYNTHETIC peptide chain(s) plus a distant nucleotide. Returns (pdb, atom list)."""
    atoms = []
    for i, chain in enumerate(chains):
        atoms += [a for a in synthetic_peptide(chain, y0=8.0 * i) if (a.chain, a.resnum) not in drop]
    atoms += far_nucleotide()
    return write_pdb(atoms), atoms


def index(atoms, chain, resnum=None, name=None):
    return np.array([i for i, a in enumerate(atoms) if a.chain == chain
                     and (resnum is None or a.resnum == resnum)
                     and (name is None or a.name == name)])


def rigid(x, rng):
    return x @ random_rotation(rng).T + rng.uniform(-50, 50, size=3)


def trajectory(path, frames, **kwargs):
    return Complex(memory_universe(path, frames, **kwargs), protein=PROTEIN, dna=DNA,
                   frame_kind=TRAJ)


# ------------------------------------------------------------------- invariance


def test_aligned_rmsd_is_invariant_to_rigid_motion(write_pdb):
    path, atoms = system(write_pdb)
    base = coords(atoms).astype(float)
    rng = np.random.default_rng(7)
    prot = index(atoms, "A")
    deformed = [base]
    for _ in range(4):
        x = base.copy()
        x[prot] += rng.normal(scale=0.4, size=(len(prot), 3))
        deformed.append(x)
    moved = [rigid(x, rng) for x in deformed]  # whole system rotated and translated

    ra = trajectory(path, deformed).backbone_rmsd().rmsd_A.to_numpy()
    rb = trajectory(path, moved).backbone_rmsd().rmsd_A.to_numpy()
    assert ra[0] == pytest.approx(0.0, abs=1e-5)
    assert ra[1:].min() > 0.1
    np.testing.assert_allclose(rb, ra, atol=1e-4)

    # Cross-check against MDAnalysis' independent (QCP) implementation.
    bb = np.array([i for i in prot if atoms[i].name in ("N", "CA", "C", "O")])
    for k, x in enumerate(moved):
        expected = mda_rmsd(x[bb], moved[0][bb], center=True, superposition=True)
        assert rb[k] == pytest.approx(expected, abs=1e-4)


def test_pure_rigid_copies_have_zero_rmsd(write_pdb):
    path, atoms = system(write_pdb)
    rng = np.random.default_rng(1)
    base = coords(atoms).astype(float)
    df = trajectory(path, [base] + [rigid(base, rng) for _ in range(3)]).backbone_rmsd()
    assert df.rmsd_A.abs().max() < 1e-4
    assert list(df.columns) == ["frame", "time_ps", "rmsd_A"]
    assert df.time_ps.tolist() == [0.0, 10.0, 20.0, 30.0]


def test_rmsf_of_a_known_oscillation(write_pdb):
    # A:ALA2 CB oscillates +-1 Å along z in the molecular frame; the backbone is rigid.
    # Every frame is also rigidly moved, which alignment on the backbone must undo.
    path, atoms = system(write_pdb)
    base = coords(atoms).astype(float)
    cb2 = index(atoms, "A", 2, "CB")
    rng = np.random.default_rng(3)
    frames = []
    for k in range(8):
        x = base.copy()
        x[cb2, 2] += 1.0 if k % 2 == 0 else -1.0
        frames.append(rigid(x, rng))
    cx = trajectory(path, frames)
    df = cx.rmsf()
    by_atom = df.set_index(["label", "atom_name"]).rmsf_A
    assert by_atom[("A:ALA2", "CB")] == pytest.approx(1.0, abs=1e-4)
    assert by_atom.drop(("A:ALA2", "CB")).max() < 1e-4
    assert df.attrs["alignment_selection"] == BACKBONE_SELECTION
    assert df.attrs["n_alignment_atoms"] == 16 and df.attrs["n_measured_atoms"] == 20

    # Including the moving atom in the alignment changes the answer: the selection matters.
    all_fit = cx.rmsf(align="all").set_index(["label", "atom_name"]).rmsf_A
    assert all_fit[("A:ALA2", "CB")] < 0.99 and all_fit.drop(("A:ALA2", "CB")).max() > 0.01


def test_alignment_never_modifies_coordinates_or_distances(write_pdb):
    path, atoms = system(write_pdb)
    rng = np.random.default_rng(5)
    base = coords(atoms).astype(float)
    cx = trajectory(path, [rigid(base, rng) for _ in range(4)])
    before = [ts.positions.copy() for ts in cx.universe.trajectory]
    distances_before = cx.min_distances(max_distance=None)
    cx.backbone_rmsd()
    cx.rmsf()
    after = [ts.positions.copy() for ts in cx.universe.trajectory]
    for b, a in zip(before, after):
        np.testing.assert_array_equal(a, b)
    pd.testing.assert_frame_equal(cx.min_distances(max_distance=None), distances_before)
    # Rigid motion of the whole system leaves every residue-nucleotide distance unchanged.
    spread = distances_before.groupby(["protein_label", "dna_label"]).min_distance_A.agg(np.ptp)
    assert spread.max() < 1e-3


# ------------------------------------------------------------ selection rules


def test_alignment_selection_is_scoped_to_protein_heavy_atoms(write_pdb):
    path, atoms = system(write_pdb)
    base = coords(atoms)
    cx = trajectory(path, [base, base])
    with pytest.raises(EmptySelectionError, match="no protein heavy atoms"):
        cx.backbone_rmsd(align="resname DC")  # DNA cannot be picked up
    with pytest.raises(SelectionError, match="at least 3"):
        cx.backbone_rmsd(align="name N and resnum 1")
    with pytest.raises(ValueError, match="collinear"):
        cx.backbone_rmsd(align="name CA")  # the synthetic CA atoms lie on a line


def test_rmsd_for_ensembles_but_rmsf_only_for_trajectories(write_pdb):
    _, atoms = system(write_pdb)
    rng = np.random.default_rng(9)
    base = coords(atoms).astype(float)
    model_atoms = [[replace(a, x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for a, p in zip(atoms, m)] for m in (base, rigid(base, rng))]
    cx = Complex.load(write_pdb(model_atoms), protein=PROTEIN, dna=DNA)
    assert cx.kind is FrameKind.ENSEMBLE
    df = cx.backbone_rmsd()
    assert df.time_ps.isna().all() and df.rmsd_A.max() < 2e-3  # PDB 3-decimal precision
    with pytest.raises(DynamicsUnavailableError, match="unordered ensemble"):
        cx.rmsf()
    static = Complex.load(write_pdb(atoms), protein=PROTEIN, dna=DNA)
    with pytest.raises(DynamicsUnavailableError):
        static.backbone_rmsd()


def test_rmsf_rows_keep_chain_identity(write_pdb):
    path, atoms = system(write_pdb, chains=("A", "B"))  # both chains number residues 1-4
    rng = np.random.default_rng(2)
    base = coords(atoms).astype(float)
    df = trajectory(path, [rigid(base, rng) for _ in range(3)]).rmsf()
    assert len(df) == 2 * 4 * 5
    assert not df.duplicated(["chain", "resnum", "atom_name"]).any()
    assert set(df[df.resnum == 1].label) == {"A:ALA1", "B:ALA1"}


# ---------------------------------------------------------- whole molecules


def test_molecule_split_across_boundary_is_rejected(write_pdb):
    path, atoms = system(write_pdb)
    box = np.array([60, 60, 60, 90, 90, 90], dtype=np.float32)
    base = coords(atoms)
    wrapped = base.copy()
    wrapped[np.r_[index(atoms, "A", 3), index(atoms, "A", 4)], 0] += 60.0  # residues 3-4 wrapped
    cx = trajectory(path, [base, wrapped], dimensions=box)
    with pytest.raises(WrappedStructureError, match=r"frame 1: .*split across the periodic "
                       r"boundary: A:ALA2:C-A:ALA3:N.*gmx trjconv"):
        cx.backbone_rmsd()
    # Distances are unaffected: they use the minimum image, not alignment.
    d = cx.min_distances(max_distance=None)
    per_frame = d.pivot_table(index=["protein_label", "dna_label"], columns="frame",
                              values="min_distance_A")
    np.testing.assert_allclose(per_frame[1], per_frame[0], atol=1e-4)


def test_broken_bond_without_box_is_rejected(write_pdb):
    path, atoms = system(write_pdb)
    base = coords(atoms)
    broken = base.copy()
    broken[np.r_[index(atoms, "A", 3), index(atoms, "A", 4)], 0] += 10.0
    with pytest.raises(WrappedStructureError, match="normal in the reference frame"):
        trajectory(path, [base, broken]).backbone_rmsd()


def test_real_chain_gap_is_not_an_error(write_pdb):
    path, atoms = system(write_pdb, drop=[("A", 3)])  # residue 3 missing
    rng = np.random.default_rng(4)
    base = coords(atoms).astype(float)
    df = trajectory(path, [base, rigid(base, rng)]).backbone_rmsd()
    assert df.rmsd_A.max() < 1e-4
    assert df.attrs["chain_gaps"] == ["A:ALA2:C-A:ALA4:N"]


def test_wrapped_side_chain_is_caught_without_topology_bonds(write_pdb):
    path, atoms = system(write_pdb)
    box = np.array([60, 60, 60, 90, 90, 90], dtype=np.float32)
    base = coords(atoms)
    wrapped = base.copy()
    wrapped[index(atoms, "A", 2, "CB"), 0] += 60.0
    cx = trajectory(path, [base, wrapped], dimensions=box)
    cx.backbone_rmsd()  # backbone atoms are whole: fine
    with pytest.raises(WrappedStructureError, match="A:ALA2:CA-A:ALA2:CB"):
        cx.rmsf()  # CB is measured, and it is split from its residue


def test_topology_bonds_are_used_when_present(write_pdb):
    path, atoms = system(write_pdb)
    base = coords(atoms)
    stretched = base.copy()
    stretched[index(atoms, "A", 2, "CB"), 2] += 2.5  # CA-CB 1.5 Å -> ~3.9 Å
    u = memory_universe(path, [base, stretched])
    cx = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=TRAJ)
    cx.rmsf()  # backbone bonds + residue extent do not notice a 3.9 Å CA-CB
    ca, cb = index(atoms, "A", 2, "CA")[0], index(atoms, "A", 2, "CB")[0]
    u.add_TopologyAttr("bonds", [(int(ca), int(cb))])
    with pytest.raises(WrappedStructureError, match="A:ALA2:CA-A:ALA2:CB"):
        Complex(u, protein=PROTEIN, dna=DNA, frame_kind=TRAJ).rmsf()


def test_chains_in_different_periodic_images_are_rejected(write_pdb):
    path, atoms = system(write_pdb, chains=("A", "B"))
    box = np.array([60, 60, 60, 90, 90, 90], dtype=np.float32)
    base = coords(atoms)
    jumped = base.copy()
    jumped[index(atoms, "B"), 0] += 60.0  # chain B whole, but one box length away
    cx = trajectory(path, [base, jumped], dimensions=box)
    with pytest.raises(WrappedStructureError, match=r"chain\(s\) \['B'\].*different periodic"):
        cx.backbone_rmsd()


# ------------------------------------------------------------------- numerics


def test_kabsch_recovers_a_rotation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    r = random_rotation(rng)
    y = x @ r.T + np.array([3.0, -2.0, 7.0])
    rotation, pc, qc = kabsch(x, y)
    np.testing.assert_allclose(rotation, r, atol=1e-10)
    assert rmsd(superpose(x, rotation, pc, qc), y) < 1e-10


def test_kabsch_never_reflects():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 3))
    mirror = x * np.array([-1.0, 1.0, 1.0])
    rotation, pc, qc = kabsch(x, mirror)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rmsd(superpose(x, rotation, pc, qc), mirror) > 0.1  # a mirror image cannot be fitted


def test_kabsch_rejects_degenerate_input():
    line = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    with pytest.raises(ValueError, match="collinear"):
        kabsch(line, line)
    with pytest.raises(ValueError, match="at least 3"):
        kabsch(line[:2], line[:2])
