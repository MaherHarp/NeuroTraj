"""Static structures vs unordered ensembles vs time-resolved trajectories."""

from __future__ import annotations

import math

import MDAnalysis as mda
import numpy as np
import pytest
from conftest import DNA, PROTEIN, moved, synthetic_atoms, write_xtc

from neurodna import (
    Complex,
    DynamicsUnavailableError,
    FrameKind,
    PeriodicBoxError,
    TimeSource,
    TrajectoryError,
)

# B:ARG1 is in contact with C:DC1 (3.5 Å) in the reference geometry; moving it
# 10 Å away along -y breaks the contact. A:ARG1–C:5CM2 (3.0 Å) never changes.
BOUND = synthetic_atoms()
UNBOUND = moved(BOUND, "B", 1, -10.0)


@pytest.fixture
def trajectory(synthetic_pdb, tmp_path):
    # 4 frames, 10 ps apart; B:ARG1 bound in frames 0 and 3 only.
    xtc = write_xtc(synthetic_pdb, tmp_path / "traj.xtc",
                    [BOUND, UNBOUND, UNBOUND, BOUND], [0.0, 10.0, 20.0, 30.0])
    return synthetic_pdb, xtc


def test_single_structure_is_static(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    assert cx.kind is FrameKind.STATIC
    assert cx.frames.n_frames == 1 and not cx.frames.is_time_resolved
    assert cx.frames.time_source is None


@pytest.mark.parametrize("method", ["contact_occupancy", "contact_episodes", "rmsf", "frame_times"])
def test_static_structure_refuses_dynamics(synthetic_pdb, method):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    with pytest.raises(DynamicsUnavailableError, match="single static structure"):
        getattr(cx, method)()


def test_static_structure_rejects_ensemble_statistics(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    with pytest.raises(DynamicsUnavailableError, match="unordered ensemble"):
        cx.ensemble_contact_frequency()


def test_multi_model_pdb_is_an_unordered_ensemble(write_pdb):
    path = write_pdb([BOUND, UNBOUND, UNBOUND, UNBOUND])
    cx = Complex.load(path, protein=PROTEIN, dna=DNA)
    assert cx.kind is FrameKind.ENSEMBLE
    with pytest.raises(DynamicsUnavailableError, match="unordered ensemble"):
        cx.contact_occupancy()
    freq = cx.ensemble_contact_frequency(cutoff=4.0).set_index(["protein_label", "dna_label"])
    assert freq.loc[("A:ARG1", "C:5CM2"), "model_fraction"] == 1.0
    assert freq.loc[("B:ARG1", "C:DC1"), "model_fraction"] == 0.25
    assert "occupancy" not in freq.columns


def test_trajectory_occupancy_and_times(trajectory):
    cx = Complex.load(*trajectory, protein=PROTEIN, dna=DNA)
    assert cx.kind is FrameKind.TRAJECTORY
    assert cx.frames.time_source is TimeSource.FILE
    assert cx.frames.timestep_ps == pytest.approx(10.0)
    np.testing.assert_allclose(cx.frame_times(), [0.0, 10.0, 20.0, 30.0])

    occ = cx.contact_occupancy(cutoff=4.0)
    got = dict(zip(zip(occ.protein_label, occ.dna_label), occ.occupancy))
    assert got == {("A:ARG1", "C:5CM2"): 1.0, ("B:ARG1", "C:DC1"): 0.5}
    assert occ.attrs["n_frames"] == 4
    assert (occ.attrs["start_time_ps"], occ.attrs["end_time_ps"]) == (0.0, 30.0)


def test_trajectory_slicing_uses_selected_frames(trajectory):
    cx = Complex.load(*trajectory, protein=PROTEIN, dna=DNA)
    occ = cx.contact_occupancy(cutoff=4.0, start=1, stop=3)
    assert list(occ.protein_label) == ["A:ARG1"]  # B:ARG1 is unbound in frames 1-2
    with pytest.raises(ValueError, match="step must be positive"):
        cx.contact_occupancy(step=-1)


def test_min_distances_per_frame_keep_real_times(trajectory):
    cx = Complex.load(*trajectory, protein=PROTEIN, dna=DNA)
    d = cx.min_distances(max_distance=4.0)
    b = d[d.protein_label == "B:ARG1"]
    assert list(b.frame) == [0, 3]
    assert list(b.time_ps) == pytest.approx([0.0, 30.0])
    assert list(b.min_distance_A) == pytest.approx([3.5, 3.5], abs=0.01)  # XTC precision


def test_trajectory_can_be_declared_unordered(trajectory):
    cx = Complex.load(*trajectory, protein=PROTEIN, dna=DNA, time_ordered=False)
    assert cx.kind is FrameKind.ENSEMBLE
    with pytest.raises(DynamicsUnavailableError):
        cx.contact_occupancy()


def test_non_monotonic_times_are_rejected(synthetic_pdb, tmp_path):
    xtc = write_xtc(synthetic_pdb, tmp_path / "bad.xtc", [BOUND] * 3, [0.0, 10.0, 5.0])
    cx = Complex.load(synthetic_pdb, xtc, protein=PROTEIN, dna=DNA)
    with pytest.raises(TrajectoryError, match="not strictly increasing"):
        cx.contact_occupancy()


def test_models_declared_time_ordered_have_no_invented_times(write_pdb):
    # PDB models carry no time; MDAnalysis would invent 1 ps steps. neurodna reports NaN.
    path = write_pdb([BOUND, UNBOUND])
    cx = Complex.load(path, protein=PROTEIN, dna=DNA, time_ordered=True)
    assert cx.kind is FrameKind.TRAJECTORY
    assert cx.frames.time_source is TimeSource.UNAVAILABLE and cx.frames.timestep_ps is None
    assert all(math.isnan(t) for t in cx.frame_times())
    assert cx.contact_occupancy().attrs["n_frames"] == 2


def test_user_timestep(write_pdb):
    path = write_pdb([BOUND, UNBOUND, BOUND])
    cx = Complex.load(path, protein=PROTEIN, dna=DNA, time_ordered=True, timestep_ps=2.0)
    assert cx.frames.time_source is TimeSource.USER
    np.testing.assert_allclose(cx.frame_times(), [0.0, 2.0, 4.0])


def test_timestep_on_static_structure_is_rejected(synthetic_pdb):
    with pytest.raises(TrajectoryError, match="only applies to trajectories"):
        Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA, timestep_ps=2.0)


def test_explicit_frame_kind_must_match_frames(synthetic_pdb):
    u = mda.Universe(str(synthetic_pdb))
    with pytest.raises(TrajectoryError, match="at least 2 frames"):
        Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY)


def test_periodic_minimum_image_contacts(write_pdb):
    # Shift B:ARG1 by one box length (40 Å) along y: NH1 lands at y = 36.5, 36.5 Å from
    # C:DC1 P directly but 3.5 Å from it through the periodic boundary.
    atoms = moved(BOUND, "B", 1, +40.0)
    path = write_pdb(atoms, cryst1=(40.0, 40.0, 40.0))
    no_pbc = Complex.load(path, protein=PROTEIN, dna=DNA, pbc=False).contacts(cutoff=4.0)
    assert "B:ARG1" not in set(no_pbc.protein_label)
    with_pbc = Complex.load(path, protein=PROTEIN, dna=DNA).contacts(cutoff=4.0)  # auto
    row = with_pbc[with_pbc.protein_label == "B:ARG1"].iloc[0]
    assert row.min_distance_A == pytest.approx(3.5, abs=1e-3)
    assert with_pbc.attrs["minimum_image"] and not no_pbc.attrs["minimum_image"]


def test_pbc_requires_a_large_enough_box(write_pdb, synthetic_pdb):
    with pytest.raises(PeriodicBoxError, match="no valid periodic box"):
        Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA, pbc=True).contacts()
    path = write_pdb(BOUND, cryst1=(40.0, 40.0, 6.0))
    with pytest.raises(PeriodicBoxError, match="twice the cutoff"):
        Complex.load(path, protein=PROTEIN, dna=DNA).contacts(cutoff=4.0)
