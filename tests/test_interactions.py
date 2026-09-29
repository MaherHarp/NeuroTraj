"""Per-frame minimum heavy-atom distances, exact cutoffs, periodic boundaries and
sampling metadata. All structures here are SYNTHETIC fixtures (see conftest)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from conftest import (
    DNA,
    PROTEIN,
    Atom,
    arginine,
    coords,
    memory_universe,
    moved,
    nucleotide,
    synthetic_atoms,
    write_xtc,
)
from MDAnalysis.transformations import rotateby

from neurodna import (
    CONTACT_DEFINITION,
    DEFAULT_CONTACT_CUTOFF,
    Complex,
    FrameKind,
    PeriodicBoxError,
    TrajectoryError,
)

BOUND = synthetic_atoms()
UNBOUND = moved(BOUND, "B", 1, -10.0)
# B:ARG1 shifted one 40 Å box length along +y: NH1 at y = 36.5, i.e. 36.5 Å from
# C:DC1 P directly and (L - 36.5) Å through the periodic boundary for box length L.
ACROSS_BOUNDARY = moved(BOUND, "B", 1, +40.0)


def cutoff_fixture() -> list[Atom]:
    """SYNTHETIC: A:ARG1 NH1 at the origin; C:DC1 P exactly 4.5 Å away; C:DG2 O3' 4.501 Å away."""
    return [
        *arginine("A", 1, (0.0, 0.0), -1),        # NH1 (0,0,0), CZ (0,-1.3,0), CA (0,-6,0)
        *nucleotide("DC", "C", 1, 4.5),           # P at (4.5, 0, 0)
        *nucleotide("DG", "C", 2, -8.901),        # O3' at (-4.501, 0, 0)
    ]


# ------------------------------------------------------------ contact definition


def test_default_cutoff_and_contact_definition(synthetic_pdb):
    assert DEFAULT_CONTACT_CUTOFF == 4.5
    df = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA).contacts()
    assert df.attrs["cutoff_A"] == 4.5
    assert df.attrs["contact_definition"] == CONTACT_DEFINITION
    assert "geometric" in CONTACT_DEFINITION and "not a hydrogen bond" in CONTACT_DEFINITION
    assert df.attrs["atoms"] == "heavy atoms only"


def test_exact_cutoff_is_inclusive(write_pdb):
    cx = Complex.load(write_pdb(cutoff_fixture()), protein=PROTEIN, dna=DNA)
    assert cx.contacts(cutoff=4.5).dna_label.tolist() == ["C:DC1"]  # exactly 4.5 Å
    assert cx.contacts(cutoff=4.5).min_distance_A.tolist() == [4.5]
    assert cx.contacts(cutoff=4.4999).empty
    assert sorted(cx.contacts(cutoff=4.501).dna_label) == ["C:DC1", "C:DG2"]
    d = cx.min_distances(max_distance=4.5)
    assert d.dna_label.tolist() == ["C:DC1"] and d.min_distance_A.tolist() == [4.5]


# --------------------------------------------------------------- known geometry


def test_min_distances_match_brute_force(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    d = cx.min_distances(max_distance=None)  # every residue-nucleotide pair
    assert len(d) == 4 * 3 and set(d.frame) == {0} and d.time_ps.isna().all()

    atoms = synthetic_atoms()
    key = lambda a: f"{a.chain}:{a.resname}{a.resnum}{a.icode}"  # noqa: E731
    expected = {}
    for p, q in itertools.product([a for a in atoms if a.chain != "C"],
                                  [a for a in atoms if a.chain == "C"]):
        dist = float(np.linalg.norm(np.subtract((p.x, p.y, p.z), (q.x, q.y, q.z))))
        pair = (key(p), key(q))
        expected[pair] = min(expected.get(pair, np.inf), dist)
    got = dict(zip(zip(d.protein_label, d.dna_label), d.min_distance_A))
    assert got.keys() == expected.keys()
    for pair, value in expected.items():
        assert got[pair] == pytest.approx(value, abs=1e-4)


def test_min_distances_reporting_radius(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    near = cx.min_distances(max_distance=5.0)
    assert set(zip(near.protein_label, near.dna_label)) == {
        ("A:ARG1", "C:5CM2"), ("B:ARG1", "C:DC1")
    }
    assert (near.min_distance_A <= 5.0).all()
    assert near.attrs["max_distance_A"] == 5.0


def test_duplicate_residue_numbers_get_separate_rows_per_frame(synthetic_pdb, tmp_path):
    xtc = write_xtc(synthetic_pdb, tmp_path / "t.xtc", [BOUND, UNBOUND], [0.0, 5.0])
    cx = Complex.load(synthetic_pdb, xtc, protein=PROTEIN, dna=DNA)
    d = cx.min_distances(max_distance=4.5)
    rows = sorted(zip(d.frame, d.protein_chain, d.protein_resnum, d.dna_label))
    # Both arginines are residue 1; only chain A stays in contact in frame 1.
    assert rows == [(0, "A", 1, "C:5CM2"), (0, "B", 1, "C:DC1"), (1, "A", 1, "C:5CM2")]


# ------------------------------------------------------------ periodic boundaries


def test_minimum_image_uses_each_frames_box(synthetic_pdb):
    # SYNTHETIC NPT-like trajectory: same coordinates, box grows 40 -> 41 -> 42 Å, so
    # the B:ARG1 NH1 - C:DC1 P image distance is 3.5, 4.5 (= cutoff) and 5.5 Å.
    dims = np.array([[40, 40, 40, 90, 90, 90], [41, 41, 41, 90, 90, 90],
                     [42, 42, 42, 90, 90, 90]], dtype=np.float32)
    u = memory_universe(synthetic_pdb, [coords(ACROSS_BOUNDARY)] * 3, dt=2.0, dimensions=dims)
    cx = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY)
    d = cx.min_distances(max_distance=6.0)
    b = d[(d.protein_label == "B:ARG1") & (d.dna_label == "C:DC1")]
    assert b.min_distance_A.tolist() == pytest.approx([3.5, 4.5, 5.5], abs=1e-5)
    assert d.attrs["minimum_image"] is True
    occ = cx.contact_occupancy().set_index(["protein_label", "dna_label"])
    assert occ.loc[("B:ARG1", "C:DC1"), "n_frames_in_contact"] == 2  # 3.5 and exactly 4.5


def test_frames_without_box_cannot_be_mixed_with_boxed_frames(synthetic_pdb):
    dims = np.array([[40, 40, 40, 90, 90, 90]] * 2 + [[0, 0, 0, 90, 90, 90]], dtype=np.float32)
    u = memory_universe(synthetic_pdb, [coords(ACROSS_BOUNDARY)] * 3, dimensions=dims)
    cx = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY)
    with pytest.raises(TrajectoryError, match="frame 2 lacks a valid periodic box"):
        cx.contact_occupancy()


def test_crystallographic_cell_is_not_a_periodic_box(write_pdb):
    # SYNTHETIC coordinates with a non-P1 CRYST1: the cell is a crystal lattice, so the
    # automatic policy must not create a contact through the "boundary".
    path = write_pdb(ACROSS_BOUNDARY, cryst1=(40.0, 40.0, 40.0), space_group="P 21 21 21")
    auto = Complex.load(path, protein=PROTEIN, dna=DNA)
    assert "crystallographic cell" in auto.periodic_policy
    df = auto.contacts()
    assert "B:ARG1" not in set(df.protein_label) and df.attrs["minimum_image"] is False
    forced = Complex.load(path, protein=PROTEIN, dna=DNA, pbc=True).contacts()
    assert "B:ARG1" in set(forced.protein_label)


def test_rotating_transformation_invalidates_the_box(synthetic_pdb):
    dims = np.array([40, 40, 40, 90, 90, 90], dtype=np.float32)
    u = memory_universe(synthetic_pdb, [coords(ACROSS_BOUNDARY)] * 2, dimensions=dims)
    u.trajectory.add_transformations(rotateby(30, [0, 0, 1], point=[0, 0, 0]))
    cx = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY)
    with pytest.raises(PeriodicBoxError, match="rotating transformations"):
        cx.contacts()
    no_pbc = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY, pbc=False)
    assert not no_pbc.contacts().empty


def test_triclinic_box_width_limits_the_cutoff(synthetic_pdb):
    # SYNTHETIC: lengths 40 Å but a 20° angle gives a perpendicular width of ~13.7 Å.
    dims = np.array([40, 40, 40, 90, 90, 20], dtype=np.float32)
    u = memory_universe(synthetic_pdb, [coords(BOUND)] * 2, dimensions=dims)
    cx = Complex(u, protein=PROTEIN, dna=DNA, frame_kind=FrameKind.TRAJECTORY)
    assert not cx.contacts(cutoff=4.5).empty
    with pytest.raises(PeriodicBoxError, match="smallest box width"):
        cx.min_distances(max_distance=7.0)


# ------------------------------------------------------------- times & sampling


def test_occupancy_records_real_nonuniform_sampling(synthetic_pdb, tmp_path):
    times = [0.0, 10.0, 20.0, 35.0, 50.0]
    xtc = write_xtc(synthetic_pdb, tmp_path / "t.xtc", [BOUND, UNBOUND, BOUND, BOUND, UNBOUND],
                    times)
    occ = Complex.load(synthetic_pdb, xtc, protein=PROTEIN, dna=DNA).contact_occupancy()
    b = occ.set_index("protein_label").loc["B:ARG1"]
    assert (b.n_frames_in_contact, b.n_frames_analyzed, b.occupancy) == (3, 5, 0.6)
    assert b.closest_distance_A == pytest.approx(3.5, abs=0.01)
    a = occ.attrs
    assert (a["start_time_ps"], a["end_time_ps"]) == (0.0, 50.0)
    assert (a["min_sampling_interval_ps"], a["max_sampling_interval_ps"]) == (10.0, 15.0)
    assert a["sampling_interval_ps"] == 12.5 and a["uniform_sampling"] is False
    assert a["time_weighted"] is False and a["units"] == {"distance": "angstrom", "time": "ps"}


def test_static_contact_analysis_needs_no_trajectory(synthetic_pdb):
    cx = Complex.load(synthetic_pdb, protein=PROTEIN, dna=DNA)
    assert cx.kind is FrameKind.STATIC
    assert len(cx.contacts()) == 2
    d = cx.min_distances()
    assert d.attrs["n_frames"] == 1 and pd.isna(d.attrs["sampling_interval_ps"])


def test_single_frame_of_a_trajectory(synthetic_pdb, tmp_path):
    xtc = write_xtc(synthetic_pdb, tmp_path / "t.xtc", [BOUND, UNBOUND, BOUND], [0.0, 1.0, 2.0])
    cx = Complex.load(synthetic_pdb, xtc, protein=PROTEIN, dna=DNA)
    d = cx.min_distances(max_distance=4.5, frame=1)
    assert set(d.frame) == {1} and set(d.time_ps) == {1.0}
    assert set(d.protein_label) == {"A:ARG1"}
    with pytest.raises(ValueError, match="either frame or start/stop/step"):
        cx.min_distances(frame=1, step=2)
