"""Contact episodes: boundaries, censoring, sampling limits (SYNTHETIC fixtures)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import DNA, PROTEIN, moved, synthetic_atoms, write_xtc

from neurodna import Complex, DynamicsUnavailableError, TimingUnavailableError
from neurodna.interactions import EPISODE_NOTE, contact_runs

BOUND = synthetic_atoms()
UNBOUND = moved(BOUND, "B", 1, -10.0)

# SYNTHETIC trajectory: B:ARG1–C:DC1 contact pattern 1 1 0 1 0 0 1 at non-uniform
# times; A:ARG1–C:5CM2 is in contact in every frame.
PATTERN = [1, 1, 0, 1, 0, 0, 1]
TIMES = [0.0, 10.0, 20.0, 30.0, 40.0, 55.0, 70.0]
NAN = math.nan


@pytest.fixture
def episode_complex(synthetic_pdb, tmp_path):
    frames = [BOUND if bound else UNBOUND for bound in PATTERN]
    xtc = write_xtc(synthetic_pdb, tmp_path / "episodes.xtc", frames, TIMES)
    return Complex.load(synthetic_pdb, xtc, protein=PROTEIN, dna=DNA)


def rows(df, protein_label):
    cols = ["episode", "first_frame", "last_frame", "n_frames", "first_time_ps", "last_time_ps",
            "preceding_time_ps", "following_time_ps", "observed_duration_ps", "max_duration_ps",
            "left_censored", "right_censored"]
    return [tuple(r) for r in df[df.protein_label == protein_label][cols].itertuples(index=False)]


def same(actual, expected):
    assert len(actual) == len(expected)
    for a_row, e_row in zip(actual, expected):
        for a, e in zip(a_row, e_row):
            if isinstance(e, float) and math.isnan(e):
                assert math.isnan(a)
            else:
                assert a == pytest.approx(e)


def test_episode_boundaries_times_and_censoring(episode_complex):
    df = episode_complex.contact_episodes()
    same(rows(df, "B:ARG1"), [
        # ep, first, last, n, t_first, t_last, t_before, t_after, observed, max, left, right
        (0, 0, 1, 2, 0.0, 10.0, NAN, 20.0, 10.0, NAN, True, False),
        (1, 3, 3, 1, 30.0, 30.0, 20.0, 40.0, 0.0, 20.0, False, False),
        (2, 6, 6, 1, 70.0, 70.0, 55.0, NAN, 0.0, NAN, False, True),
    ])
    same(rows(df, "A:ARG1"), [(0, 0, 6, 7, 0.0, 70.0, NAN, NAN, 70.0, NAN, True, True)])
    assert set(df[df.protein_label == "B:ARG1"].dna_label) == {"C:DC1"}


def test_episode_metadata_explains_sampling_limits(episode_complex):
    df = episode_complex.contact_episodes()
    assert df.attrs["note"] == EPISODE_NOTE
    assert "not protein-DNA binding residence times" in EPISODE_NOTE
    assert "sampling" in EPISODE_NOTE
    assert df.attrs["min_sampling_interval_ps"] == 10.0
    assert df.attrs["max_sampling_interval_ps"] == 15.0
    assert df.attrs["uniform_sampling"] is False
    assert df.attrs["cutoff_A"] == 4.5 and "geometric" in df.attrs["contact_definition"]
    assert "residence" not in " ".join(df.columns)


def test_stride_changes_what_is_observed(episode_complex):
    # Frames 0, 2, 4, 6 -> pattern 1 0 0 1; the episode at frame 3 is never seen.
    df = episode_complex.contact_episodes(step=2)
    same(rows(df, "B:ARG1"), [
        (0, 0, 0, 1, 0.0, 0.0, NAN, 20.0, 0.0, NAN, True, False),
        (1, 6, 6, 1, 70.0, 70.0, 40.0, NAN, 0.0, NAN, False, True),
    ])
    assert df.attrs["frame_step"] == 2
    assert (df.attrs["sampling_interval_ps"], df.attrs["max_sampling_interval_ps"]) == (20.0, 30.0)


def test_window_start_left_censors(episode_complex):
    df = episode_complex.contact_episodes(start=1)
    first = rows(df, "B:ARG1")[0]
    same([first], [(0, 1, 1, 1, 10.0, 10.0, NAN, 20.0, 0.0, NAN, True, False)])


def test_episodes_need_frame_times(write_pdb):
    path = write_pdb([BOUND, UNBOUND, BOUND])
    no_times = Complex.load(path, protein=PROTEIN, dna=DNA, time_ordered=True)
    with pytest.raises(TimingUnavailableError, match="no frame times"):
        no_times.contact_episodes()
    assert no_times.contact_occupancy().attrs["n_frames"] == 3  # frame counts need no time
    with_dt = Complex.load(path, protein=PROTEIN, dna=DNA, time_ordered=True, timestep_ps=2.0)
    b = with_dt.contact_episodes()
    assert b[b.protein_label == "B:ARG1"].first_time_ps.tolist() == [0.0, 4.0]


def test_episodes_refuse_ensembles(write_pdb):
    cx = Complex.load(write_pdb([BOUND, UNBOUND]), protein=PROTEIN, dna=DNA)
    with pytest.raises(DynamicsUnavailableError, match="unordered ensemble"):
        cx.contact_episodes()


def test_contact_runs_pure():
    per_frame = [np.array([1, 5]), np.array([1]), np.array([], dtype=np.intp), np.array([1, 5])]
    codes, first, last = contact_runs(per_frame)
    assert list(zip(codes, first, last)) == [(1, 0, 1), (1, 3, 3), (5, 0, 0), (5, 3, 3)]
    empty = contact_runs([np.array([], dtype=np.intp)] * 3)
    assert all(a.size == 0 for a in empty)
