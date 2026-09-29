"""Protein–DNA geometric contacts and distances (Å, ps).

**Contact definition.** A protein residue and a nucleotide are in *geometric
contact* in a frame when their minimum heavy-atom distance is ≤ the cutoff
(default 4.5 Å, inclusive). This is purely geometric: it is not a hydrogen
bond, salt bridge or binding energy.

Distances are computed frame by frame from the raw trajectory coordinates,
using minimum-image distances whenever the frame has a valid periodic box (see
:mod:`neurodna.geometry` and the ``pbc`` option of :class:`~neurodna.Complex`).
Nothing here applies alignment or other coordinate transformations; RMSD and
RMSF superposition lives separately in :mod:`neurodna.structure`.

All tables are tidy ``pandas.DataFrame`` objects. Units are in the column names
(``_A`` for ångström, ``_ps`` for picoseconds), and ``DataFrame.attrs`` records
the contact definition, the periodic-box usage and the frames and sampling
analysed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
import pandas as pd
from MDAnalysis.lib.distances import capped_distance

from neurodna.errors import DynamicsUnavailableError, SelectionError, TrajectoryError
from neurodna.frames import FrameKind, sampling_summary
from neurodna.geometry import pair_distances
from neurodna.residues import KEY_FIELDS, normalize_atom_name

if TYPE_CHECKING:
    from neurodna.complex import Complex

Target = Literal["dna", "modification_markers"]

CONTACT_DEFINITION = (
    "geometric contact: minimum heavy-atom distance <= cutoff (angstrom); "
    "not a hydrogen bond, salt bridge or binding energy"
)

EPISODE_NOTE = (
    "Contact episodes are runs of consecutive analysed frames with a geometric contact. "
    "Formation and breaking times are only known to within the frame sampling: an episode "
    "began after preceding_time_ps and no later than first_time_ps, and ended no earlier "
    "than last_time_ps and before following_time_ps. Separations shorter than the sampling "
    "interval are invisible. Episode durations describe individual residue-nucleotide "
    "contacts and are not protein-DNA binding residence times."
)

_PHOSPHATE_ATOMS = frozenset({"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "O3'"})
_PROTEIN_BACKBONE_ATOMS = frozenset(
    {"N", "CA", "C", "O", "OXT", "OT1", "OT2", "H", "HN", "HA", "HA2", "HA3",
     "H1", "H2", "H3", "HT1", "HT2", "HT3"}
)  # fmt: skip

_UNITS = {"distance": "angstrom", "time": "ps"}

_CANDIDATE_PAD = 0.01
"""Å added to the cutoff for the MDAnalysis candidate search (float32 slack)."""


def dna_moiety(atom_name: str) -> str:
    """Classify a nucleotide atom as ``"phosphate"``, ``"sugar"`` or ``"base"`` by name.

    Primed names (C1', H2'', ...) are sugar; P/OP1/OP2/O5'/O3' are phosphate;
    everything else, including modification atoms such as the 5mC methyl, is base.
    """
    name = normalize_atom_name(atom_name)
    if name in _PHOSPHATE_ATOMS:
        return "phosphate"
    if name.endswith("'"):
        return "sugar"
    return "base"


def protein_moiety(atom_name: str) -> str:
    """Classify an amino-acid atom as ``"backbone"`` or ``"sidechain"`` by name."""
    return "backbone" if atom_name.strip() in _PROTEIN_BACKBONE_ATOMS else "sidechain"


# ============================================================ public analyses


def contacts(
    cx: Complex,
    cutoff: float,
    *,
    frame: int = 0,
    level: Literal["residue", "atom"] = "residue",
    target: Target = "dna",
) -> pd.DataFrame:
    """Geometric contacts in one frame; see :meth:`Complex.contacts`."""
    _check_distance(cutoff, "cutoff")
    if level not in ("residue", "atom"):
        raise ValueError(f"level must be 'residue' or 'atom', got {level!r}")
    cx._goto(frame)
    dna_ag, dna_res = _target_atoms(cx, target)
    box = cx._box(cutoff)
    pi, dj, dist = _atom_pairs(cx._protein_heavy.positions, dna_ag.positions, cutoff, box)
    attrs = _attrs(cx, "contacts", cutoff=cutoff, target=target, minimum_image=box is not None)
    attrs["frame"] = int(cx._u.trajectory.ts.frame)
    if level == "atom":
        df = _atom_table(cx, pi, dj, dist, dna_ag, dna_res)
    else:
        m = _group(cx._protein_heavy_res[pi], dna_res[dj], dist, len(cx._dna_res))
        df = _pair_columns(cx, m.codes)
        df["min_distance_A"] = m.min_dist
        df["protein_atom"] = cx._protein_heavy.names[pi[m.first]].astype(str)
        df["dna_atom"] = dna_ag.names[dj[m.first]].astype(str)
        df["n_atom_pairs"] = m.counts
    df.attrs.update(attrs)
    return df


def min_distances(
    cx: Complex,
    max_distance: float | None,
    *,
    target: Target = "dna",
    frame: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Per-frame residue–nucleotide minimum heavy-atom distances; see
    :meth:`Complex.min_distances`."""
    if max_distance is not None:
        _check_distance(max_distance, "max_distance")
    dna_ag, dna_res = _target_atoms(cx, target)
    n_dna = len(cx._dna_res)
    tracker = _BoxTracker()
    frames_seen: list[int] = []
    times_seen: list[float] = []
    chunks: dict[str, list[npt.NDArray[Any]]] = {k: [] for k in ("frame", "time", "code", "d", "pa", "da")}

    for f, t in _frame_iter(cx, frame, start, stop, step):
        box = cx._box(max_distance)
        tracker.record(box is not None, f)
        pi, dj, dist = _atom_pairs(cx._protein_heavy.positions, dna_ag.positions, max_distance, box)
        m = _group(cx._protein_heavy_res[pi], dna_res[dj], dist, n_dna)
        frames_seen.append(f)
        times_seen.append(t)
        chunks["frame"].append(np.full(m.codes.size, f, dtype=np.intp))
        chunks["time"].append(np.full(m.codes.size, t, dtype=float))
        chunks["code"].append(m.codes)
        chunks["d"].append(m.min_dist)
        chunks["pa"].append(pi[m.first])
        chunks["da"].append(dj[m.first])

    cat = {k: np.concatenate(v) if v else np.array([]) for k, v in chunks.items()}
    df = _pair_columns(cx, cat["code"].astype(np.intp))
    df.insert(0, "time_ps", cat["time"].astype(float))
    df.insert(0, "frame", cat["frame"].astype(np.intp))
    df["min_distance_A"] = cat["d"].astype(float)
    df["protein_atom"] = cx._protein_heavy.names[cat["pa"].astype(np.intp)].astype(str)
    df["dna_atom"] = dna_ag.names[cat["da"].astype(np.intp)].astype(str)
    df.attrs.update(
        _attrs(cx, "min_distances", target=target, minimum_image=bool(tracker.used)),
        max_distance_A=max_distance,
        **sampling_summary(frames_seen, times_seen),
    )
    return df


def contact_occupancy(
    cx: Complex,
    cutoff: float,
    *,
    target: Target = "dna",
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Fraction of analysed trajectory frames with a geometric contact, per pair.

    Columns: residue identities, ``n_frames_in_contact``, ``n_frames_analyzed``,
    ``occupancy`` (fraction of analysed frames) and ``closest_distance_A`` (the
    smallest minimum distance seen). Only pairs in contact in at least one frame
    are listed. Frame times and sampling intervals are recorded in ``attrs``.
    Occupancy is time-weighted only for uniform sampling
    (``attrs["uniform_sampling"]``). It needs frame order but not frame times.
    """
    cx._require_trajectory("contact_occupancy")
    scan = _scan(cx, cutoff, target, start, stop, step)
    codes, counts = np.unique(np.concatenate(scan.codes_per_frame), return_counts=True)
    n = len(scan.frames)
    df = _pair_columns(cx, codes)
    df["n_frames_in_contact"] = counts
    df["n_frames_analyzed"] = n
    df["occupancy"] = counts / n
    df["closest_distance_A"] = scan.closest[codes]
    df = df.sort_values("occupancy", ascending=False, kind="stable").reset_index(drop=True)
    df.attrs.update(
        _attrs(cx, "contact_occupancy", cutoff=cutoff, target=target,
               minimum_image=bool(scan.box.used)),
        frame_step=step or 1,
        time_weighted=False,
        **sampling_summary(scan.frames, scan.times),
    )
    return df


def ensemble_contact_frequency(
    cx: Complex, cutoff: float, *, target: Target = "dna"
) -> pd.DataFrame:
    """Fraction of ensemble models with a geometric contact, per pair (not dynamics)."""
    if cx.kind is not FrameKind.ENSEMBLE:
        raise DynamicsUnavailableError(
            f"ensemble_contact_frequency needs an unordered ensemble, but this complex is "
            f"{cx.kind.value}; use contact_occupancy for trajectories or contacts for a "
            "single structure"
        )
    scan = _scan(cx, cutoff, target, None, None, None)
    codes, counts = np.unique(np.concatenate(scan.codes_per_frame), return_counts=True)
    n = len(scan.frames)
    df = _pair_columns(cx, codes)
    df["n_models_in_contact"] = counts
    df["n_models"] = n
    df["model_fraction"] = counts / n
    df["closest_distance_A"] = scan.closest[codes]
    df = df.sort_values("model_fraction", ascending=False, kind="stable").reset_index(drop=True)
    df.attrs.update(
        _attrs(cx, "ensemble_contact_frequency", cutoff=cutoff, target=target,
               minimum_image=bool(scan.box.used)),
        n_models=n,
    )
    return df


def contact_episodes(
    cx: Complex,
    cutoff: float,
    *,
    target: Target = "dna",
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Contiguous geometric-contact episodes for each residue–nucleotide pair.

    An episode is a maximal run of consecutive *analysed* frames (after
    ``start/stop/step``) in which the pair is in geometric contact. One row per
    episode:

    * ``episode``: 0-based index of the episode within its pair.
    * ``first_frame``, ``last_frame``, ``n_frames``: trajectory frame indices
      of the first and last frame in contact, and the number of analysed frames
      in the episode.
    * ``first_time_ps``, ``last_time_ps``: times of those frames (real file
      times, or ``i * timestep_ps``).
    * ``preceding_time_ps`` / ``following_time_ps``: times of the analysed
      frames just before and after the episode, where the pair was *not* in
      contact. They are NaN when the episode touches the start or end of the
      analysed window.
    * ``observed_duration_ps = last_time_ps - first_time_ps``: a lower bound on
      the episode length (0 for a single-frame episode).
    * ``max_duration_ps = following_time_ps - preceding_time_ps``: an upper
      bound; NaN when censored.
    * ``left_censored`` / ``right_censored``: the episode was already in
      progress at the first analysed frame, or still in progress at the last.

    Transition times are limited by the frame sampling (see ``attrs["note"]``).
    These durations are **not** protein–DNA binding residence times.

    Raises:
        DynamicsUnavailableError: not a trajectory.
        TimingUnavailableError: the trajectory has no frame times.
    """
    cx._require_times("contact_episodes")
    scan = _scan(cx, cutoff, target, start, stop, step)
    n = len(scan.frames)
    if n < 2:
        raise ValueError("contact_episodes needs at least 2 analysed frames")
    codes, first_pos, last_pos = contact_runs(scan.codes_per_frame)
    frames = np.asarray(scan.frames, dtype=np.intp)
    times = np.asarray(scan.times, dtype=float)

    df = _pair_columns(cx, codes)
    new_pair = np.r_[True, codes[1:] != codes[:-1]] if codes.size else np.array([], dtype=bool)
    pair_start = np.maximum.accumulate(np.where(new_pair, np.arange(codes.size), 0))
    left = first_pos == 0
    right = last_pos == n - 1
    preceding = np.where(left, math.nan, times[np.maximum(first_pos - 1, 0)])
    following = np.where(right, math.nan, times[np.minimum(last_pos + 1, n - 1)])
    df["episode"] = np.arange(codes.size) - pair_start
    df["first_frame"] = frames[first_pos]
    df["last_frame"] = frames[last_pos]
    df["n_frames"] = last_pos - first_pos + 1
    df["first_time_ps"] = times[first_pos]
    df["last_time_ps"] = times[last_pos]
    df["preceding_time_ps"] = preceding
    df["following_time_ps"] = following
    df["observed_duration_ps"] = times[last_pos] - times[first_pos]
    df["max_duration_ps"] = following - preceding
    df["left_censored"] = left
    df["right_censored"] = right
    df.attrs.update(
        _attrs(cx, "contact_episodes", cutoff=cutoff, target=target,
               minimum_image=bool(scan.box.used)),
        frame_step=step or 1,
        note=EPISODE_NOTE,
        **sampling_summary(scan.frames, scan.times),
    )
    return df


def contact_runs(
    codes_per_frame: Iterable[npt.NDArray[np.intp]],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Find runs of consecutive frame positions per contact code.

    Args:
        codes_per_frame: For each analysed frame (in order), the contact codes present.

    Returns:
        ``(codes, first_positions, last_positions)`` sorted by code then first
        position, one entry per run.
    """
    per_frame = list(codes_per_frame)
    if not per_frame or not any(len(c) for c in per_frame):
        empty = np.array([], dtype=np.intp)
        return empty, empty, empty
    pos = np.concatenate([np.full(len(c), i, dtype=np.intp) for i, c in enumerate(per_frame)])
    code = np.concatenate([np.asarray(c, dtype=np.intp) for c in per_frame])
    order = np.lexsort((pos, code))
    code, pos = code[order], pos[order]
    new_run = np.r_[True, (code[1:] != code[:-1]) | (pos[1:] != pos[:-1] + 1)]
    starts = np.flatnonzero(new_run)
    ends = np.r_[starts[1:], code.size] - 1
    return code[starts], pos[starts], pos[ends]


# ================================================================ internals


@dataclass
class _Minima:
    codes: npt.NDArray[np.intp]      # residue-pair codes (p_local * n_dna + d_local), sorted
    min_dist: npt.NDArray[np.float64]
    first: npt.NDArray[np.intp]      # index into the atom-pair arrays of the closest pair
    counts: npt.NDArray[np.intp]     # number of atom pairs per residue pair


class _BoxTracker:
    """Ensure minimum-image usage is consistent across the analysed frames."""

    def __init__(self) -> None:
        self.used: bool | None = None

    def record(self, used: bool, frame: int) -> None:
        if self.used is None:
            self.used = used
        elif self.used != used:
            raise TrajectoryError(
                f"frame {frame} {'has' if used else 'lacks'} a valid periodic box but earlier "
                "frames did not; distances would mix minimum-image and direct geometry"
            )


@dataclass
class _Scan:
    frames: list[int]
    times: list[float]
    codes_per_frame: list[npt.NDArray[np.intp]]
    closest: npt.NDArray[np.float64]
    box: _BoxTracker


def _scan(
    cx: Complex,
    cutoff: float,
    target: Target,
    start: int | None,
    stop: int | None,
    step: int | None,
) -> _Scan:
    """One pass over the frames, recording which pairs are in contact in each."""
    _check_distance(cutoff, "cutoff")
    dna_ag, dna_res = _target_atoms(cx, target)
    n_dna = len(cx._dna_res)
    closest = np.full(len(cx._protein_res) * n_dna, np.inf)
    scan = _Scan([], [], [], closest, _BoxTracker())
    for f, t in cx._frames(start, stop, step):
        box = cx._box(cutoff)
        scan.box.record(box is not None, f)
        pi, dj, dist = _atom_pairs(cx._protein_heavy.positions, dna_ag.positions, cutoff, box)
        m = _group(cx._protein_heavy_res[pi], dna_res[dj], dist, n_dna)
        np.minimum.at(closest, m.codes, m.min_dist)
        scan.frames.append(f)
        scan.times.append(t)
        scan.codes_per_frame.append(m.codes)
    return scan


def _frame_iter(
    cx: Complex, frame: int | None, start: int | None, stop: int | None, step: int | None
) -> Iterable[tuple[int, float]]:
    if frame is None:
        return cx._frames(start, stop, step)
    if (start, stop, step) != (None, None, None):
        raise ValueError("pass either frame or start/stop/step, not both")
    n = cx.frames.n_frames
    index = frame % n if -n <= frame < n else frame
    cx._goto(index)
    return cx._frames(index, index + 1, None)


def _atom_pairs(
    protein_pos: npt.NDArray[np.float32],
    dna_pos: npt.NDArray[np.float32],
    cutoff: float | None,
    box: npt.NDArray[np.float32] | None,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Heavy-atom pairs within ``cutoff`` (all pairs if ``None``) for the current frame.

    MDAnalysis finds candidate pairs within ``cutoff + _CANDIDATE_PAD``. Their
    distances are then recomputed in float64 (:func:`~neurodna.geometry.pair_distances`),
    and the inclusive ``<= cutoff`` test is applied to those.
    """
    if cutoff is None:
        pi, dj = np.divmod(np.arange(len(protein_pos) * len(dna_pos), dtype=np.intp),
                           len(dna_pos))
    else:
        pairs = capped_distance(protein_pos, dna_pos, max_cutoff=cutoff + _CANDIDATE_PAD,
                                box=box, return_distances=False)
        pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        pi, dj = pairs[:, 0], pairs[:, 1]
    dist = pair_distances(protein_pos[pi], dna_pos[dj], box)
    if cutoff is None:
        return pi, dj, dist
    keep = dist <= cutoff  # contact definition: distance <= cutoff, inclusive
    return pi[keep], dj[keep], dist[keep]


def _group(
    p_res: npt.NDArray[np.intp],
    d_res: npt.NDArray[np.intp],
    dist: npt.NDArray[np.float64],
    n_dna: int,
) -> _Minima:
    """Reduce atom pairs to residue pairs: minimum distance and closest atom pair."""
    codes = p_res * n_dna + d_res
    if codes.size == 0:
        empty = np.array([], dtype=np.intp)
        return _Minima(empty, np.array([], dtype=float), empty, empty)
    order = np.lexsort((dist, codes))
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    counts = np.diff(np.r_[starts, codes.size]).astype(np.intp)
    first = order[starts]
    return _Minima(sorted_codes[starts], dist[first], first, counts)


def _target_atoms(cx: Complex, target: Target) -> tuple[mda.AtomGroup, npt.NDArray[np.intp]]:
    if target == "dna":
        return cx._dna_heavy, cx._dna_heavy_res
    if target != "modification_markers":
        raise ValueError(f"target must be 'dna' or 'modification_markers', got {target!r}")
    mask = np.zeros(len(cx._dna_heavy), dtype=bool)
    for i, (name, local) in enumerate(zip(cx._dna_heavy.names, cx._dna_heavy_res)):
        spec = cx._dna_res[local].spec
        mask[i] = spec.is_modified and normalize_atom_name(name) in spec.marker_atoms
    if not mask.any():
        raise SelectionError(
            "target='modification_markers' but the DNA selection contains no heavy marker atoms "
            "of modified nucleotides"
        )
    return cx._dna_heavy[mask], cx._dna_heavy_res[mask]


def _pair_columns(cx: Complex, codes: npt.NDArray[np.intp]) -> pd.DataFrame:
    """Residue identity columns for residue-pair codes (vectorised)."""
    n_dna = len(cx._dna_res)
    p_idx, d_idx = np.divmod(np.asarray(codes, dtype=np.intp), n_dna)
    data: dict[str, Any] = {}
    for prefix, residues, idx in (("protein", cx._protein_res, p_idx), ("dna", cx._dna_res, d_idx)):
        for field in KEY_FIELDS:
            values = np.array([getattr(r.key, field) for r in residues], dtype=object)
            data[f"{prefix}_{field}"] = values[idx] if idx.size else values[:0]
        labels = np.array([r.key.label() for r in residues], dtype=object)
        data[f"{prefix}_label"] = labels[idx] if idx.size else labels[:0]
    mods = np.array([r.spec.modification for r in cx._dna_res], dtype=object)
    data["dna_modification"] = mods[d_idx] if d_idx.size else mods[:0]
    df = pd.DataFrame(data)
    df["protein_resnum"] = df["protein_resnum"].astype(np.int64)
    df["dna_resnum"] = df["dna_resnum"].astype(np.int64)
    return df


def _atom_table(
    cx: Complex,
    pi: npt.NDArray[np.intp],
    dj: npt.NDArray[np.intp],
    dist: npt.NDArray[np.float64],
    dna_ag: mda.AtomGroup,
    dna_res: npt.NDArray[np.intp],
) -> pd.DataFrame:
    p_res = cx._protein_heavy_res[pi]
    order = np.lexsort((dist, dna_res[dj], p_res))
    pi, dj, dist = pi[order], dj[order], dist[order]
    df = _pair_columns(cx, cx._protein_heavy_res[pi] * len(cx._dna_res) + dna_res[dj])
    p_names = cx._protein_heavy.names[pi].astype(str)
    d_names = dna_ag.names[dj].astype(str)
    df["protein_atom"] = p_names
    df["dna_atom"] = d_names
    df["distance_A"] = dist
    df["protein_moiety"] = [protein_moiety(n) for n in p_names]
    df["dna_moiety"] = [dna_moiety(n) for n in d_names]
    return df


def _attrs(
    cx: Complex,
    analysis: str,
    *,
    cutoff: float | None = None,
    target: Target = "dna",
    minimum_image: bool,
) -> dict[str, Any]:
    fi = cx.frames
    out: dict[str, Any] = {
        "analysis": analysis,
        "units": dict(_UNITS),
        "atoms": "heavy atoms only",
        "target": target,
        "frame_kind": fi.kind.value,
        "time_source": fi.time_source.value if fi.time_source else None,
        "minimum_image": minimum_image,
        "periodic_policy": cx.periodic_policy,
    }
    if cutoff is not None:
        out["cutoff_A"] = cutoff
        out["contact_definition"] = CONTACT_DEFINITION
    return out


def _check_distance(value: float, name: str) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive distance in Å, got {value}")
