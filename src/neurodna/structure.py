"""Protein backbone RMSD and per-atom RMSF (Å) after least-squares superposition.

**Alignment selection.** Each frame is superposed onto a reference frame by an
unweighted least-squares rigid-body fit (Kabsch algorithm, proper rotations
only) of the *alignment atoms*. By default these are the protein backbone:
:data:`BACKBONE_SELECTION` (``"name N CA C O"``). The ``align`` string is
evaluated with MDAnalysis selection syntax *within the protein heavy atoms* of
the :class:`~neurodna.Complex`, so it can never pick up DNA, solvent or hydrogens.
The atoms that are measured (the backbone for RMSD, or the requested atoms for
RMSF) are then compared without any further fitting.

**Coordinates are never modified.** Superposition acts on NumPy copies of the
positions. The MDAnalysis universe, and therefore every distance and contact
calculation in :mod:`neurodna.interactions`, always sees the raw coordinates.

**Preprocessing: molecules must be whole.** Superposition is meaningless if a
molecule is split across a periodic boundary, or if the chains of a multi-chain
protein sit in different periodic images. neurodna does not unwrap coordinates.
It checks every analysed frame and raises :class:`~neurodna.WrappedStructureError`
when it finds

* a bonded pair (topology bonds, or backbone N–CA, CA–C, C–O and C(i)–N(i+1)
  bonds when the topology has none) longer than :data:`BOND_SPLIT_CUTOFF`, or an
  atom farther than :data:`RESIDUE_EXTENT_CUTOFF` from its residue's CA, when
  the pair is normal under the minimum-image convention or was normal in the
  reference frame; or
* chains whose centroids jump relative to each other by more than half the
  smallest box width.

Make molecules whole before analysis, for example:

* GROMACS: ``gmx trjconv -pbc mol -center`` (then ``-pbc nojump`` or
  ``-pbc cluster`` for multi-chain complexes);
* MDAnalysis: ``u.trajectory.add_transformations(unwrap(u.atoms))`` (needs bonds)
  followed by ``nojump``, and without rotational fitting transformations;
* CPPTRAJ: ``autoimage`` or ``unwrap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
import pandas as pd
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.exceptions import SelectionError as _MDASelectionError

from neurodna.errors import (
    AmbiguousResidueError,
    DynamicsUnavailableError,
    EmptySelectionError,
    SelectionError,
    WrappedStructureError,
)
from neurodna.frames import FrameKind, sampling_summary
from neurodna.geometry import box_min_width, minimum_image_vectors
from neurodna.residues import KEY_FIELDS, format_keys

if TYPE_CHECKING:
    from neurodna.complex import Complex

BACKBONE_SELECTION = "name N CA C O"
"""Default alignment and RMSD atoms: protein backbone N, CA, C and O."""

BOND_SPLIT_CUTOFF = 3.0
"""Bonded atoms farther apart than this (Å) are treated as broken."""

RESIDUE_EXTENT_CUTOFF = 12.0
"""Max distance (Å) of any atom from its residue's CA before the residue counts as split."""

SUPERPOSITION = "unweighted least-squares rigid-body superposition (Kabsch), proper rotations only"

PREPROCESSING = (
    "Molecules must be whole and multi-chain proteins kept in one periodic image "
    "(e.g. gmx trjconv -pbc mol -center, then -pbc nojump/cluster; MDAnalysis unwrap + "
    "nojump; CPPTRAJ autoimage/unwrap). neurodna checks this but never unwraps coordinates."
)

Array = npt.NDArray[np.float64]


# ================================================================== numerics


def kabsch(mobile: npt.ArrayLike, target: npt.ArrayLike) -> tuple[Array, Array, Array]:
    """Optimal proper rotation superposing ``mobile`` onto ``target`` (both ``(n, 3)``).

    Returns:
        ``(rotation, mobile_centroid, target_centroid)`` such that
        ``(x - mobile_centroid) @ rotation.T + target_centroid`` maps ``mobile``
        coordinates into the target frame.

    Raises:
        ValueError: fewer than 3 atoms, or collinear atoms (rotation undefined).
    """
    p = np.asarray(mobile, dtype=float)
    q = np.asarray(target, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected two (n, 3) arrays of equal shape, got {p.shape} and {q.shape}")
    if p.shape[0] < 3:
        raise ValueError("superposition needs at least 3 atoms")
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    p0, q0 = p - pc, q - qc
    h = p0.T @ q0
    u, s, vt = np.linalg.svd(h)
    if s[1] <= 1e-8 * max(s[0], 1e-300):
        raise ValueError("alignment atoms are collinear; the superposition rotation is undefined")
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, pc, qc


def superpose(
    coords: npt.ArrayLike, rotation: Array, mobile_centroid: Array, target_centroid: Array
) -> Array:
    """Apply a :func:`kabsch` transform to a copy of ``coords``."""
    return (np.asarray(coords, dtype=float) - mobile_centroid) @ rotation.T + target_centroid


def rmsd(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Root-mean-square deviation (Å) between two ``(n, 3)`` coordinate sets, no fitting."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ================================================================== analyses


def backbone_rmsd(
    cx: Complex,
    *,
    align: str = BACKBONE_SELECTION,
    reference_frame: int = 0,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Backbone RMSD (Å) of each frame to ``reference_frame`` after superposition.

    The RMSD atoms are the protein backbone (:data:`BACKBONE_SELECTION`). The
    superposition uses the ``align`` atoms (default: the same backbone). Works for
    trajectories (with ``time_ps``) and unordered ensembles (per model, no time).

    Columns: ``frame``, ``time_ps``, ``rmsd_A``. ``attrs`` records the alignment
    selection, atom counts, reference frame and sampling.
    """
    if cx.kind is FrameKind.STATIC:
        raise DynamicsUnavailableError("backbone_rmsd needs several frames; this is one structure")
    align_ag = _within_protein(cx, align, "alignment")
    measure_ag = _within_protein(cx, BACKBONE_SELECTION, "backbone")
    fit = _Fitter(cx, align_ag, measure_ag, reference_frame)

    frames: list[int] = []
    times: list[float] = []
    values: list[float] = []
    for f, t in cx._frames(start, stop, step):
        moved = fit.aligned_measure()
        frames.append(f)
        times.append(t)
        values.append(rmsd(moved, fit.reference_measure))
    df = pd.DataFrame({"frame": np.asarray(frames, dtype=np.intp), "time_ps": times,
                       "rmsd_A": values})
    df.attrs.update(fit.attrs("backbone_rmsd", align, BACKBONE_SELECTION),
                    **sampling_summary(frames, times))
    return df


def rmsf(
    cx: Complex,
    atoms: str = "all",
    *,
    align: str = BACKBONE_SELECTION,
    reference_frame: int = 0,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Per-atom root-mean-square fluctuation (Å) about the mean aligned position.

    Every analysed frame is superposed onto ``reference_frame`` using the
    ``align`` atoms. The RMSF of atom *i* is ``sqrt(mean_t |x_i(t) - <x_i>|^2)``
    over the analysed frames. ``atoms`` selects the measured atoms within the
    protein heavy atoms (default: all of them). Trajectories only: fluctuations
    across an unordered ensemble are structural variability, not dynamics.

    Columns: residue identity, ``atom_name``, ``atom_index`` (0-based universe
    index) and ``rmsf_A``.
    """
    cx._require_trajectory("rmsf")
    align_ag = _within_protein(cx, align, "alignment")
    measure_ag = _within_protein(cx, atoms, "RMSF")
    fit = _Fitter(cx, align_ag, measure_ag, reference_frame)
    keys = _atom_keys(cx, measure_ag)

    n = 0
    mean = np.zeros((len(measure_ag), 3))
    m2 = np.zeros(len(measure_ag))
    frames: list[int] = []
    times: list[float] = []
    for f, t in cx._frames(start, stop, step):
        x = fit.aligned_measure()
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += np.sum(delta * (x - mean), axis=1)  # Welford update of the squared norm
        frames.append(f)
        times.append(t)
    if n < 2:
        raise ValueError("rmsf needs at least 2 analysed frames")

    data: dict[str, Any] = {f: [getattr(k, f) for k in keys] for f in KEY_FIELDS}
    data["label"] = [k.label() for k in keys]
    data["atom_name"] = measure_ag.names.astype(str)
    data["atom_index"] = measure_ag.indices.astype(np.intp)
    data["rmsf_A"] = np.sqrt(m2 / n)
    df = pd.DataFrame(data)
    df.attrs.update(fit.attrs("rmsf", align, atoms), **sampling_summary(frames, times))
    return df


# ================================================================= internals


class _Fitter:
    """Superposition onto a fixed reference, with whole-molecule checks per frame."""

    def __init__(
        self, cx: Complex, align_ag: mda.AtomGroup, measure_ag: mda.AtomGroup, reference: int
    ) -> None:
        if len(align_ag) < 3:
            raise SelectionError(
                f"the alignment selection has {len(align_ag)} heavy atom(s); at least 3 are needed"
            )
        self._cx = cx
        self._align = align_ag
        self._measure = measure_ag
        self._checker = _WholenessChecker(cx, align_ag | measure_ag)
        self._reference = reference
        cx._goto(reference)
        self._checker.set_reference()
        self.reference_align = align_ag.positions.astype(float)
        self.reference_measure = measure_ag.positions.astype(float)
        kabsch(self.reference_align, self.reference_align)  # rejects collinear alignments early

    def aligned_measure(self) -> Array:
        """Measured atoms of the current frame superposed onto the reference."""
        self._checker.check()
        rotation, pc, qc = kabsch(self._align.positions, self.reference_align)
        return superpose(self._measure.positions, rotation, pc, qc)

    def attrs(self, analysis: str, align: str, measured: str) -> dict[str, Any]:
        fi = self._cx.frames
        return {
            "analysis": analysis,
            "units": {"distance": "angstrom", "time": "ps"},
            "alignment_selection": align,
            "alignment_scope": "within protein heavy atoms",
            "n_alignment_atoms": len(self._align),
            "measured_selection": measured,
            "n_measured_atoms": len(self._measure),
            "reference_frame": self._reference,
            "superposition": SUPERPOSITION,
            "frame_kind": fi.kind.value,
            "time_source": fi.time_source.value if fi.time_source else None,
            "preprocessing": PREPROCESSING,
            "chain_gaps": self._checker.gap_labels,
        }


class _WholenessChecker:
    """Detects molecules split across periodic boundaries, frame by frame.

    Checked atom pairs (lengths compared directly and under minimum image):

    * bonds: topology bonds if available (strict: must never exceed
      :data:`BOND_SPLIT_CUTOFF`), otherwise backbone bonds (a bond already long
      in the reference frame is a chain gap, e.g. missing residues);
    * residue extent: each atom to its residue's CA (or first atom), limit
      :data:`RESIDUE_EXTENT_CUTOFF`, which catches wrapped side chains even
      without topology bonds.
    """

    _TOPOLOGY, _BACKBONE, _EXTENT = 0, 1, 2

    def __init__(self, cx: Complex, atoms: mda.AtomGroup) -> None:
        self._cx = cx
        self._atoms = atoms
        self._key_of = {r.mda_index: r.key for r in cx._protein_res}
        pos_of = {int(ix): i for i, ix in enumerate(atoms.indices)}
        bonds, topology = self._bond_pairs(atoms)
        extents = self._extent_pairs(atoms)
        kinds = [self._TOPOLOGY if topology else self._BACKBONE] * len(bonds) + [self._EXTENT] * len(extents)
        self._pairs = np.array(
            [(pos_of[a], pos_of[b]) for a, b in (*bonds, *extents)], dtype=np.intp
        ).reshape(-1, 2)
        self._kind = np.asarray(kinds, dtype=np.intp)
        self._limit = np.where(self._kind == self._EXTENT, RESIDUE_EXTENT_CUTOFF, BOND_SPLIT_CUTOFF)
        self._ref_long = np.zeros(len(self._pairs), dtype=bool)
        self.gap_labels: list[str] = []
        groups: dict[tuple[str, str], list[int]] = {}
        for i, resindex in enumerate(atoms.resindices):
            k = self._key_of[int(resindex)]
            groups.setdefault((k.segid, k.chain), []).append(i)
        self._groups = [np.asarray(v, dtype=np.intp) for v in groups.values()]
        self._group_names = [f"{s}/{c}" if s and s != c else (c or s or "-") for s, c in groups]
        self._ref_rel: Array | None = None

    def _bond_pairs(self, atoms: mda.AtomGroup) -> tuple[list[tuple[int, int]], bool]:
        try:
            bonds = atoms.intra_bonds
            return [(int(b.atoms[0].ix), int(b.atoms[1].ix)) for b in bonds], True
        except (NoDataError, AttributeError):
            pass
        pairs: list[tuple[int, int]] = []
        by_res: dict[int, dict[str, int]] = {}
        for atom in atoms:
            by_res.setdefault(int(atom.resindex), {})[atom.name.strip()] = int(atom.ix)
        prev_c: tuple[tuple[str, str], int] | None = None
        for r in self._cx._protein_res:
            names = by_res.get(r.mda_index)
            if names is None:
                prev_c = None
                continue
            for a, b in (("N", "CA"), ("CA", "C"), ("C", "O")):
                if a in names and b in names:
                    pairs.append((names[a], names[b]))
            chain_id = (r.key.segid, r.key.chain)
            if prev_c is not None and prev_c[0] == chain_id and "N" in names:
                pairs.append((prev_c[1], names["N"]))
            prev_c = (chain_id, names["C"]) if "C" in names else None
        return pairs, False

    @staticmethod
    def _extent_pairs(atoms: mda.AtomGroup) -> list[tuple[int, int]]:
        by_res: dict[int, list[Any]] = {}
        for atom in atoms:
            by_res.setdefault(int(atom.resindex), []).append(atom)
        pairs: list[tuple[int, int]] = []
        for members in by_res.values():
            anchor = next((a for a in members if a.name.strip() == "CA"), members[0])
            pairs.extend((int(anchor.ix), int(a.ix)) for a in members if a.ix != anchor.ix)
        return pairs

    def set_reference(self) -> None:
        """Check the reference frame; backbone bonds that are long here are chain gaps."""
        long_direct, long_mi = self._long(self._cx._box(None))
        self._raise_if(long_direct & ~long_mi, "split across the periodic boundary")
        self._raise_if(long_mi & (self._kind == self._TOPOLOGY),
                       "broken (longer than the limit even under minimum image)")
        self._raise_if(long_mi & (self._kind == self._EXTENT),
                       "implausibly far apart within one residue")
        self._ref_long = long_mi
        gaps = self._pairs[long_mi & (self._kind == self._BACKBONE)]
        self.gap_labels = [f"{self._label(int(a))}-{self._label(int(b))}" for a, b in gaps]
        self._ref_rel = self._relative_centroids()

    def check(self) -> None:
        box = self._cx._box(None)
        long_direct, long_mi = self._long(box)
        self._raise_if(long_direct & ~long_mi, "split across the periodic boundary")
        self._raise_if(long_mi & ~self._ref_long,
                       "stretched beyond the limit although normal in the reference frame")
        if box is not None and self._ref_rel is not None and len(self._groups) > 1:
            shift = np.linalg.norm(self._relative_centroids() - self._ref_rel, axis=1)
            limit = 0.5 * box_min_width(box)
            jumped = np.flatnonzero(shift > limit)
            if jumped.size:
                names = [self._group_names[i + 1] for i in jumped]
                raise WrappedStructureError(
                    f"frame {self._frame()}: chain(s) {names} moved by more than half the box "
                    f"width ({limit:.1f} Å) relative to chain {self._group_names[0]} since the "
                    "reference frame: the chains are in different periodic images. "
                    + PREPROCESSING
                )

    def _long(
        self, box: npt.NDArray[np.float32] | None
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        if self._pairs.size == 0:
            empty = np.zeros(0, dtype=bool)
            return empty, empty
        pos = self._atoms.positions
        vec = pos[self._pairs[:, 1]] - pos[self._pairs[:, 0]]
        direct = np.linalg.norm(vec.astype(float), axis=1)
        mi = np.linalg.norm(minimum_image_vectors(vec, box), axis=1)
        return direct > self._limit, mi > self._limit

    def _raise_if(self, mask: npt.NDArray[np.bool_], problem: str) -> None:
        if not mask.any():
            return
        bad = self._pairs[mask]
        labels = [f"{self._label(int(a))}-{self._label(int(b))}" for a, b in bad[:5]]
        more = f" ... and {len(bad) - 5} more" if len(bad) > 5 else ""
        raise WrappedStructureError(
            f"frame {self._frame()}: {len(bad)} bonded or same-residue atom pair(s) are "
            f"{problem}: {', '.join(labels)}{more}. " + PREPROCESSING
        )

    def _relative_centroids(self) -> Array:
        pos = np.asarray(self._atoms.positions, dtype=float)
        centroids = np.array([pos[g].mean(axis=0) for g in self._groups], dtype=float)
        rel: Array = centroids[1:] - centroids[0]
        return rel

    def _label(self, pos: int) -> str:
        atom = self._atoms[pos]
        return f"{self._key_of[int(atom.resindex)].label()}:{atom.name}"

    def _frame(self) -> int:
        return int(self._cx.universe.trajectory.ts.frame)


def _within_protein(cx: Complex, selection: str, what: str) -> mda.AtomGroup:
    if not isinstance(selection, str) or not selection.strip():
        raise SelectionError(f"the {what} selection must be a non-empty selection string")
    try:
        ag = cx._protein_heavy.select_atoms(selection)
    except (_MDASelectionError, ValueError) as exc:
        raise SelectionError(f"invalid {what} selection {selection!r}: {exc}") from exc
    if len(ag) == 0:
        raise EmptySelectionError(
            f"the {what} selection {selection!r} matched no protein heavy atoms"
        )
    return ag


def _atom_keys(cx: Complex, ag: mda.AtomGroup) -> list[Any]:
    key_of = {r.mda_index: r.key for r in cx._protein_res}
    keys = [key_of[int(ix)] for ix in ag.resindices]
    seen: dict[tuple[Any, str], int] = {}
    dups = []
    for k, name in zip(keys, ag.names):
        ident = (k.location, str(name))
        if ident in seen:
            dups.append(k)
        seen[ident] = 1
    if dups:
        raise AmbiguousResidueError(
            "duplicate atom names within residue(s) make per-atom results ambiguous: "
            + format_keys(dups),
            dups,
        )
    return keys
