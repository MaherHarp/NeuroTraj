"""Loading and validating a protein–DNA complex.

Units: lengths are in angstroms (Å) and times in picoseconds (ps), the native
units of MDAnalysis. See :mod:`neurodna.units`.

This module owns loading, validation, residue identity, heavy-atom
classification, frame iteration and the periodic-box policy. The analyses live
in :mod:`neurodna.interactions` (distances and contacts, raw coordinates only) and
:mod:`neurodna.structure` (superposition, RMSD, RMSF, on copies of the
coordinates). The :class:`Complex` methods delegate to them.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Union

import MDAnalysis as mda
import numpy as np
import numpy.typing as npt
import pandas as pd
from MDAnalysis.exceptions import SelectionError as _MDASelectionError
from MDAnalysis.lib.distances import capped_distance

from neurodna import interactions, structure
from neurodna.elements import classify_heavy_atoms
from neurodna.errors import (
    AmbiguousElementError,
    AmbiguousResidueError,
    DynamicsUnavailableError,
    EmptySelectionError,
    IncompleteSelectionError,
    OverlappingSelectionError,
    PeriodicBoxError,
    SelectionError,
    TimingUnavailableError,
    TrajectoryError,
    UnsupportedResidueError,
)
from neurodna.frames import FrameInfo, FrameKind, TimeSource
from neurodna.geometry import (
    box_min_width,
    crystal_cell_note,
    is_valid_box,
    rotating_transformations,
)
from neurodna.residues import (
    KEY_FIELDS,
    MAX_LISTED,
    IdentifiedResidue,
    ResidueKey,
    ResidueRegistry,
    default_dna_registry,
    default_protein_registry,
    format_keys,
    key_columns,
    normalize_atom_name,
)

PathLike = Union[str, "os.PathLike[str]"]
Target = Literal["dna", "modification_markers"]

DEFAULT_CONTACT_CUTOFF = 4.5
"""Default geometric contact cutoff: minimum heavy-atom distance ≤ 4.5 Å."""

DEFAULT_REPORT_DISTANCE = 10.0
"""Default reporting radius (Å) for :meth:`Complex.min_distances`."""

O3P_LINK_CUTOFF = 2.0
"""Max O3'(i)–P(i+1) distance in Å for two nucleotides to count as covalently linked."""

# Sugar-phosphate backbone atom names; a residue with several of them is nucleotide-like.
_NUCLEOTIDE_BACKBONE = frozenset({"P", "O5'", "C5'", "C4'", "C3'", "O3'", "C1'", "O4'"})


@dataclass(frozen=True)
class DnaStrand:
    """A covalently linked run of nucleotides, ordered 5'→3'."""

    residues: tuple[ResidueKey, ...]
    one_letter: tuple[str, ...]
    modifications: tuple[str | None, ...]
    circular: bool = False

    @property
    def sequence(self) -> str:
        """Sequence of parent bases, e.g. ``"GCGC"`` for G-5mC-G-C."""
        return "".join(self.one_letter)

    @property
    def annotated_sequence(self) -> str:
        """Sequence with modifications written in brackets, e.g. ``"G[5mC]GC"``."""
        return "".join(
            f"[{mod}]" if mod is not None else letter
            for letter, mod in zip(self.one_letter, self.modifications)
        )

    def __len__(self) -> int:
        return len(self.residues)


class Complex:
    """A validated protein–DNA complex backed by an MDAnalysis ``Universe``.

    Use :meth:`load` to read files, or construct directly from an existing
    ``Universe`` (then ``frame_kind`` must be stated explicitly; note that
    MDAnalysis in-memory readers report ``dt = 1 ps`` unless told otherwise, so
    pass ``timestep_ps`` or build the reader with the real ``dt``).

    All distances are in Å and all times in ps. Contacts and distances use heavy
    atoms only (see :mod:`neurodna.elements`).

    Args:
        universe: The MDAnalysis universe.
        protein: MDAnalysis selection string for the protein atoms.
        dna: MDAnalysis selection string for the DNA atoms. Do not rely on the
            MDAnalysis ``nucleic`` keyword alone: it does not match modified
            residues such as ``5CM``. Prefer segment/chain based selections.
        frame_kind: How to interpret the frames.
        dna_registry / protein_registry: Residue-name registries; defaults from
            :func:`~neurodna.residues.default_dna_registry` and
            :func:`~neurodna.residues.default_protein_registry`.
        timestep_ps: Time between frames (trajectories only). Overrides file
            times; frame *i* is assigned time ``i * timestep_ps``.
        pbc: Periodic-boundary policy for distances and whole-molecule checks.
            ``None`` (default): use minimum-image distances in every frame that
            has a valid periodic box (see :mod:`neurodna.geometry`), but never a
            crystallographic unit cell. ``True``: require a valid box in every
            analysed frame. ``False``: never use the box.
        allow_partial_dna: Permit leaving nucleotide-like residues out of the
            DNA selection when they share a chain/segment with selected DNA.

    Raises:
        SelectionError: invalid, empty, overlapping or incomplete selections.
        UnsupportedResidueError: residue names missing from the registries, or RNA.
        AmbiguousResidueError: residues that cannot be identified uniquely.
        AmbiguousElementError: atoms that cannot be classified as heavy/hydrogen.
        TrajectoryError: ``frame_kind`` inconsistent with the number of frames.
    """

    def __init__(
        self,
        universe: mda.Universe,
        *,
        protein: str,
        dna: str,
        frame_kind: FrameKind,
        dna_registry: ResidueRegistry | None = None,
        protein_registry: ResidueRegistry | None = None,
        timestep_ps: float | None = None,
        pbc: bool | None = None,
        allow_partial_dna: bool = False,
    ) -> None:
        self._u = universe
        if pbc not in (None, True, False):
            raise ValueError(f"pbc must be None, True or False, got {pbc!r}")
        self._pbc = pbc
        self._cell_note = crystal_cell_note(universe)
        self._dna_registry = dna_registry if dna_registry is not None else default_dna_registry()
        self._protein_registry = (
            protein_registry if protein_registry is not None else default_protein_registry()
        )
        if self._dna_registry.molecule != "dna" or self._protein_registry.molecule != "protein":
            raise ValueError("dna_registry/protein_registry have the wrong molecule type")

        self._frame_info = self._make_frame_info(frame_kind, timestep_ps)

        self._protein_ag = self._select(protein, "protein")
        self._dna_ag = self._select(dna, "DNA")
        self._check_overlap()

        self._protein_res = self._identify(self._protein_ag, self._protein_registry, "protein")
        self._dna_res = self._identify(self._dna_ag, self._dna_registry, "DNA")
        self._check_unique_keys()
        self._check_altlocs()
        self._check_dna_chemistry()
        if not allow_partial_dna:
            self._check_dna_completeness()

        self._protein_heavy, self._protein_heavy_res = self._heavy_atoms(
            self._protein_ag, self._protein_res, "protein"
        )
        self._dna_heavy, self._dna_heavy_res = self._heavy_atoms(self._dna_ag, self._dna_res, "DNA")

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(
        cls,
        topology: PathLike,
        trajectory: PathLike | Sequence[PathLike] | None = None,
        *,
        protein: str,
        dna: str,
        time_ordered: bool | None = None,
        timestep_ps: float | None = None,
        dna_registry: ResidueRegistry | None = None,
        protein_registry: ResidueRegistry | None = None,
        pbc: bool | None = None,
        allow_partial_dna: bool = False,
    ) -> Complex:
        """Load a complex from a topology file and an optional trajectory.

        Frame interpretation:

        * one frame -> :attr:`FrameKind.STATIC`;
        * several frames from ``trajectory`` -> :attr:`FrameKind.TRAJECTORY`
          (pass ``time_ordered=False`` for an unordered set of structures);
        * several models in the topology file itself (e.g. an NMR PDB) ->
          :attr:`FrameKind.ENSEMBLE` (pass ``time_ordered=True`` only if the
          models really are consecutive simulation frames).

        See :class:`Complex` for the remaining arguments.
        """
        if trajectory is None:
            universe = mda.Universe(os.fspath(topology))
        elif isinstance(trajectory, (str, os.PathLike)):
            universe = mda.Universe(os.fspath(topology), os.fspath(trajectory))
        else:
            universe = mda.Universe(os.fspath(topology), [os.fspath(t) for t in trajectory])

        n_frames = universe.trajectory.n_frames
        if n_frames == 1:
            kind = FrameKind.STATIC
        elif trajectory is None:
            kind = FrameKind.TRAJECTORY if time_ordered is True else FrameKind.ENSEMBLE
        else:
            kind = FrameKind.ENSEMBLE if time_ordered is False else FrameKind.TRAJECTORY
        return cls(
            universe,
            protein=protein,
            dna=dna,
            frame_kind=kind,
            dna_registry=dna_registry,
            protein_registry=protein_registry,
            timestep_ps=timestep_ps,
            pbc=pbc,
            allow_partial_dna=allow_partial_dna,
        )

    # --------------------------------------------------------------- properties

    @property
    def universe(self) -> mda.Universe:
        return self._u

    @property
    def protein(self) -> mda.AtomGroup:
        """All selected protein atoms."""
        return self._protein_ag

    @property
    def dna(self) -> mda.AtomGroup:
        """All selected DNA atoms."""
        return self._dna_ag

    @property
    def protein_heavy(self) -> mda.AtomGroup:
        """Selected protein heavy atoms (used for contacts, RMSD and RMSF)."""
        return self._protein_heavy

    @property
    def dna_heavy(self) -> mda.AtomGroup:
        """Selected DNA heavy atoms (used for contacts)."""
        return self._dna_heavy

    @property
    def frames(self) -> FrameInfo:
        return self._frame_info

    @property
    def kind(self) -> FrameKind:
        return self._frame_info.kind

    @property
    def periodic_policy(self) -> str:
        """``"auto"``, ``"always"`` or ``"never"``, plus any crystallographic-cell note."""
        policy = {None: "auto", True: "always", False: "never"}[self._pbc]
        if self._cell_note is not None and self._pbc is None:
            policy += f" (box ignored: {self._cell_note} is a crystallographic cell)"
        return policy

    @property
    def protein_residue_keys(self) -> tuple[ResidueKey, ...]:
        return tuple(r.key for r in self._protein_res)

    @property
    def dna_residue_keys(self) -> tuple[ResidueKey, ...]:
        return tuple(r.key for r in self._dna_res)

    def __repr__(self) -> str:
        fi = self._frame_info
        return (
            f"<Complex {fi.kind.value}, {fi.n_frames} frame(s): "
            f"{len(self._protein_res)} protein / {len(self._dna_res)} DNA residues>"
        )

    def protein_residues(self) -> pd.DataFrame:
        """Table of selected protein residues with their registry identity."""
        return self._residue_table(self._protein_res, self._protein_ag, self._protein_heavy)

    def dna_residues(self) -> pd.DataFrame:
        """Table of selected DNA residues, including modifications (e.g. 5mC)."""
        return self._residue_table(self._dna_res, self._dna_ag, self._dna_heavy)

    # ------------------------------------------------------------------ frames

    def frame_times(self) -> npt.NDArray[np.float64]:
        """Times (ps) of all frames; reads the whole trajectory.

        NaN for trajectories without time information.

        Raises:
            DynamicsUnavailableError: if the complex is not a trajectory.
            TrajectoryError: if the file's times are not strictly increasing.
        """
        self._require_trajectory("frame_times")
        return np.array([t for _, t in self._frames(None, None, None)], dtype=float)

    # ------------------------------------------------- analyses: interactions

    def contacts(
        self,
        cutoff: float = DEFAULT_CONTACT_CUTOFF,
        *,
        frame: int = 0,
        level: Literal["residue", "atom"] = "residue",
        target: Target = "dna",
    ) -> pd.DataFrame:
        """Geometric protein–DNA contacts in one frame. Works for any frame kind.

        A contact is a residue–nucleotide pair whose minimum heavy-atom distance is
        ≤ ``cutoff`` Å. This is a geometric criterion, not a hydrogen bond, salt
        bridge or binding energy.

        Args:
            cutoff: Distance cutoff in Å (inclusive).
            frame: Frame (model) index.
            level: ``"residue"``: one row per contacting pair with
                ``min_distance_A``, the closest atoms and ``n_atom_pairs``.
                ``"atom"``: one row per heavy-atom pair with ``distance_A``.
            target: ``"dna"`` for all DNA heavy atoms, ``"modification_markers"``
                for only the marker atoms of modified nucleotides (e.g. the 5mC
                methyl carbon).
        """
        return interactions.contacts(self, cutoff, frame=frame, level=level, target=target)

    def min_distances(
        self,
        max_distance: float | None = DEFAULT_REPORT_DISTANCE,
        *,
        target: Target = "dna",
        frame: int | None = None,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Residue-to-nucleotide minimum heavy-atom distances, frame by frame.

        Tidy table: one row per (frame, protein residue, nucleotide) with
        ``frame``, ``time_ps``, residue identities, ``min_distance_A`` and the
        closest atoms. Only pairs with ``min_distance_A ≤ max_distance`` are
        listed; any pair missing from a frame is farther apart than that.
        ``max_distance=None`` lists every pair in every frame.

        Works for any frame kind. ``frame`` selects one frame; otherwise
        ``start/stop/step`` slice the frames. ``time_ps`` is NaN for static
        structures, ensembles and trajectories without time information.
        """
        return interactions.min_distances(
            self, max_distance, target=target, frame=frame, start=start, stop=stop, step=step
        )

    def contact_occupancy(
        self,
        cutoff: float = DEFAULT_CONTACT_CUTOFF,
        *,
        target: Target = "dna",
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Fraction of analysed trajectory frames in which each pair is in geometric contact.

        Only for :attr:`FrameKind.TRAJECTORY`. For an unordered ensemble use
        :meth:`ensemble_contact_frequency`. A static structure has no occupancy.

        Occupancy counts frames. It equals the fraction of *time* only for
        uniformly sampled frames: see ``attrs["uniform_sampling"]`` and the
        recorded sampling intervals.
        """
        return interactions.contact_occupancy(
            self, cutoff, target=target, start=start, stop=stop, step=step
        )

    def contact_episodes(
        self,
        cutoff: float = DEFAULT_CONTACT_CUTOFF,
        *,
        target: Target = "dna",
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Contiguous runs of analysed frames in which a pair is in geometric contact.

        Needs a trajectory with known frame times. See
        :func:`neurodna.interactions.contact_episodes` for the columns, censoring
        and why these are *not* binding residence times.
        """
        return interactions.contact_episodes(
            self, cutoff, target=target, start=start, stop=stop, step=step
        )

    def ensemble_contact_frequency(
        self,
        cutoff: float = DEFAULT_CONTACT_CUTOFF,
        *,
        target: Target = "dna",
    ) -> pd.DataFrame:
        """Fraction of ensemble models in which each pair is in geometric contact.

        Only for :attr:`FrameKind.ENSEMBLE`. This is a population statistic over
        models, not a time average: it says nothing about lifetimes or kinetics.
        """
        return interactions.ensemble_contact_frequency(self, cutoff, target=target)

    # ---------------------------------------------------- analyses: structure

    def backbone_rmsd(
        self,
        *,
        align: str = structure.BACKBONE_SELECTION,
        reference_frame: int = 0,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Protein backbone RMSD (Å) per frame after least-squares superposition.

        See :func:`neurodna.structure.backbone_rmsd`.
        """
        return structure.backbone_rmsd(
            self, align=align, reference_frame=reference_frame, start=start, stop=stop, step=step
        )

    def rmsf(
        self,
        atoms: str = "all",
        *,
        align: str = structure.BACKBONE_SELECTION,
        reference_frame: int = 0,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> pd.DataFrame:
        """Per-atom RMSF (Å) of protein heavy atoms after superposition.

        See :func:`neurodna.structure.rmsf`.
        """
        return structure.rmsf(
            self, atoms, align=align, reference_frame=reference_frame,
            start=start, stop=stop, step=step,
        )

    # ---------------------------------------------------------------- sequence

    def dna_strands(self, *, frame: int = 0) -> list[DnaStrand]:
        """Split the DNA into covalently linked strands ordered 5'→3'.

        Linkage is detected geometrically in ``frame``: residue *j* follows *i*
        when O3'(i)–P(j) ≤ 2.0 Å. Residues lacking O3' or P atoms therefore end
        a strand. Numbering and file order are not used.
        """
        self._goto(frame)
        n = len(self._dna_res)
        o3_pos, o3_res, p_pos, p_res = [], [], [], []
        for local, res in enumerate(self._dna_res):
            for atom in self._u.residues[res.mda_index].atoms:
                name = normalize_atom_name(atom.name)
                if name == "O3'":
                    o3_pos.append(atom.position)
                    o3_res.append(local)
                elif name == "P":
                    p_pos.append(atom.position)
                    p_res.append(local)

        nxt: dict[int, int] = {}
        prv: dict[int, int] = {}
        if o3_pos and p_pos:
            pairs = capped_distance(
                np.asarray(o3_pos, dtype=np.float32),
                np.asarray(p_pos, dtype=np.float32),
                max_cutoff=O3P_LINK_CUTOFF,
                box=self._box(O3P_LINK_CUTOFF),
                return_distances=False,
            )
            for a, b in pairs:
                i, j = o3_res[a], p_res[b]
                if i == j:
                    continue
                if nxt.get(i, j) != j or prv.get(j, i) != i:
                    keys = [self._dna_res[k].key for k in {i, j, nxt.get(i, j), prv.get(j, i)}]
                    raise AmbiguousResidueError(
                        "branched DNA linkage (a residue bonded to two 3' or two 5' neighbours): "
                        + format_keys(keys),
                        keys,
                    )
                nxt[i] = j
                prv[j] = i

        strands: list[DnaStrand] = []
        seen: set[int] = set()

        def walk(begin: int, circular: bool) -> None:
            order = [begin]
            seen.add(begin)
            cur = begin
            while cur in nxt and nxt[cur] not in seen:
                cur = nxt[cur]
                order.append(cur)
                seen.add(cur)
            strands.append(self._strand(order, circular))

        for i in range(n):
            if i not in prv and i not in seen:
                walk(i, circular=False)
        for i in range(n):  # anything left is on a closed loop
            if i not in seen:
                walk(i, circular=True)
        return strands

    def cpg_steps(self, *, frame: int = 0) -> pd.DataFrame:
        """All 5'-CpG-3' steps within DNA strands, methylated or not.

        One row per step with the C and G residue identities and
        ``c_modification`` (e.g. ``"5mC"``, or missing for unmodified C).
        Relevant to MeCP2, whose methyl-CpG binding domain recognises mCpG.
        """
        rows: list[dict[str, Any]] = []
        for strand_index, strand in enumerate(self.dna_strands(frame=frame)):
            n = len(strand)
            last = n if strand.circular else n - 1
            for i in range(last):
                j = (i + 1) % n
                if strand.one_letter[i] == "C" and strand.one_letter[j] == "G":
                    row: dict[str, Any] = {"strand": strand_index}
                    row.update(key_columns(strand.residues[i], "c"))
                    row.update(key_columns(strand.residues[j], "g"))
                    row["c_modification"] = strand.modifications[i]
                    rows.append(row)
        columns = (
            ["strand"]
            + [f"c_{f}" for f in KEY_FIELDS]
            + [f"g_{f}" for f in KEY_FIELDS]
            + ["c_modification"]
        )
        return pd.DataFrame(rows, columns=columns)

    # ================================================ internal: frame access
    # Used by neurodna.interactions and neurodna.structure.

    def _goto(self, frame: int) -> None:
        n = self._frame_info.n_frames
        if not -n <= frame < n:
            raise IndexError(f"frame {frame} out of range for {n} frame(s)")
        self._u.trajectory[frame]

    def _frames(
        self, start: int | None, stop: int | None, step: int | None
    ) -> Iterator[tuple[int, float]]:
        """Iterate frames in order, yielding ``(frame_index, time_ps)``.

        ``time_ps`` is NaN unless the complex is a trajectory with known times.
        File times must be strictly increasing.
        """
        if step is not None and step <= 0:
            raise ValueError("step must be positive: frames are analysed in order")
        fi = self._frame_info
        previous = -math.inf
        n = 0
        for ts in self._u.trajectory[start:stop:step]:
            n += 1
            if fi.time_source is TimeSource.FILE:
                t = float(ts.time)
                if not t > previous:
                    raise TrajectoryError(
                        f"frame times are not strictly increasing (frame {ts.frame}: {t} ps after "
                        f"{previous} ps). Concatenated or unordered trajectories cannot be "
                        "analysed as time series; fix the input, or pass timestep_ps / "
                        "time_ordered=False."
                    )
                previous = t
            elif fi.time_source is TimeSource.USER:
                assert fi.timestep_ps is not None
                t = ts.frame * fi.timestep_ps
            else:
                t = math.nan
            yield int(ts.frame), t
        if n == 0:
            raise ValueError(f"no frames selected by start={start}, stop={stop}, step={step}")

    def _require_trajectory(self, what: str) -> None:
        kind = self.kind
        if kind is FrameKind.TRAJECTORY:
            return
        if kind is FrameKind.STATIC:
            raise DynamicsUnavailableError(
                f"{what} needs a time-resolved trajectory, but this complex is a single static "
                "structure; dynamics cannot be inferred from one frame. Use contacts() instead."
            )
        raise DynamicsUnavailableError(
            f"{what} needs a time-resolved trajectory, but this complex is an unordered ensemble "
            "of models; dynamics cannot be inferred from it. Use ensemble_contact_frequency(), "
            "or load with time_ordered=True if the models really are consecutive MD frames."
        )

    def _require_times(self, what: str) -> None:
        self._require_trajectory(what)
        if not self._frame_info.has_times:
            raise TimingUnavailableError(
                f"{what} is a time-dependent metric, but this trajectory has no frame times "
                "(the file format stores none). Pass timestep_ps if the sampling interval is "
                "known."
            )

    def _box(self, cutoff: float | None) -> npt.NDArray[np.float32] | None:
        """Box of the current frame to use for minimum-image distances, or ``None``.

        Applies the ``pbc`` policy. With a cutoff, also checks that every
        perpendicular box width is at least twice the cutoff.
        """
        if self._pbc is False:
            return None
        dims = self._u.dimensions
        frame = self._u.trajectory.ts.frame
        if not is_valid_box(dims):
            if self._pbc is True:
                raise PeriodicBoxError(f"pbc=True but frame {frame} has no valid periodic box")
            return None
        if self._cell_note is not None and self._pbc is None:
            return None
        rotating = rotating_transformations(self._u)
        if rotating:
            raise PeriodicBoxError(
                f"the trajectory has rotating transformations {rotating}; rotated coordinates no "
                "longer match the periodic box, so minimum-image distances would be wrong. "
                "Remove them (neurodna superposes internally for RMSD/RMSF) or pass pbc=False."
            )
        if cutoff is not None:
            width = box_min_width(dims)
            if 2 * cutoff > width:
                raise PeriodicBoxError(
                    f"frame {frame}: the smallest box width ({width:.3f} Å) is less than twice "
                    f"the cutoff ({cutoff} Å); minimum-image distances would be ambiguous"
                )
        return np.asarray(dims, dtype=np.float32)

    # ======================================================= internal: setup

    def _make_frame_info(self, kind: FrameKind, timestep_ps: float | None) -> FrameInfo:
        traj = self._u.trajectory
        n = traj.n_frames
        if kind is FrameKind.STATIC and n != 1:
            raise TrajectoryError(f"frame_kind=STATIC but the universe has {n} frames")
        if kind is not FrameKind.STATIC and n < 2:
            raise TrajectoryError(
                f"frame_kind={kind.name} needs at least 2 frames, but the universe has {n}; "
                "a single structure is STATIC"
            )
        if timestep_ps is not None:
            if kind is not FrameKind.TRAJECTORY:
                raise TrajectoryError(
                    f"timestep_ps only applies to trajectories; this complex is {kind.value}"
                )
            if not (math.isfinite(timestep_ps) and timestep_ps > 0):
                raise ValueError(f"timestep_ps must be positive, got {timestep_ps}")

        source: TimeSource | None = None
        dt: float | None = None
        if kind is FrameKind.TRAJECTORY:
            if timestep_ps is not None:
                source, dt = TimeSource.USER, float(timestep_ps)
            else:
                # MDAnalysis invents dt = 1 ps (with a warning) for formats without time.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    file_dt = float(traj.dt)
                if any("no dt information" in str(w.message) for w in caught):
                    source = TimeSource.UNAVAILABLE
                else:
                    source, dt = TimeSource.FILE, file_dt
        traj[0]
        return FrameInfo(kind, n, source, dt, has_box=self._u.dimensions is not None)

    def _select(self, selection: str, what: str) -> mda.AtomGroup:
        if not isinstance(selection, str) or not selection.strip():
            raise SelectionError(f"the {what} selection must be a non-empty selection string")
        try:
            ag = self._u.select_atoms(selection)
        except (_MDASelectionError, ValueError) as exc:
            raise SelectionError(f"invalid {what} selection {selection!r}: {exc}") from exc
        if len(ag) == 0:
            raise EmptySelectionError(f"the {what} selection {selection!r} matched no atoms")
        return ag

    def _check_overlap(self) -> None:
        shared_atoms = np.intersect1d(self._protein_ag.indices, self._dna_ag.indices)
        shared_res = np.intersect1d(self._protein_ag.resindices, self._dna_ag.resindices)
        if shared_res.size:
            keys = [self._key(self._u.residues[int(i)]) for i in shared_res]
            raise OverlappingSelectionError(
                f"the protein and DNA selections overlap: {shared_atoms.size} shared atom(s) and "
                f"{shared_res.size} residue(s) in both selections: {format_keys(keys)}"
            )

    def _key(self, residue: Any) -> ResidueKey:
        atoms = residue.atoms
        chains: set[str] = set()
        if hasattr(atoms, "chainIDs"):
            chains = {str(c).strip() for c in atoms.chainIDs}
        segid = str(residue.segid).strip() if hasattr(residue, "segid") else ""
        icode = str(residue.icode).strip() if hasattr(residue, "icode") else ""
        chain = next(iter(chains)) if len(chains) == 1 else ""
        key = ResidueKey(segid, chain, int(residue.resid), icode, str(residue.resname).strip())
        if len(chains) > 1:
            raise AmbiguousResidueError(
                f"residue {key.label()} contains atoms from several chains {sorted(chains)}; "
                "MDAnalysis groups residues by segment and number, so identical residue numbers "
                "in different chains of one segment were merged. Give each chain its own segid "
                "(or leave the PDB segid column blank so chain IDs are used).",
                [key],
            )
        return key

    def _identify(
        self, ag: mda.AtomGroup, registry: ResidueRegistry, what: str
    ) -> list[IdentifiedResidue]:
        out: list[IdentifiedResidue] = []
        unknown: list[ResidueKey] = []
        for residue in ag.residues:
            key = self._key(residue)
            spec = registry.get(key.resname)
            if spec is None:
                unknown.append(key)
            else:
                out.append(IdentifiedResidue(key, spec, int(residue.ix)))
        if unknown:
            names = sorted({k.resname for k in unknown})
            raise UnsupportedResidueError(
                f"{len(unknown)} {what} residue(s) have names not in the {what} registry "
                f"{names}: {format_keys(unknown)}. Register them with ResidueRegistry.register() "
                f"or change the {what} selection.",
                unknown,
            )
        return out

    def _check_unique_keys(self) -> None:
        by_location: dict[tuple[str, str, int, str], list[ResidueKey]] = defaultdict(list)
        for r in (*self._protein_res, *self._dna_res):
            by_location[r.key.location].append(r.key)
        dups = [keys for keys in by_location.values() if len(keys) > 1]
        if dups:
            flat = [k for keys in dups for k in keys]
            raise AmbiguousResidueError(
                f"{len(dups)} residue identifier(s) occur more than once "
                "(same segment, chain, number and insertion code): "
                + format_keys(flat)
                + ". Residue numbers must be unique within a chain/segment.",
                flat,
            )

    def _check_altlocs(self) -> None:
        if not hasattr(self._u.atoms, "altLocs"):
            return
        bad: list[ResidueKey] = []
        for ag, residues in ((self._protein_ag, self._protein_res), (self._dna_ag, self._dna_res)):
            by_res: dict[int, set[str]] = defaultdict(set)
            for resindex, alt in zip(ag.resindices, ag.altLocs):
                if str(alt).strip():
                    by_res[int(resindex)].add(str(alt).strip())
            bad.extend(r.key for r in residues if len(by_res.get(r.mda_index, ())) > 1)
        if bad:
            raise AmbiguousResidueError(
                f"{len(bad)} selected residue(s) have several alternate locations: "
                f"{format_keys(bad)}. Keep one conformer, e.g. add 'and not altloc B' to the "
                "selections.",
                bad,
            )

    def _check_dna_chemistry(self) -> None:
        rna_like: list[ResidueKey] = []
        missing: list[tuple[ResidueKey, tuple[str, ...]]] = []
        for r in self._dna_res:
            names = {normalize_atom_name(n) for n in self._u.residues[r.mda_index].atoms.names}
            if "O2'" in names:
                rna_like.append(r.key)
            absent = tuple(a for a in r.spec.marker_atoms if a not in names)
            if absent:
                missing.append((r.key, absent))
        if rna_like:
            raise UnsupportedResidueError(
                f"{len(rna_like)} residue(s) in the DNA selection have a 2'-hydroxyl (O2') and "
                f"look like RNA, which is not supported: {format_keys(rna_like)}",
                rna_like,
            )
        if missing:
            keys = [k for k, _ in missing]
            detail = "; ".join(f"{k.label()} lacks {list(a)}" for k, a in missing[:MAX_LISTED])
            raise AmbiguousResidueError(
                "residue name(s) claim a modification whose marker atoms are absent, so the "
                f"identity cannot be confirmed: {detail}. If your files name these atoms "
                "differently, register the residue with the right marker_atoms (replace=True).",
                keys,
            )

    def _check_dna_completeness(self) -> None:
        selected = {r.mda_index for r in self._dna_res} | {r.mda_index for r in self._protein_res}
        dna_segids = {r.key.segid for r in self._dna_res if r.key.segid}
        dna_chains = {r.key.chain for r in self._dna_res if r.key.chain}
        # Vectorised pre-filter: only residues sharing a segment or chain with the DNA
        # (solvated MD systems have tens of thousands of irrelevant water residues).
        residues = self._u.residues
        candidate = np.isin(np.char.strip(residues.segids.astype(str)), list(dna_segids))
        if hasattr(self._u.atoms, "chainIDs") and dna_chains:
            atoms = self._u.atoms
            in_chain = np.isin(np.char.strip(atoms.chainIDs.astype(str)), list(dna_chains))
            candidate[np.unique(atoms.resindices[in_chain])] = True
        left_out: list[ResidueKey] = []
        for residue in residues[candidate]:
            if int(residue.ix) in selected:
                continue
            segid = str(residue.segid).strip() if hasattr(residue, "segid") else ""
            chains = (
                {str(c).strip() for c in residue.atoms.chainIDs}
                if hasattr(residue.atoms, "chainIDs")
                else set()
            )
            if segid not in dna_segids and not (chains & dna_chains):
                continue
            names = {normalize_atom_name(n) for n in residue.atoms.names}
            resname = str(residue.resname).strip()
            if resname in self._dna_registry or len(names & _NUCLEOTIDE_BACKBONE) >= 3:
                chain = next(iter(chains)) if len(chains) == 1 else ""
                icode = str(residue.icode).strip() if hasattr(residue, "icode") else ""
                left_out.append(ResidueKey(segid, chain, int(residue.resid), icode, resname))
        if left_out:
            raise IncompleteSelectionError(
                f"{len(left_out)} nucleotide-like residue(s) in the same chain/segment as the "
                f"selected DNA were left out of the DNA selection: {format_keys(left_out)}. "
                "This often happens with the MDAnalysis 'nucleic' keyword, which does not match "
                "modified bases such as 5CM. Select DNA by chain/segment instead, or pass "
                "allow_partial_dna=True if the exclusion is intended."
            )

    def _heavy_atoms(
        self, ag: mda.AtomGroup, residues: list[IdentifiedResidue], what: str
    ) -> tuple[mda.AtomGroup, npt.NDArray[np.intp]]:
        elements = ag.elements if hasattr(ag, "elements") else None
        is_heavy, problems = classify_heavy_atoms(ag.names, elements)
        if problems:
            key_by_res = {r.mda_index: r.key for r in residues}
            labels = [
                f"{key_by_res[int(ag[i].resindex)].label()} {ag[i].name}: {reason}"
                for i, reason in problems
            ]
            more = f" ... and {len(labels) - MAX_LISTED} more" if len(labels) > MAX_LISTED else ""
            raise AmbiguousElementError(
                f"{len(problems)} {what} atom(s) cannot be classified as hydrogen or heavy: "
                + "; ".join(labels[:MAX_LISTED])
                + more
                + ". Fix the element records, or exclude these atoms from the selection.",
                labels,
            )
        heavy = ag[is_heavy]
        if len(heavy) == 0:
            raise EmptySelectionError(f"the {what} selection contains no heavy atoms")
        return heavy, self._atom_to_local(heavy, residues)

    @staticmethod
    def _atom_to_local(
        ag: mda.AtomGroup, residues: list[IdentifiedResidue]
    ) -> npt.NDArray[np.intp]:
        lookup = {r.mda_index: i for i, r in enumerate(residues)}
        return np.array([lookup[int(ix)] for ix in ag.resindices], dtype=np.intp)

    def _residue_table(
        self, residues: list[IdentifiedResidue], ag: mda.AtomGroup, heavy: mda.AtomGroup
    ) -> pd.DataFrame:
        n_atoms = np.bincount(self._atom_to_local(ag, residues), minlength=len(residues))
        n_heavy = np.bincount(self._atom_to_local(heavy, residues), minlength=len(residues))
        rows = []
        for r, count, count_heavy in zip(residues, n_atoms, n_heavy):
            row: dict[str, Any] = {f: getattr(r.key, f) for f in KEY_FIELDS}
            row.update(
                label=r.key.label(),
                parent=r.spec.parent,
                one_letter=r.spec.one_letter,
                modification=r.spec.modification,
                n_atoms=int(count),
                n_heavy_atoms=int(count_heavy),
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _strand(self, order: list[int], circular: bool) -> DnaStrand:
        residues = [self._dna_res[i] for i in order]
        return DnaStrand(
            residues=tuple(r.key for r in residues),
            one_letter=tuple(r.spec.one_letter for r in residues),
            modifications=tuple(r.spec.modification for r in residues),
            circular=circular,
        )
