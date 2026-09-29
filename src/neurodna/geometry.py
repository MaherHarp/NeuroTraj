"""Periodic-box handling and minimum-image geometry (lengths in Å, angles in degrees).

Minimum-image distances are used only when a frame has a *valid periodic box*:

* the box has 3 positive, finite lengths and angles strictly between 0° and 180°;
* it is not a crystallographic unit cell: a PDB ``CRYST1`` record whose space
  group is not ``P 1`` describes a crystal lattice built by symmetry operations,
  not a periodic copy of the coordinates in the file;
* no rotational fitting transformation is attached to the trajectory, since
  rotating coordinates without rotating the box breaks the periodic geometry.

When minimum-image distances are used, every perpendicular width of the box must
be at least twice the distance cutoff, so that each pair has one unique nearest
image within the cutoff.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors

# Transformations that rotate coordinates (the box is not rotated with them).
_ROTATING_TRANSFORMATIONS = frozenset({"fit_rot_trans", "rotateby"})


def is_valid_box(dimensions: npt.ArrayLike | None) -> bool:
    """True if ``[a, b, c, alpha, beta, gamma]`` describes a usable periodic box."""
    if dimensions is None:
        return False
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (6,) or not np.all(np.isfinite(dims)):
        return False
    lengths, angles = dims[:3], dims[3:]
    return bool(np.all(lengths > 0) and np.all((angles > 0) & (angles < 180)))


def box_min_width(dimensions: npt.ArrayLike) -> float:
    """Smallest perpendicular width of a (possibly triclinic) box, in Å."""
    a, b, c = triclinic_vectors(np.asarray(dimensions, dtype=np.float32)).astype(float)
    volume = abs(float(np.dot(a, np.cross(b, c))))
    widths = [volume / float(np.linalg.norm(np.cross(u, v))) for u, v in ((b, c), (c, a), (a, b))]
    return min(widths)


def crystal_cell_note(universe: Any) -> str | None:
    """Describe the box as a crystallographic cell, or return ``None``.

    Checks the ``CRYST1`` record of PDB coordinate files. Returns a note such as
    ``"CRYST1 space group C 1 2 1"`` when the space group is not ``P 1``.
    """
    from MDAnalysis.coordinates.PDB import PDBReader

    reader = universe.trajectory
    if not isinstance(reader, PDBReader):
        return None
    filename = reader.filename
    if not isinstance(filename, (str, os.PathLike)) or not os.path.exists(filename):
        return None
    with open(filename, errors="replace") as handle:
        for line in handle:
            if line.startswith("CRYST1"):
                group = line[55:66].strip()
                if group and group.replace(" ", "").upper() != "P1":
                    return f"CRYST1 space group {group}"
                return None
            if line.startswith(("ATOM", "HETATM", "MODEL")):
                return None
    return None


def rotating_transformations(universe: Any) -> list[str]:
    """Names of trajectory transformations that rotate coordinates."""
    names = [type(t).__name__ for t in getattr(universe.trajectory, "transformations", ())]
    return [n for n in names if n in _ROTATING_TRANSFORMATIONS]


def minimum_image_vectors(
    vectors: npt.NDArray[np.floating[Any]], box: npt.NDArray[np.float32] | None
) -> npt.NDArray[np.float64]:
    """Apply the minimum-image convention to difference vectors (no-op without a box)."""
    v = np.asarray(vectors, dtype=np.float64)
    if box is None or v.size == 0:
        return v
    return np.asarray(minimize_vectors(v, np.asarray(box, dtype=np.float64)), dtype=np.float64)


def pair_distances(
    a: npt.NDArray[np.floating[Any]],
    b: npt.NDArray[np.floating[Any]],
    box: npt.NDArray[np.float32] | None,
) -> npt.NDArray[np.float64]:
    """Distances (Å) between corresponding rows of ``a`` and ``b``, in float64.

    Uses the minimum-image convention when ``box`` is given. Computing in float64
    makes cutoff comparisons deterministic: MDAnalysis' float32 minimum-image
    search can place a pair that is exactly at the cutoff on either side of it,
    depending on the search method it picks for the system size.
    """
    vectors = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if box is not None and vectors.size:
        vectors = np.asarray(minimize_vectors(vectors, np.asarray(box, dtype=np.float64)),
                             dtype=np.float64)
    result: npt.NDArray[np.float64] = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    return result
