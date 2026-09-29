"""Exception hierarchy.

Every error raised deliberately by neurodna derives from :class:`NeuroDNAError`.
Errors about specific residues carry them in a ``residues`` attribute so callers
can inspect the offending residues programmatically.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurodna.residues import ResidueKey


class NeuroDNAError(Exception):
    """Base class for all neurodna errors."""


class SelectionError(NeuroDNAError, ValueError):
    """An atom selection is invalid (bad syntax, empty, overlapping, incomplete)."""


class EmptySelectionError(SelectionError):
    """An atom selection matched no atoms."""


class OverlappingSelectionError(SelectionError):
    """The protein and DNA selections share atoms or residues."""


class IncompleteSelectionError(SelectionError):
    """Nucleotide-like residues next to the selected DNA were left out of it."""


class ResidueIdentityError(NeuroDNAError, ValueError):
    """A residue cannot be identified unambiguously."""

    def __init__(self, message: str, residues: Iterable[ResidueKey] = ()) -> None:
        super().__init__(message)
        self.residues: tuple[ResidueKey, ...] = tuple(residues)


class UnsupportedResidueError(ResidueIdentityError):
    """A residue name is not in the registry, or its chemistry is not supported."""


class AmbiguousResidueError(ResidueIdentityError):
    """A residue's identity is ambiguous (duplicate keys, mixed chains, altlocs, ...)."""


class TrajectoryError(NeuroDNAError):
    """The frames are inconsistent with the requested interpretation."""


class DynamicsUnavailableError(TrajectoryError):
    """A time-resolved analysis was requested on a static structure or unordered ensemble."""


class PeriodicBoxError(NeuroDNAError, ValueError):
    """Periodic boundary handling was requested but the box is missing or too small."""


class AmbiguousElementError(NeuroDNAError, ValueError):
    """Atoms cannot be classified unambiguously as hydrogen or heavy atoms."""

    def __init__(self, message: str, atoms: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.atoms: tuple[str, ...] = tuple(atoms)


class TimingUnavailableError(DynamicsUnavailableError):
    """A time-dependent metric was requested but frame times are unknown."""


class WrappedStructureError(TrajectoryError):
    """Molecules are split across periodic boundaries (not made whole)."""
