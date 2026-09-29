"""neurodna: protein–DNA structure and MD trajectory analysis.

Distances are in angstroms (Å) and times in picoseconds (ps). Contacts are
geometric (minimum heavy-atom distance ≤ cutoff), not hydrogen bonds or energies.
"""

from neurodna.complex import (
    DEFAULT_CONTACT_CUTOFF,
    DEFAULT_REPORT_DISTANCE,
    Complex,
    DnaStrand,
)
from neurodna.errors import (
    AmbiguousElementError,
    AmbiguousResidueError,
    DynamicsUnavailableError,
    EmptySelectionError,
    IncompleteSelectionError,
    NeuroDNAError,
    OverlappingSelectionError,
    PeriodicBoxError,
    ResidueIdentityError,
    SelectionError,
    TimingUnavailableError,
    TrajectoryError,
    UnsupportedResidueError,
    WrappedStructureError,
)
from neurodna.frames import FrameInfo, FrameKind, TimeSource
from neurodna.interactions import CONTACT_DEFINITION, dna_moiety, protein_moiety
from neurodna.residues import (
    ResidueKey,
    ResidueRegistry,
    ResidueSpec,
    default_dna_registry,
    default_protein_registry,
)
from neurodna.structure import BACKBONE_SELECTION

__version__ = "0.3.0"

__all__ = [
    "BACKBONE_SELECTION",
    "CONTACT_DEFINITION",
    "DEFAULT_CONTACT_CUTOFF",
    "DEFAULT_REPORT_DISTANCE",
    "AmbiguousElementError",
    "AmbiguousResidueError",
    "Complex",
    "DnaStrand",
    "DynamicsUnavailableError",
    "EmptySelectionError",
    "FrameInfo",
    "FrameKind",
    "IncompleteSelectionError",
    "NeuroDNAError",
    "OverlappingSelectionError",
    "PeriodicBoxError",
    "ResidueIdentityError",
    "ResidueKey",
    "ResidueRegistry",
    "ResidueSpec",
    "SelectionError",
    "TimeSource",
    "TimingUnavailableError",
    "TrajectoryError",
    "UnsupportedResidueError",
    "WrappedStructureError",
    "default_dna_registry",
    "default_protein_registry",
    "dna_moiety",
    "protein_moiety",
]
