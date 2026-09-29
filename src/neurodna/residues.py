"""Residue identity and explicit residue-name registries.

Two ideas live here:

* :class:`ResidueKey` -- the identity of one residue. Residue numbers are *not*
  unique on their own (chain A and chain B both have a residue 1; insertion
  codes give 52 and 52A), so a key combines segment, chain, number, insertion
  code and name.
* :class:`ResidueRegistry` -- an explicit, extensible mapping from residue names
  to what they are. Nothing is identified by guesswork: a residue name that is
  not registered is reported as unsupported instead of being silently dropped.
  Modified nucleotides such as 5-methylcytosine are ordinary registry entries
  that record their parent nucleotide and modification.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

MoleculeType = Literal["dna", "protein"]


def normalize_atom_name(name: str) -> str:
    """Normalize an atom name for comparison (strip, old-style ``*`` -> ``'``)."""
    return name.strip().replace("*", "'")


@dataclass(frozen=True, order=True)
class ResidueKey:
    """Unique identity of a residue within a structure.

    Attributes:
        segid: Segment identifier ("" if the file has none).
        chain: Chain identifier ("" if the file has none).
        resnum: Residue number as written in the file.
        icode: PDB insertion code ("" if none).
        resname: Residue name as written in the file.
    """

    segid: str
    chain: str
    resnum: int
    icode: str
    resname: str

    @property
    def location(self) -> tuple[str, str, int, str]:
        """Positional identity (everything except the residue name)."""
        return (self.segid, self.chain, self.resnum, self.icode)

    def label(self) -> str:
        """Readable label such as ``B:5CM12`` or ``DNA1/B:DC7A``."""
        chain = self.chain or "-"
        seg = "" if self.segid in ("", self.chain) else f"{self.segid}/"
        return f"{seg}{chain}:{self.resname}{self.resnum}{self.icode}"

    def __str__(self) -> str:
        return self.label()


@dataclass(frozen=True)
class ResidueSpec:
    """What a residue name means.

    Attributes:
        resname: Residue name as it appears in structure files.
        parent: Canonical parent residue (e.g. ``"DC"`` for 5-methylcytosine).
        one_letter: One-letter code of the parent for sequences; ``""`` for
            residues that are not part of a sequence (e.g. terminal caps).
        modification: Short name of the chemical modification relative to the
            parent (e.g. ``"5mC"``), or ``None`` for unmodified residues.
        marker_atoms: Atom names that must be present for the name to be
            trusted, e.g. the 5-methyl carbon of 5-methylcytosine. They are
            also the atoms used by ``target="modification_markers"`` analyses.
    """

    resname: str
    parent: str
    one_letter: str
    modification: str | None = None
    marker_atoms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.resname or self.resname != self.resname.strip():
            raise ValueError(f"invalid residue name {self.resname!r}")
        if len(self.one_letter) > 1:
            raise ValueError(f"one_letter must be at most one character, got {self.one_letter!r}")
        if self.modification is not None and not self.marker_atoms:
            raise ValueError(
                f"modified residue {self.resname!r} needs at least one marker atom "
                "so its identity can be checked against the coordinates"
            )
        object.__setattr__(
            self, "marker_atoms", tuple(normalize_atom_name(a) for a in self.marker_atoms)
        )

    @property
    def is_modified(self) -> bool:
        return self.modification is not None


class ResidueRegistry:
    """Explicit mapping from residue names to :class:`ResidueSpec` for one molecule type.

    Example (atom names are illustrative -- use the names in your own files):
        >>> reg = default_dna_registry()
        >>> reg.register(ResidueSpec("5HC", parent="DC", one_letter="C",
        ...                          modification="5hmC", marker_atoms=("C5M", "O5M")))
    """

    def __init__(self, molecule: MoleculeType, specs: Iterable[ResidueSpec] = ()) -> None:
        self._molecule: MoleculeType = molecule
        self._specs: dict[str, ResidueSpec] = {}
        for spec in specs:
            self.register(spec)

    @property
    def molecule(self) -> MoleculeType:
        return self._molecule

    def register(self, spec: ResidueSpec, *, replace: bool = False) -> None:
        """Add a residue name. Redefining an existing name requires ``replace=True``."""
        existing = self._specs.get(spec.resname)
        if existing is not None and existing != spec and not replace:
            raise ValueError(
                f"residue name {spec.resname!r} is already registered as {existing}; "
                "pass replace=True to redefine it"
            )
        self._specs[spec.resname] = spec

    def get(self, resname: str) -> ResidueSpec | None:
        return self._specs.get(resname.strip())

    def __getitem__(self, resname: str) -> ResidueSpec:
        spec = self.get(resname)
        if spec is None:
            raise KeyError(resname)
        return spec

    def __contains__(self, resname: object) -> bool:
        return isinstance(resname, str) and resname.strip() in self._specs

    def __iter__(self) -> Iterator[ResidueSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def copy(self) -> ResidueRegistry:
        return ResidueRegistry(self._molecule, self._specs.values())

    def __repr__(self) -> str:
        return f"ResidueRegistry({self._molecule!r}, {len(self)} residue names)"


# --------------------------------------------------------------------------- DNA

_DNA_BASES = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}
_CHARMM_DNA = {"ADE": "A", "CYT": "C", "GUA": "G", "THY": "T"}


def _default_dna_specs() -> list[ResidueSpec]:
    specs: list[ResidueSpec] = []
    for letter, parent in _DNA_BASES.items():
        # PDB names plus AMBER 5'-terminal, 3'-terminal and free-nucleoside variants.
        for name in (parent, f"{parent}5", f"{parent}3", f"{parent}N"):
            specs.append(ResidueSpec(name, parent=parent, one_letter=letter))
    specs.append(ResidueSpec("DU", parent="DU", one_letter="U"))
    # CHARMM uses the same names for DNA and RNA; a DNA selection containing a
    # residue with a 2'-hydroxyl oxygen is rejected at load time.
    for name, letter in _CHARMM_DNA.items():
        specs.append(ResidueSpec(name, parent=_DNA_BASES[letter], one_letter=letter))
    # 5-methyl-2'-deoxycytidine as named in the PDB Chemical Component Dictionary;
    # C5A is the 5-methyl carbon read by MeCP2's methyl-CpG binding domain.
    specs.append(
        ResidueSpec("5CM", parent="DC", one_letter="C", modification="5mC", marker_atoms=("C5A",))
    )
    return specs


def default_dna_registry() -> ResidueRegistry:
    """A fresh registry of common DNA residue names (PDB, AMBER, CHARMM, and 5CM).

    Deliberately not included: single-letter names (``A``, ``C``, ``G``, ``U``
    denote RNA in current PDB files) and modified bases whose file naming varies
    between sources (5hmC, 5fC, 5caC). Register those explicitly for your data.
    """
    return ResidueRegistry("dna", _default_dna_specs())


# ----------------------------------------------------------------------- protein

_AMINO_ACIDS = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip

# Protonation / disulfide states used by AMBER and CHARMM. These are not
# chemical modifications in the biological sense, so modification is None.
_PROTONATION_VARIANTS = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}  # fmt: skip

_MODIFIED_AMINO_ACIDS = [
    ResidueSpec("SEP", parent="SER", one_letter="S", modification="phosphorylation", marker_atoms=("P",)),
    ResidueSpec("TPO", parent="THR", one_letter="T", modification="phosphorylation", marker_atoms=("P",)),
    ResidueSpec("PTR", parent="TYR", one_letter="Y", modification="phosphorylation", marker_atoms=("P",)),
    ResidueSpec("MSE", parent="MET", one_letter="M", modification="selenomethionine", marker_atoms=("SE",)),
]  # fmt: skip

_CAPS = ("ACE", "NME", "NHE", "NH2")


def _default_protein_specs() -> list[ResidueSpec]:
    specs: list[ResidueSpec] = []
    for name, letter in _AMINO_ACIDS.items():
        specs.append(ResidueSpec(name, parent=name, one_letter=letter))
    for name, parent in _PROTONATION_VARIANTS.items():
        specs.append(ResidueSpec(name, parent=parent, one_letter=_AMINO_ACIDS[parent]))
    # AMBER N-/C-terminal residue names (NALA, CALA, ..., NHIE, CCYX, ...).
    for name, parent in [*((n, n) for n in _AMINO_ACIDS), *_PROTONATION_VARIANTS.items()]:
        for prefix in ("N", "C"):
            specs.append(ResidueSpec(prefix + name, parent=parent, one_letter=_AMINO_ACIDS[parent]))
    specs.extend(_MODIFIED_AMINO_ACIDS)
    for name in _CAPS:
        specs.append(ResidueSpec(name, parent=name, one_letter=""))
    return specs


def default_protein_registry() -> ResidueRegistry:
    """A fresh registry of standard amino acids, MD force-field variants,
    common modified residues (SEP, TPO, PTR, MSE) and terminal caps."""
    return ResidueRegistry("protein", _default_protein_specs())


# ------------------------------------------------------------ table helpers

KEY_FIELDS = ("segid", "chain", "resnum", "icode", "resname")
"""Residue identity columns, in the order they appear in output tables."""

MAX_LISTED = 10


@dataclass(frozen=True)
class IdentifiedResidue:
    """A selected residue: its identity, registry entry and MDAnalysis residue index."""

    key: ResidueKey
    spec: ResidueSpec
    mda_index: int


def key_columns(key: ResidueKey, prefix: str) -> dict[str, object]:
    """``{prefix_segid: ..., prefix_chain: ..., ...}`` for one residue."""
    return {f"{prefix}_{f}": getattr(key, f) for f in KEY_FIELDS}


def format_keys(keys: Iterable[ResidueKey]) -> str:
    """Comma-separated labels, truncated after :data:`MAX_LISTED` entries."""
    labels = [k.label() for k in keys]
    more = f" ... and {len(labels) - MAX_LISTED} more" if len(labels) > MAX_LISTED else ""
    return ", ".join(labels[:MAX_LISTED]) + more
