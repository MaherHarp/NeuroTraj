"""Configuration for system preparation and simulation protocols (JSON-serialisable).

Choices without a scientifically neutral default must be stated explicitly in
the preparation config: how selenomethionine is handled, which engineered
residues are reverted, whether crystallographic waters are kept, and how the
protein termini are treated. Loading a config without them is an error.

Units are given in the field names: ``_nm``, ``_ps``, ``_fs``, ``_K``, ``_bar``,
``_molar``, ``_kj_mol_nm2``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from neurodna.errors import NeuroDNAError

FORCE_FIELD_FILES = ("charmm36_2024.xml", "charmm36_2024/water.xml")
"""The only supported force field: OpenMM's port of CHARMM36 (July 2024 toppar),
with CHARMM-modified TIP3P water and CHARMM ions. It contains the documented
patch for 5-methylcytosine in DNA (``5MC2``)."""


class ConfigError(NeuroDNAError, ValueError):
    """A preparation config or protocol is incomplete or invalid."""


def _choice(name: str, value: Any, allowed: tuple[Any, ...], why: str = "") -> None:
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} is not supported; allowed: {list(allowed)}. {why}".strip())


@dataclass(frozen=True)
class PreparationConfig:
    """How to turn a PDB entry into a simulation-ready system.

    Required (no defaults):
        protein_chains, dna_chains: Chain IDs to keep (other polymers are dropped).
        selenomethionine: ``"convert_to_methionine"`` (Se -> S; CHARMM36 has no
            selenium parameters). No other value is supported.
        revert_to_alanine: Residue labels such as ``"A:140"`` to truncate to Ala
            (atoms beyond CB are deleted). Use it to undo an engineered
            substitution; ``[]`` keeps the residues as deposited.
        crystal_waters: ``"keep"`` or ``"remove"`` crystallographic waters.
        protein_termini: ``"charged"`` (NH3+ / COO-, adding a missing OXT).
            Neutral caps are not implemented.
    """

    protein_chains: tuple[str, ...]
    dna_chains: tuple[str, ...]
    selenomethionine: str
    revert_to_alanine: tuple[str, ...]
    crystal_waters: str
    protein_termini: str
    ph: float = 7.0
    box_shape: str = "dodecahedron"
    padding_nm: float = 1.2
    ionic_strength_molar: float = 0.15
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    nonbonded_cutoff_nm: float = 1.2
    switch_distance_nm: float = 1.0
    ewald_error_tolerance: float = 5e-4
    constraints: str = "HBonds"
    force_field: tuple[str, ...] = FORCE_FIELD_FILES
    preparation_seed: int = 20260925
    notes: str = ""

    REQUIRED = ("protein_chains", "dna_chains", "selenomethionine", "revert_to_alanine",
                "crystal_waters", "protein_termini")

    def __post_init__(self) -> None:
        if not self.protein_chains or not self.dna_chains:
            raise ConfigError("protein_chains and dna_chains must be non-empty")
        if set(self.protein_chains) & set(self.dna_chains):
            raise ConfigError("protein_chains and dna_chains overlap")
        _choice("selenomethionine", self.selenomethionine, ("convert_to_methionine",),
                "CHARMM36 provides no selenomethionine parameters.")
        _choice("crystal_waters", self.crystal_waters, ("keep", "remove"))
        _choice("protein_termini", self.protein_termini, ("charged",),
                "Neutral ACE/NME caps are not implemented; prepare capped systems externally.")
        _choice("box_shape", self.box_shape, ("dodecahedron", "cube", "octahedron"))
        _choice("constraints", self.constraints, ("HBonds",))
        if tuple(self.force_field) != FORCE_FIELD_FILES:
            raise ConfigError(f"only the force field {list(FORCE_FIELD_FILES)} is supported")
        for label in self.revert_to_alanine:
            chain, _, number = label.partition(":")
            if not chain or not number.strip().lstrip("-").isdigit():
                raise ConfigError(f"revert_to_alanine entries look like 'A:140', got {label!r}")
        if not 0 < self.switch_distance_nm < self.nonbonded_cutoff_nm:
            raise ConfigError("need 0 < switch_distance_nm < nonbonded_cutoff_nm")
        if self.padding_nm < self.nonbonded_cutoff_nm / 2:
            raise ConfigError("padding_nm must be at least half the nonbonded cutoff")
        if not 0 <= self.preparation_seed < 2**31:
            raise ConfigError("preparation_seed must be in [0, 2**31)")
        if not 0 <= self.ionic_strength_molar < 5:
            raise ConfigError("ionic_strength_molar must be in [0, 5)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparationConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown preparation settings: {sorted(unknown)}")
        missing = [k for k in cls.REQUIRED if k not in data]
        if missing:
            raise ConfigError(
                f"preparation config must state {missing} explicitly (no defaults for "
                "choices that change the chemistry or the model)"
            )
        values = dict(data)
        for key in ("protein_chains", "dna_chains", "revert_to_alanine", "force_field"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Stage:
    """One equilibration stage with positional restraints on solute heavy atoms."""

    name: str
    ensemble: str
    steps: int
    restraint_k_kj_mol_nm2: float

    def __post_init__(self) -> None:
        _choice(f"stage {self.name!r} ensemble", self.ensemble, ("NVT", "NPT"))
        if self.steps < 0 or self.restraint_k_kj_mol_nm2 < 0:
            raise ConfigError(f"stage {self.name!r}: steps and restraint must be >= 0")


@dataclass(frozen=True)
class Protocol:
    """Minimisation, restrained equilibration and unrestrained NPT production."""

    name: str
    purpose: str
    seed: int
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    timestep_fs: float = 2.0
    friction_per_ps: float = 1.0
    barostat_interval_steps: int = 25
    minimization_max_iterations: int = 5000
    minimization_tolerance_kj_mol_nm: float = 10.0
    minimization_restraint_k_kj_mol_nm2: float = 1000.0
    equilibration: tuple[Stage, ...] = field(default_factory=tuple)
    production_steps: int = 0
    report_interval_steps: int = 5000
    checkpoint_interval_steps: int = 500000
    platform: str | None = None
    precision: str | None = None
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.timestep_fs <= 4:
            raise ConfigError("timestep_fs must be in (0, 4]")
        if self.production_steps < 0:
            raise ConfigError("production_steps must be >= 0")
        if self.report_interval_steps <= 0 or self.checkpoint_interval_steps <= 0:
            raise ConfigError("report and checkpoint intervals must be positive")
        if self.checkpoint_interval_steps % self.report_interval_steps:
            raise ConfigError("checkpoint_interval_steps must be a multiple of report_interval_steps")
        if self.production_steps % self.report_interval_steps:
            raise ConfigError("production_steps must be a multiple of report_interval_steps")
        if not 0 <= self.seed < 2**31:
            raise ConfigError("seed must be in [0, 2**31)")
        if self.platform is not None:
            _choice("platform", self.platform, ("Reference", "CPU", "OpenCL", "CUDA", "HIP", "Metal"))
        if self.precision is not None:
            _choice("precision", self.precision, ("single", "mixed", "double"))
        if self.cpu_threads is not None and not 1 <= self.cpu_threads <= 512:
            raise ConfigError("cpu_threads must be between 1 and 512")
        names = [s.name for s in self.equilibration]
        if len(set(names)) != len(names):
            raise ConfigError("equilibration stage names must be unique")

    @property
    def equilibration_ps(self) -> float:
        return sum(s.steps for s in self.equilibration) * self.timestep_fs / 1000.0

    @property
    def production_ps(self) -> float:
        return self.production_steps * self.timestep_fs / 1000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Protocol:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"unknown protocol settings: {sorted(unknown)}")
        for key in ("name", "purpose", "seed"):
            if key not in data:
                raise ConfigError(f"protocol must define {key!r}")
        values = dict(data)
        values["equilibration"] = tuple(Stage(**s) for s in values.get("equilibration", ()))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["equilibration"] = [asdict(s) for s in self.equilibration]
        return d


SMOKE = Protocol(
    name="smoke",
    purpose="INTEGRATION SMOKE TEST ONLY: checks that the pipeline runs. A few hundred "
            "steps; not equilibrated, not a scientific simulation.",
    seed=20260925,
    minimization_max_iterations=100,
    equilibration=(Stage("nvt_restrained", "NVT", 100, 1000.0),
                   Stage("npt_restrained", "NPT", 100, 100.0)),
    production_steps=200,
    report_interval_steps=50,
    checkpoint_interval_steps=100,
)

PRODUCTION = Protocol(
    name="production",
    purpose="Template for a scientific production run (100 ns). Not run in this "
            "repository; convergence must be assessed by the user, e.g. with independent "
            "replicates.",
    seed=20260925,
    minimization_max_iterations=10000,
    equilibration=(Stage("nvt_restrained", "NVT", 50000, 1000.0),
                   Stage("npt_restrained_1000", "NPT", 125000, 1000.0),
                   Stage("npt_restrained_100", "NPT", 125000, 100.0),
                   Stage("npt_restrained_10", "NPT", 125000, 10.0),
                   Stage("npt_unrestrained", "NPT", 250000, 0.0)),
    production_steps=50_000_000,
    report_interval_steps=5000,
    checkpoint_interval_steps=500_000,
)

PRESETS = {"smoke": SMOKE, "production": PRODUCTION}


def load_preparation_config(path: str | os.PathLike[str]) -> PreparationConfig:
    with open(path) as handle:
        return PreparationConfig.from_dict(json.load(handle))


def load_protocol(name_or_path: str | os.PathLike[str]) -> Protocol:
    """A preset name (``"smoke"``, ``"production"``) or a JSON file."""
    if str(name_or_path) in PRESETS:
        return PRESETS[str(name_or_path)]
    with open(name_or_path) as handle:
        return Protocol.from_dict(json.load(handle))
