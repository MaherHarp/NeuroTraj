"""Prepare an OpenMM system from a PDB entry (explicit conversions, no silent changes).

Every change to the deposited model is listed in ``preparation.json``:

* residue conversions (selenomethionine to Met, optional reversion of named
  residues to Ala);
* added atoms (hydrogens, a C-terminal OXT);
* crystal waters kept or removed, and added water and ions;
* the CHARMM template assigned to every nucleotide.

Methylated cytosine is never converted: 5CM keeps its methyl group and is
parameterised with the documented CHARMM36 5MC2 patch. The run stops if that
cannot be done.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as _platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neurodna.errors import NeuroDNAError
from neurodna.md.config import PreparationConfig
from neurodna.md.forcefield import (
    MODIFIED_RESIDUES,
    UNSUPPORTED_WITH_REASON,
    charmm_forcefield,
    force_field_sources,
    load_5cm_definitions,
    nucleotide_templates,
    require_openmm,
)

WATER_CHAIN_IDS = "WXYZ"
ION_CHAIN_ID = "I"
MAX_PDB_RESIDUES = 9999  # PDB resSeq has 4 columns; OpenMM writes hybrid-36 beyond it
PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
OXT_BOND_NM = 0.125


class PreparationBlockedError(NeuroDNAError):
    """The structure cannot be prepared with validated parameters."""

    def __init__(self, blockers: list[str]) -> None:
        super().__init__("preparation blocked:\n  - " + "\n  - ".join(blockers))
        self.blockers = blockers


@dataclass
class Solute:
    """Hydrogenated solute (protein, DNA, optional crystal waters) ready for solvation."""

    modeller: Any
    forcefield: Any
    templates: dict[Any, str]
    conversions: list[str]
    added_atoms: list[str]
    definitions: tuple[Path, Path]


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_solute(pdb_path: str | os.PathLike[str], config: PreparationConfig,
                 workdir: Path) -> Solute:
    """Apply the configured conversions, add a missing OXT and hydrogens (pH ``config.ph``)."""
    mm, app, unit = require_openmm()
    workdir.mkdir(parents=True, exist_ok=True)
    definitions = load_5cm_definitions(workdir)
    pdb = app.PDBFile(os.fspath(pdb_path))
    source_top, source_pos = pdb.topology, pdb.positions

    blockers = []
    wanted = set(config.protein_chains) | set(config.dna_chains)
    present = {c.id for c in source_top.chains()}
    if wanted - present:
        blockers.append(f"chains {sorted(wanted - present)} are not in the file")
    revert = set(config.revert_to_alanine)
    conversions: list[str] = []
    added: list[str] = []

    top = app.Topology()
    positions: list[Any] = []
    chains: dict[tuple[int, str], Any] = {}
    c_terminal: dict[str, Any] = {}
    for chain in source_top.chains():
        polymer = [r for r in chain.residues() if r.name != "HOH"]
        if chain.id in config.protein_chains and polymer:
            c_terminal[chain.id] = polymer[-1]
    for chain in source_top.chains():
        keep_polymer = chain.id in wanted
        for residue in chain.residues():
            water = residue.name == "HOH"
            if water and config.crystal_waters == "remove":
                continue
            if not water and not keep_polymer:
                continue
            if not water and chain.id in config.protein_chains:
                if residue.name not in PROTEIN_RESIDUES and residue.name != "MSE":
                    blockers.append(f"protein residue {chain.id}:{residue.name}{residue.id} "
                                    "has no supported template")
            if not water and chain.id in config.dna_chains:
                from neurodna.md.forcefield import NUCLEOTIDES

                if residue.name not in NUCLEOTIDES:
                    reason = UNSUPPORTED_WITH_REASON.get(residue.name, "no validated CHARMM36 parameters")
                    blockers.append(f"DNA residue {chain.id}:{residue.name}{residue.id}: {reason}")
            key = (chain.index, chain.id)
            if key not in chains:
                chains[key] = top.addChain(chain.id)
            name = residue.name
            label = f"{chain.id}:{residue.id}"
            drop: set[str] = set()
            if name == "MSE":
                name = "MET"
                conversions.append(f"{chain.id}:MSE{residue.id} -> MET (SE renamed SD, element S)")
            if label in revert:
                if name in ("GLY", "PRO"):
                    blockers.append(f"cannot revert {chain.id}:{name}{residue.id} to ALA")
                drop = {a.name for a in residue.atoms()} - {"N", "CA", "C", "O", "CB", "OXT"}
                conversions.append(f"{chain.id}:{name}{residue.id} -> ALA (deleted {sorted(drop)})")
                name = "ALA"
                revert.discard(label)
            new_res = top.addResidue(name, chains[key], residue.id, residue.insertionCode)
            for atom in residue.atoms():
                if atom.name in drop:
                    continue
                atom_name, element = atom.name, atom.element
                if residue.name == "MSE" and atom.name == "SE":
                    atom_name, element = "SD", app.element.sulfur
                top.addAtom(atom_name, element, new_res)
                positions.append(source_pos[atom.index].value_in_unit(unit.nanometer))
            is_c_terminal = not water and chain.id in c_terminal and c_terminal[chain.id] is residue
            if is_c_terminal:
                names = {a.name: a for a in residue.atoms()}
                if "OXT" not in names:
                    c = np.array(source_pos[names["C"].index].value_in_unit(unit.nanometer))
                    ca = np.array(source_pos[names["CA"].index].value_in_unit(unit.nanometer))
                    o = np.array(source_pos[names["O"].index].value_in_unit(unit.nanometer))
                    direction = -((ca - c) / np.linalg.norm(ca - c) + (o - c) / np.linalg.norm(o - c))
                    top.addAtom("OXT", app.element.oxygen, new_res)
                    positions.append(c + OXT_BOND_NM * direction / np.linalg.norm(direction))
                    added.append(f"{chain.id}:{name}{residue.id} OXT (C-terminal carboxylate, "
                                 "placed in the CA-C-O plane, 1.25 Å from C)")
    if revert:
        blockers.append(f"revert_to_alanine residues not found: {sorted(revert)}")
    if blockers:
        raise PreparationBlockedError(blockers)

    top.createStandardBonds()
    quantity = unit.Quantity([mm.Vec3(*p) for p in positions], unit.nanometer)
    top.createDisulfideBonds(quantity)
    forcefield = charmm_forcefield(tuple(config.force_field))
    templates = nucleotide_templates(forcefield, top, tuple(config.dna_chains))
    modeller = app.Modeller(top, quantity)
    n_before = modeller.topology.getNumAtoms()
    # OpenMM's Modeller draws initial hydrogen positions from Python's global
    # `random`; seeding it and minimising on the Reference platform makes this
    # step reproducible.
    random.seed(config.preparation_seed)
    modeller.addHydrogens(forcefield, pH=config.ph, residueTemplates=templates,
                          platform=mm.Platform.getPlatformByName("Reference"))
    added.append(f"{modeller.topology.getNumAtoms() - n_before} hydrogens (OpenMM Modeller, pH {config.ph})")
    templates = nucleotide_templates(forcefield, modeller.topology, tuple(config.dna_chains))
    _check_modified_nucleotides(modeller.topology, templates)
    return Solute(modeller, forcefield, templates, conversions, added, definitions)


def _check_modified_nucleotides(topology: Any, templates: dict[Any, str]) -> None:
    """Refuse to continue unless every 5CM still carries its methyl group and the 5MC2 patch."""
    for residue, tname in templates.items():
        if residue.name not in MODIFIED_RESIDUES:
            continue
        spec = MODIFIED_RESIDUES[residue.name]
        names = {a.name for a in residue.atoms()}
        needed = {spec["methyl_carbon"], *spec["methyl_hydrogens"]}
        if not needed <= names or "5MC2" not in tname:
            raise PreparationBlockedError([
                f"{residue.chain.id}:{residue.name}{residue.id} lost its methyl group or template "
                f"({sorted(needed - names)} missing, template {tname})"
            ])


def _relabel_solvent(modeller: Any, n_solute_chains: int) -> tuple[Any, dict[str, Any]]:
    """Put added waters in chains W, X, ... of <= 9999 residues and ions in chain I.

    The atom order is unchanged, so the System built afterwards matches the
    written PDB. Residues are numbered from 1 in each solvent chain, avoiding the
    hybrid-36 residue numbers (A000, ...) that OpenMM writes past 9999 and that
    MDAnalysis cannot parse.
    """
    _, app, _ = require_openmm()
    old = modeller.topology
    new = app.Topology()
    new.setPeriodicBoxVectors(old.getPeriodicBoxVectors())
    atom_map: dict[int, Any] = {}
    counts: dict[str, Any] = {"added_waters": 0, "added_positive_ions": 0, "added_negative_ions": 0,
                              "solvent_chains": {}}
    water_ids = iter(WATER_CHAIN_IDS)
    for i, chain in enumerate(old.chains()):
        residues = list(chain.residues())
        if i < n_solute_chains:
            blocks = [(chain.id, residues, False)]
        elif residues and all(r.name == "HOH" for r in residues):
            counts["added_waters"] += len(residues)
            blocks = []
            for start in range(0, len(residues), MAX_PDB_RESIDUES):
                try:
                    cid = next(water_ids)
                except StopIteration:
                    raise PreparationBlockedError(["too many waters for the PDB topology format; "
                                                   "reduce padding_nm"]) from None
                blocks.append((cid, residues[start:start + MAX_PDB_RESIDUES], True))
        else:
            if len(residues) > MAX_PDB_RESIDUES:
                raise PreparationBlockedError(["too many ions for one PDB chain"])
            for r in residues:
                sign = "positive" if r.name in ("NA", "K") else "negative"
                counts[f"added_{sign}_ions"] += 1
            blocks = [(ION_CHAIN_ID, residues, True)]
        for cid, block, renumber in blocks:
            if renumber:
                counts["solvent_chains"][cid] = len(block)
            new_chain = new.addChain(cid)
            for k, residue in enumerate(block, start=1):
                new_res = new.addResidue(residue.name, new_chain, str(k) if renumber else residue.id,
                                         residue.insertionCode)
                for atom in residue.atoms():
                    atom_map[atom.index] = new.addAtom(atom.name, atom.element, new_res)
    for bond in old.bonds():
        new.addBond(atom_map[bond[0].index], atom_map[bond[1].index], bond.type, bond.order)
    return app.Modeller(new, modeller.positions), counts


def create_system(modeller: Any, forcefield: Any, templates: dict[Any, str],
                  config: PreparationConfig, periodic: bool = True) -> Any:
    mm, app, unit = require_openmm()
    if periodic:
        return forcefield.createSystem(
            modeller.topology, nonbondedMethod=app.PME,
            nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
            switchDistance=config.switch_distance_nm * unit.nanometer,
            ewaldErrorTolerance=config.ewald_error_tolerance,
            constraints=app.HBonds, rigidWater=True, residueTemplates=templates,
        )
    return forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                   constraints=None, residueTemplates=templates)


def energy_fingerprint(system: Any, positions: Any) -> dict[str, float]:
    """Potential energy of each force at the prepared coordinates (CPU platform, kJ/mol)."""
    mm, _, unit = require_openmm()
    copy = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    names = []
    for i, force in enumerate(copy.getForces()):
        force.setForceGroup(i)
        names.append(f"{i}:{type(force).__name__}")
    context = mm.Context(copy, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    out = {name: round(float(context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                             .value_in_unit(unit.kilojoule_per_mole)), 3)
           for i, name in enumerate(names)}
    del context
    return out


def net_charge(system: Any) -> float:
    mm, _, unit = require_openmm()
    nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    return float(sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                     for i in range(nb.getNumParticles())))


def prepare(pdb_path: str | os.PathLike[str], config: PreparationConfig,
            output_dir: str | os.PathLike[str]) -> Path:
    """Build, solvate and parameterise the system; write it with a full preparation report.

    Outputs in ``output_dir``: ``prepared.pdb`` (topology and coordinates),
    ``system.xml`` (serialised OpenMM System), ``preparation.json`` (report) and
    the 5CM definition files used.
    """
    mm, app, unit = require_openmm()
    from neurodna.md.assess import _assess

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report, solute = _assess(pdb_path, config, out)
    if not report["supported"] or solute is None:
        raise PreparationBlockedError(report["blockers"] or ["force-field compatibility not checked"])
    modeller, forcefield = solute.modeller, solute.forcefield
    solute_charge = report["force_field"]["solute_net_charge_e"]
    audit = report["force_field"]["modified_residue_audit"]

    n_solute_chains = modeller.topology.getNumChains()
    random.seed(config.preparation_seed + 1)  # Modeller picks the waters replaced by ions at random
    modeller.addSolvent(
        forcefield, model="tip3p", padding=config.padding_nm * unit.nanometer,
        boxShape=config.box_shape, ionicStrength=config.ionic_strength_molar * unit.molar,
        positiveIon=config.positive_ion, negativeIon=config.negative_ion, neutralize=True,
        residueTemplates=solute.templates,
    )
    modeller, counts = _relabel_solvent(modeller, n_solute_chains)
    templates = nucleotide_templates(forcefield, modeller.topology, tuple(config.dna_chains))
    system = create_system(modeller, forcefield, templates, config, periodic=True)
    total_charge = net_charge(system)
    if abs(total_charge) > 1e-3:
        raise PreparationBlockedError([f"solvated system is not neutral ({total_charge:+.4f} e)"])

    pdb_out, system_out = out / "prepared.pdb", out / "system.xml"
    with open(pdb_out, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    system_out.write_text(mm.XmlSerializer.serialize(system))

    fingerprint = energy_fingerprint(system, modeller.positions)
    box = modeller.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    composition = {
        "atoms": modeller.topology.getNumAtoms(),
        "crystal_waters_kept": sum(1 for r in modeller.topology.residues()
                                   if r.name == "HOH" and r.chain.id not in WATER_CHAIN_IDS),
        **counts,
        "solute_net_charge_e": round(solute_charge, 4),
        "system_net_charge_e": round(total_charge, 6),
        "box_vectors_nm": [[round(float(x), 5) for x in v] for v in box],
        "box_volume_nm3": round(float(abs(np.linalg.det(np.array(box, dtype=float)))), 3),
    }
    preparation = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "neurodna",
        "input": {"path": os.fspath(pdb_path), "sha256": sha256(pdb_path)},
        "config": config.to_dict(),
        "assessment": report,
        "conversions": solute.conversions,
        "added_atoms": solute.added_atoms,
        "nucleotide_templates": {f"{r.chain.id}:{r.name}{r.id}": t for r, t in templates.items()},
        "modified_residue_audit": audit,
        "composition": composition,
        "selections": selections(config),
        "force_field": {"files": list(config.force_field),
                        "sources": force_field_sources(tuple(config.force_field)),
                        "modified_residues": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                                  for kk, vv in v.items()}
                                              for k, v in MODIFIED_RESIDUES.items()}},
        "reproducibility": {"preparation_seed": config.preparation_seed,
                            "hydrogen_placement_platform": "Reference",
                            "note": "Python's random module is seeded before OpenMM Modeller adds "
                                    "hydrogens (seed) and ions (seed + 1). prepared.pdb is then "
                                    "reproducible; system.xml may differ byte-wise between processes "
                                    "because OpenMM numbers CHARMM NBFIX atom-type tables in hash "
                                    "order, so compare energy_fingerprint_kj_mol instead (relative tolerance ~1e-6: "
                                    "the CPU platform's multithreaded PME summation is not bitwise "
                                    "deterministic).",
                            "energy_fingerprint_kj_mol": fingerprint},
        "nonbonded": {"method": "PME", "cutoff_nm": config.nonbonded_cutoff_nm,
                      "switch_distance_nm": config.switch_distance_nm,
                      "switching": "OpenMM potential switching (CHARMM itself recommends force "
                                   "switching; this is an approximation)",
                      "ewald_error_tolerance": config.ewald_error_tolerance,
                      "constraints": "HBonds", "rigid_water": True},
        "files": {"prepared_pdb": {"name": pdb_out.name, "sha256": sha256(pdb_out)},
                  "system_xml": {"name": system_out.name, "sha256": sha256(system_out)},
                  "5cm_hydrogens": solute.definitions[0].name,
                  "5cm_residues": solute.definitions[1].name},
        "software": software_versions(),
        "units": {"length": "nm (OpenMM files) / angstrom (neurodna tables)", "time": "ps",
                  "energy": "kJ/mol"},
    }
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    return out


def selections(config: PreparationConfig) -> dict[str, str]:
    """neurodna / MDAnalysis selections for the prepared topology."""
    return {"protein": f"chainID {' '.join(config.protein_chains)} and not resname HOH",
            "dna": f"chainID {' '.join(config.dna_chains)} and not resname HOH"}


def software_versions() -> dict[str, str]:
    import MDAnalysis

    import neurodna

    mm, _, _ = require_openmm()
    return {"neurodna": neurodna.__version__, "openmm": mm.__version__,
            "MDAnalysis": MDAnalysis.__version__, "numpy": np.__version__,
            "python": _platform.python_version(), "machine": _platform.platform()}
