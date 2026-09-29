"""Assess whether a structure can be prepared with documented force-field parameters.

The report covers:

* missing residues and atoms;
* termini;
* titratable residues (for protonation);
* crystallographic waters, ions and other heterogens;
* alternate locations and the biological assembly;
* force-field compatibility: a template for every residue, and every bonded
  term present for each modified residue.

It separates *blockers* (preparation is impossible without new, unvalidated
parameters or unsupported modelling) from *decisions* (explicit choices recorded
in the preparation config) and *notes* (caveats to report with any result).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from neurodna.md.config import PreparationConfig
from neurodna.md.forcefield import MODIFIED_RESIDUES, NUCLEOTIDES, UNSUPPORTED_WITH_REASON
from neurodna.pdbheader import read_pdb_header

PROTEIN = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
TITRATABLE = ("ASP", "GLU", "HIS", "CYS", "LYS", "TYR", "ARG")


def _ranges(numbers: list[int]) -> list[str]:
    out: list[str] = []
    for n in sorted(numbers):
        if out and int(out[-1].split("-")[-1]) == n - 1:
            out[-1] = f"{out[-1].split('-')[0]}-{n}"
        else:
            out.append(str(n))
    return out


def structural_findings(pdb_path: str | os.PathLike[str], config: PreparationConfig) -> dict[str, Any]:
    """Checks that need only the PDB file (no OpenMM)."""
    import warnings

    import MDAnalysis as mda

    header = read_pdb_header(pdb_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(os.fspath(pdb_path))
    blockers: list[str] = []
    decisions: list[str] = []
    notes: list[str] = []

    altlocs = sorted({str(a) for a in u.atoms.altLocs if str(a).strip()}) if hasattr(u.atoms, "altLocs") else []
    if altlocs:
        blockers.append(f"alternate locations {altlocs} present; select one conformer before preparation")

    chains: dict[str, Any] = {}
    polymer_names = PROTEIN | set(NUCLEOTIDES) | {"MSE"}
    dna = u.select_atoms(f"chainID {' '.join(config.dna_chains)} and not resname HOH")
    for cid in (*config.protein_chains, *config.dna_chains):
        ag = u.select_atoms(f"chainID {cid} and not resname HOH")
        if len(ag) == 0:
            blockers.append(f"chain {cid} not found")
            continue
        residues = ag.residues
        numbers = [int(r) for r in residues.resids]
        gaps = [f"{a}-{b}" for a, b in zip(numbers, numbers[1:]) if b != a + 1]
        missing = [r.resnum for r in header.missing_residues if r.chain == cid]
        entry: dict[str, Any] = {
            "observed": f"{numbers[0]}-{numbers[-1]}",
            "n_residues": len(numbers),
            "missing_residues": _ranges(missing),
            "internal_numbering_gaps": gaps,
            "missing_atoms": [f"{r.label()}: {' '.join(r.atoms)}" for r in header.missing_atoms if r.chain == cid],
        }
        if gaps:
            blockers.append(f"chain {cid} has internal gaps {gaps}; missing internal residues are not modelled")
        if entry["missing_atoms"]:
            blockers.append(f"chain {cid} has missing atoms {entry['missing_atoms']}; model them before preparation")
        if cid in config.protein_chains:
            first, last = residues[0], residues[-1]
            before = [m for m in missing if m < numbers[0]]
            after = [m for m in missing if m > numbers[-1]]
            titr = {name: int(sum(residues.resnames == name)) for name in TITRATABLE}
            entry["termini"] = {"n_terminal": f"{cid}:{first.resname}{first.resid}",
                                "c_terminal": f"{cid}:{last.resname}{last.resid}",
                                "n_truncated": bool(before), "c_truncated": bool(after),
                                "c_terminal_has_OXT": "OXT" in last.atoms.names}
            entry["titratable_residues"] = titr
            for end, res in (("N", first), ("C", last)):
                atom = res.atoms.select_atoms("name N" if end == "N" else "name C")
                if len(atom) and len(dna):
                    d = float(np.min(np.linalg.norm(dna.positions - atom.positions[0], axis=1)))
                    entry["termini"][f"{end.lower()}_terminal_to_dna_A"] = round(d, 2)
            if before or after:
                notes.append(
                    f"chain {cid} is truncated ({'N' if before else ''}{'C' if after else ''}-terminal residues "
                    f"not observed); protein_termini='charged' places NH3+/COO- charges at "
                    f"{entry['termini']['n_terminal']} and {entry['termini']['c_terminal']} that do not "
                    "exist in the full-length protein"
                )
            if titr["HIS"]:
                notes.append(f"{titr['HIS']} histidine(s): tautomers chosen by OpenMM's hydrogen-bond heuristic")
            notes.append(f"protonation: standard states at pH {config.ph} (OpenMM Modeller rules); "
                         "no pKa calculation was performed")
        else:
            first, last = residues[0], residues[-1]
            entry["termini"] = {"5prime": f"{cid}:{first.resname}{first.resid}",
                                "5prime_phosphate": "P" in first.atoms.names,
                                "3prime": f"{cid}:{last.resname}{last.resid}"}
        chains[cid] = entry

    resnames = set(u.residues.resnames)
    other = sorted(resnames - polymer_names - {"HOH"})
    selected = u.select_atoms(f"chainID {' '.join((*config.protein_chains, *config.dna_chains))}")
    names_in_chains = {str(r) for r in selected.residues.resnames}
    nonstandard = sorted(names_in_chains - PROTEIN - {"DA", "DC", "DG", "DT", "HOH"})
    if other:
        blockers.append(f"heterogens {other} have no preparation route (ions/ligands are not handled)")
    modified: dict[str, str] = {}
    for name in nonstandard:
        if name in MODIFIED_RESIDUES:
            modified[name] = f"supported: {MODIFIED_RESIDUES[name]['charmm']}"
        elif name == "MSE":
            modified[name] = f"requires conversion: {UNSUPPORTED_WITH_REASON['MSE']}"
            decisions.append(f"selenomethionine -> {config.selenomethionine}")
        else:
            modified[name] = "BLOCKED: no validated CHARMM36 parameters"
            blockers.append(f"{name}: no validated CHARMM36 parameters in the supported force field; "
                            "prepare the system externally with validated parameters")

    engineered = [f"{r.label()}: {r.detail}" for r in header.seqadv if "ENGINEERED" in r.detail]
    if engineered:
        decisions.append(f"engineered residues {engineered}: revert_to_alanine={list(config.revert_to_alanine)}")
    waters = {c: int(u.select_atoms(f"resname HOH and chainID {c}").n_residues)
              for c in sorted(set(u.select_atoms("resname HOH").chainIDs))}
    decisions.append(f"crystallographic waters ({sum(waters.values())}): {config.crystal_waters}")
    decisions.append(f"protein termini: {config.protein_termini}")
    decisions.append(f"solvent: CHARMM TIP3P, {config.box_shape} box, {config.padding_nm} nm padding, "
                     f"{config.ionic_strength_molar} M {config.positive_ion}/{config.negative_ion}, neutralised")

    assemblies: list[dict[str, Any]] = [{"biomolecule": a.biomolecule, "chains": list(a.chains),
                   "identity_operator_only": a.identity_only} for a in header.assemblies]
    kept = set(config.protein_chains) | set(config.dna_chains)
    if assemblies and not any(set(a["chains"]) == kept and a["identity_operator_only"] for a in assemblies):
        notes.append("the selected chains are not an identity-operator biological assembly; the deposited "
                     "coordinates may not represent the biological complex")
    return {
        "entry": {"idcode": header.idcode, "experiment": header.experiment,
                  "resolution_A": header.resolution_A, "title": header.title},
        "chains": chains,
        "nonstandard_residues": modified,
        "engineered_residues": engineered,
        "crystallographic_waters_by_chain": waters,
        "ions_or_other_heterogens": other,
        "alternate_locations": altlocs,
        "assemblies": assemblies,
        "blockers": blockers,
        "decisions": decisions,
        "notes": notes,
    }


def assess(pdb_path: str | os.PathLike[str], config: PreparationConfig,
           workdir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Full assessment; the force-field part runs only if OpenMM is installed."""
    report, _ = _assess(pdb_path, config, workdir)
    return report


def _assess(pdb_path: str | os.PathLike[str], config: PreparationConfig,
            workdir: str | os.PathLike[str] | None) -> tuple[dict[str, Any], Any]:
    report = structural_findings(pdb_path, config)
    report["force_field"] = {"files": list(config.force_field), "checked": False}
    solute = None
    if not report["blockers"]:
        try:
            from neurodna.md.forcefield import bonded_term_audit, require_openmm

            require_openmm()
        except ImportError as exc:
            report["force_field"]["reason_not_checked"] = str(exc)
        else:
            import tempfile

            from neurodna.md.prepare import PreparationBlockedError, build_solute, create_system, net_charge

            directory = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="neurodna_md_"))
            try:
                solute = build_solute(pdb_path, config, directory)
            except PreparationBlockedError as exc:
                report["blockers"].extend(exc.blockers)
            except Exception as exc:  # template matching failures from OpenMM
                report["blockers"].append(f"force-field template matching failed: {exc}")
            else:
                system = create_system(solute.modeller, solute.forcefield, solute.templates, config,
                                       periodic=False)
                audit = bonded_term_audit(solute.forcefield, solute.modeller.topology, system,
                                          solute.templates)
                for record in audit:
                    if not record["complete"]:
                        report["blockers"].append(f"{record['residue']}: missing parameters {record['missing']}")
                report["force_field"].update(
                    checked=True,
                    all_residues_matched=True,
                    modified_nucleotides={r["residue"]: r["template"] for r in audit},
                    modified_residue_audit=audit,
                    solute_net_charge_e=round(net_charge(system), 4),
                    conversions=solute.conversions,
                )
    report["supported"] = not report["blockers"] and report["force_field"]["checked"]
    return report, solute
