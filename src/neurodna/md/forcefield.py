"""CHARMM36 (July 2024) templates for the MeCP2–DNA system, including 5-methylcytosine.

What is used, and where it is documented:

* Protein: CHARMM36m (``par_all36m_prot.prm``), matched automatically by OpenMM.
* DNA: CHARMM36 nucleic acids (``top_all36_na.rtf``/``par_all36_na.prm``). CHARMM
  builds DNA by patching its nucleobase residues (ADE, CYT, GUA, THY) with a deoxy
  patch (``DEOX``), plus terminal patches at strand ends.
* 5-methyl-2'-deoxycytidine (PDB ``5CM``): CYT + DEOX + ``5MC2``. The patch comes
  from ``stream/na/toppar_all36_na_modifications.str``, where it reads
  ``PRES 5MC2 -0.06 ! Patch to convert cytosine in DNA to 5-methylcytosine``.
  Its bonded and Lennard-Jones terms for atom type CN3D are annotated
  ``!5mc, adm jr. 9/9/93`` in that file.

Templates are assigned explicitly for every nucleotide. OpenMM's automatic
patch matching is ambiguous for these residues: for example, thymine matches
both THY+DEOX and URA+5MC2+DEOX, which have different parameters. It could
also pick CHARMM's RNA ``5MC`` residue instead of the documented DNA patch.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from neurodna.errors import NeuroDNAError

# PDB residue name -> (CHARMM base template, extra single-residue patches)
NUCLEOTIDES: dict[str, tuple[str, tuple[str, ...]]] = {
    "DA": ("ADE", ()),
    "DC": ("CYT", ()),
    "DG": ("GUA", ()),
    "DT": ("THY", ()),
    "5CM": ("CYT", ("5MC2_4",)),
}
SUGAR_PATCHES = {"internal": ("DEOX_0",), "5prime_hydroxyl": ("DEO5TER",),
                 "3prime_hydroxyl": ("DEOX_0", "3TER_1")}

MODIFIED_RESIDUES: dict[str, dict[str, Any]] = {
    "5CM": {
        "description": "5-methyl-2'-deoxycytidine 5'-monophosphate (PDB CCD 5CM)",
        "charmm": "CYT + DEOX + 5MC2 (OpenMM patch variant 5MC2_4)",
        "source": "CHARMM36 toppar_c36_jul24: stream/na/toppar_all36_na_modifications.str "
                  "(md5 cfa6730ae4b0d3d2763659859892ab6b, as recorded in OpenMM's "
                  "charmm36_2024.xml)",
        "source_comment": "PRES 5MC2 -0.06 ! Patch to convert cytosine in DNA to 5-methylcytosine",
        "parameter_provenance": "CN3D bonded/LJ terms annotated '5mc, adm jr. 9/9/93' "
                                "(MacKerell) in the same stream file",
        "methyl_carbon": "C5A",
        "methyl_hydrogens": ("H5A1", "H5A2", "H5A3"),
    },
}
UNSUPPORTED_WITH_REASON = {
    "MSE": "selenomethionine: CHARMM36 has no selenium parameters; convert to MET "
           "(config selenomethionine='convert_to_methionine')",
}

# Proper torsions that have no PeriodicTorsionForce term because CHARMM36 defines
# them with a zero force constant (verified against the jul24 source files whose
# md5 hashes OpenMM's charmm36_2024.xml records). Keys are CHARMM atom classes.
VERIFIED_ZERO_TORSIONS: dict[tuple[str, str, str, str], str] = {
    ("CN3", "CN3D", "CN9", "HN9"): "toppar_all36_na_modifications.str: CN3 CN3D CN9 HN9 0.0 3 0.0",
    ("NN3", "CN2", "CN3D", "CN9"): "toppar_all36_na_modifications.str: NN3 CN2 CN3D CN9 0.0 2 180.0",
    ("NN1", "CN2", "CN3D", "CN9"): "toppar_all36_na_modifications.str: NN1 CN2 CN3D CN9 0.0 2 180.0",
    ("NN1", "CN2", "CN3D", "CN3"): "toppar_all36_na_modifications.str: NN1 CN2 CN3D CN3 0.0 2 180.0",
    ("CN3", "NN2", "CN7B", "CN8"): "par_all36_na.prm: CN8 CN7B NN2 CN3 0.0 3 180.0",
}


class OpenMMUnavailableError(NeuroDNAError, ImportError):
    """OpenMM is not installed; the MD workflow is optional."""


def require_openmm() -> tuple[Any, Any, Any]:
    """Import OpenMM lazily: ``(openmm, openmm.app, openmm.unit)``."""
    try:
        import openmm
        import openmm.app
        import openmm.unit
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OpenMMUnavailableError(
            "the MD workflow needs OpenMM >= 8.1: pip install 'neurodna[md]'"
        ) from exc
    return openmm, openmm.app, openmm.unit


def openmm_data_dir() -> Path:
    _, app, _ = require_openmm()
    return Path(os.path.dirname(app.__file__)) / "data"


def force_field_sources(files: tuple[str, ...]) -> list[dict[str, str]]:
    """Source files and md5 hashes recorded in the OpenMM force-field XML headers."""
    out: list[dict[str, str]] = []
    for name in files:
        root = ET.parse(openmm_data_dir() / name).getroot()
        info = root.find("Info")
        for src in (info.findall("Source") if info is not None else []):
            out.append({"force_field_file": name, "source": src.get("Source", ""),
                        "md5": src.get("md5hash", ""), "package": src.get("sourcePackageVersion", "")})
    return out


def write_5cm_definitions(directory: Path) -> tuple[Path, Path]:
    """Hydrogen and bond definitions for 5CM, derived from OpenMM's own DC definitions.

    DC's H5 is removed and three methyl hydrogens are placed on C5A. OpenMM
    positions them geometrically; no parameters are involved.
    """
    data = openmm_data_dir()
    spec = MODIFIED_RESIDUES["5CM"]
    methyl, hydrogens = spec["methyl_carbon"], spec["methyl_hydrogens"]
    outputs = []
    for source, tag, out_name in (("hydrogens.xml", "H", "5cm_hydrogens.xml"),
                                  ("residues.xml", "Bond", "5cm_residues.xml")):
        dc = next(r for r in ET.parse(data / source).getroot() if r.get("name") == "DC")
        root = ET.Element("Residues")
        res = ET.SubElement(root, "Residue", name="5CM")
        for element in dc:
            names = {element.get("name"), element.get("from"), element.get("to")}
            if "H5" not in names:
                res.append(element)
        if tag == "H":
            for h in hydrogens:
                ET.SubElement(res, "H", {"name": h, "parent": methyl})
        else:
            ET.SubElement(res, "Bond", {"from": "C5", "to": methyl})
            for h in hydrogens:
                ET.SubElement(res, "Bond", {"from": methyl, "to": h})
        path = directory / out_name
        ET.ElementTree(root).write(path, encoding="unicode")
        outputs.append(path)
    return outputs[0], outputs[1]


def load_5cm_definitions(directory: Path) -> tuple[Path, Path]:
    """Write and register the 5CM definitions with OpenMM (process-wide)."""
    _, app, _ = require_openmm()
    hydrogens, bonds = write_5cm_definitions(directory)
    app.Topology.loadBondDefinitions(str(bonds))
    app.Modeller.loadHydrogenDefinitions(str(hydrogens))
    return hydrogens, bonds


def charmm_forcefield(files: tuple[str, ...]) -> Any:
    _, app, _ = require_openmm()
    return app.ForceField(*files)


def template_name(forcefield: Any, base: str, patches: tuple[str, ...]) -> str:
    """Build (once) and register the patched template ``base-patch1-patch2``."""
    name = "-".join((base, *patches))
    if name not in forcefield._templates:
        template = forcefield._templates[base]
        for patch in patches:
            allowed = {p for p, _ in forcefield._templatePatches.get(template.name.split("-")[0], ())}
            if patch not in allowed:
                raise NeuroDNAError(f"patch {patch} is not allowed on CHARMM template {base}")
            template = forcefield._patches[patch].createPatchedTemplates([template])[0]
        template.name = name
        forcefield.registerResidueTemplate(template)
    return name


def nucleotide_templates(forcefield: Any, topology: Any, dna_chains: tuple[str, ...]) -> dict[Any, str]:
    """Explicit CHARMM template for every nucleotide in the DNA chains, in file order."""
    templates: dict[Any, str] = {}
    for chain in topology.chains():
        if chain.id not in dna_chains:
            continue
        residues = [r for r in chain.residues() if r.name != "HOH"]
        for i, residue in enumerate(residues):
            if residue.name not in NUCLEOTIDES:
                raise NeuroDNAError(
                    f"no validated CHARMM36 template for nucleotide {chain.id}:{residue.name}{residue.id}"
                )
            base, extra = NUCLEOTIDES[residue.name]
            names = {a.name for a in residue.atoms()}
            if i == 0:
                if "P" in names:
                    raise NeuroDNAError(
                        f"5'-phosphorylated terminus {chain.id}:{residue.name}{residue.id} is not supported"
                    )
                sugar = SUGAR_PATCHES["5prime_hydroxyl"]
            elif i == len(residues) - 1:
                sugar = SUGAR_PATCHES["3prime_hydroxyl"]
            else:
                sugar = SUGAR_PATCHES["internal"]
            templates[residue] = template_name(forcefield, base, (*sugar, *extra))
    return templates


def bonded_term_audit(forcefield: Any, topology: Any, system: Any,
                      templates: dict[Any, str]) -> list[dict[str, Any]]:
    """Check every bond, angle and proper torsion touching each modified residue's base.

    A torsion without a PeriodicTorsionForce term is accepted only if its CHARMM
    atom classes are in :data:`VERIFIED_ZERO_TORSIONS`. Returns one record per
    modified residue with ``complete`` True/False.
    """
    mm, _, _ = require_openmm()
    from openmm.app.internal import compiled

    bonds: set[frozenset[int]] = set()
    angles: set[tuple[int, int, int]] = set()
    torsions: set[tuple[int, int, int, int]] = set()
    constrained = {frozenset(system.getConstraintParameters(i)[:2]) for i in range(system.getNumConstraints())}
    for force in system.getForces():
        if isinstance(force, mm.HarmonicBondForce):
            bonds |= {frozenset(force.getBondParameters(i)[:2]) for i in range(force.getNumBonds())}
        elif isinstance(force, mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a, b, c = force.getAngleParameters(i)[:3]
                angles.add((min(a, c), b, max(a, c)))
        elif isinstance(force, mm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                a, b, c, d = force.getTorsionParameters(i)[:4]
                torsions.add((a, b, c, d) if b < c else (d, c, b, a))
    neighbours: dict[int, set[int]] = {a.index: set() for a in topology.atoms()}
    for a, b in topology.bonds():
        neighbours[a.index].add(b.index)
        neighbours[b.index].add(a.index)
    bonded_to = [sorted(neighbours[i]) for i in range(topology.getNumAtoms())]
    atoms = list(topology.atoms())

    report = []
    for residue, tname in templates.items():
        if residue.name not in MODIFIED_RESIDUES:
            continue
        template = forcefield._templates[tname]
        match = compiled.matchResidueToTemplate(residue, template, bonded_to, False, False)
        classes = {atom.index: forcefield._atomTypes[template.atoms[t].type].atomClass
                   for atom, t in zip(residue.atoms(), match)}
        focus = {a.index for a in residue.atoms()
                 if a.name in ("C4", "C5", "C6", MODIFIED_RESIDUES[residue.name]["methyl_carbon"],
                               *MODIFIED_RESIDUES[residue.name]["methyl_hydrogens"])}
        missing: list[str] = []
        zero: list[str] = []
        n = {"bonds": 0, "angles": 0, "propers": 0}

        def name(q: tuple[int, ...]) -> str:
            return "-".join(atoms[i].name for i in q)

        for pair in {frozenset((i, j)) for i in focus for j in neighbours[i]}:
            n["bonds"] += 1
            if pair not in bonds and pair not in constrained:
                missing.append(f"bond {name(tuple(sorted(pair)))}")
        for j in neighbours:
            for i in neighbours[j]:
                for k in neighbours[j]:
                    if i < k and focus & {i, j, k}:
                        n["angles"] += 1
                        if (i, j, k) not in angles:
                            missing.append(f"angle {name((i, j, k))}")
        seen: set[tuple[int, int, int, int]] = set()
        for b in neighbours:
            for c in neighbours[b]:
                if b >= c:
                    continue
                for a in neighbours[b] - {c}:
                    for d in neighbours[c] - {b}:
                        q = (a, b, c, d)
                        if a == d or not focus & set(q) or q in seen:
                            continue
                        seen.add(q)
                        n["propers"] += 1
                        if q in torsions:
                            continue
                        cls: tuple[str, str, str, str] = (classes.get(a, "?"), classes.get(b, "?"),
                                                          classes.get(c, "?"), classes.get(d, "?"))
                        key = cls if cls in VERIFIED_ZERO_TORSIONS else (cls[3], cls[2], cls[1], cls[0])
                        if key in VERIFIED_ZERO_TORSIONS:
                            zero.append(f"{name(q)} ({VERIFIED_ZERO_TORSIONS[key]})")
                        else:
                            missing.append(f"proper {name(q)} classes {'-'.join(cls)}")
        report.append({
            "residue": f"{residue.chain.id}:{residue.name}{residue.id}",
            "template": tname,
            "checked": n,
            "zero_force_constant_torsions": sorted(zero),
            "missing": sorted(missing),
            "complete": not missing,
        })
    return report
