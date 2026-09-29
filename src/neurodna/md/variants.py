"""Derived comparison models, built by explicit and minimal edits to a PDB file.

:func:`demethylate_cytosines` turns listed 5-methyl-2'-deoxycytidines (PDB 5CM)
into 2'-deoxycytidines (DC). The only coordinate change is deleting the 5-methyl
carbon (C5A) and any methyl hydrogens. The hydrogen that replaces it on C5 (H5)
is added later by the ordinary preparation step, like every other hydrogen.

Everything else is copied byte for byte: the protein, DNA backbone and base
coordinates, crystallographic waters and B-factors. The result is a *derived
model*, not an experimental structure. A REMARK line says so, and a JSON
record lists every change.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from neurodna.errors import NeuroDNAError

METHYL_CARBON = "C5A"
METHYL_HYDROGENS = ("H5A1", "H5A2", "H5A3", "H71", "H72", "H73")
_COORD = ("ATOM  ", "HETATM", "ANISOU")


def _sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demethylate_cytosines(pdb_in: str | os.PathLike[str], pdb_out: str | os.PathLike[str],
                          residues: list[str]) -> dict[str, Any]:
    """Write ``pdb_out`` with the listed 5CM residues (``"B:8"``) converted to DC.

    Every 5CM residue in the file must be listed. That keeps header records
    (SEQRES, COMPND, MODRES, HET, ...) unambiguous. Returns the provenance
    record, which is also written to ``<pdb_out>.json``.

    Raises:
        NeuroDNAError: a listed residue is missing or not 5CM, an unlisted 5CM
            remains, or a residue has atoms other than the expected ones.
    """
    targets: set[tuple[str, int]] = set()
    for label in residues:
        chain, _, number = label.partition(":")
        if len(chain) != 1 or not number.strip().lstrip("-").isdigit():
            raise NeuroDNAError(f"residue labels look like 'B:8', got {label!r}")
        targets.add((chain, int(number)))

    lines = Path(pdb_in).read_text().splitlines(keepends=True)
    found: dict[tuple[str, int], list[str]] = {}
    all_5cm: set[tuple[str, int]] = set()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            key = (line[21], int(line[22:26]))
            if line[17:20].strip() == "5CM":
                all_5cm.add(key)
            if key in targets:
                if line[17:20].strip() != "5CM":
                    raise NeuroDNAError(f"{key[0]}:{line[17:20].strip()}{key[1]} is not 5CM")
                found.setdefault(key, []).append(line[12:16].strip())
    missing = targets - set(found)
    if missing:
        raise NeuroDNAError(f"residues not found: {sorted(missing)}")
    if all_5cm - targets:
        raise NeuroDNAError(f"unlisted 5CM residues remain: {sorted(all_5cm - targets)}; list every "
                            "5CM so that header records stay unambiguous")

    removed: list[dict[str, Any]] = []
    removed_serials: set[str] = set()
    touched_serials: set[str] = set()
    out: list[str] = []
    header_changes: list[str] = []
    for line in lines:
        record = line[:6]
        if record in _COORD and (line[21], int(line[22:26])) in targets:
            name = line[12:16].strip()
            serial = line[6:11].strip()
            touched_serials.add(serial)
            if name == METHYL_CARBON or name in METHYL_HYDROGENS:
                removed_serials.add(serial)
                if record != "ANISOU":
                    removed.append({"residue": f"{line[21]}:5CM{int(line[22:26])}", "atom": name,
                                    "serial": int(serial),
                                    "xyz_A": [float(line[30:38]), float(line[38:46]), float(line[46:54])]})
                continue
            new_record = "ATOM  " if record == "HETATM" else record
            out.append(new_record + line[6:17] + " DC" + line[20:])
            continue
        if record == "CONECT":
            serials = {line[i:i + 5].strip() for i in range(6, 31, 5)} - {""}
            if serials & touched_serials:
                header_changes.append(f"dropped CONECT {line.rstrip()[6:].strip()}")
                continue
        elif record in ("MODRES", "HET   ", "HETNAM", "HETSYN", "FORMUL") and " 5CM " in f" {line[6:]} ":
            header_changes.append(f"dropped {record.strip()} record for 5CM")
            continue
        elif record == "LINK  " and "5CM" in line:
            header_changes.append("dropped LINK record involving 5CM")
            continue
        elif record == "SEQRES" and "5CM" in line:
            line = line.replace("5CM", " DC")
            header_changes.append(f"SEQRES chain {line[11]}: 5CM -> DC")
        elif record == "COMPND" and "(5CM)" in line:
            width = len(line.rstrip("\n"))
            line = line.rstrip("\n").replace("(5CM)", "DC").ljust(width) + "\n"
            header_changes.append("COMPND molecule text: (5CM) -> DC")
        out.append(line)

    expected = len(targets)
    if len([r for r in removed if r["atom"] == METHYL_CARBON]) != expected:
        raise NeuroDNAError("each listed residue must contain exactly one methyl carbon C5A")

    labels = ", ".join(f"{c}:5CM{n}" for c, n in sorted(targets))
    texts = ["NEURODNA DERIVED MODEL - NOT AN EXPERIMENTAL STRUCTURE.",
             f"5-METHYL GROUPS REMOVED FROM {labels}",
             "(ATOM C5A DELETED, RESIDUES RENAMED DC).",
             "ALL OTHER COORDINATES AS DEPOSITED."]
    if any(len(t) > 69 for t in texts):
        raise NeuroDNAError("too many residues to describe in REMARK 999 lines")
    remark = [f"REMARK 999 {t}".ljust(80) + "\n" for t in texts]
    first_remark = next((i for i, line in enumerate(out) if line.startswith("REMARK")), 1)
    out[first_remark:first_remark] = remark

    Path(pdb_out).write_text("".join(out))
    provenance = {
        "kind": "neurodna derived model (computational demethylation)",
        "source": {"path": os.fspath(pdb_in), "sha256": _sha256(pdb_in)},
        "output": {"path": os.fspath(pdb_out), "sha256": _sha256(pdb_out)},
        "residues": [f"{c}:{n}" for c, n in sorted(targets)],
        "conversion": "5CM -> DC: 5-methyl carbon C5A deleted; H5 on C5 is added during preparation",
        "removed_atoms": removed,
        "header_changes": header_changes,
        "unchanged": "all other atom records (protein, DNA, crystallographic waters), coordinates "
                     "and B-factors copied byte for byte",
    }
    Path(f"{pdb_out}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
