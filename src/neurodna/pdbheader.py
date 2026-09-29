"""Read the header records of a legacy-format PDB file.

Used to verify an entry before analysis: identity and revisions, molecules and
chains (COMPND), sequence database mapping and engineered differences (DBREF,
SEQADV), modified residues (MODRES/HETNAM), residues and atoms missing from the
model (REMARK 465/470) and biological assemblies (REMARK 350).

Only the records listed here are parsed. Column positions follow the wwPDB
format specification v3.3.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Compound:
    mol_id: int
    molecule: str
    chains: tuple[str, ...]
    fragment: str | None = None
    engineered: bool | None = None
    mutation: bool | None = None


@dataclass(frozen=True)
class ResidueRecord:
    """A residue named in a header record (MODRES, SEQADV, REMARK 465/470)."""

    resname: str
    chain: str
    resnum: int
    icode: str = ""
    detail: str = ""
    atoms: tuple[str, ...] = ()

    def label(self) -> str:
        return f"{self.chain}:{self.resname}{self.resnum}{self.icode}"


@dataclass(frozen=True)
class DbRef:
    chain: str
    seq_begin: int
    seq_end: int
    database: str
    accession: str
    db_begin: int
    db_end: int


@dataclass(frozen=True)
class Assembly:
    """One REMARK 350 biomolecule."""

    biomolecule: int
    author_unit: str | None
    software_unit: str | None
    chains: tuple[str, ...]
    operators: tuple[npt.NDArray[np.float64], ...]  # 3x4 [R|t] matrices

    @property
    def identity_only(self) -> bool:
        """True if the assembly is the deposited coordinates themselves (one identity operator)."""
        ident = np.hstack([np.eye(3), np.zeros((3, 1))])
        return len(self.operators) == 1 and bool(np.allclose(self.operators[0], ident))


@dataclass(frozen=True)
class PdbHeader:
    idcode: str
    classification: str
    deposition_date: str
    title: str
    experiment: str
    resolution_A: float | None
    revisions: tuple[tuple[int, str, str], ...]  # (number, date, records modified)
    compounds: tuple[Compound, ...]
    dbrefs: tuple[DbRef, ...]
    seqadv: tuple[ResidueRecord, ...]
    modres: tuple[ResidueRecord, ...]
    hetnam: dict[str, str] = field(default_factory=dict)
    missing_residues: tuple[ResidueRecord, ...] = ()
    missing_atoms: tuple[ResidueRecord, ...] = ()
    assemblies: tuple[Assembly, ...] = ()

    def compound_for_chain(self, chain: str) -> Compound | None:
        return next((c for c in self.compounds if chain in c.chains), None)


def read_pdb_header(path: str | os.PathLike[str]) -> PdbHeader:
    """Parse the header of a PDB-format file (stops at the first coordinate record)."""
    lines: list[str] = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM", "MODEL ")):
                break
            lines.append(line.rstrip("\n"))

    def rec(name: str) -> list[str]:
        return [l for l in lines if l[:6] == name.ljust(6)]

    def remark(number: int) -> list[str]:
        return [l for l in lines if l.startswith(f"REMARK {number:3d}")]

    header = rec("HEADER")[0] if rec("HEADER") else ""
    title = " ".join(l[10:80].strip() for l in rec("TITLE")).strip()
    expdta = " ".join(l[10:79].strip() for l in rec("EXPDTA")).strip()

    resolution = None
    for l in remark(2):
        m = re.search(r"RESOLUTION\.\s+([\d.]+)\s+ANGSTROMS", l)
        if m:
            resolution = float(m.group(1))

    revisions = tuple(
        (int(l[7:10]), l[13:22].strip(), l[39:66].strip())
        for l in rec("REVDAT") if not l[10:12].strip()  # continuation lines have a number here
    )
    return PdbHeader(
        idcode=header[62:66].strip(),
        classification=header[10:50].strip(),
        deposition_date=header[50:59].strip(),
        title=title,
        experiment=expdta,
        resolution_A=resolution,
        revisions=revisions,
        compounds=_compounds(rec("COMPND")),
        dbrefs=tuple(
            DbRef(l[12], int(l[14:18]), int(l[20:24]), l[26:32].strip(), l[33:41].strip(),
                  int(l[55:60]), int(l[62:67]))
            for l in rec("DBREF")
        ),
        seqadv=tuple(
            ResidueRecord(l[12:15].strip(), l[16], int(l[18:22]), l[22].strip(), _seqadv_detail(l))
            for l in rec("SEQADV")
        ),
        modres=tuple(
            ResidueRecord(l[12:15].strip(), l[16], int(l[18:22]), l[22].strip(),
                          f"{l[24:27].strip()} {l[29:80].strip()}".strip())
            for l in rec("MODRES")
        ),
        hetnam=_hetnam(rec("HETNAM")),
        missing_residues=_remark_residues(remark(465), "M RES C SSSEQI", atoms=False),
        missing_atoms=_remark_residues(remark(470), "M RES CSSEQI  ATOMS", atoms=True),
        assemblies=_assemblies(remark(350)),
    )


def _seqadv_detail(line: str) -> str:
    """``"ENGINEERED MUTATION (UNP P51608 ALA140)"``; database residue omitted for insertions."""
    conflict = line[49:70].strip()
    db_res, db_seq = line[39:42].strip(), line[43:48].strip()
    source = f"{line[24:28].strip()} {line[29:38].strip()}"
    return f"{conflict} ({source} {db_res}{db_seq})" if db_res else f"{conflict} ({source})"


def _yes_no(fields: dict[str, str], key: str) -> bool | None:
    return None if key not in fields else fields[key].upper() == "YES"


def _compounds(lines: list[str]) -> tuple[Compound, ...]:
    text = " ".join(l[10:80].strip() for l in lines)
    out: list[Compound] = []
    for block in re.split(r"MOL_ID:", text)[1:]:
        fields: dict[str, str] = {}
        parts = block.split(";")
        fields["MOL_ID"] = parts[0].strip()
        for part in parts[1:]:
            if ":" in part:
                k, v = part.split(":", 1)
                fields[k.strip()] = v.strip()

        out.append(Compound(
            mol_id=int(fields["MOL_ID"]),
            molecule=fields.get("MOLECULE", ""),
            chains=tuple(c.strip() for c in fields.get("CHAIN", "").split(",") if c.strip()),
            fragment=fields.get("FRAGMENT"),
            engineered=_yes_no(fields, "ENGINEERED"),
            mutation=_yes_no(fields, "MUTATION"),
        ))
    return tuple(out)


def _hetnam(lines: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for l in lines:
        code = l[11:14].strip()
        names[code] = (names.get(code, "") + " " + l[15:70].strip()).strip()
    return names


def _remark_residues(lines: list[str], column_header: str, *, atoms: bool) -> tuple[ResidueRecord, ...]:
    out: list[ResidueRecord] = []
    started = False
    for l in lines:
        body = l[10:]
        if not started:
            started = column_header in body
            continue
        if not body.strip():
            continue
        try:
            if atoms:  # REMARK 470: RES 16-18, C 20, SSEQ 21-24, I 25, atoms from 29
                out.append(ResidueRecord(l[15:18].strip(), l[19], int(l[20:24]), l[24].strip(),
                                         atoms=tuple(l[28:].split())))
            else:      # REMARK 465: RES 16-18, C 20, SSSEQ 22-26, I 27
                out.append(ResidueRecord(l[15:18].strip(), l[19], int(l[21:26]),
                                         l[26:27].strip()))
        except (ValueError, IndexError):
            continue
    return tuple(out)


@dataclass
class _AssemblyDraft:
    biomolecule: int
    author: str | None = None
    software: str | None = None
    chains: list[str] = field(default_factory=list)
    rows: dict[int, list[list[float]]] = field(default_factory=dict)

    def build(self) -> Assembly:
        ops = tuple(np.array(self.rows[k], dtype=float).reshape(3, 4)
                    for k in sorted(self.rows) if len(self.rows[k]) == 3)
        return Assembly(self.biomolecule, self.author, self.software, tuple(self.chains), ops)


def _assemblies(lines: list[str]) -> tuple[Assembly, ...]:
    drafts: list[_AssemblyDraft] = []
    for l in lines:
        body = l[10:].strip()
        if body.startswith("BIOMOLECULE:"):
            drafts.append(_AssemblyDraft(int(body.split(":", 1)[1])))
            continue
        if not drafts:
            continue
        current = drafts[-1]
        value = body.split(":", 1)[1].strip() if ":" in body else ""
        if body.startswith("AUTHOR DETERMINED BIOLOGICAL UNIT:"):
            current.author = value
        elif body.startswith("SOFTWARE DETERMINED QUATERNARY STRUCTURE:"):
            current.software = value
        elif body.startswith(("APPLY THE FOLLOWING TO CHAINS:", "AND CHAINS:")):
            current.chains += [c.strip() for c in value.split(",") if c.strip()]
        elif body.startswith("BIOMT"):
            parts = body.split()
            current.rows.setdefault(int(parts[1]), []).append([float(x) for x in parts[2:6]])
    return tuple(d.build() for d in drafts)
