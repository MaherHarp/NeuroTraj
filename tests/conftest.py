"""SYNTHETIC protein–DNA test systems with hand-placed coordinates (Å).

Everything built here is a synthetic test fixture, not an experimental structure;
every PDB file written by these helpers carries a ``REMARK   1 SYNTHETIC TEST
FIXTURE`` record. The only real structure in the test suite is ``data/3C2I.pdb``.

The basic synthetic system mimics the MeCP2 MBD / mCpG geometry in miniature:

DNA strand, chain C, 5'→3' (one nucleotide every 6 Å along x; O3'(i)–P(i+1) = 1.6 Å):
    C:DC1   P at x=0
    C:5CM2  P at x=6, methyl carbon C5A at (8, 6, 0)
    C:DG3   P at x=12

Protein:
    A:ARG1   NH1 at (8, 9, 0)   -> 3.0 Å from C:5CM2 C5A (the "arginine–methyl" contact)
    B:ARG1   NH1 at (0, -3.5, 0) -> 3.5 Å from C:DC1 P   (same residue number, other chain)
    A:GLY52 and A:SER52A far away (insertion code; no contacts)

The only other protein–DNA atom pair within 4.5 Å is A:ARG1 CZ – C:5CM2 C5A at 4.3 Å;
all remaining pairs are ≥ 4.7 Å apart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader


@dataclass(frozen=True)
class Atom:
    name: str
    resname: str
    chain: str
    resnum: int
    x: float
    y: float
    z: float = 0.0
    icode: str = ""
    segid: str = ""
    altloc: str = ""
    element: str = ""

    def pdb_line(self, serial: int) -> str:
        name = self.name if len(self.name) == 4 else f" {self.name:<3s}"
        element = self.element or self.name.lstrip("0123456789")[0]
        return (
            f"ATOM  {serial:5d} {name:<4s}{self.altloc or ' ':1s}{self.resname:>3s} "
            f"{self.chain:1s}{self.resnum:4d}{self.icode or ' ':1s}   "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}{1.0:6.2f}{0.0:6.2f}      "
            f"{self.segid:<4s}{element:>2s}"
        )


def nucleotide(resname: str, chain: str, resnum: int, x0: float, *, methyl: bool = False,
               segid: str = "") -> list[Atom]:
    atoms = [
        Atom("P", resname, chain, resnum, x0, 0.0, segid=segid, element="P"),
        Atom("OP1", resname, chain, resnum, x0, 1.2, segid=segid, element="O"),
        Atom("C1'", resname, chain, resnum, x0 + 2, 2.0, segid=segid, element="C"),
        Atom("N1", resname, chain, resnum, x0 + 2, 4.0, segid=segid, element="N"),
        Atom("O3'", resname, chain, resnum, x0 + 4.4, 0.0, segid=segid, element="O"),
    ]
    if methyl:
        atoms.append(Atom("C5A", resname, chain, resnum, x0 + 2, 6.0, segid=segid, element="C"))
    return atoms


def arginine(chain: str, resnum: int, nh1: tuple[float, float], direction: float,
             segid: str = "") -> list[Atom]:
    """Arg with NH1 at ``nh1``; other atoms extend away from the DNA along ±y."""
    x, y = nh1
    return [
        Atom("NH1", "ARG", chain, resnum, x, y, segid=segid, element="N"),
        Atom("CZ", "ARG", chain, resnum, x, y + 1.3 * direction, segid=segid, element="C"),
        Atom("CA", "ARG", chain, resnum, x, y + 6.0 * direction, segid=segid, element="C"),
    ]


def synthetic_atoms() -> list[Atom]:
    return [
        *arginine("A", 1, (8.0, 9.0), +1),
        Atom("CA", "GLY", "A", 52, 30.0, 30.0, element="C"),
        Atom("CA", "SER", "A", 52, 34.0, 30.0, icode="A", element="C"),
        *arginine("B", 1, (0.0, -3.5), -1),
        *nucleotide("DC", "C", 1, 0.0),
        *nucleotide("5CM", "C", 2, 6.0, methyl=True),
        *nucleotide("DG", "C", 3, 12.0),
    ]


def pdb_text(models: Iterable[list[Atom]], cryst1: tuple[float, float, float] | None = None,
             space_group: str = "P 1") -> str:
    models = list(models)
    lines = ["REMARK   1 SYNTHETIC TEST FIXTURE - NOT AN EXPERIMENTAL STRUCTURE"]
    if cryst1 is not None:
        a, b, c = cryst1
        lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{90:7.2f}{90:7.2f}{90:7.2f} "
                     f"{space_group:<11s}{1:4d}")
    multi = len(models) > 1
    for m, atoms in enumerate(models, start=1):
        if multi:
            lines.append(f"MODEL     {m:4d}")
        lines.extend(atom.pdb_line(i) for i, atom in enumerate(atoms, start=1))
        if multi:
            lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"


PROTEIN = "protein"
DNA = "chainID C"


@pytest.fixture
def write_pdb(tmp_path: Path) -> Callable[..., Path]:
    counter = iter(range(1000))

    def _write(atoms: list[Atom] | list[list[Atom]], *,
               cryst1: tuple[float, float, float] | None = None,
               space_group: str = "P 1") -> Path:
        models = atoms if atoms and isinstance(atoms[0], list) else [atoms]
        path = tmp_path / f"synthetic{next(counter)}.pdb"
        path.write_text(pdb_text(models, cryst1, space_group))  # type: ignore[arg-type]
        return path

    return _write


@pytest.fixture
def synthetic_pdb(write_pdb: Callable[..., Path]) -> Path:
    """SYNTHETIC reference complex (see module docstring)."""
    return write_pdb(synthetic_atoms())


def moved(atoms: list[Atom], chain: str, resnum: int, dy: float) -> list[Atom]:
    """Copy of ``atoms`` with residue chain:resnum shifted by ``dy`` along y."""
    return [replace(a, y=a.y + dy) if (a.chain, a.resnum) == (chain, resnum) else a for a in atoms]


def coords(atoms: list[Atom]) -> np.ndarray:
    return np.array([[a.x, a.y, a.z] for a in atoms], dtype=np.float32)


def memory_universe(topology: Path, frames: list[np.ndarray], *, dt: float = 10.0,
                    dimensions: np.ndarray | None = None) -> mda.Universe:
    """SYNTHETIC in-memory trajectory: topology from a PDB, coordinates per frame.

    Uses float32 coordinates without file precision loss, so numerical tests can
    be exact. ``dt`` (ps) becomes the frame time step; ``dimensions`` may be
    ``(6,)`` or ``(n_frames, 6)``.
    """
    u = mda.Universe(str(topology))
    u.load_new(np.stack(frames).astype(np.float32), format=MemoryReader, dt=dt,
               dimensions=dimensions)
    return u


# ---------------------------------------------------------------- synthetic peptide

def synthetic_peptide(chain: str = "A", n_residues: int = 4, x0: float = 0.0,
                      y0: float = 0.0) -> list[Atom]:
    """SYNTHETIC zig-zag poly-Ala backbone (N, CA, C, O) plus CB, bond lengths 1.3-1.5 Å.

    Residue i starts at x = x0 + 3.8 i; peptide C(i)-N(i+1) = 1.32 Å.
    """
    atoms: list[Atom] = []
    for i in range(n_residues):
        x = x0 + 3.8 * i
        r = i + 1
        atoms += [
            Atom("N", "ALA", chain, r, x, y0, 0.0, element="N"),
            Atom("CA", "ALA", chain, r, x + 1.2, y0 + 0.8, 0.0, element="C"),
            Atom("C", "ALA", chain, r, x + 2.5, y0 + 0.2, 0.0, element="C"),
            Atom("O", "ALA", chain, r, x + 2.7, y0 - 1.0, 0.5, element="O"),
            Atom("CB", "ALA", chain, r, x + 1.2, y0 + 1.6, 1.3, element="C"),
        ]
    return atoms


def far_nucleotide(chain: str = "D", x0: float = 0.0, y0: float = 20.0) -> list[Atom]:
    """SYNTHETIC single nucleotide, far from the synthetic peptide."""
    return [replace(a, y=a.y + y0) for a in nucleotide("DC", chain, 1, x0)]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random proper rotation matrix."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def write_xtc(topology: Path, path: Path, frames: list[list[Atom]], times: list[float]) -> Path:
    """SYNTHETIC XTC trajectory with explicit frame times (ps); XTC precision is 0.01 Å."""
    u = mda.Universe(str(topology))
    with mda.Writer(str(path), n_atoms=u.atoms.n_atoms) as w:
        for atoms, t in zip(frames, times):
            u.atoms.positions = coords(atoms)
            u.trajectory.ts.time = t
            w.write(u.atoms)
    return path
