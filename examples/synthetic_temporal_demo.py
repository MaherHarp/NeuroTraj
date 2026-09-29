"""Demonstration of neurodna's time-resolved APIs on a SYNTHETIC toy system.

No verified MeCP2–DNA MD trajectory accompanies this package, so the temporal
APIs are shown here on an invented system: one arginine and two nucleotides
placed by hand, with the arginine moved in and out of contact over seven
frames at chosen times. The coordinates are not derived from 3C2I or any real
structure, and the numbers printed are properties of this toy only. They say
nothing about MeCP2.

    python examples/synthetic_temporal_demo.py
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import MDAnalysis as mda
import pandas as pd

from neurodna import Complex

BANNER = "SYNTHETIC TOY SYSTEM - not a real structure or simulation"
TIMES_PS = [0.0, 10.0, 20.0, 30.0, 40.0, 55.0, 70.0]   # deliberately non-uniform
IN_CONTACT = [1, 1, 0, 1, 0, 0, 1]                      # arginine near the phosphate?

ATOMS = [  # name, resname, chain, resnum, x, y, z, element
    ("NH1", "ARG", "P", 1, 0.0, -3.5, 0.0, "N"),
    ("CZ", "ARG", "P", 1, 0.0, -4.8, 0.0, "C"),
    ("CA", "ARG", "P", 1, 0.0, -9.5, 0.0, "C"),
    ("P", "DC", "X", 1, 0.0, 0.0, 0.0, "P"),
    ("OP1", "DC", "X", 1, 0.0, 1.2, 0.0, "O"),
    ("C1'", "DC", "X", 1, 2.0, 2.0, 0.0, "C"),
    ("O3'", "DC", "X", 1, 4.4, 0.0, 0.0, "O"),
    ("P", "DG", "X", 2, 6.0, 0.0, 0.0, "P"),
    ("OP1", "DG", "X", 2, 6.0, 1.2, 0.0, "O"),
    ("C1'", "DG", "X", 2, 8.0, 2.0, 0.0, "C"),
    ("O3'", "DG", "X", 2, 10.4, 0.0, 0.0, "O"),
]


def write_toy(directory: Path) -> tuple[Path, Path]:
    lines = ["REMARK   1 SYNTHETIC TEST FIXTURE - NOT AN EXPERIMENTAL STRUCTURE"]
    for i, (name, res, chain, num, x, y, z, el) in enumerate(ATOMS, start=1):
        lines.append(f"ATOM  {i:5d}  {name:<3s} {res:>3s} {chain}{num:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
    pdb = directory / "synthetic_toy.pdb"
    pdb.write_text("\n".join(lines + ["END"]) + "\n")
    u = mda.Universe(str(pdb))
    base = u.atoms.positions.copy()
    arg = u.select_atoms("resname ARG").indices
    xtc = directory / "synthetic_toy.xtc"
    with mda.Writer(str(xtc), n_atoms=u.atoms.n_atoms) as w:
        for t, bound in zip(TIMES_PS, IN_CONTACT):
            pos = base.copy()
            if not bound:
                pos[arg, 1] -= 10.0  # move the arginine 10 Å away
            u.atoms.positions = pos
            u.trajectory.ts.time = t
            w.write(u.atoms)
    return pdb, xtc


def main() -> None:
    pd.set_option("display.width", 140)
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdb, xtc = write_toy(Path(tmp))
        cx = Complex.load(pdb, xtc, protein="chainID P", dna="chainID X")
        print(f"[{BANNER}]  {cx}")

        occ = cx.contact_occupancy()
        print(f"\n[SYNTHETIC] contact occupancy (cutoff {occ.attrs['cutoff_A']} Å); "
              f"sampling {occ.attrs['min_sampling_interval_ps']}-"
              f"{occ.attrs['max_sampling_interval_ps']} ps, "
              f"uniform={occ.attrs['uniform_sampling']}:")
        print(occ[["protein_label", "dna_label", "n_frames_in_contact", "n_frames_analyzed",
                   "occupancy"]].to_string(index=False))

        ep = cx.contact_episodes()
        print("\n[SYNTHETIC] contact episodes (times in ps):")
        print(ep[["protein_label", "dna_label", "episode", "first_time_ps", "last_time_ps",
                  "preceding_time_ps", "following_time_ps", "observed_duration_ps",
                  "max_duration_ps", "left_censored", "right_censored"]].to_string(index=False))
        print("\nNote:", ep.attrs["note"])


if __name__ == "__main__":
    main()
