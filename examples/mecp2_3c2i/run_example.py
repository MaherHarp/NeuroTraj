"""Static MeCP2 MBD – methylated DNA contact analysis of RCSB PDB entry 3C2I.

3C2I (Ho et al., Mol Cell 2008) is an X-ray structure (2.5 Å) of the
methyl-CpG-binding domain (MBD) of human MeCP2, UniProt P51608 residues 77-167
(residues 91-162 are modelled), bound to a 20-bp DNA duplex from the BDNF gene.
Each strand carries one 5-methylcytosine (5CM). It is a structure of a domain,
not of full-length MeCP2, and it is a single static model: this example
computes no dynamics.

Usage (downloads 3C2I once into the cache, then reuses it):

    python examples/mecp2_3c2i/run_example.py [--cache-dir DIR] [--output-dir DIR]

Offline, from an existing copy (its checksum is recorded and compared):

    python examples/mecp2_3c2i/run_example.py --pdb-file path/to/3C2I.pdb
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import MDAnalysis  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import neurodna  # noqa: E402
from neurodna import Complex, FrameKind  # noqa: E402
from neurodna.fetch import fetch_pdb, sha256_file  # noqa: E402
from neurodna.pdbheader import PdbHeader, read_pdb_header  # noqa: E402
from neurodna.plotting import plot_contact_map  # noqa: E402

ENTRY_ID = "3C2I"
# SHA-256 of the PDB-format file as retrieved on 2026-09-25 (entry revision 1.4,
# 2024-11-20). If RCSB remediates the entry, the download stops with a checksum
# error: inspect the new revision before updating this value.
EXPECTED_SHA256 = "f96ee9192eabe5eafc50ff0b161c51275ad6ad8f03a2c0d0e447f18a820ed7bf"

# Chains identified from the entry (COMPND / REMARK 350), not guessed:
PROTEIN_CHAIN = "A"          # MeCP2 MBD, UniProt P51608 77-167 (+ His tag)
DNA_CHAINS = ("B", "C")      # 20-nt strands from BDNF, one 5CM each
PROTEIN_SELECTION = f"chainID {PROTEIN_CHAIN} and not resname HOH"
DNA_SELECTION = f"chainID {' '.join(DNA_CHAINS)} and not resname HOH"
EXPECTED_MODIFIED_DNA = {"B:5CM8", "C:5CM33"}

CUTOFF_A = neurodna.DEFAULT_CONTACT_CUTOFF  # 4.5 Å, minimum heavy-atom distance
MAP_MAX_DISTANCE_A = 8.0                     # reporting radius for the contact map

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- verification


def verify_entry(header: PdbHeader) -> dict[str, Any]:
    """Check that the file is the expected entry and describe it. Raises on surprises."""
    problems: list[str] = []
    if header.idcode != ENTRY_ID:
        problems.append(f"file is entry {header.idcode!r}, not {ENTRY_ID}")
    protein = header.compound_for_chain(PROTEIN_CHAIN)
    if protein is None or "METHYL-CPG-BINDING PROTEIN 2" not in protein.molecule:
        problems.append(f"chain {PROTEIN_CHAIN} is not MeCP2: {protein}")
    for chain in DNA_CHAINS:
        compound = header.compound_for_chain(chain)
        if compound is None or not compound.molecule.startswith("DNA"):
            problems.append(f"chain {chain} is not DNA: {compound}")
    modified = {r.label() for r in header.modres if r.resname == "5CM"}
    if modified != EXPECTED_MODIFIED_DNA:
        problems.append(f"5CM residues are {sorted(modified)}, expected {sorted(EXPECTED_MODIFIED_DNA)}")
    if len(header.assemblies) != 1:
        problems.append(f"expected one biological assembly, found {len(header.assemblies)}")
    else:
        assembly = header.assemblies[0]
        if set(assembly.chains) != {PROTEIN_CHAIN, *DNA_CHAINS} or not assembly.identity_only:
            problems.append(
                f"assembly 1 is chains {assembly.chains} with {len(assembly.operators)} "
                "operator(s); the deposited coordinates would not be the complex"
            )
    if problems:
        raise RuntimeError("3C2I verification failed:\n  - " + "\n  - ".join(problems))

    unp = next(d for d in header.dbrefs if d.chain == PROTEIN_CHAIN)
    return {
        "idcode": header.idcode,
        "title": header.title,
        "experiment": header.experiment,
        "resolution_A": header.resolution_A,
        "deposition_date": header.deposition_date,
        "revisions": [{"number": n, "date": d, "records": r} for n, d, r in header.revisions],
        "protein": {
            "chain": PROTEIN_CHAIN,
            "molecule": protein.molecule if protein else None,
            "fragment": protein.fragment if protein else None,
            "uniprot": f"{unp.accession} residues {unp.db_begin}-{unp.db_end}",
            "sequence_differences": [f"{r.label()}: {r.detail}" for r in header.seqadv],
            "note": "Methyl-CpG-binding domain only; full-length MeCP2 (UniProt P51608) is "
                    "486 residues.",
        },
        "dna_chains": {c: header.compound_for_chain(c).molecule  # type: ignore[union-attr]
                       for c in DNA_CHAINS},
        "modified_residues": [f"{r.label()} ({r.detail})" for r in header.modres],
        "heterogen_names": header.hetnam,
        "missing_residues": [r.label() for r in header.missing_residues],
        "missing_atoms": [f"{r.label()}: {' '.join(r.atoms)}" for r in header.missing_atoms],
        "assembly": {
            "biomolecule": header.assemblies[0].biomolecule,
            "author_unit": header.assemblies[0].author_unit,
            "software_unit": header.assemblies[0].software_unit,
            "chains": list(header.assemblies[0].chains),
            "identity_operator_only": header.assemblies[0].identity_only,
        },
    }


# --------------------------------------------------------------------- analysis


def residue_notes(header: PdbHeader) -> dict[str, str]:
    """Per-residue caveats from the header (engineered mutations, SeMet, missing atoms)."""
    notes: dict[str, list[str]] = {}
    for r in header.seqadv:
        if "EXPRESSION TAG" not in r.detail:
            notes.setdefault(r.label(), []).append(r.detail.lower())
    for r in header.modres:
        if r.resname == "MSE":
            notes.setdefault(r.label(), []).append("selenomethionine in place of Met")
    for r in header.missing_atoms:
        notes.setdefault(r.label(), []).append("missing atoms: " + " ".join(r.atoms))
    return {k: "; ".join(v) for k, v in notes.items()}


def analyze(pdb_path: Path, output_dir: Path, provenance: dict[str, Any]) -> dict[str, Path]:
    """Run the static analysis on a local 3C2I file and write all outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    header = read_pdb_header(pdb_path)
    entry = verify_entry(header)
    notes = residue_notes(header)
    unp = next(d for d in header.dbrefs if d.chain == PROTEIN_CHAIN)

    cx = Complex.load(pdb_path, protein=PROTEIN_SELECTION, dna=DNA_SELECTION)
    if cx.kind is not FrameKind.STATIC:
        raise RuntimeError(f"expected one static model, got {cx.kind.value}")
    dna_table = cx.dna_residues()
    kept = set(dna_table.loc[dna_table.modification == "5mC", "label"])
    if kept != EXPECTED_MODIFIED_DNA:
        raise RuntimeError(f"methylated residues not preserved: {sorted(kept)}")

    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        in_unp = df.protein_resnum.between(unp.seq_begin, unp.seq_end)
        df.insert(df.columns.get_loc("protein_label") + 1, "protein_uniprot_resnum",
                  (df.protein_resnum - unp.seq_begin + unp.db_begin).where(in_unp).astype("Int64"))
        df["protein_note"] = df.protein_label.map(notes).fillna("")
        return df

    contacts = annotate(cx.contacts(CUTOFF_A))
    atom_contacts = annotate(cx.contacts(CUTOFF_A, level="atom"))
    methyl = annotate(cx.contacts(CUTOFF_A, level="atom", target="modification_markers"))
    distances = cx.min_distances(max_distance=MAP_MAX_DISTANCE_A)
    strands = cx.dna_strands()
    steps = cx.cpg_steps()
    waters = cx.universe.select_atoms("resname HOH").residues

    outputs = {
        "residue_contacts": output_dir / "residue_contacts.csv",
        "atom_contacts": output_dir / "atom_contacts.csv",
        "methyl_contacts": output_dir / "methyl_contacts.csv",
        "min_distances": output_dir / "min_distances.csv",
        "dna_residues": output_dir / "dna_residues.csv",
        "contact_map": output_dir / "contact_map.png",
        "parameters": output_dir / "parameters.json",
    }
    contacts.to_csv(outputs["residue_contacts"], index=False)
    atom_contacts.to_csv(outputs["atom_contacts"], index=False)
    methyl.to_csv(outputs["methyl_contacts"], index=False)
    distances.to_csv(outputs["min_distances"], index=False)
    dna_table.to_csv(outputs["dna_residues"], index=False)

    # Contact map: every protein residue within the reporting radius, strands 5'->3'.
    dna_order = [k.label() for s in strands for k in s.residues]
    ax = plot_contact_map(
        distances,
        cutoff=CUTOFF_A,
        dna_order=dna_order,
        title="MeCP2 MBD – methylated BDNF DNA, PDB 3C2I (static X-ray model, 2.5 Å)",
        subtitle="Rows: residues within 8 Å of DNA; crystallographic waters excluded",
    )
    ax.figure.savefig(outputs["contact_map"], dpi=200)
    plt.close(ax.figure)

    parameters = {
        "description": "Static geometric contact analysis of one crystallographic model. "
                       "No dynamics, energies, hydrogen bonds or affinities are computed.",
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"path": str(pdb_path), "sha256": sha256_file(pdb_path),
                  "pinned_sha256": EXPECTED_SHA256, **provenance},
        "entry": entry,
        "selections": {"protein": PROTEIN_SELECTION, "dna": DNA_SELECTION},
        "contact_definition": neurodna.CONTACT_DEFINITION,
        "cutoff_A": CUTOFF_A,
        "contact_map_max_distance_A": MAP_MAX_DISTANCE_A,
        "atoms": "heavy atoms only (element records cross-checked with atom names)",
        "units": {"distance": "angstrom", "time": "ps (not applicable: static)"},
        "frame_kind": cx.kind.value,
        "periodic_policy": cx.periodic_policy,
        "minimum_image_used": bool(contacts.attrs["minimum_image"]),
        "excluded": {"crystallographic_waters": int(len(waters)),
                     "reason": "protein-DNA heavy-atom contacts only; water-mediated contacts "
                               "are not analysed"},
        "counts": {"protein_residues": len(cx.protein_residue_keys),
                   "dna_residues": len(cx.dna_residue_keys),
                   "protein_heavy_atoms": len(cx.protein_heavy),
                   "dna_heavy_atoms": len(cx.dna_heavy),
                   "residue_contacts": len(contacts),
                   "atom_contacts": len(atom_contacts),
                   "methyl_atom_contacts": len(methyl)},
        "dna_strands": [{"chain": s.residues[0].chain, "annotated_sequence_5to3": s.annotated_sequence}
                        for s in strands],
        "cpg_steps": [f"{r.c_chain}:{r.c_resname}{r.c_resnum}-{r.g_resname}{r.g_resnum} "
                      f"({r.c_modification if isinstance(r.c_modification, str) else 'unmodified'})"
                      for r in steps.itertuples()],
        "software": {"neurodna": neurodna.__version__, "MDAnalysis": MDAnalysis.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__, "python": platform.python_version()},
        "outputs": {k: v.name for k, v in outputs.items()},
    }
    outputs["parameters"].write_text(json.dumps(parameters, indent=2, default=str) + "\n")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="download cache (default: neurodna.fetch.default_cache_dir())")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--pdb-file", type=Path, default=None,
                        help="use an existing local copy instead of downloading")
    parser.add_argument("--refresh", action="store_true", help="download again")
    args = parser.parse_args(argv)

    if args.pdb_file is not None:
        pdb_path = args.pdb_file
        provenance: dict[str, Any] = {"source": "local file supplied with --pdb-file",
                                      "matches_pinned_sha256":
                                          sha256_file(pdb_path) == EXPECTED_SHA256}
    else:
        fetched = fetch_pdb(ENTRY_ID, cache_dir=args.cache_dir, expected_sha256=EXPECTED_SHA256,
                            refresh=args.refresh)
        pdb_path = fetched.path
        provenance = {"source": "RCSB download", "from_cache": fetched.from_cache,
                      **fetched.metadata}

    outputs = analyze(pdb_path, args.output_dir, provenance)
    for name, path in outputs.items():
        print(f"{name:18s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
