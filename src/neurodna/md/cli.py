"""``neurodna-md``: assess, prepare, run and analyse OpenMM simulations.

    neurodna-md assess 3C2I.pdb --config preparation.json
    neurodna-md prepare 3C2I.pdb --config preparation.json --output-dir prepared/
    neurodna-md run prepared/ --protocol smoke --output-dir runs/smoke
    neurodna-md analyze runs/smoke --output-dir runs/smoke/analysis
    neurodna-md derive-unmethylated 3C2I.pdb --residues B:8 C:33 --output 3C2I_CpG.pdb
    neurodna-md import-external --topology sys.pdb --system system.xml \\
        --protein "segid PROA" --dna "segid DNAA DNAB" --output-dir prepared_external/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from neurodna.errors import NeuroDNAError


def import_external(topology: Path, system_xml: Path, protein: str, dna: str,
                    output_dir: Path, notes: str = "") -> Path:
    """Register an externally prepared OpenMM system so ``run``/``analyze`` can use it.

    neurodna does not check the external system's parameters; provenance of the
    preparation is the user's responsibility and is recorded as such.
    """
    from neurodna.md.prepare import sha256

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(topology, out / "prepared.pdb")
    shutil.copyfile(system_xml, out / "system.xml")
    preparation = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": "external",
        "statement": "Prepared outside neurodna; parameters and preparation choices were not "
                     "checked by neurodna.",
        "notes": notes,
        "input": {"topology": str(topology), "system_xml": str(system_xml)},
        "selections": {"protein": protein, "dna": dna},
        "files": {"prepared_pdb": {"name": "prepared.pdb", "sha256": sha256(out / "prepared.pdb")},
                  "system_xml": {"name": "system.xml", "sha256": sha256(out / "system.xml")}},
    }
    (out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurodna-md", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("assess", help="check force-field compatibility and report decisions")
    p.add_argument("pdb", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="write the report as JSON")
    p = sub.add_parser("prepare", help="build the solvated, parameterised system")
    p.add_argument("pdb", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("run", help="minimise, equilibrate and run production")
    p.add_argument("prepared_dir", type=Path)
    p.add_argument("--protocol", required=True, help="'smoke', 'production' or a JSON file")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--platform", default=None, help="override the protocol platform (CPU, OpenCL, CUDA)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=None, help="override the protocol's cpu_threads")
    p.add_argument("--stop-after-stage", default=None,
                   help="end the session after this stage (minimization or an equilibration stage)")
    p.add_argument("--stop-at-production-step", type=int, default=None,
                   help="end the session at this production step (a checkpoint multiple)")
    p = sub.add_parser("analyze", help="neurodna analysis of a run's production trajectory")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cutoff", type=float, default=4.5)
    p.add_argument("--burn-in-ps", type=float, default=0.0,
                   help="exclude production frames with time <= this (ps) from every table (0: keep all)")
    p.add_argument("--prepared-dir", type=Path, default=None,
                   help="prepared-system directory, if the path recorded in simulation.json does not resolve")
    p = sub.add_parser("derive-unmethylated",
                       help="derived model: delete 5-methyl groups of listed 5CM residues (-> DC)")
    p.add_argument("pdb", type=Path)
    p.add_argument("--residues", nargs="+", required=True, help="e.g. B:8 C:33")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("import-external", help="use an externally prepared OpenMM system")
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--system", type=Path, required=True)
    p.add_argument("--protein", required=True)
    p.add_argument("--dna", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    try:
        if args.command == "assess":
            from neurodna.md.assess import assess
            from neurodna.md.config import load_preparation_config

            report = assess(args.pdb, load_preparation_config(args.config))
            text = json.dumps(report, indent=2, default=str)
            if args.output:
                args.output.write_text(text + "\n")
            print(text)
            return 0 if report["supported"] else 2
        if args.command == "prepare":
            from neurodna.md.config import load_preparation_config
            from neurodna.md.prepare import prepare

            print(prepare(args.pdb, load_preparation_config(args.config), args.output_dir))
            return 0
        if args.command == "run":
            from dataclasses import replace

            from neurodna.md.config import load_protocol
            from neurodna.md.run import run

            protocol = load_protocol(args.protocol)
            if args.platform:
                protocol = replace(protocol, platform=args.platform)
            if args.cpu_threads:
                protocol = replace(protocol, cpu_threads=args.cpu_threads)
            meta = run(args.prepared_dir, protocol, args.output_dir, resume=args.resume,
                       stop_after_stage=args.stop_after_stage,
                       stop_at_production_step=args.stop_at_production_step)
            print(json.dumps({"status": meta["status"], "simulated_ps": meta["simulated_ps"],
                              "stages": meta["stages"], "platform": meta["platform"]}, indent=2, default=str))
            return 0
        if args.command == "analyze":
            from neurodna.md.analyze import analyze_run

            outputs = analyze_run(args.run_dir, args.output_dir, args.cutoff, args.burn_in_ps, args.prepared_dir)
            for name, path in outputs.items():
                print(f"{name:18s} {path}")
            return 0
        if args.command == "derive-unmethylated":
            from neurodna.md.variants import demethylate_cytosines

            record = demethylate_cytosines(args.pdb, args.output, args.residues)
            print(json.dumps(record, indent=2))
            return 0
        if args.command == "import-external":
            print(import_external(args.topology, args.system, args.protein, args.dna,
                                  args.output_dir, args.notes))
            return 0
    except (NeuroDNAError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
