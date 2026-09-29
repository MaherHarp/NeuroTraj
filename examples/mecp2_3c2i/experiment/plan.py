"""Validate the MeCP2-3C2I experiment design and write per-run configs + launch commands.

This script never starts a simulation. It checks that:

* every preparation config and protocol loads with neurodna's own validators;
* all seeds are unique, including the +1/+2 seeds each run derives;
* the stronger tier can resume the pilot runs (only production_steps may change);
* the analysis plan refers to residues present in both conditions.

It then prints the simulated time, trajectory frames and storage per tier, and
writes ``<plan-dir>/<tier>/<condition>/r<i>/{preparation,protocol}.json`` and
``<plan-dir>/commands_<tier>.sh``.

    python examples/mecp2_3c2i/experiment/plan.py --plan-dir PLAN --experiment-dir RUNS
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from neurodna.md.config import PreparationConfig, Protocol
from neurodna.md.run import _check_resumable

HERE = Path(__file__).resolve().parent
TIERS = ("development", "pilot", "stronger")


def load_design(directory: Path = HERE) -> dict[str, Any]:
    design = json.loads((directory / "experiment.json").read_text())
    design["_prep"] = json.loads((directory / design["shared_preparation_config"]).read_text())
    design["_protocols"] = {t: Protocol.from_dict(json.loads((directory / design["tiers"][t]["protocol"])
                                                             .read_text())) for t in TIERS}
    design["_analysis"] = json.loads((directory / design["analysis_plan"]).read_text())
    return design


def runs(design: dict[str, Any], tier: str) -> list[dict[str, Any]]:
    """One entry per (condition, replicate) of a tier, with its configs."""
    out = []
    for condition in design["conditions"]:
        for r in design["tiers"][tier]["replicates"]:
            seeds = design["seeds"][f"{tier}/{condition}/r{r}"]
            prep = PreparationConfig.from_dict({**design["_prep"], "preparation_seed": seeds["preparation_seed"]})
            proto = replace(design["_protocols"][tier], seed=seeds["run_seed"])
            extends = design["tiers"][tier].get("extends")
            reuse = extends is not None and r in extends["replicates"]
            out.append({"tier": tier, "condition": condition, "replicate": r, "preparation": prep,
                        "protocol": proto, "run_tier": extends["tier"] if reuse else tier, "resume": reuse})
    return out


def validate(design: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty if the design is consistent)."""
    problems: list[str] = []
    used: dict[int, str] = {}
    for tier in TIERS:
        for run in runs(design, tier):
            if run["resume"]:
                continue  # an extension of a pilot run: same seeds by construction
            name = f"{tier}/{run['condition']}/r{run['replicate']}"
            for seed in (run["protocol"].seed, run["protocol"].seed + 1, run["protocol"].seed + 2):
                if seed in used:
                    problems.append(f"run seed {seed} of {name} collides with {used[seed]}")
                used[seed] = name
            p = run["preparation"].preparation_seed
            for seed in (p, p + 1):
                key = -seed - 1  # separate namespace from run seeds
                if key in used:
                    problems.append(f"preparation seed {seed} of {name} collides with {used[key]}")
                used[key] = name
    pilot, stronger = design["_protocols"]["pilot"], design["_protocols"]["stronger"]
    try:
        _check_resumable(pilot.to_dict(), stronger.to_dict())
    except Exception as exc:
        problems.append(f"stronger tier cannot resume the pilot runs: {exc}")
    for tier in ("pilot", "stronger"):
        if design["_protocols"][tier].report_interval_steps != pilot.report_interval_steps:
            problems.append(f"{tier} must save frames at the pilot interval to allow extension")
    for r in design["tiers"]["stronger"]["extends"]["replicates"]:
        for condition in design["conditions"]:
            if design["seeds"][f"pilot/{condition}/r{r}"] != design["seeds"][f"stronger/{condition}/r{r}"]:
                problems.append(f"stronger/{condition}/r{r} must reuse the pilot seeds it extends")
    plan = design["_analysis"]
    positions = {p["dna_position"] for p in plan["primary_pairs"]["pairs"]}
    for pair in plan["primary_pairs"]["pairs"]:
        if set(pair["labels"]) != set(design["conditions"]):
            problems.append(f"primary pair {pair['protein']}-{pair['dna_position']} lacks labels for every condition")
    for protein, position in plan["confirmatory_family"]["pairs"]:
        if position not in positions:
            problems.append(f"confirmatory pair {protein}-{position} is not a primary pair")
    return problems


def budget(design: dict[str, Any], tier: str, atoms: int) -> dict[str, float]:
    """Simulated ns, frames and storage (GB) for a tier, counting only new simulation."""
    c = design["measured_constants"]
    proto = design["_protocols"][tier]
    frames_per_run = proto.production_steps // proto.report_interval_steps
    new_ns = 0.0
    new_frames = 0
    for run in runs(design, tier):
        if run["resume"]:
            base = design["_protocols"][run["run_tier"]]
            new_ns += (proto.production_steps - base.production_steps) * proto.timestep_fs * 1e-6
            new_frames += (proto.production_steps - base.production_steps) // proto.report_interval_steps
        else:
            new_ns += (proto.production_steps + sum(s.steps for s in proto.equilibration)) * proto.timestep_fs * 1e-6
            new_frames += frames_per_run
    n_runs = len(runs(design, tier))
    per_frame = c["xtc_bytes_per_atom_per_frame"] * atoms
    overhead_per_run = (len(proto.equilibration) + 2) * (c["stage_state_xml_bytes"] + c["checkpoint_bytes"]) \
        + c["system_xml_bytes"] + 2 * c["prepared_pdb_bytes"]
    return {
        "runs": n_runs,
        "new_simulated_ns": round(new_ns, 2),
        "production_ns_per_replicate": proto.production_ps / 1000,
        "frames_per_replicate": frames_per_run,
        "trajectory_GB_per_replicate": round(frames_per_run * per_frame / 1e9, 2),
        "trajectory_GB_total": round(n_runs * frames_per_run * per_frame / 1e9, 2),
        "new_trajectory_GB": round(new_frames * per_frame / 1e9, 2),
        "other_files_GB_total": round(n_runs * overhead_per_run / 1e9, 2),
    }


def write_plan(design: dict[str, Any], tier: str, plan_dir: Path, experiment_dir: Path,
               input_pdb: Path, platform: str) -> Path:
    q = shlex.quote
    lines = ["#!/usr/bin/env bash", f"# {tier}: generated by plan.py; review before running.",
             "set -euo pipefail", ""]
    derived = experiment_dir / "inputs" / "3C2I_CpG_derived.pdb"
    lines += ["# inputs (idempotent)",
              f"neurodna-fetch 3C2I --cache-dir {q(str(input_pdb.parent.parent))} --sha256 {design['input']['sha256']}",
              f"mkdir -p {q(str(derived.parent))}",
              f"neurodna-md derive-unmethylated {q(str(input_pdb))} --residues "
              f"{' '.join(design['conditions']['CpG']['derivation']['residues'])} --output {q(str(derived))}", ""]
    for run in runs(design, tier):
        rel = Path(tier) / run["condition"] / f"r{run['replicate']}"
        cfg = plan_dir / rel
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "preparation.json").write_text(json.dumps(run["preparation"].to_dict(), indent=2) + "\n")
        (cfg / "protocol.json").write_text(json.dumps(run["protocol"].to_dict(), indent=2) + "\n")
        base = experiment_dir / run["run_tier"] / run["condition"] / f"r{run['replicate']}"
        source = input_pdb if run["condition"] == "mCpG" else derived
        lines.append(f"# {run['condition']} replicate {run['replicate']}"
                     + (" (extends the pilot run)" if run["resume"] else ""))
        if not run["resume"]:
            lines.append(f"neurodna-md prepare {q(str(source))} --config {q(str(cfg / 'preparation.json'))} "
                         f"--output-dir {q(str(base / 'prepared'))}")
        lines.append(f"neurodna-md run {q(str(base / 'prepared'))} --protocol {q(str(cfg / 'protocol.json'))} "
                     f"--output-dir {q(str(base / 'md'))} --platform {platform}"
                     + (" --resume" if run["resume"] else ""))
        lines.append("")
    script = plan_dir / f"commands_{tier}.sh"
    script.write_text("\n".join(lines))
    return script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="where prepared systems and trajectories would be written")
    parser.add_argument("--input-pdb", type=Path, default=Path("examples/mecp2_3c2i/cache/pdb/3C2I.pdb"))
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--atoms", type=int, default=49302,
                        help="atoms per system for storage estimates (measured prepared 3C2I system)")
    args = parser.parse_args(argv)
    design = load_design()
    problems = validate(design)
    if problems:
        print("design is inconsistent:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1
    print(f"{'tier':12s} {'runs':>4s} {'new ns':>9s} {'prod ns/rep':>11s} {'frames/rep':>10s} "
          f"{'GB/rep':>7s} {'GB traj':>8s} {'GB new':>7s} {'GB other':>8s}")
    for tier in TIERS:
        b = budget(design, tier, args.atoms)
        print(f"{tier:12s} {b['runs']:4d} {b['new_simulated_ns']:9.1f} {b['production_ns_per_replicate']:11.0f} "
              f"{b['frames_per_replicate']:10d} {b['trajectory_GB_per_replicate']:7.2f} "
              f"{b['trajectory_GB_total']:8.2f} {b['new_trajectory_GB']:7.2f} {b['other_files_GB_total']:8.2f}")
        script = write_plan(design, tier, args.plan_dir, args.experiment_dir, args.input_pdb, args.platform)
        print(f"{'':12s} wrote {script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
