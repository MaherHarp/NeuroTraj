"""Write and validate the configs for the next 3C2I stages. Never starts a simulation.

    python examples/mecp2_3c2i/md/next_stage/make_configs.py

Writes, next to this script:

* ``tiers.json``: the three tiers (A laptop validation, B modest GPU pilot, C
  research scale) with durations, frames, storage and wall-clock estimates
  computed from measured constants, and what each tier can and cannot support.
* ``tier_a_laptop/``: ``preparation.json``, ``protocol.json``, ``commands.sh``.
* ``tier_b_gpu/<condition>_r<i>/``: ``preparation.json``, ``protocol.json``, and
  ``tier_b_gpu/commands.sh``.

Everything derives from the experiment design (``../../experiment``): the shared
preparation config, the development protocol and the seed scheme. Every config
is loaded with neurodna's own validators, and all seeds are checked for
uniqueness. The Tier B replicate-1 runs deliberately reuse the design's
development seeds, so they *are* the preregistered development runs.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from neurodna.md.config import PreparationConfig, Protocol, Stage

HERE = Path(__file__).resolve().parent
MD = HERE.parent
EXPERIMENT = MD.parent / "experiment"
ROOT = MD.parents[2]

# ---------------------------------------------------------------- measured constants
# All from the laptop micro-pilot / smoke runs on this machine (see pilot_analysis_report.md).
MEASURED = {
    "hardware": "Apple M3 Max (10 performance + 4 efficiency cores), OpenMM 8.6.1 CPU platform",
    "cpu_ns_per_day_6_threads": 0.899,       # micro-pilot, all 25 ps of dynamics, 49,302 atoms
    "cpu_ns_per_day_14_threads": [1.55, 1.72],  # smoke run, 49,302 atoms (not allowed for Tier A)
    "minimization_s_per_iteration_6_threads": 0.785,  # 100 iterations in 78.5 s
    "preparation_s": 60,                       # Met140 build (hydrogens on the Reference platform)
    "diagnostics_s_per_frame": 1.5,            # diagnose_run.py, 80 frames in ~121 s (bulk density dominates)
    "atoms_measured_system": 49302,
    "atoms_design_system": 46348,              # docs/experiment_design.md section 10 (Ala140 build)
    "xtc_bytes_per_atom_per_frame": 3.63928,
    "checkpoint_bytes": 2404353,
    "stage_state_xml_bytes_per_atom": 8194399 / 49302,
    "system_xml_bytes": 17094732,
    "pdb_bytes": 3995390,
    "gpu_ns_per_day": None,
}
ILLUSTRATIVE_GPU_NS_PER_DAY = (50, 100, 200, 400)  # NOT measurements; same grid as the design document


def load_design() -> dict[str, Any]:
    design = json.loads((EXPERIMENT / "experiment.json").read_text())
    design["_prep"] = json.loads((EXPERIMENT / design["shared_preparation_config"]).read_text())
    design["_development"] = json.loads((EXPERIMENT / design["tiers"]["development"]["protocol"]).read_text())
    return design


# ---------------------------------------------------------------------- tier A
TIER_A_PREP_SEED = 20261001
TIER_A_RUN_SEED = 20261011  # uses 20261011, +1, +2
TIER_A_PROTOCOL = Protocol(
    name="tier_a_laptop_validation",
    purpose=("TIER A LAPTOP VALIDATION: the design's mCpG system (Ala140, crystal waters, charged termini) "
             "through the design's staged restraint ladder, compressed to picoseconds (27.5 ps equilibration "
             "+ 50 ps production). Checks preparation, stage transitions, barostat behaviour under restraints, "
             "checkpoint/resume and QC code on the real design system. Not equilibrated, not converged; "
             "cannot establish MeCP2-DNA behaviour."),
    seed=TIER_A_RUN_SEED,
    temperature_K=300.0, pressure_bar=1.0, timestep_fs=2.0, friction_per_ps=1.0, barostat_interval_steps=25,
    minimization_max_iterations=1000, minimization_tolerance_kj_mol_nm=10.0,
    minimization_restraint_k_kj_mol_nm2=1000.0,
    equilibration=(
        Stage("nvt_restrained", "NVT", 1250, 1000.0),       # 2.5 ps   (design: 100 ps)
        Stage("npt_restrained_1000", "NPT", 2500, 1000.0),  # 5 ps     (design: 250 ps)
        Stage("npt_restrained_100", "NPT", 2500, 100.0),    # 5 ps     (design: 250 ps)
        Stage("npt_restrained_10", "NPT", 2500, 10.0),      # 5 ps     (design: 250 ps)
        Stage("npt_unrestrained", "NPT", 5000, 0.0),        # 10 ps    (design: 500 ps)
    ),
    production_steps=25000,          # 50 ps
    report_interval_steps=250,       # 0.5 ps -> 100 frames
    checkpoint_interval_steps=2500,  # 5 ps (about 8 min of wall time at 6 threads)
    platform="CPU", precision=None, cpu_threads=6,
)

# ---------------------------------------------------------------------- tier B
TIER_B_RUNS = (("mCpG", 1), ("mCpG", 2), ("mCpG", 3), ("CpG", 1))


def tier_b_seeds(condition: str, replicate: int) -> dict[str, int]:
    """The design's seed scheme (experiment.json) applied to the development tier."""
    base = 1_000_000 + {"mCpG": 100_000, "CpG": 200_000}[condition] + 1_000 + 10 * replicate
    return {"preparation_seed": base, "run_seed": base + 1_000_000}


def storage_gb(atoms: int, frames: int, n_stage_files: int) -> dict[str, float]:
    m = MEASURED
    xtc = frames * atoms * m["xtc_bytes_per_atom_per_frame"]
    other = (n_stage_files * (m["checkpoint_bytes"] + m["stage_state_xml_bytes_per_atom"] * atoms)
             + m["checkpoint_bytes"] + m["system_xml_bytes"] + 2 * m["pdb_bytes"])
    return {"trajectory_GB": round(xtc / 1e9, 3), "other_files_GB": round(other / 1e9, 3),
            "total_GB": round((xtc + other) / 1e9, 3)}


def hours(ns: float, ns_per_day: float) -> float:
    return round(ns / ns_per_day * 24, 2)


def tiers(design: dict[str, Any], dev: Protocol) -> dict[str, Any]:
    m = MEASURED
    atoms = m["atoms_design_system"]
    a = TIER_A_PROTOCOL
    a_eq_ps, a_prod_ps = a.equilibration_ps, a.production_ps
    a_frames = a.production_steps // a.report_interval_steps
    a_dyn_h = hours((a_eq_ps + a_prod_ps) / 1000, m["cpu_ns_per_day_6_threads"])
    a_min_h = round(a.minimization_max_iterations * m["minimization_s_per_iteration_6_threads"] / 3600, 2)
    b_eq_ns, b_prod_ns = dev.equilibration_ps / 1000, dev.production_ps / 1000
    b_frames = dev.production_steps // dev.report_interval_steps
    b_total_ns = len(TIER_B_RUNS) * (b_eq_ns + b_prod_ns)
    b_store = storage_gb(atoms, b_frames, len(dev.equilibration) + 2)
    return {
        "status": "configs only; nothing here has been run",
        "measured_constants": m,
        "illustrative_gpu_ns_per_day": list(ILLUSTRATIVE_GPU_NS_PER_DAY),
        "no_duration_guarantees_convergence": True,
        "A_laptop_validation": {
            "recommended_next": True,
            "system": "design mCpG condition: deposited 3C2I, Ala140, crystal waters kept, charged termini "
                      "(experiment/preparation.json), preparation_seed 20261001",
            "replicates": 1, "parallel": False, "cpu_threads": a.cpu_threads,
            "minimization_max_iterations": a.minimization_max_iterations,
            "equilibration_ps": a_eq_ps, "equilibration_stages": a.to_dict()["equilibration"],
            "production_ps": a_prod_ps, "output_interval_ps": a.report_interval_steps * a.timestep_fs / 1000,
            "frames": a_frames,
            "checkpoint_interval_ps": a.checkpoint_interval_steps * a.timestep_fs / 1000,
            "storage": storage_gb(atoms, a_frames, len(a.equilibration) + 2),
            "wall_clock_h": {"preparation": round(m["preparation_s"] / 3600, 2),
                             "minimization_worst_case": a_min_h, "dynamics": a_dyn_h,
                             "diagnostics": round(a_frames * m["diagnostics_s_per_frame"] / 3600, 2),
                             "total": round(m["preparation_s"] / 3600 + a_min_h + a_dyn_h
                                            + a_frames * m["diagnostics_s_per_frame"] / 3600, 1),
                             "basis": "measured 0.899 ns/day and 0.785 s/iteration at 6 threads on the 49,302-atom "
                                      "system; the 46,348-atom design system should be slightly faster"},
            "purpose": "run the design's own system and staged protocol end to end before any GPU time is "
                       "spent: Ala140 preparation, five-stage restraint ladder with its restraint releases, "
                       "barostat under k = 1000 restraints, an unrestrained NPT stage, QC code on the real "
                       "design system, and stage-by-stage 6-thread timings",
            "can_support": ["the design system prepares, minimises and integrates stably through every "
                            "restraint transition", "barostat and volume behaviour through the ladder "
                            "(qualitative)", "checkpoint/resume and QC code on the design system",
                            "refined per-stage laptop timings and file sizes"],
            "cannot_support": ["equilibration or convergence", "contact persistence or occupancy estimates",
                               "replicate variability", "any mCpG-CpG comparison", "GPU throughput"],
            "optional_A2_CpG_build_check": "derive the CpG model, prepare it and run only minimisation "
                                           "(--stop-after-stage minimization): about 15 min",
        },
        "B_gpu_pilot": {
            "recommended_next": False,
            "runs": [f"{c} r{r}" for c, r in TIER_B_RUNS],
            "design_relation": "the design's development tier (mCpG r1 + CpG r1, same protocol and seeds) "
                               "plus two extra mCpG replicates for consistency information",
            "minimization_max_iterations": dev.minimization_max_iterations,
            "equilibration_ns_per_run": b_eq_ns, "production_ns_per_run": b_prod_ns,
            "output_interval_ps": dev.report_interval_steps * dev.timestep_fs / 1000,
            "frames_per_run": b_frames,
            "checkpoint_interval_ns": dev.checkpoint_interval_steps * dev.timestep_fs / 1e6,
            "total_simulated_ns": round(b_total_ns, 2),
            "storage_per_run": b_store,
            "storage_total_GB": round(len(TIER_B_RUNS) * b_store["total_GB"], 2),
            "gpu_hours_total_at_illustrative_throughput": {str(r): hours(b_total_ns, r)
                                                           for r in ILLUSTRATIVE_GPU_NS_PER_DAY},
            "gpu_hours_per_run_formula": f"{b_eq_ns + b_prod_ns:g} ns / (measured ns/day) * 24",
            "laptop_cpu_days_for_comparison": round(b_total_ns / m["cpu_ns_per_day_6_threads"], 1),
            "why_5_ns": "equals the preregistered development tier, so replicate-1 runs satisfy its decision "
                        "rule; gives the >= 4 blocks of >= 1 ns that the design's thermodynamic flag needs "
                        "(after a 1 ns sensitivity burn-in); shorter runs could not evaluate that flag",
            "why_replicates": "three independent mCpG preparations (solvent/ion placement, velocities) give a "
                              "first look at between-replicate spread of QC quantities and primary-pair "
                              "occupancy; one long run cannot",
            "prerequisites": ["Tier A completed without technical failure",
                              "GPU throughput measured with the step-0 benchmark",
                              "construct decision recorded: charged termini as designed, or neutral caps "
                              "adopted as a documented design amendment (requires implemented and validated "
                              "capping or an external preparation) - decided on modelling grounds, not on "
                              "simulation outcomes"],
            "can_support": ["measured GPU throughput and file sizes", "the full 10,000-iteration minimisation "
                            "and 1.35 ns ladder on GPU", "whether the design's thermodynamic flags clear after "
                            "1.35 ns in three independent replicates", "whether DNA core pairs stay intact "
                            "(>= 0.8) over 5 ns", "crude between-replicate spread of primary-pair occupancy "
                            "over 5 ns, to judge the pilot tier", "the CpG pipeline end to end",
                            "the design's development -> pilot decision rule"],
            "cannot_support": ["mCpG-CpG differences (one CpG run; interpretation 'none' as preregistered)",
                               "contact persistence beyond ~5 ns or residence times", "convergence",
                               "the size of the charged-termini effect"],
        },
        "C_research_scale": {
            "run_now": False,
            "tiers": {"pilot": {"runs": "3 replicates x 2 conditions", "production_ns_per_run": 100,
                                "new_simulated_ns": 608.1, "trajectory_GB": 10.1, "other_GB": 0.6,
                                "gpu_days_total_at_100_ns_per_day": 6.1},
                      "stronger": {"runs": "5 replicates x 2 conditions (pilot runs extended to 500 ns + 4 new)",
                                   "production_ns_per_run": 500, "new_simulated_ns": 4405.4,
                                   "trajectory_GB_total": 84.3, "other_GB": 1.0,
                                   "gpu_days_new_at_100_ns_per_day": 44.1}},
            "source": "docs/experiment_design.md sections 6 and 10 (unchanged)",
            "prerequisites": ["Tier B passes the development -> pilot rule", "construct decision implemented "
                              "and validated", "measured GPU throughput", "QC and aggregation code validated "
                              "on Tier B output"],
            "can_support": ["pilot: descriptive replicate variability and effect sizes",
                            "stronger: the preregistered confirmatory family (R111-B9, R133-C34)"],
            "cannot_support": ["binding affinity or specificity", "residence times", "full-length MeCP2 or "
                               "chromatin behaviour", "disease or pathogenicity claims"],
        },
    }


def main() -> int:
    q = shlex.quote
    design = load_design()
    dev = Protocol.from_dict(design["_development"])
    design_seeds = {v for s in design["seeds"].values() for k, base in s.items() for v in
                    ((base,) if k == "preparation_seed" else (base, base + 1, base + 2))}
    problems: list[str] = []

    # tier A
    a_dir = HERE / "tier_a_laptop"
    a_dir.mkdir(exist_ok=True)
    a_prep = PreparationConfig.from_dict({**design["_prep"], "preparation_seed": TIER_A_PREP_SEED})
    Protocol.from_dict(TIER_A_PROTOCOL.to_dict())  # round trip through the validator
    a_seeds = {TIER_A_PREP_SEED, TIER_A_RUN_SEED, TIER_A_RUN_SEED + 1, TIER_A_RUN_SEED + 2}
    if a_seeds & design_seeds:
        problems.append("Tier A seeds collide with design seeds")
    (a_dir / "preparation.json").write_text(json.dumps(a_prep.to_dict(), indent=2) + "\n")
    (a_dir / "protocol.json").write_text(json.dumps(TIER_A_PROTOCOL.to_dict(), indent=2) + "\n")
    rel = a_dir.relative_to(ROOT)
    out = "examples/mecp2_3c2i/md/output/tier_a"
    a_cmds = f"""#!/usr/bin/env bash
# TIER A - laptop validation of the design's mCpG system. Generated by make_configs.py.
# Review before running. Foreground only; 6 threads; one run at a time. Estimated ~2.4 h in total.
#
# Stop safely at any time with Ctrl-C. Resume by re-running step 3 with --resume.
#   Lost work on Ctrl-C: minimisation (<= ~13 min) restarts; an equilibration stage restarts from the
#   previous stage's end (<= ~16 min); production restarts from the last 5 ps checkpoint (<= ~8 min).
#   Frames and log rows written after that checkpoint are discarded on resume.
# Planned stops: add --stop-after-stage <stage> or --stop-at-production-step <multiple of 2500>.
# Each resume restarts the barostat's step-size adaptation, so resume as rarely as practical.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."  # project root
export OPENMM_CPU_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
O={out}

# 1. input (cached, checksum-verified)
.venv/bin/neurodna-fetch 3C2I --cache-dir examples/mecp2_3c2i/cache --sha256 {design['input']['sha256']}

# 2. prepare the design mCpG system (~1-2 min)
.venv/bin/neurodna-md prepare examples/mecp2_3c2i/cache/pdb/3C2I.pdb --config {rel}/preparation.json \\
    --output-dir $O/prepared

# 3. minimise + 27.5 ps staged equilibration + 50 ps production (~2.2 h at 6 threads)
.venv/bin/neurodna-md run $O/prepared --protocol {rel}/protocol.json --output-dir $O/run
#    after an interruption:
#    .venv/bin/neurodna-md run $O/prepared --protocol {rel}/protocol.json --output-dir $O/run --resume

# 4. diagnostics + independent neurodna validation (~3 min)
.venv/bin/python examples/mecp2_3c2i/md/diagnose_run.py --run-dir $O/run --out $O/diagnostics

# Optional A2 - CpG build check (~15 min): derived model, preparation, minimisation only.
# .venv/bin/neurodna-md derive-unmethylated examples/mecp2_3c2i/cache/pdb/3C2I.pdb --residues B:8 C:33 \\
#     --output $O/3C2I_CpG_derived.pdb
# .venv/bin/neurodna-md prepare $O/3C2I_CpG_derived.pdb --config {rel}/preparation.json --output-dir $O/prepared_cpg
# .venv/bin/neurodna-md run $O/prepared_cpg --protocol {rel}/protocol.json --output-dir $O/run_cpg \\
#     --stop-after-stage minimization
"""
    (a_dir / "commands.sh").write_text(a_cmds)

    # tier B
    b_dir = HERE / "tier_b_gpu"
    b_dir.mkdir(exist_ok=True)
    runs_root = "examples/mecp2_3c2i/experiment/runs/tier_b"
    derived = f"{runs_root}/inputs/3C2I_CpG_derived.pdb"
    lines = ["#!/usr/bin/env bash",
             "# TIER B - modest GPU pilot. Generated by make_configs.py. NOT RUN. Review before running.",
             "# mCpG r1 and CpG r1 are the design's development runs (same protocol and seeds);",
             "# mCpG r2 and r3 add consistency information. Runs are independent: one per GPU in parallel is fine.",
             "# PLATFORM=CUDA (NVIDIA) or OpenCL (e.g. the Apple M3 Max GPU). Needs: pip install -e '.[md,plot]'",
             "set -euo pipefail", 'cd "$(dirname "$0")/../../../../.."  # project root', 'PLATFORM="${PLATFORM:-CUDA}"',
             f"R={runs_root}", "",
             "# inputs",
             f"neurodna-fetch 3C2I --cache-dir examples/mecp2_3c2i/cache --sha256 {design['input']['sha256']}",
             "mkdir -p $R/inputs",
             f"neurodna-md derive-unmethylated examples/mecp2_3c2i/cache/pdb/3C2I.pdb --residues B:8 C:33 "
             f"--output {derived}", ""]
    seen = set(a_seeds)
    for condition, r in TIER_B_RUNS:
        seeds = tier_b_seeds(condition, r)
        key = f"development/{condition}/r{r}"
        if key in design["seeds"] and design["seeds"][key] != seeds:
            problems.append(f"{key}: seed scheme disagrees with experiment.json")
        run_seeds = {seeds["preparation_seed"], seeds["run_seed"], seeds["run_seed"] + 1, seeds["run_seed"] + 2}
        if run_seeds & seen:
            problems.append(f"{condition} r{r}: seed collision")
        if key not in design["seeds"] and run_seeds & design_seeds:
            problems.append(f"{condition} r{r}: collides with a design seed")
        seen |= run_seeds
        prep = PreparationConfig.from_dict({**design["_prep"], "preparation_seed": seeds["preparation_seed"]})
        proto = replace(dev, seed=seeds["run_seed"])
        d = b_dir / f"{condition}_r{r}"
        d.mkdir(exist_ok=True)
        (d / "preparation.json").write_text(json.dumps(prep.to_dict(), indent=2) + "\n")
        (d / "protocol.json").write_text(json.dumps(proto.to_dict(), indent=2) + "\n")
        cfg = d.relative_to(ROOT)
        base = f"$R/{condition}/r{r}"
        source = "examples/mecp2_3c2i/cache/pdb/3C2I.pdb" if condition == "mCpG" else derived
        lines += [f"# {condition} replicate {r}" + (" (= design development run)" if key in design["seeds"] else ""),
                  f"neurodna-md prepare {q(source)} --config {cfg}/preparation.json --output-dir {base}/prepared"]
        if (condition, r) == TIER_B_RUNS[0]:
            lines += ["# step 0: GPU benchmark (0.8 ps smoke protocol; "
                      "read ns_per_day in simulation.json, then decide)",
                      f"neurodna-md run {base}/prepared --protocol smoke --output-dir {base}/benchmark "
                      "--platform $PLATFORM"]
        lines += [f"neurodna-md run {base}/prepared --protocol {cfg}/protocol.json --output-dir {base}/md "
                  "--platform $PLATFORM",
                  f"python examples/mecp2_3c2i/md/diagnose_run.py --run-dir {base}/md --out {base}/diagnostics "
                  f"--condition {condition}",
                  f"python examples/mecp2_3c2i/md/diagnose_run.py --run-dir {base}/md "
                  f"--out {base}/diagnostics_burn1ns --tables {base}/md_analysis_burn1ns --condition {condition} "
                  "--burn-in-ps 1000", ""]
    (b_dir / "commands.sh").write_text("\n".join(lines))

    manifest = tiers(design, dev)
    (HERE / "tiers.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    a = manifest["A_laptop_validation"]
    b = manifest["B_gpu_pilot"]
    print(f"Tier A: {a['equilibration_ps']:g} ps eq + {a['production_ps']:g} ps production, {a['frames']} frames, "
          f"~{a['wall_clock_h']['total']} h at 6 threads, {a['storage']['total_GB']} GB")
    print(f"Tier B: {len(TIER_B_RUNS)} runs x ({b['equilibration_ns_per_run']:g} + {b['production_ns_per_run']:g}) ns "
          f"= {b['total_simulated_ns']} ns, {b['frames_per_run']} frames/run, {b['storage_total_GB']} GB; GPU hours "
          f"{b['gpu_hours_total_at_illustrative_throughput']} (illustrative ns/day)")
    print("configs validated; nothing was run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
