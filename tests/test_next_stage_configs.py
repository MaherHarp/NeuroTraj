"""The committed next-stage configs (examples/mecp2_3c2i/md/next_stage) load and match the design."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from neurodna.md.config import load_preparation_config, load_protocol

ROOT = Path(__file__).resolve().parents[1]
NEXT = ROOT / "examples" / "mecp2_3c2i" / "md" / "next_stage"
EXPERIMENT = ROOT / "examples" / "mecp2_3c2i" / "experiment"


def test_tier_a_is_laptop_safe_and_uses_the_design_system():
    proto = load_protocol(NEXT / "tier_a_laptop" / "protocol.json")
    prep = load_preparation_config(NEXT / "tier_a_laptop" / "preparation.json")
    design_prep = load_preparation_config(EXPERIMENT / "preparation.json").to_dict()
    assert proto.platform == "CPU" and proto.cpu_threads is not None and proto.cpu_threads <= 6
    assert proto.production_ps + proto.equilibration_ps <= 100  # picosecond-scale validation only
    assert proto.production_steps % proto.checkpoint_interval_steps == 0
    assert "Not equilibrated" in proto.purpose
    assert {k: v for k, v in prep.to_dict().items() if k != "preparation_seed"} == \
        {k: v for k, v in design_prep.items() if k != "preparation_seed"}


def test_tier_b_replicate_1_runs_are_the_design_development_runs():
    design = json.loads((EXPERIMENT / "experiment.json").read_text())
    development = load_protocol(EXPERIMENT / "protocols" / "development.json")
    seeds = set()
    for d in sorted((NEXT / "tier_b_gpu").iterdir()):
        if not d.is_dir():
            continue
        condition, r = d.name.split("_r")
        proto = load_protocol(d / "protocol.json")
        prep = load_preparation_config(d / "preparation.json")
        assert proto == replace(development, seed=proto.seed)  # only the seed differs
        key = f"development/{condition}/r{r}"
        if key in design["seeds"]:
            assert (prep.preparation_seed, proto.seed) == (design["seeds"][key]["preparation_seed"],
                                                           design["seeds"][key]["run_seed"])
        seeds |= {prep.preparation_seed, proto.seed, proto.seed + 1, proto.seed + 2}
    assert len(seeds) == 4 * 4


def test_tiers_manifest_records_no_runs_and_no_convergence_claim():
    tiers = json.loads((NEXT / "tiers.json").read_text())
    assert tiers["status"].startswith("configs only") and tiers["no_duration_guarantees_convergence"]
    assert tiers["A_laptop_validation"]["recommended_next"] and not tiers["C_research_scale"]["run_now"]
    assert tiers["measured_constants"]["gpu_ns_per_day"] is None
