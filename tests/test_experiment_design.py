"""The MeCP2-3C2I experiment design stays internally consistent (no simulation is run)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "examples" / "mecp2_3c2i" / "experiment"


@pytest.fixture(scope="module")
def plan():
    spec = importlib.util.spec_from_file_location("experiment_plan", EXPERIMENT / "plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_design_validates(plan):
    assert plan.validate(plan.load_design()) == []


def test_conditions_share_everything_but_the_input(plan):
    design = plan.load_design()
    runs = plan.runs(design, "pilot")
    by_cond = {c: [r for r in runs if r["condition"] == c] for c in design["conditions"]}
    for m, u in zip(by_cond["mCpG"], by_cond["CpG"]):
        a, b = m["preparation"].to_dict(), u["preparation"].to_dict()
        a.pop("preparation_seed"), b.pop("preparation_seed")
        assert a == b
        pa, pb = m["protocol"].to_dict(), u["protocol"].to_dict()
        assert pa.pop("seed") != pb.pop("seed") and pa == pb
    assert design["_prep"]["revert_to_alanine"] == ["A:140"] and design["_prep"]["crystal_waters"] == "keep"


def test_budget_and_commands(plan, tmp_path):
    design = plan.load_design()
    pilot = plan.budget(design, "pilot", 46348)
    assert pilot["runs"] == 6 and pilot["new_simulated_ns"] == pytest.approx(608.1)
    assert pilot["frames_per_replicate"] == 10000
    stronger = plan.budget(design, "stronger", 46348)
    assert stronger["new_simulated_ns"] == pytest.approx(3 * 2 * 400 + 2 * 2 * 501.35)
    script = plan.write_plan(design, "stronger", tmp_path / "plan", tmp_path / "runs",
                             Path("3C2I.pdb"), "CUDA")
    text = script.read_text()
    assert text.count("--resume") == 6 and text.count("neurodna-md prepare") == 4
    assert "neurodna-md derive-unmethylated 3C2I.pdb --residues B:8 C:33" in text
    proto = json.loads((tmp_path / "plan" / "stronger" / "CpG" / "r1" / "protocol.json").read_text())
    assert proto["seed"] == design["seeds"]["pilot/CpG/r1"]["run_seed"]


def test_confirmatory_pairs_are_not_methyl_mediated(plan):
    analysis = plan.load_design()["_analysis"]
    pairs = {(p["protein"], p["dna_position"]): p for p in analysis["primary_pairs"]["pairs"]}
    assert len(pairs) == 19
    for protein, position in analysis["confirmatory_family"]["pairs"]:
        assert not pairs[(protein, position)]["methyl_mediated_in_crystal"]
    assert {k for k, p in pairs.items() if p["methyl_mediated_in_crystal"]} == {
        ("A:ASP121", "B:8"), ("A:TYR123", "B:8")}
