"""MD workflow pieces that need no OpenMM: configs, protocols, structural assessment."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from neurodna.md.assess import structural_findings
from neurodna.md.config import (
    PRODUCTION,
    SMOKE,
    ConfigError,
    PreparationConfig,
    Protocol,
    load_preparation_config,
    load_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "data" / "3C2I.pdb"
EXAMPLE_CONFIG = ROOT / "examples" / "mecp2_3c2i" / "md" / "preparation.json"

BASE = {"protein_chains": ["A"], "dna_chains": ["B", "C"], "selenomethionine": "convert_to_methionine",
        "revert_to_alanine": [], "crystal_waters": "keep", "protein_termini": "charged"}


def test_example_config_loads():
    config = load_preparation_config(EXAMPLE_CONFIG)
    assert config.protein_chains == ("A",) and config.dna_chains == ("B", "C")
    assert config.force_field == ("charmm36_2024.xml", "charmm36_2024/water.xml")


@pytest.mark.parametrize("key", PreparationConfig.REQUIRED)
def test_choices_that_change_the_model_must_be_explicit(key):
    data = {k: v for k, v in BASE.items() if k != key}
    with pytest.raises(ConfigError, match=f"must state \\['{key}'\\] explicitly"):
        PreparationConfig.from_dict(data)


@pytest.mark.parametrize(("key", "value", "match"), [
    ("selenomethionine", "keep", "no selenomethionine parameters"),
    ("protein_termini", "neutral_caps", "not implemented"),
    ("crystal_waters", "some", "not supported"),
    ("revert_to_alanine", ["140"], "look like 'A:140'"),
])
def test_unsupported_choices_are_rejected(key, value, match):
    with pytest.raises(ConfigError, match=match):
        PreparationConfig.from_dict({**BASE, key: value})


def test_other_force_fields_and_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="only the force field"):
        PreparationConfig.from_dict({**BASE, "force_field": ["amber14-all.xml", "amber14/tip3p.xml"]})
    with pytest.raises(ConfigError, match="unknown preparation settings"):
        PreparationConfig.from_dict({**BASE, "replace_5mc_with_c": True})


def test_presets_separate_smoke_from_production():
    assert "SMOKE TEST ONLY" in SMOKE.purpose and SMOKE.production_ps == pytest.approx(0.4)
    assert PRODUCTION.production_ps == pytest.approx(100_000.0)  # 100 ns template
    assert "Not run in this repository" in PRODUCTION.purpose
    assert load_protocol("smoke") is SMOKE
    assert Protocol.from_dict(json.loads(json.dumps(SMOKE.to_dict()))) == SMOKE


def test_protocol_validation():
    with pytest.raises(ConfigError, match="multiple of report_interval_steps"):
        replace(SMOKE, production_steps=210)
    with pytest.raises(ConfigError, match="must define 'seed'"):
        Protocol.from_dict({"name": "x", "purpose": "y"})


def test_resume_only_extends_production():
    from neurodna.md.run import SimulationError, _check_resumable

    old = SMOKE.to_dict()
    _check_resumable(old, replace(SMOKE, production_steps=400).to_dict())
    with pytest.raises(SimulationError, match="temperature_K"):
        _check_resumable(old, replace(SMOKE, production_steps=400, temperature_K=310.0).to_dict())
    with pytest.raises(SimulationError, match="fewer production steps"):
        _check_resumable(old, replace(SMOKE, production_steps=100).to_dict())


def test_importing_the_md_package_does_not_import_openmm():
    code = "import neurodna, neurodna.md, neurodna.md.cli, sys; print('openmm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_structural_assessment_of_3c2i():
    r = structural_findings(REAL, load_preparation_config(EXAMPLE_CONFIG))
    assert r["blockers"] == []
    assert r["chains"]["A"]["missing_residues"] == ["77-90", "163-173"]
    assert r["chains"]["A"]["termini"]["n_truncated"] and not r["chains"]["A"]["termini"]["c_terminal_has_OXT"]
    assert r["chains"]["B"]["termini"]["5prime_phosphate"] is False
    assert r["nonstandard_residues"]["5CM"].startswith("supported: CYT + DEOX + 5MC2")
    assert r["nonstandard_residues"]["MSE"].startswith("requires conversion")
    assert r["engineered_residues"] == ["A:MSE140: ENGINEERED MUTATION (UNP P51608 ALA140)"]
    assert r["crystallographic_waters_by_chain"] == {"A": 17, "B": 14, "C": 16}
    assert any("NH3+/COO-" in n for n in r["notes"])


def test_unparameterised_modified_nucleotide_is_a_blocker(tmp_path):
    # Rename 5CM to a (hypothetical here) code with no validated parameters: the
    # assessment must block, not silently map it to cytosine.
    text = REAL.read_text().replace("5CM", "5HC")
    path = tmp_path / "renamed.pdb"
    path.write_text(text)
    r = structural_findings(path, load_preparation_config(EXAMPLE_CONFIG))
    assert r["nonstandard_residues"]["5HC"].startswith("BLOCKED")
    assert any("5HC: no validated CHARMM36 parameters" in b for b in r["blockers"])


def test_cli_reports_config_errors(tmp_path, capsys):
    from neurodna.md.cli import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({k: v for k, v in BASE.items() if k != "crystal_waters"}))
    assert main(["assess", str(REAL), "--config", str(bad)]) == 1
    assert "crystal_waters" in capsys.readouterr().err


def test_import_external_records_unchecked_provenance(tmp_path):
    from neurodna.md.cli import import_external
    from neurodna.md.prepare import sha256

    topology, system = tmp_path / "sys.pdb", tmp_path / "system.xml"
    topology.write_text(REAL.read_text())
    system.write_text("<System/>\n")
    out = import_external(topology, system, "chainID A and not resname HOH",
                          "chainID B C and not resname HOH", tmp_path / "ext", notes="CHARMM-GUI")
    p = json.loads((out / "preparation.json").read_text())
    assert p["kind"] == "external" and "not checked by neurodna" in p["statement"]
    assert p["files"]["system_xml"]["sha256"] == sha256(system)
    assert p["selections"]["dna"] == "chainID B C and not resname HOH"


def test_demethylation_changes_only_the_methyl_groups(tmp_path):
    import warnings

    from neurodna import Complex
    from neurodna.errors import NeuroDNAError
    from neurodna.md.variants import demethylate_cytosines

    out = tmp_path / "3C2I_CpG.pdb"
    record = demethylate_cytosines(REAL, out, ["B:8", "C:33"])
    assert [(a["residue"], a["atom"]) for a in record["removed_atoms"]] == [("B:5CM8", "C5A"), ("C:5CM33", "C5A")]
    src = [l for l in REAL.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    new = [l for l in out.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert len(new) == len(src) - 2
    untouched = [l for l in src if l[17:20] != "5CM"]
    converted = {("B", "8"), ("C", "33")}
    assert untouched == [l for l in new if l[17:20] != " DC" or (l[21], l[22:26].strip()) not in converted]
    renamed = [l for l in new if (l[21], l[22:26].strip()) in {("B", "8"), ("C", "33")}]
    assert all(l.startswith("ATOM  ") and l[17:20] == " DC" for l in renamed) and len(renamed) == 38
    text = out.read_text()
    assert "NEURODNA DERIVED MODEL" in text
    assert not [l for l in text.splitlines() if "5CM" in l and not l.startswith("REMARK 999")]
    assert all(len(l) <= 80 for l in text.splitlines() if l.startswith("REMARK 999"))
    assert json.loads((tmp_path / "3C2I_CpG.pdb.json").read_text())["output"]["sha256"] == record["output"]["sha256"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = Complex.load(out, protein="chainID A and not resname HOH", dna="chainID B C and not resname HOH")
    assert cx.dna_residues().modification.isna().all()
    strands = {s.residues[0].chain: s.annotated_sequence for s in cx.dna_strands()}
    assert strands == {"B": "TCTGGAACGGAATTCTTCTA", "C": "ATAGAAGAATTCCGTTCCAG"}
    with pytest.raises(NeuroDNAError, match="unlisted 5CM"):
        demethylate_cytosines(REAL, tmp_path / "half.pdb", ["B:8"])
    with pytest.raises(NeuroDNAError, match="is not 5CM"):
        demethylate_cytosines(REAL, tmp_path / "bad.pdb", ["B:8", "C:33", "B:9"])


def test_cpu_threads_and_clean_stop_points():
    from neurodna.md.run import SimulationError, check_stop_options

    with pytest.raises(ConfigError, match="cpu_threads"):
        replace(SMOKE, cpu_threads=0)
    _check_resumable_ok = __import__("neurodna.md.run", fromlist=["_check_resumable"])._check_resumable
    _check_resumable_ok(SMOKE.to_dict(), replace(SMOKE, cpu_threads=6).to_dict())  # threads may change on resume
    old = {k: v for k, v in SMOKE.to_dict().items() if k != "cpu_threads"}  # runs made before the field existed
    _check_resumable_ok(old, replace(SMOKE, cpu_threads=6).to_dict())
    check_stop_options(SMOKE, "nvt_restrained", None)
    check_stop_options(SMOKE, None, SMOKE.checkpoint_interval_steps)
    with pytest.raises(SimulationError, match="must be one of"):
        check_stop_options(SMOKE, "heating", None)
    with pytest.raises(SimulationError, match="multiple of checkpoint_interval_steps"):
        check_stop_options(SMOKE, None, SMOKE.checkpoint_interval_steps // 2)
    with pytest.raises(SimulationError, match="either"):
        check_stop_options(SMOKE, "minimization", 100)


def test_resume_drops_log_rows_after_the_checkpoint(tmp_path):
    """SYNTHETIC log: rows after an interruption must not survive a resume (no duplicate steps)."""
    from neurodna.md.run import _truncate_log

    header = '#"Step","Time (ps)","Potential Energy (kJ/mole)"\n'
    log = tmp_path / "production_log.csv"
    log.write_text(header + "".join(f"{s},{s * 0.002},-1.0\n" for s in (125, 250, 375, 500, 625))
                   + header + "750,1.5,-1.0\n")
    _truncate_log(log, 500)
    assert log.read_text() == header + "".join(f"{s},{s * 0.002},-1.0\n" for s in (125, 250, 375, 500)) + header
    _truncate_log(tmp_path / "missing.csv", 0)  # nothing to do
    assert not (tmp_path / "missing.csv").exists()
