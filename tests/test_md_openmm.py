"""OpenMM integration SMOKE TEST on the real 3C2I structure (opt-in: ``pytest -m openmm``).

Checks that assessment, preparation, a few hundred MD steps, resume and neurodna
analysis work together. It is not a scientific simulation: the run is far too
short to be equilibrated, let alone converged.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("openmm")

from neurodna import FrameKind, TimeSource  # noqa: E402
from neurodna.md.analyze import analyze_run, load_run  # noqa: E402
from neurodna.md.config import SMOKE, PreparationConfig, load_preparation_config  # noqa: E402
from neurodna.md.prepare import PreparationBlockedError, prepare  # noqa: E402
from neurodna.md.run import run  # noqa: E402

pytestmark = pytest.mark.openmm

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "data" / "3C2I.pdb"
TINY = replace(SMOKE, name="pytest-smoke", minimization_max_iterations=20,
               equilibration=tuple(replace(s, steps=20) for s in SMOKE.equilibration),
               production_steps=40, report_interval_steps=20, checkpoint_interval_steps=20,
               platform="CPU")


@pytest.fixture(scope="module")
def config() -> PreparationConfig:
    base = load_preparation_config(ROOT / "examples" / "mecp2_3c2i" / "md" / "preparation.json")
    return replace(base, padding_nm=1.0)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, config) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare(REAL, config, tmp_path_factory.mktemp("prepared"))


@pytest.fixture(scope="module")
def smoke_run(prepared, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("run")
    run(prepared, TINY, out)
    return out


def test_preparation_keeps_5mc_with_documented_patch(prepared):
    p = json.loads((prepared / "preparation.json").read_text())
    assert p["assessment"]["supported"] and p["assessment"]["blockers"] == []
    templates = p["nucleotide_templates"]
    assert templates["B:5CM8"] == templates["C:5CM33"] == "CYT-DEOX_0-5MC2_4"
    assert templates["B:DT1"] == "THY-DEO5TER"  # not URA+5MC2 (graph-identical, different parameters)
    assert all(a["complete"] for a in p["modified_residue_audit"])
    assert p["composition"]["system_net_charge_e"] == 0.0
    assert p["conversions"] == ["A:MSE94 -> MET (SE renamed SD, element S)",
                                "A:MSE140 -> MET (SE renamed SD, element S)"]
    text = (prepared / "prepared.pdb").read_text()
    methyl = [l for l in text.splitlines() if l[17:20] == "5CM" and l[12:16].strip() in
              ("C5A", "H5A1", "H5A2", "H5A3")]
    assert len(methyl) == 8  # 2 residues x (C5A + 3 H)


def test_preparation_is_reproducible(prepared, config, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = prepare(REAL, config, tmp_path / "again")
    a = json.loads((prepared / "preparation.json").read_text())
    b = json.loads((again / "preparation.json").read_text())
    assert a["files"]["prepared_pdb"]["sha256"] == b["files"]["prepared_pdb"]["sha256"]
    fa, fb = (x["reproducibility"]["energy_fingerprint_kj_mol"] for x in (a, b))
    assert fa.keys() == fb.keys()
    for key in fa:  # CPU-platform PME summation order is not bitwise deterministic
        assert fb[key] == pytest.approx(fa[key], rel=1e-6, abs=1e-3)


def test_blocked_preparation(tmp_path, config):
    bad = tmp_path / "renamed.pdb"
    bad.write_text(REAL.read_text().replace("5CM", "5HC"))
    with pytest.raises(PreparationBlockedError, match="5HC"):
        prepare(bad, config, tmp_path / "out")


def test_run_metadata_seeds_checkpoints(smoke_run):
    m = json.loads((smoke_run / "simulation.json").read_text())
    assert m["seeds"] == {"velocities": TINY.seed, "integrator": TINY.seed + 1, "barostat": TINY.seed + 2}
    assert m["convergence_assessed"] is False and "SMOKE TEST ONLY" in m["statement"]
    assert m["simulated_ps"] == {"equilibration": 0.08, "production": 0.08}
    assert m["production"]["n_frames"] == 2 and m["production"]["frame_interval_ps"] == 0.04
    for name in ("minimization", "nvt_restrained", "npt_restrained", "production"):
        assert (smoke_run / "stages" / f"{name}.chk").exists()
        assert (smoke_run / "stages" / f"{name}.xml").exists()
    assert all(s["wall_time_s"] > 0 for s in m["stages"])


def test_resume_extends_production(smoke_run, prepared):
    # emulate an interruption after the last checkpoint (step 40): a log row that resume must drop
    with open(smoke_run / "production_log.csv", "a") as fh:
        fh.write("60,0.12,-1.0,300.0,400.0,1.0,0\n")
    run(prepared, replace(TINY, production_steps=80), smoke_run, resume=True)
    m = json.loads((smoke_run / "simulation.json").read_text())
    assert m["production"]["n_frames"] == 4 and m["simulated_ps"]["production"] == pytest.approx(0.16)
    cx = load_run(smoke_run)
    assert list(cx.frame_times()) == pytest.approx([0.04, 0.08, 0.12, 0.16], abs=1e-4)
    steps = [int(line.split(",")[0]) for line in (smoke_run / "production_log.csv").read_text().splitlines()
             if not line.startswith("#")]
    assert steps == [20, 40, 60, 80]
    assert not list(smoke_run.glob("**/.*.tmp"))


def test_neurodna_reads_the_run(smoke_run, tmp_path):
    cx = load_run(smoke_run)
    assert cx.kind is FrameKind.TRAJECTORY and cx.frames.time_source is TimeSource.FILE
    assert set(cx.dna_residues().loc[lambda d: d.modification == "5mC", "label"]) == {"B:5CM8", "C:5CM33"}
    paths = analyze_run(smoke_run, tmp_path / "analysis")
    summary = json.loads(paths["summary"].read_text())
    assert summary["convergence_assessed"] is False and summary["minimum_image"] is True
    import pandas as pd

    occ = pd.read_csv(paths["contact_occupancy"])
    assert ("A:ARG133", "C:5CM33") in set(zip(occ.protein_label, occ.dna_label))


def test_automatic_platform_is_the_fastest_usable_one():
    import openmm as mm

    from neurodna.md.run import _platform

    probe = mm.System()
    probe.addParticle(1.0)
    usable = []
    for i in range(mm.Platform.getNumPlatforms()):
        platform = mm.Platform.getPlatform(i)
        try:
            context = mm.Context(probe, mm.VerletIntegrator(0.001), platform)
            del context
            usable.append(platform)
        except Exception:
            continue
    fastest = max(usable, key=lambda p: p.getSpeed()).getName()
    chosen, _ = _platform(mm, replace(TINY, platform=None))
    assert chosen.getName() == fastest  # an empty probe System used to make every platform fail -> always CPU
