"""Offline tests for neurodna.md.qc and the run loader/analysis provenance.

All inputs are SYNTHETIC: hand-built logs, OpenMM-style XML with hand-computable
kinetic energies, lattice "water" boxes of known density, and a short
trajectory made by translating the 3C2I test structure. None is a simulation.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

from neurodna.errors import TrajectoryError
from neurodna.fetch import sha256_file
from neurodna.md import qc
from neurodna.md.analyze import analyze_run, load_run, prepared_dir_of

DATA = Path(__file__).parent / "data"
PDB = DATA / "3C2I.pdb"
SELECTIONS = {"protein": "chainID A and not resname HOH", "dna": "chainID B C and not resname HOH"}


# ------------------------------------------------------------------- qc: logs

def test_drift_summary_of_a_linear_series():
    t = np.arange(8, dtype=float)
    s = qc.drift_summary(2.0 * t + 1.0, t, n_blocks=4)
    assert s["slope_per_ps"] == pytest.approx(2.0)
    assert s["first_half_mean"] == pytest.approx(4.0) and s["second_half_mean"] == pytest.approx(12.0)
    assert s["block_means"] == pytest.approx([2.0, 6.0, 10.0, 14.0])
    assert s["block_se"] == pytest.approx(np.std([2, 6, 10, 14], ddof=1) / 2)
    assert s["half_difference_over_block_se"] == pytest.approx(8.0 / s["block_se"])


def _write_log(path: Path, rows: list[tuple[float, ...]], repeat_header_after: int | None = None) -> None:
    header = '#"Step","Time (ps)","Potential Energy (kJ/mole)","Temperature (K)","Box Volume (nm^3)",' \
             '"Density (g/mL)","Speed (ns/day)"'
    lines = [header]
    for i, r in enumerate(rows):
        lines.append(",".join(str(x) for x in r))
        if repeat_header_after is not None and i == repeat_header_after:
            lines.append(header)  # a resumed session appends a new header
    path.write_text("\n".join(lines) + "\n")


def test_thermo_table_labels_stages_and_puts_production_after_equilibration(tmp_path):
    meta = {"stages": [{"name": "minimization"},
                       {"name": "nvt", "ensemble": "NVT", "steps": 2, "simulated_ps": 0.004},
                       {"name": "npt", "ensemble": "NPT", "steps": 2, "simulated_ps": 0.004},
                       {"name": "production", "ensemble": "NPT", "steps": 2, "simulated_ps": 0.004}]}
    (tmp_path / "simulation.json").write_text(json.dumps(meta))
    _write_log(tmp_path / "equilibration_log.csv",
               [(s, s * 0.002, -1.0, 300.0, 10.0, 1.0, 0) for s in (1, 2, 3, 4)], repeat_header_after=1)
    _write_log(tmp_path / "production_log.csv", [(s, s * 0.002, -2.0, 301.0, 9.9, 1.01, 0) for s in (1, 2)])
    df = qc.thermo_table(tmp_path)
    assert list(df.stage) == ["nvt", "nvt", "npt", "npt", "production", "production"]
    assert list(df.ensemble) == ["NVT", "NVT", "NPT", "NPT", "NPT", "NPT"]
    assert df.t_dyn_ps.tolist() == pytest.approx([0.002, 0.004, 0.006, 0.008, 0.010, 0.012])


# --------------------------------------------------------- qc: kinetic energy

def test_kinetic_temperature_uses_masses_constraints_and_com_removal(tmp_path):
    (tmp_path / "system.xml").write_text(
        '<System><Particles><Particle mass="12"/><Particle mass="1"/><Particle mass="0"/></Particles>'
        '<Constraints><Constraint p1="0" p2="1" d="0.1"/></Constraints>'
        '<Forces><Force type="CMMotionRemover"/></Forces></System>')
    (tmp_path / "state.xml").write_text(
        '<State time="2.5" stepCount="1250"><Velocities><Velocity x="1" y="0" z="0"/>'
        '<Velocity x="0" y="2" z="0"/><Velocity x="5" y="5" z="5"/></Velocities></State>')
    k = qc.kinetic_temperature(tmp_path / "state.xml", tmp_path / "system.xml")
    assert k["kinetic_energy_kj_mol"] == pytest.approx(8.0)  # 0.5 (12*1 + 1*4); massless excluded
    assert k["degrees_of_freedom"] == 2                        # 3*2 - 1 - 3
    assert k["temperature_K"] == pytest.approx(2 * 8.0 / (2 * qc.BOLTZMANN_KJ_MOL_K))
    assert (k["step"], k["time_ps"]) == (1250, 2.5)


def test_kinetic_temperature_rejects_mismatched_files(tmp_path):
    (tmp_path / "system.xml").write_text('<System><Particles><Particle mass="1"/></Particles><Forces/></System>')
    (tmp_path / "state.xml").write_text('<State><Velocities><Velocity x="1" y="0" z="0"/>'
                                        '<Velocity x="1" y="0" z="0"/></Velocities></State>')
    with pytest.raises(ValueError, match="particle counts"):
        qc.kinetic_temperature(tmp_path / "state.xml", tmp_path / "system.xml")


# ------------------------------------------------------- qc: structure series

def _universe(frames: np.ndarray, names: list[str], box: float = 100.0) -> mda.Universe:
    n = frames.shape[1]
    u = mda.Universe.empty(n, n_residues=1, atom_resindex=np.zeros(n, dtype=int), trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("resname", ["SYN"])
    dims = np.tile([box, box, box, 90.0, 90.0, 90.0], (len(frames), 1))
    u.load_new(frames.astype(np.float32), format=MemoryReader, dimensions=dims, dt=0.5)
    return u


def test_rmsd_to_reference_is_invariant_to_rigid_motion():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(6, 3)) * 3
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = ref @ rot.T + [10.0, -4.0, 2.0]
    shifted = ref.copy()
    shifted[:, 0] += np.array([1, -1, 1, -1, 1, -1]) * 0.5  # zero-mean, centroid unchanged
    u = _universe(np.stack([ref, moved, shifted]), [f"X{i}" for i in range(6)])
    r = qc.rmsd_to_reference(u, "all", ref)
    assert r[0] == pytest.approx(0, abs=1e-4) and r[1] == pytest.approx(0, abs=1e-4)
    assert 0 < r[2] <= 0.5 + 1e-4  # superposition can only lower the unfitted 0.5 A
    with pytest.raises(ValueError, match="shape"):
        qc.rmsd_to_reference(u, "all", ref[:3])


def test_pair_distance_series_reports_named_distances():
    f0 = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=float)
    f1 = f0 * 2
    u = _universe(np.stack([f0, f1]), ["A", "B", "C"])
    df = qc.pair_distance_series(u, [("AB", "name A", "name B"), ("BC", "name B", "name C")])
    assert df.AB.tolist() == pytest.approx([3, 6]) and df.BC.tolist() == pytest.approx([5, 10])
    assert df.time_ps.tolist() == pytest.approx([0.0, 0.5])
    with pytest.raises(ValueError, match="one atom each"):
        qc.pair_distance_series(u, [("bad", "name A B", "name C")])


@pytest.mark.parametrize("solute_at", [(15.0, 15.0, 15.0), (0.2, 0.2, 0.2)])
def test_bulk_water_density_of_a_lattice_with_periodic_images(solute_at):
    spacing, n = 3.0, 10  # 1000 "waters" in a 30 A cube: 1/27 per A^3 = 37.04 per nm^3
    grid = (np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) * spacing
    frame = np.vstack([grid, [solute_at]])
    u = _universe(frame[None], ["OW"] * len(grid) + ["S"], box=n * spacing)
    df = qc.bulk_water_density(u, "name S", "name OW", exclusion_A=6.0, grid_spacing_A=0.5)
    expected = 1000 / 27.0
    assert df.molecules_per_nm3.iloc[0] == pytest.approx(expected, rel=0.03)
    assert df.g_per_ml.iloc[0] == pytest.approx(expected * qc.WATER_G_MOL / qc.AVOGADRO * 1e21, rel=0.03)
    # the exclusion sphere removes the same volume wherever the solute sits (minimum image)
    assert df.bulk_volume_nm3.iloc[0] == pytest.approx(27.0 - 4 / 3 * np.pi * 0.216, rel=0.01)


# ------------------------------------------------ run loader and provenance

def _synthetic_run(tmp_path: Path, times_ps: np.ndarray, relative_to: Path | None = None) -> Path:
    """A run directory in the layout neurodna.md.run writes, with a SYNTHETIC trajectory."""
    prep = tmp_path / "prepared"
    prep.mkdir()
    (prep / "preparation.json").write_text(json.dumps({"selections": SELECTIONS}))
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(PDB, run / "topology.pdb")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(PDB))
        xyz = u.atoms.positions
        frames = np.stack([xyz + [0.1 * i, 0.0, 0.0] for i in range(len(times_ps))])
        dims = np.tile([200.0, 200.0, 200.0, 90.0, 90.0, 90.0], (len(times_ps), 1))
        u.load_new(frames, format=MemoryReader, dimensions=dims)
        with mda.Writer(str(run / "production.xtc"), u.atoms.n_atoms) as w:
            for ts, t in zip(u.trajectory, times_ps):
                ts.time = float(t)
                w.write(u.atoms)
    recorded = prep if relative_to is None else prep.relative_to(relative_to)
    meta = {
        "inputs": {"prepared_dir": str(recorded), "prepared_dir_resolved": str(prep.resolve()),
                   "preparation_json_sha256": sha256_file(prep / "preparation.json")},
        "production": {"trajectory": "production.xtc", "sha256": sha256_file(run / "production.xtc"),
                       "n_frames": len(times_ps), "frame_interval_ps": float(times_ps[1] - times_ps[0])},
        "protocol": {"name": "synthetic", "purpose": "SYNTHETIC TEST DATA, not a simulation"},
        "statement": "synthetic", "simulated_ps": float(times_ps[-1]),
    }
    (run / "simulation.json").write_text(json.dumps(meta))
    return run


def test_load_run_checks_checksums_and_resolves_the_prepared_dir(tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd_at_run_time"
    elsewhere.mkdir()
    run = _synthetic_run(tmp_path, np.arange(4) * 0.25 + 0.25, relative_to=tmp_path)
    monkeypatch.chdir(elsewhere)  # the recorded relative path does not resolve from here
    assert prepared_dir_of(run) == (tmp_path / "prepared").resolve()
    cx = load_run(run)
    assert cx.frame_times().tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])

    (tmp_path / "prepared" / "preparation.json").write_text(json.dumps({"selections": SELECTIONS}, indent=1))
    with pytest.raises(TrajectoryError, match="preparation.json does not match"):
        load_run(run)


def test_load_run_rejects_a_modified_trajectory(tmp_path):
    run = _synthetic_run(tmp_path, np.arange(3) * 0.5)
    with open(run / "production.xtc", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(TrajectoryError, match="checksum"):
        load_run(run)


def test_analyze_run_burn_in_and_provenance(tmp_path):
    times = np.arange(5) * 0.25  # includes t = 0
    run = _synthetic_run(tmp_path, times)
    out = analyze_run(run, tmp_path / "all", cutoff=4.5)
    summary = json.loads(out["summary"].read_text())
    window = summary["frame_window"]
    assert (window["first_frame"], window["n_frames"], window["first_time_ps"]) == (0, 5, 0.0)
    assert summary["inputs"]["trajectory"]["sha256"] == sha256_file(run / "production.xtc")
    assert summary["inputs"]["topology"]["sha256"] == sha256_file(run / "topology.pdb")
    assert summary["selections"] == SELECTIONS and summary["cutoff_A"] == 4.5
    assert summary["units"] == {"distance": "angstrom", "time": "ps"}
    assert summary["time_source"] == "file" and summary["convergence_assessed"] is False
    assert set(summary["software"]) >= {"neurodna", "MDAnalysis", "numpy", "pandas", "python"}

    out = analyze_run(run, tmp_path / "burn", cutoff=4.5, burn_in_ps=0.5)
    summary = json.loads(out["summary"].read_text())
    assert (summary["frame_window"]["first_frame"], summary["frame_window"]["n_frames"]) == (3, 2)
    rmsd = __import__("pandas").read_csv(out["backbone_rmsd"])
    assert rmsd.frame.tolist() == [3, 4] and rmsd.rmsd_A.iloc[0] == pytest.approx(0, abs=1e-3)
    assert "--burn-in-ps 0.5" in summary["command"]

    with pytest.raises(TrajectoryError, match="fewer than 2 frames"):
        analyze_run(run, tmp_path / "none", burn_in_ps=0.75)
    with pytest.raises(ValueError, match=">= 0"):
        analyze_run(run, tmp_path / "neg", burn_in_ps=-1)
