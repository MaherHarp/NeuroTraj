"""Run minimisation, equilibration and production for a prepared system.

Directory layout written to ``output_dir``:

* ``topology.pdb`` -- copy of the prepared topology (atom order = trajectory);
* ``production.xtc`` -- production trajectory with real times (ps). It is written
  with ``enforcePeriodicBox=False``, so molecules stay whole and never jump
  between periodic images;
* ``production.chk`` / ``stages/*.chk`` -- binary OpenMM checkpoints
  (platform-specific), plus portable ``stages/*.xml`` states;
* ``equilibration_log.csv``, ``production_log.csv`` -- energies, temperature,
  volume and speed;
* ``simulation.json`` -- protocol, seeds, platform, software, checksums, and the
  measured wall time and throughput of every stage;
* ``progress.json`` -- completed stages (used by ``resume=True``).

Nothing here judges convergence. ``simulation.json`` records the simulated
durations; whether they suffice is a scientific question for the user (for
example, with independent replicates).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neurodna.errors import NeuroDNAError
from neurodna.md.config import Protocol
from neurodna.md.forcefield import require_openmm
from neurodna.md.prepare import sha256, software_versions

RESTRAINT_ENERGY = "0.5*k_restraint*periodicdistance(x, y, z, x0, y0, z0)^2"


class SimulationError(NeuroDNAError):
    """The run cannot start or resume consistently."""


def _seeds(seed: int) -> dict[str, int]:
    return {"velocities": seed, "integrator": seed + 1, "barostat": seed + 2}


def run(prepared_dir: str | os.PathLike[str], protocol: Protocol,
        output_dir: str | os.PathLike[str], *, resume: bool = False,
        stop_after_stage: str | None = None,
        stop_at_production_step: int | None = None) -> dict[str, Any]:
    """Run ``protocol`` on the system in ``prepared_dir``; returns the metadata dict.

    ``stop_after_stage`` ("minimization" or an equilibration stage name) and
    ``stop_at_production_step`` (a multiple of ``checkpoint_interval_steps``)
    end the session cleanly at a checkpoint. ``resume=True`` continues later.
    ``metadata["status"]`` says whether the protocol is complete.
    """
    mm, app, unit = require_openmm()
    check_stop_options(protocol, stop_after_stage, stop_at_production_step)
    prep_dir, out = Path(prepared_dir), Path(output_dir)
    preparation = json.loads((prep_dir / "preparation.json").read_text())
    for key in ("prepared_pdb", "system_xml"):
        f = preparation["files"][key]
        if sha256(prep_dir / f["name"]) != f["sha256"]:
            raise SimulationError(f"{f['name']} does not match the checksum in preparation.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stages").mkdir(exist_ok=True)
    progress_path = out / "progress.json"
    progress: dict[str, Any] = {"completed": [], "production_steps_done": 0}
    if progress_path.exists():
        if not resume:
            raise SimulationError(f"{out} already contains a run; pass resume=True or use a new directory")
        progress = json.loads(progress_path.read_text())
        previous = json.loads((out / "simulation.json").read_text())["protocol"]
        _check_resumable(previous, protocol.to_dict())

    pdb = app.PDBFile(str(prep_dir / preparation["files"]["prepared_pdb"]["name"]))
    shutil.copyfile(prep_dir / preparation["files"]["prepared_pdb"]["name"], out / "topology.pdb")
    system = mm.XmlSerializer.deserialize((prep_dir / preparation["files"]["system_xml"]["name"]).read_text())
    seeds = _seeds(protocol.seed)

    # Positional restraints on solute heavy atoms (reference: prepared coordinates).
    import MDAnalysis as mda

    u = mda.Universe(str(out / "topology.pdb"))
    solute = u.select_atoms(f"({preparation['selections']['protein']}) or "
                            f"({preparation['selections']['dna']})")
    heavy = solute.select_atoms("not element H")
    restraint = mm.CustomExternalForce(RESTRAINT_ENERGY)
    restraint.addGlobalParameter("k_restraint", 0.0)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    ref = pdb.positions.value_in_unit(unit.nanometer)
    for i in heavy.indices:
        restraint.addParticle(int(i), list(ref[int(i)]))
    system.addForce(restraint)
    barostat = mm.MonteCarloBarostat(protocol.pressure_bar * unit.bar,
                                     protocol.temperature_K * unit.kelvin, 0)
    barostat.setRandomNumberSeed(seeds["barostat"])
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(protocol.temperature_K * unit.kelvin,
                                             protocol.friction_per_ps / unit.picosecond,
                                             protocol.timestep_fs * unit.femtoseconds)
    integrator.setRandomNumberSeed(seeds["integrator"])
    platform, properties = _platform(mm, protocol)
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    metadata: dict[str, Any] = (json.loads((out / "simulation.json").read_text())
                                if progress_path.exists() else {"stages": []})
    metadata.update({
        "protocol": protocol.to_dict(),
        "seeds": seeds,
        "platform": _platform_info(simulation, platform),
        "software": software_versions(),
        "inputs": {"prepared_dir": str(prep_dir), "prepared_dir_resolved": str(prep_dir.resolve()),
                   **{k: preparation["files"][k] for k in ("prepared_pdb", "system_xml")},
                   "preparation_json_sha256": sha256(prep_dir / "preparation.json")},
        "restraints": {"atoms": "solute heavy atoms", "n_atoms": int(len(heavy)),
                       "reference": "prepared coordinates", "energy": RESTRAINT_ENERGY},
        "n_atoms": system.getNumParticles(),
        "units": {"time": "ps", "length": "nm (OpenMM) / angstrom (neurodna)", "energy": "kJ/mol",
                  "restraint_k": "kJ/mol/nm^2"},
        "convergence_assessed": False,
        "statement": "Simulated durations are reported as run; no claim of equilibration or "
                     "convergence is made. " + protocol.purpose,
    })

    def save_progress() -> None:
        _write_atomic(progress_path, json.dumps(progress, indent=2) + "\n")
        _write_atomic(out / "simulation.json", json.dumps(metadata, indent=2, default=str) + "\n")

    def set_stage(k: float, npt: bool) -> None:
        simulation.context.setParameter("k_restraint", k)
        frequency = protocol.barostat_interval_steps if npt else 0
        if barostat.getFrequency() != frequency:
            barostat.setFrequency(frequency)
            simulation.context.reinitialize(preserveState=True)

    def finish(status: str) -> dict[str, Any]:
        metadata["simulated_ps"] = {
            "equilibration": round(sum(s.get("simulated_ps", 0.0) for s in metadata["stages"]
                                       if s["name"] not in ("minimization", "production")), 6),
            "production": round(sum(s.get("simulated_ps", 0.0) for s in metadata["stages"]
                                    if s["name"] == "production"), 6),
        }
        metadata["status"] = status
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_progress()
        return metadata

    last_stage = progress["completed"][-1] if progress["completed"] else None
    if last_stage is not None and last_stage != "production":
        simulation.loadCheckpoint(str(out / "stages" / f"{last_stage}.chk"))

    # ---------------------------------------------------------------- minimisation
    if "minimization" not in progress["completed"]:
        set_stage(protocol.minimization_restraint_k_kj_mol_nm2, npt=False)
        e0 = _energy(simulation, unit)
        t0 = time.perf_counter()
        simulation.minimizeEnergy(tolerance=protocol.minimization_tolerance_kj_mol_nm
                                  * unit.kilojoule_per_mole / unit.nanometer,
                                  maxIterations=protocol.minimization_max_iterations)
        wall = time.perf_counter() - t0
        e1 = _energy(simulation, unit)
        simulation.context.setVelocitiesToTemperature(protocol.temperature_K * unit.kelvin,
                                                      seeds["velocities"])
        _save_stage(simulation, out, "minimization")
        metadata["stages"].append({"name": "minimization", "max_iterations": protocol.minimization_max_iterations,
                                   "restraint_k": protocol.minimization_restraint_k_kj_mol_nm2,
                                   "potential_energy_before_kj_mol": e0, "potential_energy_after_kj_mol": e1,
                                   "wall_time_s": round(wall, 2)})
        progress["completed"].append("minimization")
        save_progress()
    if stop_after_stage == "minimization":
        return finish("stopped after minimization (resume to continue)")

    # --------------------------------------------------------------- equilibration
    # rows of an interrupted stage are dropped; the stage reruns from the last stage checkpoint
    _truncate_log(out / "equilibration_log.csv",
                  sum(s.steps for s in protocol.equilibration if s.name in progress["completed"]))
    with open(out / "equilibration_log.csv", "a") as log:
        for stage in protocol.equilibration:
            if stage.name in progress["completed"]:
                if stage.name == stop_after_stage:
                    return finish(f"stopped after {stage.name} (resume to continue)")
                continue
            set_stage(stage.restraint_k_kj_mol_nm2, npt=stage.ensemble == "NPT")
            reporter = app.StateDataReporter(log, max(1, min(stage.steps, protocol.report_interval_steps)),
                                             step=True, time=True, potentialEnergy=True, temperature=True,
                                             volume=True, density=True, speed=True)
            simulation.reporters = [reporter]
            t0 = time.perf_counter()
            simulation.step(stage.steps)
            wall = time.perf_counter() - t0
            _save_stage(simulation, out, stage.name)
            metadata["stages"].append(_stage_record(stage.name, stage.ensemble, stage.steps, protocol,
                                                    wall, stage.restraint_k_kj_mol_nm2, simulation, unit))
            progress["completed"].append(stage.name)
            save_progress()
            if stage.name == stop_after_stage:
                return finish(f"stopped after {stage.name} (resume to continue)")
    simulation.reporters = []

    # ------------------------------------------------------------------ production
    xtc = out / "production.xtc"
    chk = out / "production.chk"
    if protocol.production_steps > 0:
        set_stage(0.0, npt=True)
        appending = progress["production_steps_done"] > 0
        if appending:
            simulation.loadCheckpoint(str(chk))
            done = simulation.currentStep
            expected_frames = done // protocol.report_interval_steps
            _truncate_xtc(out / "topology.pdb", xtc, expected_frames)
            _truncate_log(out / "production_log.csv", done)
        else:
            simulation.context.setTime(0.0)
            simulation.context.setStepCount(0)
            simulation.currentStep = 0
            for stale in (xtc, chk, out / "production_log.csv"):
                if stale.exists():
                    stale.unlink()
        target = protocol.production_steps
        if stop_at_production_step is not None:
            target = min(target, stop_at_production_step)
        remaining = target - simulation.currentStep
        if remaining < 0:
            raise SimulationError("the checkpoint is beyond the requested production length")
        simulation.reporters = [
            app.XTCReporter(str(xtc), protocol.report_interval_steps, append=appending and xtc.exists(),
                            enforcePeriodicBox=False),
            app.StateDataReporter(str(out / "production_log.csv"), protocol.report_interval_steps,
                                  step=True, time=True, potentialEnergy=True, temperature=True,
                                  volume=True, density=True, speed=True, append=appending),
        ]
        t0 = time.perf_counter()
        chunk = protocol.checkpoint_interval_steps
        steps_run = 0
        while remaining > 0:
            n = min(chunk, remaining)
            simulation.step(n)
            _save_checkpoint(simulation, chk)
            remaining -= n
            steps_run += n
            progress["production_steps_done"] = simulation.currentStep
            save_progress()
        wall = time.perf_counter() - t0
        simulation.reporters = []
        _save_stage(simulation, out, "production")
        record = _stage_record("production", "NPT", steps_run, protocol, wall, 0.0, simulation, unit)
        record["resumed_from_step"] = simulation.currentStep - steps_run
        metadata["stages"].append(record)
        complete = simulation.currentStep == protocol.production_steps
        if complete and "production" not in progress["completed"]:
            progress["completed"].append("production")
        n_frames = simulation.currentStep // protocol.report_interval_steps
        metadata["production"] = {
            "trajectory": xtc.name, "sha256": sha256(xtc), "size_bytes": xtc.stat().st_size,
            "n_frames": n_frames,
            "frame_interval_ps": protocol.report_interval_steps * protocol.timestep_fs / 1000.0,
            "first_frame_time_ps": protocol.report_interval_steps * protocol.timestep_fs / 1000.0,
            "simulated_ps": simulation.currentStep * protocol.timestep_fs / 1000.0,
            "time_origin": "t = 0 at the start of production (equilibration not included)",
            "periodic_wrapping": "none (enforcePeriodicBox=False): molecules are whole",
            "complete": complete,
        }
        if not complete:
            return finish(f"stopped at production step {simulation.currentStep} of "
                          f"{protocol.production_steps} (resume to continue)")
    return finish("complete")


def check_stop_options(protocol: Protocol, stop_after_stage: str | None,
                       stop_at_production_step: int | None) -> None:
    """Stops must coincide with a saved checkpoint."""
    if stop_after_stage is not None and stop_at_production_step is not None:
        raise SimulationError("use either stop_after_stage or stop_at_production_step")
    stages = ["minimization", *(s.name for s in protocol.equilibration)]
    if stop_after_stage is not None and stop_after_stage not in stages:
        raise SimulationError(f"stop_after_stage must be one of {stages}")
    if stop_at_production_step is not None:
        if not 0 < stop_at_production_step <= protocol.production_steps:
            raise SimulationError("stop_at_production_step must be in (0, production_steps]")
        if stop_at_production_step % protocol.checkpoint_interval_steps:
            raise SimulationError("stop_at_production_step must be a multiple of checkpoint_interval_steps "
                                  "so the session ends at a checkpoint")


def _check_resumable(previous: dict[str, Any], current: dict[str, Any]) -> None:
    # Thread count changes speed, not the physics; an older run without the field used the default.
    free = ("production_steps", "purpose", "name", "cpu_threads")
    fixed = {k: v for k, v in previous.items() if k not in free}
    now = {k: v for k, v in current.items() if k not in free}
    if fixed != now:
        changed = sorted(k for k in set(fixed) | set(now) if fixed.get(k) != now.get(k))
        raise SimulationError(f"cannot resume: protocol settings changed {changed}; only "
                              "production_steps may be extended")
    if current["production_steps"] < previous["production_steps"]:
        raise SimulationError("cannot resume with fewer production steps than already requested")


def _platform(mm: Any, protocol: Protocol) -> tuple[Any, dict[str, str]]:
    name = protocol.platform
    if name is None:
        speeds = sorted((mm.Platform.getPlatform(i) for i in range(mm.Platform.getNumPlatforms())),
                        key=lambda p: p.getSpeed(), reverse=True)
        for candidate in speeds:
            try:
                context = mm.Context(mm.System(), mm.VerletIntegrator(0.001), candidate)
                del context
                return _platform_with(mm, candidate.getName(), protocol)
            except Exception:
                continue
    return _platform_with(mm, name or "CPU", protocol)


def _platform_with(mm: Any, name: str, protocol: Protocol) -> tuple[Any, dict[str, str]]:
    platform = mm.Platform.getPlatformByName(name)
    props: dict[str, str] = {}
    if name == "CPU" and protocol.cpu_threads:
        props["Threads"] = str(protocol.cpu_threads)
    if protocol.precision and name in ("CUDA", "OpenCL", "HIP", "Metal"):
        props["Precision"] = protocol.precision
    return platform, props


def _platform_info(simulation: Any, platform: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"name": platform.getName()}
    for prop in platform.getPropertyNames():
        try:
            info[prop] = platform.getPropertyValue(simulation.context, prop)
        except Exception:
            continue
    return info


def _energy(simulation: Any, unit: Any) -> float:
    state = simulation.context.getState(getEnergy=True)
    return round(float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)), 3)


def _save_stage(simulation: Any, out: Path, name: str) -> None:
    _save_checkpoint(simulation, out / "stages" / f"{name}.chk")
    tmp = out / "stages" / f".{name}.xml.tmp"
    simulation.saveState(str(tmp))
    os.replace(tmp, out / "stages" / f"{name}.xml")


def _save_checkpoint(simulation: Any, path: Path) -> None:
    """Write a checkpoint atomically, so an interruption never leaves a truncated file."""
    tmp = path.with_name(f".{path.name}.tmp")
    simulation.saveCheckpoint(str(tmp))
    os.replace(tmp, path)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _truncate_log(path: Path, max_step: int) -> None:
    """Drop log rows after ``max_step``: rows written after the checkpoint being resumed.

    They belong to dynamics that are discarded and regenerated on resume; keeping
    them would duplicate steps. Header lines (including the one each session
    appends) are kept.
    """
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines(keepends=True):
        first = line.split(",", 1)[0].strip()
        if first.lstrip("-").isdigit() and int(first) > max_step:
            continue
        kept.append(line)
    _write_atomic(path, "".join(kept))


def _stage_record(name: str, ensemble: str, steps: int, protocol: Protocol, wall: float,
                  k: float, simulation: Any, unit: Any) -> dict[str, Any]:
    simulated_ps = steps * protocol.timestep_fs / 1000.0
    state = simulation.context.getState(getEnergy=True)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    return {
        "name": name, "ensemble": ensemble, "steps": steps, "simulated_ps": simulated_ps,
        "restraint_k": k, "wall_time_s": round(wall, 2),
        "ns_per_day": round(simulated_ps / 1000.0 / wall * 86400, 3) if wall > 0 else None,
        "final_potential_energy_kj_mol": round(float(state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)), 3),
        "final_box_volume_nm3": round(float(abs(np.linalg.det(np.asarray(box)))), 4),
    }


def _truncate_xtc(topology: Path, xtc: Path, n_frames: int) -> None:
    """Drop frames written after the last checkpoint (they will be regenerated)."""
    if not xtc.exists():
        return
    import MDAnalysis as mda

    u = mda.Universe(str(topology), str(xtc))
    if u.trajectory.n_frames == n_frames:
        return
    if u.trajectory.n_frames < n_frames:
        raise SimulationError(f"{xtc} has {u.trajectory.n_frames} frames but the checkpoint expects {n_frames}")
    tmp = xtc.with_suffix(".tmp.xtc")
    with mda.Writer(str(tmp), n_atoms=u.atoms.n_atoms) as writer:
        for _ in u.trajectory[:n_frames]:
            writer.write(u.atoms)
    os.replace(tmp, xtc)
