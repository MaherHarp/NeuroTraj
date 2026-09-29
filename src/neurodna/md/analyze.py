"""Feed an OpenMM run (from :mod:`neurodna.md.run`) into neurodna. Needs no OpenMM.

:func:`load_run` checks the trajectory against ``simulation.json`` (checksum,
frame count, frame spacing) before building a :class:`~neurodna.Complex` with the
selections recorded at preparation. :func:`analyze_run` exports contact
occupancy, contact episodes, backbone RMSD and RMSF, each labelled with the
run's purpose and simulated duration.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from neurodna.complex import Complex
from neurodna.errors import TrajectoryError
from neurodna.fetch import sha256_file


def prepared_dir_of(run_dir: str | os.PathLike[str],
                    prepared_dir: str | os.PathLike[str] | None = None) -> Path:
    """The prepared-system directory of a run, checked against ``simulation.json``.

    Uses ``prepared_dir`` if given; otherwise the recorded path (as typed when the
    run started, relative to the working directory then) and, if that does not
    exist from here, the recorded absolute path. The directory's
    ``preparation.json`` must match the SHA-256 recorded by the run.
    """
    meta = json.loads((Path(run_dir) / "simulation.json").read_text())
    inputs = meta["inputs"]
    if prepared_dir is not None:
        candidates = [Path(prepared_dir)]
    else:
        candidates = [Path(inputs["prepared_dir"])]
        if inputs.get("prepared_dir_resolved"):
            candidates.append(Path(inputs["prepared_dir_resolved"]))
    found = next((c for c in candidates if (c / "preparation.json").is_file()), None)
    if found is None:
        raise TrajectoryError(
            f"prepared directory not found (tried {', '.join(map(str, candidates))}); "
            "run from the directory the simulation was started in or pass prepared_dir")
    expected = inputs.get("preparation_json_sha256")
    if expected and sha256_file(found / "preparation.json") != expected:
        raise TrajectoryError(f"{found / 'preparation.json'} does not match the checksum in simulation.json")
    return found


def load_run(run_dir: str | os.PathLike[str],
             prepared_dir: str | os.PathLike[str] | None = None) -> Complex:
    """A :class:`Complex` for the production trajectory of a run directory.

    ``prepared_dir`` overrides the recorded prepared-system path (see
    :func:`prepared_dir_of`).
    """
    run = Path(run_dir)
    meta = json.loads((run / "simulation.json").read_text())
    prep = json.loads((prepared_dir_of(run, prepared_dir) / "preparation.json").read_text())
    production = meta.get("production")
    if not production:
        raise TrajectoryError(f"{run} has no production trajectory")
    xtc = run / production["trajectory"]
    if sha256_file(xtc) != production["sha256"]:
        raise TrajectoryError(f"{xtc} does not match the checksum in simulation.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx = Complex.load(run / "topology.pdb", xtc, protein=prep["selections"]["protein"],
                          dna=prep["selections"]["dna"])
    if cx.frames.n_frames != production["n_frames"]:
        raise TrajectoryError(f"trajectory has {cx.frames.n_frames} frames, metadata says "
                              f"{production['n_frames']}")
    times = cx.frame_times()
    if len(times) > 1 and not np.allclose(np.diff(times), production["frame_interval_ps"], atol=1e-3):
        raise TrajectoryError("frame spacing in the trajectory differs from simulation.json")
    return cx


def analyze_run(run_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str],
                cutoff: float = 4.5, burn_in_ps: float = 0.0,
                prepared_dir: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Occupancy, episodes, RMSD and RMSF of the production trajectory, with provenance.

    With ``burn_in_ps > 0``, frames with time <= ``burn_in_ps`` are excluded from
    every table (``0`` keeps every frame). The RMSD/RMSF
    reference is the first analysed frame. ``analysis_summary.json`` records the
    inputs (paths and SHA-256), selections, frame window, parameters and
    software versions needed to reproduce the tables.
    """
    import platform
    from datetime import datetime, timezone

    import MDAnalysis
    import pandas as pd

    import neurodna

    if burn_in_ps < 0:
        raise ValueError("burn_in_ps must be >= 0")
    run, out = Path(run_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "simulation.json").read_text())
    prep_dir = prepared_dir_of(run, prepared_dir)
    prep = json.loads((prep_dir / "preparation.json").read_text())
    cx = load_run(run, prep_dir)
    times = cx.frame_times()
    start = int(np.searchsorted(times, burn_in_ps, side="right")) if burn_in_ps > 0 else 0
    if cx.frames.n_frames - start < 2:
        raise TrajectoryError(f"burn-in of {burn_in_ps} ps leaves fewer than 2 frames")
    tables = {
        "contact_occupancy": cx.contact_occupancy(cutoff, start=start),
        "contact_episodes": cx.contact_episodes(cutoff, start=start),
        "backbone_rmsd": cx.backbone_rmsd(reference_frame=start, start=start),
        "rmsf": cx.rmsf(reference_frame=start, start=start),
    }
    paths: dict[str, Path] = {}
    for name, df in tables.items():
        paths[name] = out / f"{name}.csv"
        df.to_csv(paths[name], index=False)
    occupancy = tables["contact_occupancy"]
    xtc = run / meta["production"]["trajectory"]
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_dir": str(run),
        "protocol": meta["protocol"]["name"],
        "purpose": meta["protocol"]["purpose"],
        "statement": meta["statement"],
        "convergence_assessed": False,
        "simulated_ps": meta["simulated_ps"],
        "inputs": {
            "topology": {"path": str(run / "topology.pdb"), "sha256": sha256_file(run / "topology.pdb")},
            "trajectory": {"path": str(xtc), "sha256": sha256_file(xtc)},
            "simulation_json_sha256": sha256_file(run / "simulation.json"),
            "prepared_dir": str(prep_dir.resolve()),
            "preparation_json_sha256": sha256_file(prep_dir / "preparation.json"),
        },
        "selections": prep["selections"],
        "frame_window": {"burn_in_ps": burn_in_ps, "first_frame": start, "last_frame": cx.frames.n_frames - 1,
                         "n_frames": cx.frames.n_frames - start, "first_time_ps": float(times[start]),
                         "last_time_ps": float(times[-1])},
        "frame_interval_ps": meta["production"]["frame_interval_ps"],
        "units": {"distance": "angstrom", "time": "ps"},
        "cutoff_A": cutoff,
        "contact_definition": occupancy.attrs["contact_definition"],
        "atoms": occupancy.attrs["atoms"],
        "minimum_image": occupancy.attrs["minimum_image"],
        "periodic_policy": cx.periodic_policy,
        "time_source": occupancy.attrs["time_source"],
        "sampling": {k: occupancy.attrs[k] for k in ("sampling_interval_ps", "uniform_sampling",
                                                     "start_time_ps", "end_time_ps")},
        "rmsd": {"alignment": tables["backbone_rmsd"].attrs["alignment_selection"],
                 "reference_frame": start, "superposition": tables["backbone_rmsd"].attrs["superposition"]},
        "rmsf": {"atoms": tables["rmsf"].attrs["measured_selection"], "reference_frame": start},
        "episode_note": tables["contact_episodes"].attrs["note"],
        "software": {"neurodna": neurodna.__version__, "MDAnalysis": MDAnalysis.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "python": platform.python_version()},
        "command": (f"neurodna-md analyze {run.resolve()} --output-dir {out.resolve()} --cutoff {cutoff} "
                    f"--burn-in-ps {burn_in_ps}"
                    + (f" --prepared-dir {prep_dir.resolve()}" if prepared_dir is not None else "")),
        "outputs": {k: v.name for k, v in paths.items()},
    }
    paths["summary"] = out / "analysis_summary.json"
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return paths
