"""Quality-control diagnostics for MD runs (numerical, physical, structural).

These functions describe a run; none of them establishes equilibration or
convergence. Units: Å, ps, kJ/mol, K, g/mL. Needs no OpenMM.

* :func:`thermo_table` / :func:`drift_summary`: the energy/temperature/volume
  logs written by :mod:`neurodna.md.run`, on one continuous time axis.
* :func:`kinetic_temperature`: temperature recomputed from the velocities in
  a saved OpenMM state, independently of the reporter.
* :func:`rmsd_to_reference`: RMSD of an atom selection to reference coordinates
  after optimal superposition on the same atoms (Kabsch).
* :func:`pair_distance_series`: per-frame distances of named atom pairs
  (Watson–Crick N1–N3, methyl C5–C5A, ...).
* :func:`bulk_water_density`: water density away from the solute (a check on
  the model's water density that is independent of solute mass).
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from neurodna.structure import kabsch, superpose

BOLTZMANN_KJ_MOL_K = 0.0083144626
AVOGADRO = 6.02214076e23
WATER_G_MOL = 18.01528
_LOG_COLUMNS = ["step", "time_ps", "potential_kj_mol", "temperature_K", "volume_nm3",
                "density_g_ml", "speed_ns_day"]


def _read_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment=None)
    df = df[pd.to_numeric(df.iloc[:, 0], errors="coerce").notna()].astype(float)  # drop repeated headers
    df.columns = _LOG_COLUMNS
    return df.reset_index(drop=True)


def thermo_table(run_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """Equilibration and production logs on one axis: ``t_dyn_ps`` = time since dynamics began."""
    run = Path(run_dir)
    meta = json.loads((run / "simulation.json").read_text())
    frames: list[pd.DataFrame] = []
    eq_path = run / "equilibration_log.csv"
    offset = 0.0
    if eq_path.exists():
        eq = _read_log(eq_path)
        stages = [s for s in meta["stages"] if s["name"] not in ("minimization", "production")]
        bounds, total = [], 0
        for s in stages:
            total += s["steps"]
            bounds.append((total, s["name"], s["ensemble"]))
        eq["stage"] = [next(n for b, n, _ in bounds if step <= b) for step in eq.step]
        eq["ensemble"] = [next(e for b, _, e in bounds if step <= b) for step in eq.step]
        eq["t_dyn_ps"] = eq.time_ps
        offset = sum(s["simulated_ps"] for s in stages)
        frames.append(eq)
    pr_path = run / "production_log.csv"
    if pr_path.exists():
        pr = _read_log(pr_path)
        pr["stage"], pr["ensemble"] = "production", "NPT"
        pr["t_dyn_ps"] = pr.time_ps + offset
        frames.append(pr)
    return pd.concat(frames, ignore_index=True)


def drift_summary(values: Sequence[float], times: Sequence[float], n_blocks: int = 4) -> dict[str, Any]:
    """Mean, SD, half means, linear slope and block standard error of a series.

    ``half_difference_over_block_se`` compares the two half means with the block
    standard error. Serial correlation makes every value here approximate, and a
    short series gives unreliable block estimates.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    n = v.size
    half = n // 2
    blocks = np.array_split(v, n_blocks) if n >= n_blocks else [v]
    block_means = np.array([b.mean() for b in blocks])
    block_se = float(block_means.std(ddof=1) / math.sqrt(len(block_means))) if len(block_means) > 1 else math.nan
    slope = float(np.polyfit(t, v, 1)[0]) if n >= 2 else math.nan
    diff = float(v[half:].mean() - v[:half].mean()) if n >= 2 else math.nan
    return {
        "n": int(n), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else math.nan,
        "min": float(v.min()), "max": float(v.max()),
        "first_half_mean": float(v[:half].mean()) if half else math.nan,
        "second_half_mean": float(v[half:].mean()),
        "half_difference": diff,
        "slope_per_ps": slope,
        "block_means": [float(x) for x in block_means],
        "block_se": block_se,
        "half_difference_over_block_se": float(abs(diff) / block_se) if block_se and block_se > 0 else math.nan,
    }


def kinetic_temperature(state_xml: str | os.PathLike[str],
                        system_xml: str | os.PathLike[str]) -> dict[str, float]:
    """Kinetic energy (kJ/mol) and temperature (K) from a saved OpenMM state.

    Degrees of freedom = 3 N_massive - N_constraints - 3 (if the system removes
    centre-of-mass motion), the same convention OpenMM uses.
    """
    def child(root: ET.Element, tag: str) -> ET.Element:
        found = root.find(tag)
        if found is None:
            raise ValueError(f"{tag} missing from the XML")
        return found

    system = ET.parse(system_xml).getroot()
    masses = np.array([float(p.get("mass", "0")) for p in child(system, "Particles")], dtype=np.float64)
    constraints = system.find("Constraints")
    n_constraints = len(constraints) if constraints is not None else 0
    com_removed = any(f.get("type") == "CMMotionRemover" for f in child(system, "Forces"))
    state = ET.parse(state_xml).getroot()
    vel = np.array([[float(v.get(c, "nan")) for c in "xyz"] for v in child(state, "Velocities")],
                   dtype=np.float64)
    if len(vel) != len(masses):
        raise ValueError("state and system have different particle counts")
    ke = 0.5 * float(np.sum(masses[:, None] * vel * vel))  # amu nm^2 ps^-2 = kJ/mol
    dof = 3 * int(np.count_nonzero(masses)) - n_constraints - (3 if com_removed else 0)
    return {"kinetic_energy_kj_mol": ke, "degrees_of_freedom": dof,
            "temperature_K": 2.0 * ke / (dof * BOLTZMANN_KJ_MOL_K),
            "step": int(state.get("stepCount", -1)), "time_ps": float(state.get("time", "nan"))}


def rmsd_to_reference(universe: Any, selection: str,
                      reference: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """Per-frame RMSD (Å) of ``selection`` to ``reference`` (same atoms, same order)
    after least-squares superposition on those atoms. Coordinates must be whole."""
    atoms = universe.select_atoms(selection)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (len(atoms), 3):
        raise ValueError(f"reference has shape {ref.shape}, selection has {len(atoms)} atoms")
    out = []
    for _ in universe.trajectory:
        x = atoms.positions.astype(float)
        rotation, pc, qc = kabsch(x, ref)
        moved = superpose(x, rotation, pc, qc)
        out.append(float(np.sqrt(np.mean(np.sum((moved - ref) ** 2, axis=1)))))
    return np.asarray(out)


def pair_distance_series(universe: Any, pairs: Sequence[tuple[str, str, str]]) -> pd.DataFrame:
    """Per-frame distances (Å) for named atom pairs.

    ``pairs`` holds ``(label, selection_1, selection_2)``, each selection naming
    exactly one atom. Distances are direct (molecules must be whole): use this for
    intramolecular pairs such as base-pair N1–N3 or C5–C5A bonds.
    """
    resolved = []
    for label, s1, s2 in pairs:
        a, b = universe.select_atoms(s1), universe.select_atoms(s2)
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f"{label}: selections must match one atom each ({len(a)}, {len(b)})")
        resolved.append((label, a[0].index, b[0].index))
    rows = []
    for ts in universe.trajectory:
        pos = universe.atoms.positions
        row: dict[str, float] = {"frame": ts.frame, "time_ps": float(ts.time)}
        for label, i, j in resolved:
            row[label] = float(np.linalg.norm(pos[i] - pos[j]))
        rows.append(row)
    return pd.DataFrame(rows)


def bulk_water_density(universe: Any, solute_selection: str, water_oxygen_selection: str,
                       exclusion_A: float = 10.0, grid_spacing_A: float = 2.0,
                       step: int = 1) -> pd.DataFrame:
    """Water density (molecules/nm³ and g/mL) in the region > ``exclusion_A`` from the solute.

    The bulk region's volume comes from a grid over the (triclinic) cell. Grid
    points and water oxygens are both classified with minimum-image distances,
    so the count and the volume use the same definition. This tests the water
    model's density without the solute and ions dominating the mass.
    """
    from MDAnalysis.lib.distances import capped_distance
    from MDAnalysis.lib.mdamath import triclinic_vectors

    solute = universe.select_atoms(solute_selection)
    oxygens = universe.select_atoms(water_oxygen_selection)
    rows = []
    for ts in universe.trajectory[::step]:
        box = ts.dimensions
        vectors = triclinic_vectors(box).astype(float)
        n = np.maximum(1, np.ceil(np.linalg.norm(vectors, axis=1) / grid_spacing_A)).astype(int)
        frac = np.stack(np.meshgrid(*[(np.arange(k) + 0.5) / k for k in n], indexing="ij"), -1).reshape(-1, 3)
        grid = (frac @ vectors).astype(np.float32)
        near_grid = capped_distance(grid, solute.positions, exclusion_A, box=box, return_distances=False)
        bulk_fraction = 1.0 - len(np.unique(near_grid[:, 0])) / len(grid)
        near_water = capped_distance(oxygens.positions, solute.positions, exclusion_A, box=box,
                                     return_distances=False)
        n_bulk = len(oxygens) - len(np.unique(near_water[:, 0]))
        volume_nm3 = abs(float(np.linalg.det(vectors))) / 1000.0
        bulk_volume = bulk_fraction * volume_nm3
        number_density = n_bulk / bulk_volume
        rows.append({"frame": ts.frame, "time_ps": float(ts.time), "bulk_volume_nm3": bulk_volume,
                     "bulk_waters": n_bulk, "molecules_per_nm3": number_density,
                     "g_per_ml": number_density * WATER_G_MOL / AVOGADRO * 1e21})
    return pd.DataFrame(rows)
