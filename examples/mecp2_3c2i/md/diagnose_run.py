"""Technical diagnostics and independent neurodna validation for one 3C2I MD run.

Recomputes everything from the raw outputs of ``neurodna-md run``: logs, saved
states, the XTC, ``simulation.json`` and the prepared system. With no arguments
it analyses the laptop micro-pilot:

    OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 python examples/mecp2_3c2i/md/diagnose_run.py

For another run (e.g. the Tier A / Tier B runs in ``next_stage/``):

    python examples/mecp2_3c2i/md/diagnose_run.py --run-dir RUN --out OUT [--prepared-dir PREP]
        [--condition mCpG|CpG] [--burn-in-ps PS] [--stride N]

Writes ``diagnostics.json``, ``primary_pair_distances.csv`` and PNG figures to
``--out``, plus neurodna's own tables (``neurodna-md analyze``) to ``--tables``.

Every quantity is a diagnostic of the software, the integration or physical
plausibility. None establishes equilibration, convergence or biology.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import MDAnalysis as mda  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from MDAnalysis.lib.distances import distance_array  # noqa: E402
from MDAnalysis.lib.mdamath import box_volume  # noqa: E402

from neurodna.fetch import sha256_file  # noqa: E402
from neurodna.md import qc  # noqa: E402
from neurodna.md.analyze import analyze_run, load_run, prepared_dir_of  # noqa: E402

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "experiment" / "analysis_plan.json"
CUTOFF = 4.5
WC_MAX = 3.2  # design QC criterion (docs/experiment_design.md section 8)
MCPG_POSITIONS = {("B", 8), ("B", 9), ("C", 33), ("C", 34)}
INK, MUTED, BLUE, GRID = "#1f1f1e", "#6b6b66", "#2a78d6", "#d9d9d6"
ORANGE = "#eb6834"  # second categorical slot, used only to highlight


def style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5)


def stage_lines(ax: Any, bounds: list[tuple[float, str]]) -> None:
    for t, name in bounds:
        ax.axvline(t, color=MUTED, linewidth=0.6, linestyle=":")
        ax.text(t, 1.0, f" {name}", transform=ax.get_xaxis_transform(), fontsize=6, color=MUTED,
                va="top", ha="left", rotation=0)


def resnum(label: str) -> int:
    match = re.search(r"(-?\d+)$", label)
    if match is None:
        raise ValueError(f"no residue number in {label!r}")
    return int(match.group(1))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, default=HERE / "output" / "micro_pilot")
    ap.add_argument("--prepared-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "pilot_analysis")
    ap.add_argument("--tables", type=Path, default=None, help="default: <run-dir>_analysis")
    ap.add_argument("--condition", choices=("mCpG", "CpG"), default=None, help="default: from the DNA residues")
    ap.add_argument("--burn-in-ps", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1, help="frame stride for the bulk-water density only")
    args = ap.parse_args(argv)
    run, out = args.run_dir, args.out
    out.mkdir(parents=True, exist_ok=True)
    prep_dir = prepared_dir_of(run, args.prepared_dir)
    tables_dir = args.tables or run.with_name(run.name + "_analysis")
    meta = json.loads((run / "simulation.json").read_text())
    progress = json.loads((run / "progress.json").read_text())
    prep = json.loads((prep_dir / "preparation.json").read_text())
    plan = json.loads(PLAN.read_text())
    proto = meta["protocol"]
    dt_ps = proto["timestep_fs"] / 1000.0
    eq_ps = sum(s["steps"] for s in proto["equilibration"]) * dt_ps
    label = f"{proto['name']}, {eq_ps:g} ps equilibration + {proto['production_steps'] * dt_ps:g} ps production"
    d: dict[str, Any] = {"run": str(run.resolve()), "label": label,
                         "note": "technical diagnostics only; not equilibration, convergence or biology"}

    # ------------------------------------------------------------ completion / integrity
    d["completion"] = {
        "status": meta["status"], "completed_stages": progress["completed"],
        "production_steps_done": progress["production_steps_done"],
        "protocol_production_steps": proto["production_steps"],
        "trajectory_sha256_matches_metadata": sha256_file(run / "production.xtc") == meta["production"]["sha256"],
        "prepared_files_match_preparation_json": all(
            sha256_file(prep_dir / prep["files"][k]["name"]) == prep["files"][k]["sha256"]
            for k in ("prepared_pdb", "system_xml")),
        "topology_is_prepared_pdb": sha256_file(run / "topology.pdb") == prep["files"]["prepared_pdb"]["sha256"],
        "stage_checkpoints": sorted(p.name for p in (run / "stages").glob("*.chk")),
        "sessions": [{"name": s["name"], "resumed_from_step": s.get("resumed_from_step"),
                      "wall_time_s": s.get("wall_time_s"), "ns_per_day": s.get("ns_per_day")}
                     for s in meta["stages"]],
    }

    # ------------------------------------------------------------------ thermodynamics
    thermo = qc.thermo_table(run)
    prod = thermo[thermo.stage == "production"]
    prod = prod[prod.time_ps > args.burn_in_ps] if args.burn_in_ps > 0 else prod
    finite = bool(np.isfinite(thermo[["potential_kj_mol", "temperature_K", "volume_nm3",
                                      "density_g_ml"]].to_numpy()).all())
    kin_prod = qc.kinetic_temperature(run / "stages" / "production.xml", prep_dir / "system.xml")
    d["thermodynamics"] = {
        "all_logged_values_finite": finite,
        "rows": {"equilibration": int((thermo.stage != "production").sum()),
                 "production": int((thermo.stage == "production").sum())},
        "minimization": {k: v for k, v in meta["stages"][0].items() if k.startswith("potential")},
        "production_window_ps": [float(prod.time_ps.iloc[0]), float(prod.time_ps.iloc[-1])],
        "production": {q: qc.drift_summary(prod[q], prod.time_ps) for q in
                       ("potential_kj_mol", "temperature_K", "volume_nm3", "density_g_ml")},
        "temperature_sd_expected_canonical_K": float(proto["temperature_K"]
                                                     * np.sqrt(2.0 / kin_prod["degrees_of_freedom"])),
        "equilibration_temperature_first_last_K": [float(thermo.temperature_K.iloc[0]),
                                                   float(thermo[thermo.stage != "production"].temperature_K.iloc[-1])],
        "pressure": "not logged: the Monte Carlo barostat samples volume at the set pressure; OpenMM's reporter "
                    "does not report instantaneous pressure, whose fluctuations in a ~50k-atom box are hundreds "
                    "of bar. Volume and density are used instead.",
    }
    npt = thermo[thermo.ensemble == "NPT"]
    changes = np.diff(npt.volume_nm3.to_numpy()) != 0
    d["barostat"] = {
        "npt_log_intervals": int(changes.size),
        "intervals_with_volume_change_fraction": float(changes.mean()) if changes.size else None,
        "attempts_per_interval": proto["report_interval_steps"] // proto["barostat_interval_steps"],
        "volume_npt_start_end_nm3": [float(npt.volume_nm3.iloc[0]), float(npt.volume_nm3.iloc[-1])],
        "note": "per-attempt acceptance is not recorded; this is the fraction of log intervals in which the "
                "volume changed. Every resume restarts the barostat's adaptive trial step size (see md/README.md).",
    }
    kin = []
    for name in [s["name"] for s in proto["equilibration"]] + ["production"]:
        path = run / "stages" / f"{name}.xml"
        if not path.exists():
            continue
        k = qc.kinetic_temperature(path, prep_dir / "system.xml")
        rep = thermo[(thermo.stage == name) & (thermo.step == k["step"])]
        kin.append({"state": name, "step": k["step"], "kinetic_energy_kj_mol": round(k["kinetic_energy_kj_mol"], 1),
                    "degrees_of_freedom": k["degrees_of_freedom"], "T_from_velocities_K": round(k["temperature_K"], 3),
                    "T_reported_K": round(float(rep.temperature_K.iloc[0]), 3) if len(rep) else None})
    d["kinetic_temperature_check"] = kin

    # --------------------------------------------------------------------- trajectory
    cx = load_run(run, prep_dir)  # checks checksums, frame count and spacing
    u = cx.universe
    times = cx.frame_times()
    xtc_times = np.array([ts.time for ts in u.trajectory])
    dims = np.array([ts.dimensions.copy() for ts in u.trajectory])
    d["trajectory"] = {
        "frames": int(len(times)),
        "expected_frames": proto["production_steps"] // proto["report_interval_steps"],
        "first_last_ps": [float(times[0]), float(times[-1])],
        "strictly_increasing": bool(np.all(np.diff(times) > 0)),
        "spacing_ps": sorted(set(np.round(np.diff(times), 6).tolist())),
        "neurodna_times_equal_xtc_times": bool(np.allclose(times, xtc_times)),
        "box_angles_constant": bool(np.allclose(dims[:, 3:], dims[0, 3:])),
        "box_lengths_isotropic": bool(np.allclose(dims[:, :3] / dims[:, :1], 1.0)),
        "box_angles_deg": [float(x) for x in dims[0, 3:]],
        "box_length_A_first_last": [float(dims[0, 0]), float(dims[-1, 0])],
    }

    # ----------------------------------------------------------- whole molecules / PBC
    protein, dna = cx.protein, cx.dna
    links = []
    for a in dna.select_atoms("name O3'"):
        nxt = dna.select_atoms(f"chainID {a.chainID} and resid {a.resid + 1} and name P")
        if len(nxt):
            links.append((a.index, nxt[0].index))
    links_arr = np.array(links)
    ow, h1, h2 = (u.select_atoms(f"resname HOH and name {n}") for n in ("O", "H1", "H2"))
    dna_res = [r.atoms for r in dna.residues]
    com_pd, link_max, oh_max, res_extent = [], 0.0, 0.0, 0.0
    for _ in u.trajectory:
        pos = u.atoms.positions
        link_max = max(link_max, float(np.linalg.norm(pos[links_arr[:, 0]] - pos[links_arr[:, 1]], axis=1).max()))
        oh_max = max(oh_max, float(np.linalg.norm(ow.positions - h1.positions, axis=1).max()),
                     float(np.linalg.norm(ow.positions - h2.positions, axis=1).max()))
        res_extent = max(res_extent, max(float(np.linalg.norm(r.positions - r.positions.mean(0), axis=1).max())
                                         for r in dna_res))
        com_pd.append(float(np.linalg.norm(protein.center_of_mass() - dna.center_of_mass())))
    d["whole_molecules"] = {
        "protein": "neurodna whole-molecule check (backbone bonds, residue extent, chain images) passed in "
                   "backbone_rmsd and rmsf for every frame",
        "dna_O3prime_P_links": int(len(links)), "dna_O3prime_P_max_A": round(link_max, 3),
        "dna_max_atom_to_residue_centroid_A": round(res_extent, 3),
        "water_O_H_max_A": round(oh_max, 3),
        "water_O_H_note": "rigid water (0.9572 A); XTC stores coordinates to 0.01 A, so up to ~0.017 A deviation "
                          "is quantisation",
        "protein_dna_com_distance_A": {"first": round(com_pd[0], 2), "last": round(com_pd[-1], 2),
                                       "min": round(min(com_pd), 2), "max": round(max(com_pd), 2)},
        "wrapping": "none: XTC written with enforcePeriodicBox=False",
    }

    # ------------------------------------------------------------------ structure
    ref_u = mda.Universe(str(prep_dir / "prepared.pdb"))
    bb_sel = "chainID A and not resname HOH and name N CA C O"
    core_sel = ("((chainID B and resid 4:18) or (chainID C and resid 24:38)) and not resname HOH "
                "and not element H")
    rmsd_bb_ref = qc.rmsd_to_reference(u, bb_sel, ref_u.select_atoms(bb_sel).positions)
    rmsd_dna_ref = qc.rmsd_to_reference(u, core_sel, ref_u.select_atoms(core_sel).positions)
    rmsd_bb_first = cx.backbone_rmsd().rmsd_A.to_numpy()
    rmsf = cx.rmsf("name CA")
    wc_pairs, wc_highlight = [], []
    for i in range(2, 21):
        j = 42 - i
        rb = u.select_atoms(f"chainID B and resid {i} and not resname HOH").residues[0]
        rc = u.select_atoms(f"chainID C and resid {j} and not resname HOH").residues[0]

        def wc_atom(r: Any) -> str:
            return "N1" if r.resname in ("DA", "DG") else "N3"

        lab = f"B:{rb.resname}{i}-C:{rc.resname}{j}"
        wc_pairs.append((lab, f"chainID B and resid {i} and not resname HOH and name {wc_atom(rb)}",
                         f"chainID C and resid {j} and not resname HOH and name {wc_atom(rc)}"))
        if ("B", i) in MCPG_POSITIONS:
            wc_highlight.append(lab)
    wc = qc.pair_distance_series(u, wc_pairs)
    wc_labels = [p[0] for p in wc_pairs]
    d["structure"] = {
        "protein_backbone_rmsd_to_prepared_A": qc.drift_summary(rmsd_bb_ref, times),
        "protein_backbone_rmsd_to_first_frame_A": {"max": float(rmsd_bb_first.max()),
                                                   "last": float(rmsd_bb_first[-1])},
        "dna_core_heavy_rmsd_to_prepared_A": qc.drift_summary(rmsd_dna_ref, times),
        "dna_core_selection": core_sel,
        "watson_crick_fraction_N1N3_le_3.2A": {lab: float((wc[lab] <= WC_MAX).mean()) for lab in wc_labels},
        "watson_crick_max_A": {lab: round(float(wc[lab].max()), 2) for lab in wc_labels},
        "watson_crick_min_fraction": float(min((wc[lab] <= WC_MAX).mean() for lab in wc_labels)),
        "ca_rmsf_A": {"median": float(rmsf.rmsf_A.median()), "max": float(rmsf.rmsf_A.max()),
                      "max_residue": str(rmsf.loc[rmsf.rmsf_A.idxmax(), "label"])},
    }

    # ------------------------------------------------------------ bulk water density
    bulk = qc.bulk_water_density(u, "chainID A B C and not resname HOH", "resname HOH and name O",
                                 exclusion_A=10.0, grid_spacing_A=2.0, step=args.stride)
    volumes = np.array([box_volume(x) for x in dims[::args.stride]]) / 1000.0
    d["bulk_water_density"] = {
        "definition": "water O farther than 10 A (minimum image) from any protein/DNA atom; region volume from a "
                      "2 A grid over the triclinic cell classified the same way; ions remain in the region",
        "g_per_ml": qc.drift_summary(bulk.g_per_ml, bulk.time_ps),
        "molecules_per_nm3_mean": float(bulk.molecules_per_nm3.mean()),
        "bulk_volume_fraction_mean": float((bulk.bulk_volume_nm3.to_numpy() / volumes).mean()),
        "ions_total": int(len(u.select_atoms("resname NA CL"))),
    }

    # ----------------------------------------------------- 5-methylcytosine / condition
    dres = cx.dna_residues()
    five_mc = dres.loc[dres.modification == "5mC", "label"].tolist()
    condition = args.condition or ("mCpG" if five_mc else "CpG")
    d["condition"] = condition
    if condition == "mCpG":
        methyl = qc.pair_distance_series(u, [
            (f"{c}{r} C5-C5A", f"chainID {c} and resid {r} and name C5", f"chainID {c} and resid {r} and name C5A")
            for c, r in (("B", 8), ("C", 33))] + [
            (f"{c}{r} C5A-{h}", f"chainID {c} and resid {r} and name C5A", f"chainID {c} and resid {r} and name {h}")
            for c, r in (("B", 8), ("C", 33)) for h in ("H5A1", "H5A2", "H5A3")])
        hcols = [c for c in methyl.columns if "C5A-H" in c]
        d["methylcytosine"] = {
            "neurodna_5mC_residues": five_mc,
            "C5_C5A_A": {c: [round(float(methyl[c].min()), 3), round(float(methyl[c].max()), 3)]
                         for c in methyl.columns if c.endswith("C5-C5A")},
            "C5A_H_A_range": [round(float(methyl[hcols].min().min()), 3), round(float(methyl[hcols].max().max()), 3)],
            "template_in_preparation": {k: v for k, v in prep["nucleotide_templates"].items() if "5CM" in k},
        }
    else:
        c5a = u.select_atoms("chainID B C and not resname HOH and name C5A")
        d["methylcytosine"] = {"neurodna_5mC_residues": five_mc, "methyl_carbons_present": int(len(c5a)),
                               "expected": "none (CpG condition)"}

    # ------------------------------------------------------------ charged termini
    dna_phos_o = dna.select_atoms("name OP1 OP2")
    first_res, last_res = int(protein.residues.resids.min()), int(protein.residues.resids.max())
    nterm = u.select_atoms(f"chainID A and resid {first_res} and name N")
    cterm = u.select_atoms(f"chainID A and resid {last_res} and name O OXT")
    term: dict[str, list[float]] = {"n_to_phosphate_O": [], "c_to_phosphate_O": []}
    for ts in u.trajectory:
        term["n_to_phosphate_O"].append(float(distance_array(nterm.positions, dna_phos_o.positions,
                                                             box=ts.dimensions).min()))
        term["c_to_phosphate_O"].append(float(distance_array(cterm.positions, dna_phos_o.positions,
                                                             box=ts.dimensions).min()))
    d["charged_termini"] = {"residues": [first_res, last_res],
                            **{k: {"min": round(min(v), 2), "max": round(max(v), 2), "last": round(v[-1], 2)}
                               for k, v in term.items()}}

    # ---------------------------------------------------------- neurodna analysis
    paths = analyze_run(run, tables_dir, cutoff=CUTOFF, burn_in_ps=args.burn_in_ps, prepared_dir=prep_dir)
    occ = pd.read_csv(paths["contact_occupancy"])
    eps = pd.read_csv(paths["contact_episodes"])
    summary = json.loads(paths["summary"].read_text())
    start = summary["frame_window"]["first_frame"]
    primary = [(p["protein"], p["labels"][condition], p) for p in plan["primary_pairs"]["pairs"]]
    omap = dict(zip(zip(occ.protein_label, occ.dna_label), occ.occupancy))
    d["neurodna"] = {
        "analysis_summary": {k: summary[k] for k in ("inputs", "selections", "frame_window", "cutoff_A",
                                                   "minimum_image", "time_source", "software", "units")},
        "pairs_in_contact_any_frame": int(len(occ)), "episodes": int(len(eps)), "primary": [],
    }
    for prot, dl, p in primary:
        e = eps[(eps.protein_label == prot) & (eps.dna_label == dl)]
        d["neurodna"]["primary"].append({
            "pair": f"{prot}-{dl}", "occupancy": round(float(omap.get((prot, dl), 0.0)), 4),
            "episodes": int(len(e)), "left_censored": int(e.left_censored.sum()),
            "right_censored": int(e.right_censored.sum()),
            "crystal_A": p["crystal_min_distance_A"], "methyl_mediated": p["methyl_mediated_in_crystal"]})

    # ------------------------------------------- independent validation (brute force)
    dist = cx.min_distances(12.0, start=start)
    nd: dict[str, dict[int, float]] = {}
    for r in dist.itertuples():
        nd.setdefault(f"{r.protein_label}-{r.dna_label}", {})[int(r.frame)] = float(r.min_distance_A)
    groups = {}
    for prot, dl, _ in primary:
        a = cx.protein_heavy.select_atoms(f"chainID {prot.split(':')[0]} and resid {resnum(prot)}")
        b = cx.dna_heavy.select_atoms(f"chainID {dl.split(':')[0]} and resid {resnum(dl)}")
        if len(a.residues) != 1 or len(b.residues) != 1:
            raise RuntimeError(f"{prot}-{dl}: selection does not match one residue each")
        groups[f"{prot}-{dl}"] = (a, b)
    bf: dict[str, list[float]] = {k: [] for k in groups}
    direct: dict[str, list[float]] = {k: [] for k in groups}
    for ts in u.trajectory[start:]:
        for key, (a, b) in groups.items():
            bf[key].append(float(distance_array(a.positions, b.positions, box=ts.dimensions).min()))
            direct[key].append(float(distance_array(a.positions, b.positions).min()))
    frames = np.arange(start, len(times))
    max_diff = max_pbc_diff = occ_diff = 0.0
    ep_mismatch = []
    for idx, key in enumerate(bf):
        series = np.array(bf[key])
        neuro = np.array([nd.get(key, {}).get(int(f), np.inf) for f in frames])
        within = series <= 12.0
        if within.any():
            max_diff = max(max_diff, float(np.abs(series[within] - neuro[within]).max()))
        if np.any(np.isfinite(neuro) & ~within):
            ep_mismatch.append(f"{key}: neurodna lists a distance > 12 A")
        max_pbc_diff = max(max_pbc_diff, float(np.abs(series - np.array(direct[key])).max()))
        in_contact = series <= CUTOFF
        prot, dl, _ = primary[idx]
        occ_diff = max(occ_diff, abs(float(in_contact.mean()) - float(omap.get((prot, dl), 0.0))))
        runs: list[tuple[int, int]] = []
        begin = None  # plain loop, independent of neurodna.interactions.contact_runs
        for i, c in enumerate(in_contact):
            if c and begin is None:
                begin = i
            if not c and begin is not None:
                runs.append((int(frames[begin]), int(frames[i - 1])))
                begin = None
        if begin is not None:
            runs.append((int(frames[begin]), int(frames[-1])))
        e = eps[(eps.protein_label + "-" + eps.dna_label) == key]
        mine = list(zip(e.first_frame.astype(int).tolist(), e.last_frame.astype(int).tolist()))
        censor_ok = all(bool(row.left_censored) == (row.first_frame == frames[0])
                        and bool(row.right_censored) == (row.last_frame == frames[-1]) for row in e.itertuples())
        if runs != mine or not censor_ok:
            ep_mismatch.append(key)
    u.trajectory[0]
    o3p = (u.select_atoms("chainID B and resid 8 and name O3'")[0].index,
           u.select_atoms("chainID B and resid 9 and name P")[0].index)
    d["independent_validation"] = {
        "pairs_checked": len(bf), "frames_checked": int(len(frames)),
        "min_distance_max_abs_diff_A": max_diff,
        "minimum_image_vs_direct_max_abs_diff_A": max_pbc_diff,
        "occupancy_max_abs_diff": occ_diff,
        "episode_boundaries_or_censoring_mismatches": ep_mismatch,
        "units_check_B8_O3prime_to_B9_P_A": round(float(np.linalg.norm(u.atoms.positions[o3p[0]]
                                                                      - u.atoms.positions[o3p[1]])), 3),
        "residue_identities_unique": len(set(cx.protein_residue_keys + cx.dna_residue_keys))
        == len(cx.protein_residue_keys) + len(cx.dna_residue_keys),
        "method": "MDAnalysis distance_array over neurodna's heavy-atom groups, per frame, with and without the "
                  "periodic box; contact runs recomputed by a plain loop; censoring = run touches the first or "
                  "last analysed frame",
    }

    # -------------------------------------------------------------------- figures
    bounds, t0 = [], 0.0
    for s in proto["equilibration"]:
        bounds.append((t0, s["name"].replace("_restrained", " r").replace("npt_", "").replace("nvt", "NVT")))
        t0 += s["steps"] * dt_ps
    bounds.append((t0, "production"))
    t = thermo.t_dyn_ps
    xlab = "time since start of dynamics (ps)"

    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.plot(t, thermo.potential_kj_mol / 1000, color=BLUE, linewidth=1.5)
    stage_lines(ax, bounds)
    style(ax, f"Potential energy ({label}) - diagnostic only", xlab, "potential energy (MJ/mol)")
    fig.savefig(out / "fig1_potential_energy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.plot(t, thermo.temperature_K, color=BLUE, linewidth=1.5)
    ax.axhline(proto["temperature_K"], color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(t.iloc[-1], proto["temperature_K"], f" {proto['temperature_K']:g} K target", fontsize=6,
            color=MUTED, va="bottom", ha="right")
    stage_lines(ax, bounds)
    style(ax, "Temperature (reporter) - diagnostic only", xlab, "temperature (K)")
    fig.savefig(out / "fig2_temperature.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, thermo.volume_nm3, color=BLUE, linewidth=1.5)
    axes[1].plot(t, thermo.density_g_ml, color=BLUE, linewidth=1.5)
    for a in axes:
        stage_lines(a, bounds)
    style(axes[0], "Box volume (barostat in NPT stages only)", "", "volume (nm³)")
    style(axes[1], "System density (protein + DNA + water + ions)", xlab, "density (g/mL)")
    fig.savefig(out / "fig3_volume_density.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.plot(times, rmsd_bb_ref, color=BLUE, linewidth=1.5)
    ax.plot(times, rmsd_bb_first, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(times[-1], rmsd_bb_ref[-1], " vs prepared (crystal) model", fontsize=6, color=BLUE, va="center")
    ax.text(times[-1], rmsd_bb_first[-1], " vs first production frame", fontsize=6, color=MUTED, va="center")
    ax.set_xlim(0, times[-1] * 1.45)
    style(ax, "Protein backbone RMSD - diagnostic only (flattening is not convergence)", "production time (ps)",
          "RMSD (Å)")
    fig.savefig(out / "fig4_protein_rmsd.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True, constrained_layout=True)
    axes[0].plot(times, rmsd_dna_ref, color=BLUE, linewidth=1.5)
    style(axes[0], "DNA core duplex heavy-atom RMSD vs prepared model (B4-B18 / C24-C38)", "", "RMSD (Å)")
    for lab in wc_labels:
        hl = lab in wc_highlight
        axes[1].plot(times, wc[lab], color=ORANGE if hl else GRID, linewidth=1.5 if hl else 0.8, zorder=3 if hl else 1)
    axes[1].axhline(WC_MAX, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].text(times[-1], WC_MAX, f" {WC_MAX} Å", fontsize=6, color=MUTED, va="bottom", ha="right")
    axes[1].text(0.0, 1.02, "orange: CpG-step pairs B8·C34, B9·C33; grey: the other 17 pairs",
                 transform=axes[1].transAxes, fontsize=6, color=MUTED)
    style(axes[1], "Watson-Crick N1-N3 distances, all 19 base pairs\n", "production time (ps)", "distance (Å)")
    fig.savefig(out / "fig5_dna_structure.png", dpi=160)
    plt.close(fig)

    ptimes = times[start:]
    fig, axes = plt.subplots(4, 5, figsize=(10, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, (prot, dl, p) in zip(axes.flat, primary):
        ax.plot(ptimes, bf[f"{prot}-{dl}"], color=BLUE, linewidth=1.0)
        ax.axhline(CUTOFF, color=MUTED, linewidth=0.6, linestyle="--")
        ax.scatter([ptimes[0]], [p["crystal_min_distance_A"]], color=INK, s=8, zorder=3)
        ax.set_title(f"{prot.split(':')[1]}-{dl}{' *' if p['methyl_mediated_in_crystal'] else ''}",
                     fontsize=7, color=INK)
        ax.tick_params(labelsize=6, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(primary):]:
        ax.axis("off")
    fig.suptitle("All 19 preregistered primary pairs: minimum heavy-atom distance (Å) vs production time (ps). "
                 "Dashed: 4.5 Å cutoff; dot: crystal value; * methyl-mediated in crystal. DIAGNOSTIC ONLY",
                 fontsize=8, color=INK, x=0.01, ha="left")
    fig.savefig(out / "fig6_primary_pair_distances.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    labels = [f"{x['pair']}{' *' if x['methyl_mediated'] else ''}" for x in d["neurodna"]["primary"]]
    vals = [x["occupancy"] for x in d["neurodna"]["primary"]]
    ax.barh(range(len(vals))[::-1], vals, color=BLUE, height=0.7)
    ax.set_yticks(range(len(vals))[::-1], labels, fontsize=6)
    ax.set_xlim(0, 1)
    style(ax, f"Primary-pair contact occupancy (≤ {CUTOFF} Å), {len(frames)} correlated frames / "
              f"{ptimes[-1] - ptimes[0] + (ptimes[1] - ptimes[0]):g} ps\n"
              "DIAGNOSTIC ONLY - not a persistence estimate; * methyl-mediated in crystal",
          "fraction of analysed frames in contact", "")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    fig.savefig(out / "fig7_primary_occupancy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0), constrained_layout=True)
    ax.plot(times, term["n_to_phosphate_O"], color=BLUE, linewidth=1.5)
    ax.plot(times, term["c_to_phosphate_O"], color=ORANGE, linewidth=1.5)
    ax.text(times[-1], term["n_to_phosphate_O"][-1], f" N-terminal N (res {first_res})", fontsize=6, color=BLUE,
            va="center")
    ax.text(times[-1], term["c_to_phosphate_O"][-1], f" C-terminal O/OXT (res {last_res})", fontsize=6,
            color=ORANGE, va="center")
    ax.set_xlim(0, times[-1] * 1.4)
    style(ax, "Charged termini at truncation points: distance to nearest DNA phosphate O", "production time (ps)",
          "distance (Å)")
    fig.savefig(out / "fig8_termini_to_dna.png", dpi=160)
    plt.close(fig)

    (out / "diagnostics.json").write_text(json.dumps(d, indent=1, default=float) + "\n")
    pd.DataFrame({"time_ps": ptimes, **bf}).to_csv(out / "primary_pair_distances.csv", index=False)
    return d


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: result[k] for k in ("completion", "trajectory", "independent_validation")}, indent=1,
                     default=float))
