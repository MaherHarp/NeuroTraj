"""Render docs/assets/neurotraj.gif, the README animation, from the 3C2I micro-pilot.

Every frame is a real frame of the OpenMM micro-pilot trajectory (20 ps of
unrestrained NPT, 0.25 ps per frame; see examples/mecp2_3c2i/md/). Coordinates
are not smoothed, exaggerated or interpolated. The only additions are a camera
rotation, motion trails built from the preceding real frames, and glow. The
trajectory is played forwards then backwards so the loop is seamless.

The trajectory is git-ignored, so this script only runs where the micro-pilot
was simulated::

    python docs/make_readme_gif.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "examples" / "mecp2_3c2i" / "md" / "output" / "micro_pilot"
OUT = ROOT / "docs" / "assets" / "neurotraj.gif"

W, H, DPI = 800, 400, 100
TRAIL = 3  # preceding real frames drawn as fading trails
FRAME_MS = 40
COLORS = 112
BG = "#05060f"
MUTED = "#8b93b8"
PROTEIN_CMAP = LinearSegmentedColormap.from_list("protein", ["#00e5ff", "#5b6cff", "#c04dff"])
STRANDS = {"B": "#ff3d9a", "C": "#ff9f1c"}
METHYL = "#ffe14d"
ARG = "#39ff88"
DNA = "DA DC DG DT 5CM"
# Measured in this trajectory: Arg111 holds the 5mC methyl of B8, Arg133 that of C33.
PAIRS = [(111, "B", 8), (133, "C", 33)]
PARTNER_OFFSET = 42  # strand B residue i pairs with strand C residue 42 - i (C1'-C1' about 10.8 A)


def load() -> dict[str, mda.AtomGroup]:
    warnings.filterwarnings("ignore")
    u = mda.Universe(str(RUN / "topology.pdb"), str(RUN / "production.xtc"))
    b = u.select_atoms("segid B and name C1'")
    c_by_resid = {a.resid: a for a in u.select_atoms("segid C and name C1'")}
    paired = [(x, c_by_resid[PARTNER_OFFSET - x.resid]) for x in b if PARTNER_OFFSET - x.resid in c_by_resid]
    groups = {
        "protein": u.select_atoms("segid A and protein and not name H*"),
        "ca": u.select_atoms("segid A and name CA"),
        "dna": u.select_atoms(f"segid B C and resname {DNA} and not name H*"),
        "methyl": u.select_atoms("resname 5CM and name C5A"),
        "rung_b": mda.AtomGroup([x for x, _ in paired]),
        "rung_c": mda.AtomGroup([y for _, y in paired]),
    }
    for segid in STRANDS:
        groups[f"p{segid}"] = u.select_atoms(f"segid {segid} and resname {DNA} and name P")
    for resid, _, _ in PAIRS:
        groups[f"arg{resid}"] = u.select_atoms(f"segid A and resid {resid} and name NE CZ NH1 NH2")
    return groups


def collect(groups: dict[str, mda.AtomGroup]) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    u = groups["protein"].universe
    frames, times = [], []
    for ts in u.trajectory:
        frames.append({k: v.positions.copy() for k, v in groups.items()})
        times.append(ts.time)
    return frames, np.array(times)


def rotation(angle: float, tilt: float = 0.3) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    ct, st = np.cos(tilt), np.sin(tilt)
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return rx @ ry


def glow_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, width: float) -> None:
    for w, alpha in ((width * 5, 0.07), (width * 2.2, 0.16), (width, 0.95)):
        ax.plot(x, y, color=color, lw=w, alpha=alpha, solid_capstyle="round")


def render(
    frames: list[dict[str, np.ndarray]], times: np.ndarray, groups: dict[str, mda.AtomGroup]
) -> list[Image.Image]:
    center = np.concatenate([frames[0]["protein"], frames[0]["dna"]]).mean(axis=0)
    ca_resids = groups["ca"].resids
    seq = np.searchsorted(ca_resids, groups["protein"].resids).clip(0, len(ca_resids) - 1) / (len(ca_resids) - 1)
    protein_rgba = PROTEIN_CMAP(seq)
    dna_rgba = np.array([to_rgba(STRANDS[s]) for s in groups["dna"].segids])
    ca_rgba = PROTEIN_CMAP(np.linspace(0, 1, len(ca_resids) - 1))
    distances = np.array([[np.linalg.norm(f[f"arg{r}"] - f["methyl"][i], axis=1).min()
                           for i, (r, _, _) in enumerate(PAIRS)] for f in frames])

    order = list(range(len(frames))) + list(range(len(frames) - 2, 0, -1))
    images = []
    for n, idx in enumerate(order):
        R = rotation(2 * np.pi * n / len(order))

        def project(xyz: np.ndarray, R: np.ndarray = R) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            p = (xyz - center) @ R.T
            return p[:, 0] - 9, p[:, 1] - 1.5, np.clip((p[:, 2] + 30) / 60, 0.15, 1.0)

        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        ax = fig.add_axes((0, 0, 1, 1), facecolor=BG)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-25, 25)
        ax.set_aspect("equal")
        ax.axis("off")

        for back in range(TRAIL, 0, -1):
            f_old = frames[order[max(0, n - back)]]
            for key, rgba in (("protein", protein_rgba), ("dna", dna_rgba)):
                x, y, _ = project(f_old[key])
                ax.scatter(x, y, s=4, c=rgba, alpha=0.08 * (1 - back / (TRAIL + 1)), linewidths=0)

        f = frames[idx]
        for key, rgba, size in (("protein", protein_rgba, 9), ("dna", dna_rgba, 9)):
            x, y, depth = project(f[key])
            z = np.argsort(depth)
            core = rgba[z].copy()
            core[:, 3] = 0.3 + 0.7 * depth[z]
            ax.scatter(x[z], y[z], s=size * 4 * depth[z], c=rgba[z], alpha=0.06, linewidths=0)
            ax.scatter(x[z], y[z], s=size * depth[z], c=core, linewidths=0)

        bx, by, _ = project(f["rung_b"])
        cx, cy, _ = project(f["rung_c"])
        for w, alpha in ((4, 0.05), (1.0, 0.3)):
            ax.add_collection(LineCollection(np.stack([np.c_[bx, by], np.c_[cx, cy]], axis=1),
                                             colors="#ffd6f0", linewidths=w, alpha=alpha))
        for segid, color in STRANDS.items():
            x, y, _ = project(f[f"p{segid}"])
            glow_line(ax, x, y, color, 2.4)

        x, y, _ = project(f["ca"])
        segments = np.stack([np.c_[x[:-1], y[:-1]], np.c_[x[1:], y[1:]]], axis=1)
        for w, alpha in ((9, 0.07), (4, 0.16), (1.8, 0.95)):
            ax.add_collection(LineCollection(segments, colors=ca_rgba, linewidths=w, alpha=alpha,
                                             capstyle="round"))

        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * n / 16)
        mx, my, _ = project(f["methyl"])
        for i, (resid, _, _) in enumerate(PAIRS):
            gx, gy, _ = project(f[f"arg{resid}"])
            nh = int(np.argmin(np.linalg.norm(f[f"arg{resid}"] - f["methyl"][i], axis=1)))
            grip = float(np.clip((5.0 - distances[idx, i]) / 1.8, 0.15, 1.0))
            for w, alpha in ((12, 0.14 * grip), (5, 0.3 * grip), (2.0, grip)):
                ax.plot([gx[nh], mx[i]], [gy[nh], my[i]], color=METHYL, lw=w, alpha=alpha, solid_capstyle="round")
            ax.scatter(gx, gy, s=90, c=ARG, alpha=0.15, linewidths=0)
            ax.scatter(gx, gy, s=16, c=ARG, linewidths=0)
            ax.text(gx.mean() + 1.4, gy.mean() + 1.4, f"R{resid}", color=ARG, fontsize=10,
                    fontweight="bold", family="monospace")
        for size, alpha in ((1400, 0.04 + 0.04 * pulse), (520, 0.1 + 0.1 * pulse), (170, 0.4), (50, 1.0)):
            ax.scatter(mx, my, s=size, c=METHYL, alpha=alpha, linewidths=0)

        ax.text(-49, 22.2, "MeCP2 reading methylated DNA", color="white", fontsize=16, fontweight="bold")
        ax.text(-49, 19.6, "real OpenMM frames  ·  PDB 3C2I  ·  49,302 atoms  ·  5mC methyls in gold",
                color=MUTED, fontsize=8.5, family="monospace")
        ax.text(-49, -24, f"t = {times[idx]:5.2f} ps", color="white", fontsize=12, family="monospace")

        inset = fig.add_axes((0.71, 0.08, 0.27, 0.2), facecolor=BG)
        for i, ((resid, _, _), color) in enumerate(zip(PAIRS, (ARG, "#7dffd0"))):
            inset.plot(times[: idx + 1], distances[: idx + 1, i], color=color, lw=1.2)
            inset.plot(times[idx:], distances[idx:, i], color=color, lw=0.8, alpha=0.2)
            inset.scatter([times[idx]], [distances[idx, i]], s=22, c=METHYL, zorder=3)
            inset.text(0.03 + 0.2 * i, 0.08, f"R{resid}", color=color, fontsize=6.5,
                       transform=inset.transAxes, family="monospace", fontweight="bold")
        inset.set_ylim(2.8, 5.0)
        inset.set_xlim(times[0], times[-1])
        inset.tick_params(colors=MUTED, labelsize=6.5, length=2)
        for side in inset.spines.values():
            side.set_color("#2a2f4a")
        inset.set_title("Arg to 5mC methyl distance (Å)", color=MUTED, fontsize=7, loc="left", pad=3)

        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    return images


def main() -> None:
    groups = load()
    frames, times = collect(groups)
    images = render(frames, times, groups)
    # one shared palette sampled across the loop, so small bright features (the methyls) keep their colour
    samples = images[:: max(1, len(images) // 8)]
    sheet = Image.new("RGB", (W * len(samples), H))
    for k, im in enumerate(samples):
        sheet.paste(im, (W * k, 0))
    palette = sheet.quantize(colors=COLORS, method=Image.Quantize.MAXCOVERAGE)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames_p[0].save(OUT, save_all=True, append_images=frames_p[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"{OUT.relative_to(ROOT)}  {len(frames_p)} frames  {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
