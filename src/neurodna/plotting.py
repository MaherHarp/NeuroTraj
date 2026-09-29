"""Optional matplotlib plots. Requires ``pip install neurodna[plot]``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plotting needs matplotlib: pip install 'neurodna[plot]'") from exc
    return plt


def plot_contact_occupancy(
    occupancy: pd.DataFrame, *, top: int | None = 20, ax: Axes | None = None
) -> Axes:
    """Horizontal bar chart of the output of :meth:`Complex.contact_occupancy`
    (or :meth:`Complex.ensemble_contact_frequency`)."""
    if "occupancy" in occupancy:
        column, xlabel = "occupancy", "Contact occupancy (fraction of frames)"
    elif "model_fraction" in occupancy:
        column, xlabel = "model_fraction", "Contact frequency (fraction of models)"
    else:
        raise ValueError("expected an 'occupancy' or 'model_fraction' column")
    df = occupancy.sort_values(column, ascending=False)
    if top is not None:
        df = df.head(top)
    labels = df["protein_label"] + " – " + df["dna_label"]
    if ax is None:
        plt = _require_matplotlib()
        _, ax = plt.subplots(figsize=(6, 0.3 * len(df) + 1))
    assert ax is not None
    ax.barh(labels[::-1], df[column][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    cutoff = occupancy.attrs.get("cutoff_A")
    if cutoff is not None:
        ax.set_title(f"Geometric protein–DNA contacts (min. heavy-atom distance ≤ {cutoff:g} Å)")
    return ax


# Sequential blue ramp (steps 100 -> 700 of the reference palette): light = far, dark = close.
_BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_INK, _INK_MUTED, _SURFACE = "#1f1f1e", "#6b6b66", "#ffffff"


def plot_contact_map(
    distances: pd.DataFrame,
    *,
    cutoff: float,
    dna_order: list[str] | None = None,
    protein_order: list[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Residue × nucleotide map of minimum heavy-atom distances for one frame.

    Args:
        distances: Output of :meth:`Complex.min_distances` for a single frame.
            Pairs absent from the table (farther than its ``max_distance``) are
            drawn as empty cells.
        cutoff: Contact cutoff (Å). Pairs with distance ≤ cutoff are outlined,
            so contact status is not conveyed by colour alone.
        dna_order: Nucleotide labels in column order (e.g. each strand 5'→3',
            from :meth:`Complex.dna_strands`). Default: chain, then number.
        protein_order: Protein residue labels in row order. Default: residues
            present in ``distances``, by chain then number.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    if distances["frame"].nunique() > 1:
        raise ValueError("plot_contact_map needs distances from a single frame")
    plt = _require_matplotlib()
    if protein_order is None:
        rows = distances.drop_duplicates("protein_label").sort_values(
            ["protein_segid", "protein_chain", "protein_resnum", "protein_icode"])
        protein_order = rows["protein_label"].tolist()
    if dna_order is None:
        cols = distances.drop_duplicates("dna_label").sort_values(
            ["dna_segid", "dna_chain", "dna_resnum", "dna_icode"])
        dna_order = cols["dna_label"].tolist()
    r_index = {label: i for i, label in enumerate(protein_order)}
    c_index = {label: j for j, label in enumerate(dna_order)}
    grid = np.full((len(protein_order), len(dna_order)), np.nan)
    for p, d, v in zip(distances.protein_label, distances.dna_label, distances.min_distance_A):
        if p in r_index and d in c_index:
            grid[r_index[p], c_index[d]] = v

    max_distance = distances.attrs.get("max_distance_A") or float(np.nanmax(grid))
    vmin = float(np.floor(np.nanmin(grid))) if np.isfinite(grid).any() else 0.0
    cmap = LinearSegmentedColormap.from_list("neurodna_blue", _BLUE_RAMP[::-1])
    cmap.set_bad(_SURFACE)
    norm = Normalize(vmin=vmin, vmax=max_distance)

    if ax is None:
        width = max(6.0, 0.28 * len(dna_order) + 2.5)
        height = max(4.0, 0.22 * len(protein_order) + 2.2)
        _, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    assert ax is not None
    mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=norm,
                         edgecolors=_SURFACE, linewidth=1.0)
    for i, j in zip(*np.nonzero(np.nan_to_num(grid, nan=np.inf) <= cutoff)):
        ax.add_patch(Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, fill=False,
                               edgecolor=_INK, linewidth=1.1))

    # Strand boundaries: a surface-coloured gap where the chain changes. The chain is
    # read from the label ("[segid/]chain:resname+number"), so nucleotides with no
    # nearby residue (absent from the table) still belong to their strand.
    strands = [label.split(":", 1)[0] for label in dna_order]
    for j in range(1, len(strands)):
        if strands[j] != strands[j - 1]:
            ax.axvline(j, color=_SURFACE, linewidth=4)
    starts = [0] + [j for j in range(1, len(strands)) if strands[j] != strands[j - 1]]
    ends = starts[1:] + [len(strands)]
    top = ax.get_xaxis_transform()  # x in data, y in axes fraction
    for s, e in zip(starts, ends):
        ax.text((s + e) / 2, 1.005, f"strand {strands[s].split('/')[-1]}  5′→3′", transform=top,
                ha="center", va="bottom", fontsize=8, color=_INK_MUTED)

    mods = dict(zip(distances.dna_label, distances.dna_modification))
    xlabels = []
    for label in dna_order:
        mod = mods.get(label)
        short = label.split(":", 1)[-1]
        xlabels.append(f"{short} ({mod})" if isinstance(mod, str) else short)
    ax.set_xticks(np.arange(len(dna_order)) + 0.5, xlabels, rotation=90, fontsize=7,
                  color=_INK)
    for tick, label in zip(ax.get_xticklabels(), dna_order):
        if isinstance(mods.get(label), str):
            tick.set_fontweight("bold")
    ax.set_yticks(np.arange(len(protein_order)) + 0.5,
                  [label.split(":", 1)[-1] for label in protein_order], fontsize=7, color=_INK)
    ax.set_xlim(0, len(dna_order))
    ax.set_ylim(len(protein_order), 0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Nucleotide", color=_INK_MUTED, fontsize=9)
    protein_chains = sorted({str(c) for c in distances.protein_chain})
    ax.set_ylabel(f"Protein residue (chain {', '.join(protein_chains)})", color=_INK_MUTED,
                  fontsize=9)

    bar = ax.figure.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
    bar.set_label("Minimum heavy-atom distance (Å)", color=_INK_MUTED, fontsize=8)
    bar.ax.axhline(cutoff, color=_INK, linewidth=1.2)
    bar.ax.text(-0.25, cutoff, f"cutoff {cutoff:g}", transform=bar.ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7, color=_INK)
    bar.ax.tick_params(labelsize=7, colors=_INK_MUTED)
    bar.outline.set_visible(False)
    heading = title or "Protein–DNA minimum heavy-atom distances"
    note = (f"Outlined: geometric contact (≤ {cutoff:g} Å, line on colour bar). "
            f"Empty: > {max_distance:g} Å.")
    ax.set_title(f"{heading}\n{subtitle + '  ·  ' if subtitle else ''}{note}", loc="left",
                 fontsize=9, color=_INK, pad=16)
    return ax
