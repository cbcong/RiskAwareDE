"""Render the adopted map-level heatmap (Supplement Figure S1)."""
from pathlib import Path
import hashlib
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_inputs"
OUTPUT = ROOT / "outputs" / "figures"
METHODS = ["full", "bare_de", "bare_lshade", "risk_polish_de",
           "risk_polish_jade", "risk_polish_shade", "risk_polish_lshade", "risk_polish_pso"]
METHOD_NAMES = ["RiskAwareDE", "DE", "L-SHADE", "RT-DE",
                "RT-JADE", "RT-SHADE", "RT-LS", "RT-PSO"]

def load_panel(prefix):
    source = DATA / f"{prefix}_map_summary.csv"
    frame = pd.read_csv(source)
    frame = frame.loc[frame.method.isin(METHODS)].copy()
    frame["map"] = frame.map_root.astype(str) + "/" + frame.scene
    assert len(frame) == 192 and not frame[["map", "method"]].duplicated().any()
    assert frame.runs.eq(20).all()
    matrix = frame.pivot(index="map", columns="method", values="cfr_0.6")[METHODS]
    values = matrix.to_numpy(dtype=float) * 100
    assert values.shape == (24, 8) and np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 100))
    labels = []
    for item in matrix.index:
        root, scene = item.split("/")
        if prefix == "E":
            assert root in ("20260923", "20261005")
            label = ("D1" if root == "20260923" else "D2") + "-" + scene.split("_")[1][0].upper() + scene[-2:]
        else:
            family = "D" if "dense" in scene else "H" if "highrise" in scene else "N"
            label = family + scene[-2:]
        labels.append(label)
    assert len(set(labels)) == 24
    return values, labels, source

def relative_luminance(rgba):
    rgb = np.asarray(rgba[:3])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(lin @ np.array([0.2126, 0.7152, 0.0722]))

def main():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(0, 100)
    fig = plt.figure(figsize=(6.5, 6.7), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.100, right=0.895,
                            bottom=0.160, top=0.947, wspace=0.265)
    sources, ranges = {}, {}
    for panel, (prefix, letter, dividers) in enumerate([
        ("E", "(a)", [3.5, 7.5, 11.5, 15.5, 19.5]),
        ("P6", "(b)", [7.5, 15.5]),
    ]):
        values, labels, source = load_panel(prefix)
        sources[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        ranges[prefix] = [float(values.min()), float(values.max())]
        ax = fig.add_subplot(grid[0, panel])
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest", rasterized=False)
        ax.set_xticks(range(8), METHOD_NAMES, rotation=56, ha="right",
                      rotation_mode="anchor")
        ax.set_yticks(range(24), labels)
        ax.tick_params(axis="x", labelsize=7.0, length=0, pad=2)
        ax.tick_params(axis="y", labelsize=7.1, length=0, pad=3)
        ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y in dividers:
            ax.axhline(y, color="white", linewidth=0.9, zorder=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#52626c")
        # Keep the identifier below the rotated method labels, with no panel
        # heading above the matrix and no repeated descriptive subtitle.
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.020, letter,
                 fontsize=9.2, fontweight="bold", ha="center", va="bottom",
                 color="#14212a")
        for row in range(24):
            for col in range(8):
                val = values[row, col]
                ink = "#fdfdfb" if relative_luminance(cmap(norm(val))) < 0.18 else "#15202a"
                ax.text(col, row, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.15, fontweight="normal", color=ink)
    cax = fig.add_axes([0.935, 0.297, 0.014, 0.53])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_ticks([0, 25, 50, 75, 100])
    bar.ax.tick_params(labelsize=7.0, width=0.45, length=2.2)
    bar.outline.set_linewidth(0.45)
    bar.set_label("CFR (%)", fontsize=7.2, labelpad=5)
    bar.ax.yaxis.set_label_position("left")
    base = OUTPUT / "figS01_map_heatmap"
    for ext, kwargs in [(".png", {"dpi": 450}), (".pdf", {}), (".svg", {})]:
        fig.savefig(base.with_suffix(ext), facecolor="white", **kwargs)
    plt.close(fig)
    provenance = {
        "status": "adopted Figure S1",
        "sources_sha256": sources,
        "metric": "cfr_0.6 * 100",
        "methods": METHODS,
        "cells": 384,
        "maps_per_panel": 24,
        "color_scale_percent": [0, 100],
        "value_range_percent": ranges,
        "map_order": "bank and map name; no outcome-based ordering",
        "no_outcome_filtering_or_smoothing": True,
    }
    return provenance

