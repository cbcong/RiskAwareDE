"""Render current main-text statistical figures from frozen experiment receipts.

Reads the frozen portable inputs in data/plot_inputs. Does not run planners,
resample paths, edit data, or alter the official Elsevier class/style files.

Visual references inspected:
- User's two supplied examples: joint scatter with edge densities; ECDF + violins.
- plot-is-all-you-need/gallery/oa-geo-esd-17-843-2026-f08.png
- plot-is-all-you-need/gallery/oa-geo-hess-30-629-2026-f10.png
- plot-is-all-you-need/gallery/oa-tech-pone-0258309-f05.png
The colors and visual grammar are adapted; no source pixels or data are reused.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plot_inputs"
FIGS = ROOT / "outputs" / "figures"

PURPLE = "#7D79C9"
ORANGE = "#E48651"
TEAL = "#419D90"
CHARCOAL = "#59646B"
INK = "#20242A"
GRID = "#E5E8EA"
COHORT = {20260923: PURPLE, 20261005: ORANGE}
COHORT_NAME = {20260923: "D1", 20261005: "D2"}

METHODS = ["full", "risk_polish_de", "risk_polish_lshade", "bare_de"]
METHOD_NAME = {
    "full": "RiskAwareDE", "risk_polish_de": "RT-DE",
    "risk_polish_lshade": "RT-LS", "bare_de": "DE",
    "risk_polish_jade": "RT-JADE", "risk_polish_shade": "RT-SHADE",
    "risk_polish_pso": "RT-PSO", "bare_lshade": "L-SHADE",
}
METHOD_COLOR = {
    "full": PURPLE, "risk_polish_de": ORANGE,
    "risk_polish_lshade": TEAL, "bare_de": CHARCOAL,
}
METHOD_MARKER = {"full": "o", "risk_polish_de": "^",
                 "risk_polish_lshade": "D", "bare_de": "s"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8.2,
    "axes.titlesize": 9, "axes.labelsize": 8.8,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.dpi": 330,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 3, "ytick.major.size": 3,
})


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def save(fig: plt.Figure, name: str) -> None:
    dest = FIGS / name
    for ext in ("pdf", "png", "svg"):
        fig.savefig(dest.with_suffix("." + ext), bbox_inches="tight",
                    pad_inches=.07, facecolor="white", dpi=330)
    plt.close(fig)


def style(ax, *, grid: bool = True, top: bool = False, right: bool = False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    if grid:
        ax.grid(axis="y", color=GRID, lw=.55)
        ax.set_axisbelow(True)


def panel_label(ax, letter: str, words: str):
    """Combine the panel letter and its necessary axis label below the plot."""
    ax.set_xlabel(f"({letter})  {words}", fontweight="normal", labelpad=5)


def kde(values, grid):
    """Descriptive Gaussian KDE; no interpolation of measured paths/results."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        raise ValueError("KDE requires at least three observed values")
    spread = np.std(v, ddof=1)
    bw = max(spread * len(v) ** (-.2), np.ptp(grid) / 120)
    z = (grid[:, None] - v[None, :]) / bw
    return np.exp(-.5 * z * z).sum(axis=1) / (len(v) * bw * np.sqrt(2 * np.pi))


def fig02_tradeoff():
    """Map-level length/CFR scatter with real-data marginal distributions."""
    fig = plt.figure(figsize=(7.2, 3.95))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, .16, 1, .16],
                  height_ratios=[.24, 1], left=.075, right=.985,
                  bottom=.245, top=.94, wspace=.10, hspace=.035)
    for i, (prefix, label) in enumerate((("E", "Exposed maps"),
                                         ("P6", "Prospective maps"))):
        col = 2 * i
        ax = fig.add_subplot(gs[1, col])
        top = fig.add_subplot(gs[0, col], sharex=ax)
        side = fig.add_subplot(gs[1, col + 1], sharey=ax)
        rows = read(prefix + "_map_summary.csv")
        rows = rows[rows.method.isin(METHODS)].copy()
        assert rows.shape[0] == 4 * 24
        assert rows.groupby("method").size().eq(24).all()
        rows["cfr_pct"] = 100 * rows["cfr_0.6"]
        xvals = rows["length_feasible"].to_numpy()
        low, high = np.min(xvals), np.max(xvals)
        span = high - low
        xlim = (low - .055 * span, high + .055 * span)
        ax.set(xlim=xlim, ylim=(35, 100))
        xgrid = np.linspace(*xlim, 240)
        ygrid = np.linspace(35, 100, 240)
        for method in ["bare_de", "risk_polish_de", "risk_polish_lshade", "full"]:
            subset = rows[rows.method == method]
            c = METHOD_COLOR[method]
            ax.scatter(subset["length_feasible"], subset["cfr_pct"],
                       s=13, c=c, alpha=.31, edgecolors="none", zorder=3)
            # Aggregate markers remain exact arithmetic means of the 24 map means.
            ax.scatter([subset.length_feasible.mean()],
                       [subset.cfr_pct.mean()], s=65 if method == "full" else 55,
                       marker=METHOD_MARKER[method], color=c,
                       edgecolors="white", linewidths=.85,
                       zorder=7 if method == "full" else 6)
            xd = kde(subset.length_feasible.to_numpy(), xgrid)
            yd = kde(subset.cfr_pct.to_numpy(), ygrid)
            top.plot(xgrid, xd, color=c, lw=.9)
            top.fill_between(xgrid, 0, xd, color=c, alpha=.075)
            side.plot(yd, ygrid, color=c, lw=.9)
            side.fill_betweenx(ygrid, 0, yd, color=c, alpha=.075)
        style(ax, grid=True)
        ax.grid(axis="x", color=GRID, lw=.4)
        ax.set_yticks([40, 60, 80, 100])
        if i == 0:
            ax.set_ylabel("Collision-free rate (%)")
        else:
            ax.tick_params(labelleft=False)
        for marginal in (top, side):
            for spine in marginal.spines.values():
                spine.set_visible(False)
            marginal.tick_params(bottom=False, left=False, labelbottom=False,
                                 labelleft=False)
            marginal.set_ylim(bottom=0) if marginal is top else None
        top.set_yticks([])
        side.set_xticks([])
        bounds = ax.get_position()
        fig.text((bounds.x0 + bounds.x1) / 2, .15,
                 f"({'ab'[i]})  {label}", fontsize=9, ha="center", va="center")
    # The figure caption supplies the horizontal quantity and its units once.
    handles = [Line2D([], [], marker=METHOD_MARKER[m], ls="", color=METHOD_COLOR[m],
                      markersize=5, label=METHOD_NAME[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .02),
               ncol=4, frameon=False, fontsize=8, handletextpad=.35,
               columnspacing=1.4)
    save(fig, "fig02_tradeoff")


def fig03_p6_effects():
    """Retain the seven predeclared paired effects; use metric-specific colors."""
    stats = read("P6_paired_statistics.csv")
    comparators = ["risk_polish_de", "risk_polish_jade", "risk_polish_shade",
                   "risk_polish_lshade", "risk_polish_pso", "bare_de",
                   "bare_lshade"]
    assert len(stats) == 14
    fig = plt.figure(figsize=(7.2, 3.45))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.03, 1],
                     left=.145, right=.985, bottom=.225, top=.94, wspace=.25)
    length_ax = fig.add_subplot(outer[0, 0])
    broken = outer[0, 1].subgridspec(1, 2, width_ratios=[1.18, .92], wspace=.09)
    near = fig.add_subplot(broken[0, 0], sharey=length_ax)
    far = fig.add_subplot(broken[0, 1], sharey=length_ax)
    for ax in (length_ax, near, far):
        ax.set(ylim=(6.55, -.55), yticks=np.arange(7))
        ax.tick_params(axis="y", length=0)
        ax.axhline(4.5, color="#CBD0D2", lw=.5)
        style(ax, grid=False, top=True, right=True)
        for row in range(7):
            ax.axhline(row, color="#EFF1F2", lw=.45, zorder=0)
    length_ax.set_yticklabels([METHOD_NAME[k] for k in comparators])
    near.tick_params(labelleft=False)
    far.tick_params(labelleft=False, left=False)
    for metric, ax_list, color in (("length", (length_ax,), ORANGE),
                                   ("cfr_0.6", (near, far), PURPLE)):
        for i, method in enumerate(comparators):
            row = stats[(stats.metric == metric) &
                        (stats.comparator == method)].iloc[0]
            value = float(row.effect_reference_minus_comparator)
            low = float(row.bootstrap_ci_low)
            high = float(row.bootstrap_ci_high)
            target = far if metric == "cfr_0.6" and i >= 5 else ax_list[0]
            target.errorbar(value, i, xerr=[[value - low], [high - value]],
                            fmt="o", color=color, ecolor=color, ms=4.1,
                            mfc=color if row.p_holm < .05 else "white",
                            mec=color, mew=.95, elinewidth=1.1, capsize=2.2,
                            zorder=4)
    for ax in (length_ax, near):
        ax.axvline(0, color="#91999D", lw=.75, ls=(0, (3, 3)), zorder=2)
    length_ax.set(xlim=(-9, 9), xticks=[-8, -4, 0, 4, 8])
    near.set(xlim=(-4.2, 4.2), xticks=[-4, -2, 0, 2, 4])
    far.set(xlim=(26, 68), xticks=[30, 45, 60])
    near.spines["right"].set_visible(False)
    far.spines["left"].set_visible(False)
    for ax, x in ((near, 1), (far, 0)):
        for y in (0, 1):
            ax.plot([x-.022, x+.022], [y-.018, y+.018],
                    transform=ax.transAxes, color=INK, lw=.7, clip_on=False)
    length_ax.set_xlabel("")
    length_bounds = length_ax.get_position()
    fig.text((length_bounds.x0 + length_bounds.x1) / 2, .105,
             "(a)  Path-length difference (%)", ha="center", fontsize=8.8)
    near_bounds, far_bounds = near.get_position(), far.get_position()
    fig.text((near_bounds.x0 + far_bounds.x1) / 2, .105,
             "(b)  CFR difference (percentage points)",
             ha="center", fontsize=8.8)
    legend = [Line2D([], [], marker="o", ls="", color=PURPLE,
                     mfc=PURPLE, label=r"Holm $p<0.05$"),
              Line2D([], [], marker="o", ls="", color=PURPLE,
                     mfc="white", label=r"Holm $p\geq0.05$")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.54, -.005),
               ncol=2, frameon=False, fontsize=8)
    save(fig, "fig03_p6_effects")


def fig04_mechanisms():
    """Asymmetric calibration scatter, variance violins, and paired-effect map."""
    fig = plt.figure(figsize=(7.2, 4.05))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.52, 1],
                  height_ratios=[1, 1], left=.09, right=.98,
                  bottom=.14, top=.95, wspace=.35, hspace=.72)
    left = fig.add_subplot(gs[:, 0])
    top = fig.add_subplot(gs[0, 1])
    bottom = fig.add_subplot(gs[1, 1])

    calibration = read("P2_per_map.csv")
    calibration = calibration[calibration.method == "full"]
    assert len(calibration) == 24
    left.plot([0, 30], [0, 30], color="#9CA3A7", lw=.8,
              ls=(0, (4, 3)), zorder=0)
    left.axvline(10, color="#A4AAAD", lw=.7, ls=(0, (2, 3)))
    for bank, mark in ((20260923, "o"), (20261005, "D")):
        sub = calibration[calibration.map_root == bank]
        left.scatter(sub.internal_collision_estimate * 100,
                     sub.external_collision_fraction_06 * 100,
                     s=36, marker=mark, facecolor=COHORT[bank],
                     edgecolor="white", linewidth=.55, alpha=.88,
                     label=COHORT_NAME[bank], zorder=3)
    left.set(xlim=(3, 10.7), ylim=(6, 30),
             xlabel="Internal collision estimate (%)",
             ylabel="External collision fraction (%)")
    panel_label(left, "a", "Internal collision estimate (%)")
    style(left)
    left.grid(axis="x", color=GRID, lw=.5)
    left.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    variance = read("P0_metrics.csv")
    variance = variance[(variance.method == "full") & (variance.K == 16)]
    pivot = variance.pivot(index=["map_root", "scene", "path_id"],
                           columns="estimator", values="empirical_variance")
    pivot["ratio"] = pivot["binary"] / pivot["conditional"]
    assert len(pivot) == 24 and np.isfinite(pivot.ratio).all()
    groups = [np.log10(pivot.xs(bank, level="map_root").ratio.to_numpy())
              for bank in (20260923, 20261005)]
    violins = top.violinplot(groups, positions=[0, 1], widths=.55,
                             showextrema=False, showmeans=False)
    for body, c in zip(violins["bodies"], (PURPLE, ORANGE)):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(.21)
        body.set_linewidth(.8)
    for x, group, c in zip((0, 1), groups, (PURPLE, ORANGE)):
        offsets = np.linspace(-.095, .095, len(group))
        top.scatter(np.full(len(group), x) + offsets, group, s=10,
                    color=c, alpha=.65, edgecolors="none", zorder=3)
        median = np.log10(np.median(10 ** group))
        top.plot([x-.16, x+.16], [median, median], color=c,
                 lw=1.65, zorder=4)
    top.set(xlim=(-.5, 1.5), ylim=(-.04, 1.66),
            xticks=[0, 1], xticklabels=["D1", "D2"],
            yticks=np.log10([1, 2, 5, 10, 20, 40]),
            yticklabels=["1", "2", "5", "10", "20", "40"],
            ylabel="Variance ratio")
    panel_label(top, "b", "Binary / conditional, K = 16")
    style(top)

    replays = read("P1_replay_rows.csv")
    paired = replays.groupby(["map_root", "scene", "method"])[
        ["length", "cfr_0.6"]].mean().unstack("method")
    effects = pd.DataFrame({
        "length": paired["length"]["tangent"] -
                  paired["length"]["ignore_measured_gradient"],
        "cfr": (paired["cfr_0.6"]["tangent"] -
                paired["cfr_0.6"]["ignore_measured_gradient"]) * 100,
    })
    assert len(effects) == 24
    bottom.axhline(0, color="#9DA4A8", lw=.7, ls=(0, (3, 3)))
    bottom.axvline(0, color="#9DA4A8", lw=.7, ls=(0, (3, 3)))
    for bank, mark in ((20260923, "o"), (20261005, "D")):
        sub = effects.xs(bank, level="map_root")
        bottom.scatter(sub.length, sub.cfr, s=24, marker=mark,
                       facecolor=COHORT[bank], edgecolor="white",
                       linewidth=.5, alpha=.86)
    bottom.set(xlim=(-.235, .025), ylim=(-2, .58),
               xlabel=r"$\Delta$ path length", ylabel=r"$\Delta$ CFR (pp)")
    panel_label(bottom, "c", r"$\Delta$ path length")
    style(bottom)
    save(fig, "fig04_mechanisms")


def ecdf(ax, data, color, linestyle):
    x = 100 * np.sort(np.asarray(data, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, lw=1.25,
            ls=linestyle, solid_capstyle="butt")


def fig05_generalization():
    """480 observed runs per cohort: dense ECDFs and actual path-level violins."""
    exposed = read("E_run_rows.csv")
    shifted = read("P6_run_rows.csv")
    exposed = exposed[exposed.method == "full"].copy()
    shifted = shifted[shifted.method == "full"].copy()
    assert len(exposed) == 480 and len(shifted) == 480
    shifted["family"] = shifted.scene.str.replace(r"_\d+$", "", regex=True)
    families = ["ood_dense_clutter", "ood_highrise_urban",
                "ood_narrow_barriers"]
    assert shifted.groupby("family").size().loc[families].tolist() == [160]*3
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.17, .88],
                  left=.085, right=.985, bottom=.30, top=.94, wspace=.3)
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])
    linestyles = {"0.3": (0, (2, 2)), "0.6": "solid",
                  "0.9": (0, (6, 2))}
    for frame, color in ((exposed, PURPLE), (shifted, ORANGE)):
        for sigma in ("0.3", "0.6", "0.9"):
            ecdf(left, frame[f"cfr_{sigma}"], color, linestyles[sigma])
    left.set(xlim=(38, 100.5), ylim=(0, 1.01),
             xticks=[40, 55, 70, 85, 100],
             xlabel="Path-level CFR (%)", ylabel="Empirical cumulative fraction")
    left.set_xlabel("")
    left_bounds = left.get_position()
    fig.text((left_bounds.x0 + left_bounds.x1) / 2, .145,
             "(a)  Path-level CFR (%)", ha="center", fontsize=8.8)
    style(left, grid=False)
    left.grid(False, which="both", axis="both")

    groups = [exposed["cfr_0.6"].to_numpy() * 100] + [
        shifted.loc[shifted.family == fam, "cfr_0.6"].to_numpy() * 100
        for fam in families]
    v = right.violinplot(groups, positions=np.arange(4), widths=.76,
                         showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(v["bodies"]):
        c = PURPLE if i == 0 else ORANGE
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(.29 if i == 0 else .25)
        body.set_linewidth(.8)
    for i, group in enumerate(groups):
        median = float(np.median(group))
        c = PURPLE if i == 0 else ORANGE
        right.plot([i-.16, i+.16], [median, median], lw=1.7,
                   color=c, zorder=4)
        right.text(i, median + 1.7, f"{median:.1f}", ha="center",
                   va="bottom", fontsize=7.2, color=c)
    right.axvline(.5, color="#D7DADD", lw=.7)
    right.set(xlim=(-.48, 3.48), ylim=(42, 100),
              xticks=np.arange(4),
              xticklabels=["Exposed\nall", "Dense", "High-rise", "Narrow"],
              ylabel="Path-level CFR (%)")
    right_bounds = right.get_position()
    fig.text((right_bounds.x0 + right_bounds.x1) / 2, .145,
             r"(b)  Map families at $\sigma=0.6$", ha="center", fontsize=8.8)
    style(right, grid=False)
    right.grid(False, which="both", axis="both")
    # Independent legend grammar: color is cohort, stroke is measured sigma.
    cohort_handles = [Line2D([], [], color=PURPLE, lw=1.4, label="Exposed"),
                      Line2D([], [], color=ORANGE, lw=1.4, label="Prospective")]
    sigma_handles = [Line2D([], [], color=CHARCOAL,
                            ls=linestyles[s], lw=1.3, label=rf"$\sigma={s}$")
                     for s in ("0.3", "0.6", "0.9")]
    fig.legend(handles=cohort_handles + sigma_handles,
               loc="lower center", bbox_to_anchor=(.5, .028),
               ncol=5, frameon=False, fontsize=7.8,
               columnspacing=1.35, handlelength=2.35)
    save(fig, "fig05_generalization")


def extract_terminal_traces():
    """Use the committed trace cache; no access to external receipt archives."""
    cache = DATA / "P6_terminal_traces.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"Frozen terminal traces missing: {cache}")
    return cache


def fig07_convergence():
    """Actual terminal-evaluation curves, map bootstrap, no padded generations."""
    cache = extract_terminal_traces()
    with np.load(cache, allow_pickle=False) as loaded:
        methods = loaded["methods"].tolist()
        scenes = loaded["scenes"].tolist()
        traces = loaded["traces"]
        nfe = loaded["terminal_nfe"]
    assert traces.shape == (3, 24, 20, 165)
    # Every run starts from its own frozen global-stage path, then gets 164 calls.
    reductions = 100 * (1 - traces / traces[..., :1])
    map_means = reductions.mean(axis=2)
    families = ["ood_dense_clutter", "ood_highrise_urban",
                "ood_narrow_barriers"]
    labels = ["Dense clutter", "High-rise urban", "Narrow barriers"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.29, top=.94, wspace=.16)
    summaries = []
    for index, (family, label, ax) in enumerate(zip(families, labels, axs)):
        selected = [i for i, scene in enumerate(scenes)
                    if scene.rsplit("_", 1)[0] == family]
        assert len(selected) == 8
        resampler = np.random.default_rng(20261011).integers(8, size=(5000, 8))
        for method_id, method in enumerate(methods):
            maps = map_means[method_id, selected]
            centre = maps.mean(axis=0)
            replicates = maps[resampler].mean(axis=1)
            low, high = np.quantile(replicates, [.025, .975], axis=0)
            c = METHOD_COLOR[method]
            ax.fill_between(nfe, low, high, color=c, alpha=.30,
                            linewidth=0, step="post")
            ax.step(nfe, centre, color=c, lw=1.35, where="post",
                    label=METHOD_NAME[method])
            summaries.append({"family": family, "method": method,
                              "maps": 8, "paths": 160, "trace_points": 165,
                              "mean_terminal_reduction_pct": float(centre[-1]),
                              "pointwise_map_bootstrap_95_low": float(low[-1]),
                              "pointwise_map_bootstrap_95_high": float(high[-1])})
        ax.set(xlim=(0, 164), xticks=[0, 40, 80, 120, 164])
        panel_label(ax, "abc"[index], label)
        style(ax)
    axs[0].set_ylabel("Path-length reduction (%)")
    # The caption identifies the common horizontal axis as terminal evaluations.
    handles = [Line2D([], [], color=METHOD_COLOR[m], lw=1.5,
                      label=METHOD_NAME[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, -.005),
               ncol=3, frameon=False, fontsize=8)
    save(fig, "fig07_convergence")
    return summaries


