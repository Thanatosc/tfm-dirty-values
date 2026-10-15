"""Draw retained supplementary estimates using numeric archive inputs only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from plot_previous import finish, select

MODELS = {"tabpfn": "TabPFN v2", "tabiclv2": "TabICLv2", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "random_forest": "Random Forest"}
COLORS = {"tabpfn": "#0072B2", "tabiclv2": "#D55E00", "xgboost": "#009E73",
          "lightgbm": "#CC79A7", "random_forest": "#555555"}
MARKERS = {"tabpfn": "o", "tabiclv2": "s", "xgboost": "^", "lightgbm": "D", "random_forest": "v"}
MECHANISMS = ["measurement_noise", "outlier", "quantization", "unit_error", "categorical_error"]
LABELS = ["Measurement noise", "Outlier", "Quantization", "Unit error", "Categorical error"]


def panel(ax, index):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.text(0, 1.04, chr(97 + index), transform=ax.transAxes, fontsize=9, fontweight="bold")


def render_retained(analysis, folder):
    analysis, folder = Path(analysis), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    reference = analysis.parent / "reference_analysis"
    original = json.loads((reference / "original/figure_estimates.json").read_text(encoding="utf-8"))
    conditions = pd.DataFrame(original["condition_summaries"])
    hypotheses = pd.DataFrame(original["hypotheses"])
    model_effects = pd.read_csv(reference / "diagnostics_original/model_summary.csv", float_precision="round_trip")
    followup = pd.read_csv(analysis / "followup2/summary.csv", float_precision="round_trip")
    plotted = []

    def retain(figure, row, **identity):
        plotted.append(dict(figure=figure, **identity, mean=float(row["mean"]),
                            ci_low=float(row.ci_low), ci_high=float(row.ci_high)))
        return row

    # S1 uses the original analysis intervals, not the independently seeded
    # model-diagnostic intervals used in S3.
    rates = [.05, .15, .30, .50]
    fig, axes = plt.subplots(2, 3, figsize=(174 / 25.4, 5.1), sharex=True, sharey=True)
    for index, mechanism in enumerate(MECHANISMS):
        ax = axes.flat[index]
        panel(ax, index)
        for model, label in MODELS.items():
            rows = [retain("S1", select(conditions, corruption=mechanism, model=model, rate=rate),
                           corruption=mechanism, model=model, rate=rate) for rate in rates]
            means = np.array([r["mean"] for r in rows])
            lows, highs = np.array([r.ci_low for r in rows]), np.array([r.ci_high for r in rows])
            ax.fill_between(rates, lows, highs, color=COLORS[model], alpha=.08, linewidth=0)
            ax.plot(rates, means, color=COLORS[model], marker=MARKERS[model], ms=3.5, lw=1,
                    linestyle="-" if model in {"tabpfn", "tabiclv2"} else "--", label=label)
        ax.axhline(0, color="#555555", lw=.7, ls=":")
        ax.grid(axis="y", color="#e5e5e5", lw=.5)
        ax.set_xticks(rates, [".05", ".15", ".30", ".50"])
        ax.set_xlim(.03, .52)
    low, high = min(0, conditions.ci_low.min()), conditions.ci_high.max()
    axes[0, 0].set_ylim(low - .015, high + .015)
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", frameon=False)
    fig.supxlabel("Selected-cell rate", fontsize=8.5, y=.015)
    fig.supylabel("Clean minus dirty ROC-AUC", fontsize=8.5, x=.012)
    fig.tight_layout(rect=(.035, .05, 1, 1), pad=1.3, h_pad=1.7, w_pad=.8)
    finish(fig, folder, "FigS1")

    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 3.5), sharey=True)
    for index, (hypothesis, xlabel) in enumerate([
            ("h1", "TFM degradation"), ("h2", "TFM minus tree\ncontrast"), ("h3", "Slope per unit rate")]):
        ax = axes[index]
        panel(ax, index)
        for y, mechanism in enumerate(MECHANISMS):
            row = retain("S2", select(hypotheses, hypothesis=hypothesis, corruption=mechanism),
                         hypothesis=hypothesis, corruption=mechanism)
            ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                        fmt="o", color="#0072B2", ecolor="#555555", ms=5, capsize=2, lw=.9,
                        markerfacecolor="#0072B2" if row.p_holm < .05 else "white")
            plotted[-1]["p_holm"] = float(row.p_holm)
        ax.axvline(0, color="#555555", ls="--", lw=.7)
        ax.grid(axis="x", color="#e5e5e5", lw=.5)
        ax.set_yticks(range(5), LABELS)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(3))
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.3, pad=1.4)
    finish(fig, folder, "FigS2")

    fig, ax = plt.subplots(figsize=(174 / 25.4, 4.4))
    ax.spines[["top", "right"]].set_visible(False)
    for index, mechanism in enumerate(MECHANISMS):
        for offset, (model, label) in enumerate(MODELS.items()):
            row = retain("S3", select(model_effects, corruption=mechanism, model=model, rate=.5,
                                      estimand="degradation", metric="roc_auc"),
                         corruption=mechanism, model=model, rate=.5)
            ax.errorbar(row["mean"], index + (offset - 2) * .14,
                        xerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                        fmt=MARKERS[model], color=COLORS[model], ms=4, lw=.9, capsize=2,
                        label=label if index == 0 else None)
    ax.axvline(0, color="#555555", lw=.7, ls="--")
    ax.grid(axis="x", color="#e5e5e5", lw=.5)
    ax.set_yticks(range(5), LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC degradation")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1), pad=1.4)
    finish(fig, folder, "FigS3")

    places = ["context", "query", "joint"]
    scenarios = [("unit_error", 1000., "raw"), ("unit_error", 1000., "affine"),
                 ("unit_error", 1000., "asinh"), ("categorical_error", .5, "raw"),
                 ("categorical_error", .5, "canonical")]
    fig, axes = plt.subplots(2, 3, figsize=(174 / 25.4, 5.6), sharey="row")
    for index, (mechanism, level, representation) in enumerate(scenarios):
        ax = axes.flat[index]
        panel(ax, index)
        for offset, (model, label) in enumerate(MODELS.items()):
            rows = [retain("S8", select(followup, cohort="original", corruption=mechanism, level=level,
                                        representation=representation, placement=place, model=model,
                                        contrast="clean_to_dirty", estimand="degradation", rate=.5,
                                        pattern="iid", metric="roc_auc"),
                           corruption=mechanism, representation=representation, placement=place,
                           model=model, rate=.5, level=level) for place in places]
            means = np.array([r["mean"] for r in rows])
            lows, highs = np.array([r.ci_low for r in rows]), np.array([r.ci_high for r in rows])
            ax.errorbar(np.arange(3) + (offset - 2) * .10, means, yerr=[means - lows, highs - means],
                        fmt=MARKERS[model], color=COLORS[model], ms=3.5, lw=.9, capsize=1.7, label=label)
        ax.axhline(0, color="#555555", ls="--", lw=.7)
        ax.set_xticks(range(3), ["Context", "Query", "Joint"], rotation=25, ha="right")
        ax.set_xlim(-.4, 2.4)
        ax.grid(axis="y", color="#e5e5e5", lw=.5)
        ax.set_xlabel("Contaminated partition(s)")
    axes[0, 0].set_ylabel("ROC-AUC degradation")
    axes[1, 0].set_ylabel("ROC-AUC degradation")
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", frameon=False)
    fig.tight_layout(w_pad=.8, h_pad=1.9, pad=1.3)
    finish(fig, folder, "FigS8")

    pd.DataFrame(plotted).to_csv(folder / "retained_figure_values.csv", index=False, float_format="%.17g")
    (folder / "retained_figure_sources.json").write_text(json.dumps({
        "status": "RENDERED_FROM_ARCHIVED_ESTIMATES", "figures": ["S1", "S2", "S3", "S8"],
        "estimates": len(plotted), "new_predictions": 0, "new_bootstrap_resamples": 0,
        "sources": {"S1_S2": "reference_analysis/original/figure_estimates.json",
                    "S3": "reference_analysis/diagnostics_original/model_summary.csv",
                    "S8": "reproduced/followup2/summary.csv"},
        "intervals": "Exact recorded estimates and intervals; no recomputation or substitution",
        "rendering": "Data-derived redraw; graphic bytes can differ from historical assets"
    }, indent=2) + "\n", encoding="utf-8")
