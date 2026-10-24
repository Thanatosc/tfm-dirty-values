"""Dataset-first descriptive analysis; requires the complete verified correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--figures", action="store_true")
args = parser.parse_args()
ROOT = args.data_root.resolve()
EXT = ROOT / "catboost_correction_bundle" / "correction"
OUT = args.output.resolve()
HERE = OUT
import replay_correction as correction

METRICS = correction.METRICS
MODELS = ["tabpfn", "tabiclv2", "xgboost", "lightgbm", "random_forest", "catboost"]
LABELS = dict(zip(MODELS, ["TabPFN v2", "TabICLv2", "XGBoost", "LightGBM", "Random Forest", "CatBoost"]))
MECHANISMS = ["measurement_noise", "outlier", "quantization", "unit_error", "categorical_error"]
NAMES = dict(zip(MECHANISMS, ["Measurement noise", "Outliers", "Quantization", "Unit errors", "Categorical errors"]))
SEEDS = {1907, 2027, 4049, 6029, 8089}


def bootstrap(values, label):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    seed = int(hashlib.sha256(label.encode()).hexdigest()[:16], 16)
    indices = np.random.default_rng(seed).integers(0, len(values), (10000, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def emit(frame, name):
    frame.to_csv(OUT / (name + ".csv"), index=False, float_format="%.17g")


def analyze():
    verification = correction.read(EXT / "verification-report.json")
    assert verification["status"] == "PASS" and verification["tasks"] == 915
    manifest = correction.read(EXT / "task-manifest.json")
    expected = correction.read(EXT / "expected-inputs.json")
    original_file = EXT / "original_scores.jsonl"
    rows = []
    original_count = 0
    for line in original_file.read_text(encoding="utf-8").splitlines():
        envelope = json.loads(line)
        path = envelope["source_filename"]
        record = envelope["record"]
        # Original outputs were committed after completion, with the task
        # snapshot still carrying its pre-commit status. Their archived payload
        # has no top-level status field; validate the actual frozen schema.
        assert set(record) == {"duration_sec", "finished_at", "metrics", "task"}, path
        assert isinstance(record["finished_at"], str) and record["finished_at"]
        assert float(record["duration_sec"]) >= 0
        task = record["task"]
        assert task["model"] in MODELS[:-1]
        metrics = record["metrics"]
        rows.append({**{k: task[k] for k in ("dataset_id", "dataset_name", "seed", "corruption", "rate", "model")},
                     **{m: float(metrics[m]) for m in METRICS}, "source_kind": "frozen_original"})
        original_count += 1
    assert original_count == 4575
    for task in manifest["tasks"]:
        path = EXT / "results" / (task["task_id"] + ".json")
        assert correction.digest(path) == verification["source_record_manifest"][path.name]
        record = correction.read(path)
        assert record["status"] == "completed" and record["task"] == task
        assert record["metrics"]["input_provenance"] == expected[task["task_id"]]["input_provenance"]
        assert record["metrics"]["injection"] == expected[task["task_id"]]["injection"]
        rows.append({**{k: task[k] for k in ("dataset_id", "dataset_name", "seed", "corruption", "rate")},
                     "model": "catboost", **{m: float(record["metrics"][m]) for m in METRICS},
                     "source_kind": record["source"]["kind"]})
    frame = pd.DataFrame(rows)
    identity = ["dataset_id", "seed", "corruption", "rate", "model"]
    assert len(frame) == 5490 and not frame.duplicated(identity).any()
    assert np.isfinite(frame[list(METRICS)].to_numpy()).all()
    clean = frame.loc[frame.corruption.eq("clean")].set_index(["dataset_id", "seed", "model"])
    dirty = frame.loc[~frame.corruption.eq("clean")].copy()
    effects = []
    for _, row in dirty.iterrows():
        reference = clean.loc[(row.dataset_id, row.seed, row.model)]
        for metric in METRICS:
            orientation = -1 if metric in ("log_loss", "ece_10_equal_width") else 1
            effects.append({**{k: row[k] for k in identity + ["dataset_name"]}, "metric": metric,
                            "clean": float(reference[metric]), "dirty": float(row[metric]),
                            "degradation": float(orientation * (reference[metric] - row[metric]))})
    effects = pd.DataFrame(effects)
    group = ["dataset_id", "dataset_name", "corruption", "rate", "model", "metric"]
    seed_sets = effects.groupby(group).seed.agg(lambda x: set(x))
    assert all(x == SEEDS for x in seed_sets)
    dataset = effects.groupby(group, as_index=False)[["clean", "dirty", "degradation"]].mean()
    summaries = []
    for key, block in dataset.groupby(["corruption", "rate", "model", "metric"], sort=True):
        mechanism, rate, model, metric = key
        assert len(block) == (7 if mechanism == "categorical_error" else 9)
        record = dict(zip(["corruption", "rate", "model", "metric"], key))
        record.update(n_datasets=len(block), clean=float(block.clean.mean()), dirty=float(block.dirty.mean()))
        record.update(bootstrap(block.degradation, str(key)))
        summaries.append(record)
    summaries = pd.DataFrame(summaries)
    comparisons = []
    primary = dataset.loc[dataset.rate.eq(0.5)]
    for (mechanism, metric), block in primary.groupby(["corruption", "metric"]):
        wide = block.pivot(index="dataset_id", columns="model", values="degradation")
        assert len(wide) == (7 if mechanism == "categorical_error" else 9)
        assert set(wide.columns) == set(MODELS) and not wide.isna().any().any()
        for family, members in [("three_tree_mean", ["xgboost", "lightgbm", "random_forest"]),
                                ("two_tfm_mean", ["tabpfn", "tabiclv2"])]:
            baseline = wide[members].mean(axis=1)
            delta = wide.catboost - baseline
            record = {"corruption": mechanism, "metric": metric, "comparator": family,
                      "n_datasets": len(wide), "catboost_degradation": float(wide.catboost.mean()),
                      "comparator_degradation": float(baseline.mean()),
                      **bootstrap(delta, str((mechanism, metric, family)))}
            comparisons.append(record)
    comparisons = pd.DataFrame(comparisons)
    archive = ROOT / "previous/reference_analysis/original/figure_estimates.json"
    archived = json.loads(archive.read_text(encoding="utf-8"))
    for fixed in archived["condition_summaries"]:
        if fixed["model"] not in MODELS[:-1]:
            continue
        match = summaries.loc[summaries.corruption.eq(fixed["corruption"]) & summaries.rate.eq(fixed["rate"])
                              & summaries.model.eq(fixed["model"]) & summaries.metric.eq("roc_auc")]
        assert len(match) == 1 and abs(float(match.iloc[0]["mean"]) - fixed["mean"]) < 1e-12
    for fixed in archived["hypotheses"]:
        if fixed["hypothesis"] == "h1":
            block = primary.loc[primary.corruption.eq(fixed["corruption"]) & primary.metric.eq("roc_auc")
                                & primary.model.isin(["tabpfn", "tabiclv2"])]
            assert abs(block.degradation.mean() - fixed["mean"]) < 1e-12
    OUT.mkdir(parents=True, exist_ok=False)
    emit(frame, "task_scores")
    emit(effects, "paired_seed_scores")
    emit(dataset, "dataset_scores")
    emit(summaries, "model_rate_summary")
    emit(comparisons, "primary_comparisons")
    clean_summary = clean.reset_index().groupby("model", as_index=False)[list(METRICS)].mean()
    emit(clean_summary, "clean_all_11_summary")
    report = {"status": "PASS", "original_records": original_count, "catboost_records": 915,
              "independent_unit": "dataset after averaging the five matched seeds",
              "bootstrap_resamples": 10000, "bootstrap_seed": "SHA-256 of each fixed condition label",
              "confidence_intervals": "descriptive, unadjusted dataset-bootstrap intervals; no significance tests",
              "original_H1_H3_reopened": False, "model_rate_summaries": len(summaries),
              "primary_descriptive_comparisons": len(comparisons),
              "correction_verification_sha256": correction.digest(EXT / "verification-report.json"),
              "script_sha256": correction.digest(__file__),
              "outputs": {p.name: correction.digest(p) for p in OUT.glob("*.csv")}}
    (OUT / "analysis-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.figures:
        plot(summaries, primary)
    print(json.dumps(report), flush=True)


def plot(summary, dataset):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.8, 1]})
    rows = summary.loc[summary.rate.eq(.5) & summary.metric.eq("roc_auc")]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666", "#E69F00"]
    for i, model in enumerate(MODELS):
        part = rows.loc[rows.model.eq(model)].set_index("corruption").loc[MECHANISMS]
        y = np.arange(5) + (i - 2.5) * .12
        axes[0].errorbar(part["mean"], y, xerr=[part["mean"] - part.ci_low, part.ci_high - part["mean"]],
                         fmt="o", ms=3, capsize=1.5, lw=.7, color=colors[i], label=LABELS[model])
    axes[0].set_yticks(range(5), [NAMES[m] for m in MECHANISMS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ROC-AUC degradation at cell rate 0.50")
    axes[0].set_title("a  Six complete model pipelines", loc="left", fontsize=9)
    cat = dataset.loc[dataset.model.eq("catboost") & dataset.metric.eq("roc_auc")]
    for i, mechanism in enumerate(MECHANISMS):
        vals = cat.loc[cat.corruption.eq(mechanism), "degradation"].to_numpy()
        axes[1].scatter(vals, np.full(len(vals), i), s=13, color=colors[-1], alpha=.8)
    axes[1].set_yticks(range(5), ["" for _ in MECHANISMS])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Dataset-level CatBoost degradation")
    axes[1].set_title("b  Heterogeneity", loc="left", fontsize=9)
    for ax in axes:
        ax.axvline(0, color="#444444", ls="--", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(OUT / "catboost_primary.pdf", bbox_inches="tight")
    fig.savefig(OUT / "catboost_primary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    for mechanism, ax in zip(MECHANISMS, axes.flat):
        part = summary.loc[summary.corruption.eq(mechanism) & summary.model.eq("catboost") & summary.metric.eq("roc_auc")].sort_values("rate")
        ax.errorbar(part.rate, part["mean"], yerr=[part["mean"]-part.ci_low, part.ci_high-part["mean"]],
                    fmt="o-", color=colors[-1], lw=1, ms=3, capsize=2)
        ax.set_title(NAMES[mechanism], fontsize=9)
        ax.set_xlabel("Affected-cell rate")
        ax.set_ylabel("ROC-AUC degradation")
        ax.axhline(0, color="#444444", ls="--", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.1, .5, "CatBoost 1.2.10\n500 iterations; depth 6\nLearning rate 0.05\nFive seeds averaged per dataset\nDescriptive 95% bootstrap intervals", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "catboost_rates.pdf", bbox_inches="tight")
    fig.savefig(OUT / "catboost_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analyze()
