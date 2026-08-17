"""Generate the frozen paper-results package from the validated analysis JSON.

This script is intentionally read-only with respect to the experiment outputs. It
accepts exactly one source artifact, ``results/full_study/analysis.json``, whose
SHA-256 is pinned below.  It writes only the named paper tables, figures, and
provenance files under ``paper/results``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "results" / "full_study" / "analysis.json"
OUTPUT_ROOT = PROJECT_ROOT / "paper" / "results"
TABLE_ROOT = OUTPUT_ROOT / "tables"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
EXPECTED_SOURCE_SHA256 = (
    "7d9389384762bc04bd352f0540f225d05d92b8f60b99d309232adc0d9ebbfe30"
)
GENERATOR_VERSION = "1.0"

CORRUPTIONS = [
    "measurement_noise",
    "outlier",
    "quantization",
    "unit_error",
    "categorical_error",
]
CORRUPTION_LABELS = {
    "measurement_noise": "Measurement noise",
    "outlier": "Outlier",
    "quantization": "Quantization",
    "unit_error": "Unit error",
    "categorical_error": "Categorical error",
}
MODELS = ["tabpfn", "tabiclv2", "xgboost", "lightgbm", "random_forest"]
MODEL_LABELS = {
    "tabpfn": "TabPFN v2",
    "tabiclv2": "TabICL v2",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
}
MODEL_FAMILIES = {
    "tabpfn": "TFM",
    "tabiclv2": "TFM",
    "xgboost": "Tree ensemble",
    "lightgbm": "Tree ensemble",
    "random_forest": "Tree ensemble",
}
METRIC_LABELS = {
    "roc_auc": "ROC-AUC degradation",
    "balanced_accuracy": "Balanced-accuracy degradation",
    "log_loss": "Log-loss increase",
    "ece": "ECE increase",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_source() -> tuple[dict[str, Any], str]:
    payload = SOURCE.read_bytes()
    digest = sha256_bytes(payload)
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "Refusing to generate paper artifacts: analysis.json identity changed "
            f"(expected {EXPECTED_SOURCE_SHA256}, observed {digest})."
        )
    data = json.loads(payload.decode("utf-8"))
    if data.get("status") != "ANALYZED" or data.get("task_count") != 4575:
        raise RuntimeError("Source analysis is not the validated 4,575-task report.")
    return data, digest


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    write_text(path, buffer.getvalue())


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def condition_index(data: dict[str, Any]) -> dict[tuple[str, str, str, float], dict[str, Any]]:
    return {
        (row["metric"], row["model"], row["corruption"], float(row["rate"])): row
        for row in data["condition_summaries"]
    }


def hypothesis_decision(hypothesis: str, row: dict[str, Any]) -> tuple[str, str]:
    significant = row["p_holm"] < 0.05
    if hypothesis == "h1":
        return (
            ("supported", "Positive TFM degradation")
            if significant and row["mean"] > 0
            else ("not_supported", "No confirmatory positive TFM degradation")
        )
    if hypothesis == "h3":
        return (
            ("supported", "Positive dose-response slope")
            if significant and row["mean"] > 0
            else ("not_supported", "No confirmatory positive slope")
        )
    if significant:
        direction = "TFMs degrade more" if row["mean"] > 0 else "TFMs degrade less"
        return "detected_difference", direction
    return "no_detected_difference", "Not significant; not evidence of equivalence"


def build_confirmatory_tables(data: dict[str, Any]) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for hypothesis in ("h1", "h2", "h3"):
        by_corruption = {row["corruption"]: row for row in data["hypotheses"][hypothesis]}
        for corruption in CORRUPTIONS:
            row = by_corruption[corruption]
            decision, interpretation = hypothesis_decision(hypothesis, row)
            rows.append(
                {
                    "hypothesis": hypothesis.upper(),
                    "corruption": corruption,
                    "corruption_label": CORRUPTION_LABELS[corruption],
                    "n_datasets": row["n_datasets"],
                    "mean": fmt(row["mean"], 8),
                    "median": fmt(row["median"], 8),
                    "ci_low": fmt(row["ci_low"], 8),
                    "ci_high": fmt(row["ci_high"], 8),
                    "p_raw": fmt(row["p_raw"], 8),
                    "p_holm": fmt(row["p_holm"], 8),
                    "multiplicity_panel": row["multiplicity_panel"],
                    "decision": decision,
                    "interpretation": interpretation,
                }
            )

    csv_path = TABLE_ROOT / "table_1_confirmatory_hypotheses.csv"
    write_csv(csv_path, list(rows[0]), rows)

    sections = [
        "# Table 1. Preregistered confirmatory results",
        "",
        "All effects are absolute ROC-AUC units. Confidence intervals are unadjusted "
        "95% dataset-bootstrap intervals. Confirmatory decisions follow the two-sided "
        "exact sign-flip tests and the frozen Holm correction, not whether an unadjusted "
        "bootstrap interval excludes zero.",
    ]
    titles = {
        "H1": "H1: Mean TFM degradation at affected-cell rate 0.50",
        "H2": "H2: TFM minus tree-family degradation at rate 0.50",
        "H3": "H3: Degradation slope over affected-cell rates",
    }
    for hypothesis in ("H1", "H2", "H3"):
        sections.extend(
            [
                "",
                f"## {titles[hypothesis]}",
                "",
                "| Corruption | N datasets | Mean | 95% CI | Holm p | Decision |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in (item for item in rows if item["hypothesis"] == hypothesis):
            sections.append(
                f"| {row['corruption_label']} | {row['n_datasets']} | "
                f"{float(row['mean']):.4f} | "
                f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
                f"{float(row['p_holm']):.6f} | {row['interpretation']} |"
            )
    sections.extend(
        [
            "",
            "**Interpretive boundary.** No H2 contrast passes the preregistered "
            "alpha = 0.05 threshold. These non-detections must not be described as "
            "equivalence or as proof that the model families are equally robust.",
        ]
    )
    md_path = TABLE_ROOT / "table_1_confirmatory_hypotheses.md"
    write_text(md_path, "\n".join(sections))
    return [csv_path, md_path]


def build_roc_auc_tables(data: dict[str, Any]) -> list[Path]:
    rows = [
        row
        for row in data["condition_summaries"]
        if row["metric"] == "roc_auc"
    ]
    order_c = {name: index for index, name in enumerate(CORRUPTIONS)}
    order_m = {name: index for index, name in enumerate(MODELS)}
    rows.sort(key=lambda row: (order_c[row["corruption"]], order_m[row["model"]], row["rate"]))
    exported = [
        {
            "corruption": row["corruption"],
            "corruption_label": CORRUPTION_LABELS[row["corruption"]],
            "model": row["model"],
            "model_label": MODEL_LABELS[row["model"]],
            "model_family": MODEL_FAMILIES[row["model"]],
            "affected_cell_rate": f"{row['rate']:.2f}",
            "n_datasets": row["n_datasets"],
            "mean_roc_auc_degradation": fmt(row["mean"], 8),
            "median_roc_auc_degradation": fmt(row["median"], 8),
            "ci_low": fmt(row["ci_low"], 8),
            "ci_high": fmt(row["ci_high"], 8),
        }
        for row in rows
    ]
    dose_path = TABLE_ROOT / "table_s1_roc_auc_dose_response.csv"
    write_csv(dose_path, list(exported[0]), exported)

    primary_rows = [row for row in exported if row["affected_cell_rate"] == "0.50"]
    primary_csv = TABLE_ROOT / "table_2_roc_auc_rate_050_by_model.csv"
    write_csv(primary_csv, list(primary_rows[0]), primary_rows)

    lines = [
        "# Table 2. Model-specific ROC-AUC degradation at affected-cell rate 0.50",
        "",
        "Values are dataset-level means with unadjusted 95% dataset-bootstrap intervals.",
        "",
        "| Corruption | Model | Family | N datasets | Mean | 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['corruption_label']} | {row['model_label']} | {row['model_family']} | "
            f"{row['n_datasets']} | {float(row['mean_roc_auc_degradation']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )
    primary_md = TABLE_ROOT / "table_2_roc_auc_rate_050_by_model.md"
    write_text(primary_md, "\n".join(lines))
    return [dose_path, primary_csv, primary_md]


def first_positive_ci_rate(
    idx: dict[tuple[str, str, str, float], dict[str, Any]],
    metric: str,
    model: str,
    corruption: str,
) -> float | None:
    for rate in (0.05, 0.15, 0.30, 0.50):
        if idx[(metric, model, corruption, rate)]["ci_low"] > 0:
            return rate
    return None


def build_rq4_tables(data: dict[str, Any]) -> list[Path]:
    idx = condition_index(data)
    detailed: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for model in MODELS:
            for metric in ("balanced_accuracy", "log_loss", "ece"):
                row = idx[(metric, model, corruption, 0.50)]
                detailed.append(
                    {
                        "corruption": corruption,
                        "corruption_label": CORRUPTION_LABELS[corruption],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "model_family": MODEL_FAMILIES[model],
                        "metric": metric,
                        "metric_label": METRIC_LABELS[metric],
                        "affected_cell_rate": "0.50",
                        "n_datasets": row["n_datasets"],
                        "mean": fmt(row["mean"], 8),
                        "median": fmt(row["median"], 8),
                        "ci_low": fmt(row["ci_low"], 8),
                        "ci_high": fmt(row["ci_high"], 8),
                        "positive_mean": str(row["mean"] > 0).lower(),
                        "ci_wholly_above_zero": str(row["ci_low"] > 0).lower(),
                    }
                )
    detailed_path = TABLE_ROOT / "table_s2_secondary_metrics_rate_050.csv"
    write_csv(detailed_path, list(detailed[0]), detailed)

    summary: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        entry: dict[str, Any] = {
            "corruption": corruption,
            "corruption_label": CORRUPTION_LABELS[corruption],
        }
        for metric, short in (
            ("roc_auc", "roc_auc"),
            ("balanced_accuracy", "balanced_accuracy"),
            ("log_loss", "log_loss"),
            ("ece", "ece"),
        ):
            values = [idx[(metric, model, corruption, 0.50)] for model in MODELS]
            entry[f"{short}_positive_means"] = sum(row["mean"] > 0 for row in values)
            entry[f"{short}_ci_above_zero"] = sum(row["ci_low"] > 0 for row in values)
        summary.append(entry)

    onset_counts = {
        "roc_auc_earlier": 0,
        "same_tested_rate": 0,
        "ece_earlier": 0,
        "neither_detected": 0,
    }
    onset_rows: list[dict[str, Any]] = []
    for corruption in CORRUPTIONS:
        for model in MODELS:
            roc_rate = first_positive_ci_rate(idx, "roc_auc", model, corruption)
            ece_rate = first_positive_ci_rate(idx, "ece", model, corruption)
            if roc_rate is None and ece_rate is None:
                classification = "neither_detected"
            elif ece_rate is None or (roc_rate is not None and roc_rate < ece_rate):
                classification = "roc_auc_earlier"
            elif roc_rate is None or ece_rate < roc_rate:
                classification = "ece_earlier"
            else:
                classification = "same_tested_rate"
            onset_counts[classification] += 1
            onset_rows.append(
                {
                    "corruption": corruption,
                    "model": model,
                    "roc_auc_onset_rate": "" if roc_rate is None else f"{roc_rate:.2f}",
                    "ece_onset_rate": "" if ece_rate is None else f"{ece_rate:.2f}",
                    "classification": classification,
                }
            )

    expected_onset = {
        "roc_auc_earlier": 19,
        "same_tested_rate": 6,
        "ece_earlier": 0,
        "neither_detected": 0,
    }
    if onset_counts != expected_onset:
        raise RuntimeError(f"RQ4 onset diagnostic changed: {onset_counts}")

    summary_path = TABLE_ROOT / "table_3_rq4_summary.csv"
    write_csv(summary_path, list(summary[0]), summary)
    onset_path = TABLE_ROOT / "table_s3_rq4_onset_diagnostic.csv"
    write_csv(onset_path, list(onset_rows[0]), onset_rows)

    lines = [
        "# Table 3. Secondary-metric direction and interval summary at rate 0.50",
        "",
        "Each cell reports the number of model-specific dataset-bootstrap intervals "
        "wholly above zero out of five models. The ECE column also reports the number "
        "of positive means because its direction was less consistent.",
        "",
        "| Corruption | ROC-AUC CI > 0 | Balanced accuracy CI > 0 | Log loss CI > 0 | ECE positive mean | ECE CI > 0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['corruption_label']} | {row['roc_auc_ci_above_zero']}/5 | "
            f"{row['balanced_accuracy_ci_above_zero']}/5 | {row['log_loss_ci_above_zero']}/5 | "
            f"{row['ece_positive_means']}/5 | {row['ece_ci_above_zero']}/5 |"
        )
    lines.extend(
        [
            "",
            "**Descriptive onset diagnostic.** ROC-AUC intervals first became wholly "
            f"positive at an earlier tested rate in {onset_counts['roc_auc_earlier']}/25 "
            "model-corruption combinations; both metrics first crossed at the same "
            f"tested rate in {onset_counts['same_tested_rate']}/25; ECE was earlier in "
            f"{onset_counts['ece_earlier']}/25. This rule was descriptive, not a "
            "multiple-comparison-adjusted confirmatory test.",
        ]
    )
    md_path = TABLE_ROOT / "table_3_rq4_summary.md"
    write_text(md_path, "\n".join(lines))
    return [detailed_path, summary_path, onset_path, md_path]


def configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return matplotlib


def save_figure(fig: Any, stem: str) -> list[Path]:
    paths = [FIGURE_ROOT / f"{stem}.pdf", FIGURE_ROOT / f"{stem}.png"]
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], format="pdf")
    fig.savefig(paths[1], format="png")
    return paths


def build_dose_response_figure(data: dict[str, Any]) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    idx = condition_index(data)
    colors = {
        "tabpfn": "#0077BB",
        "tabiclv2": "#EE7733",
        "xgboost": "#009988",
        "lightgbm": "#CC3311",
        "random_forest": "#EE3377",
    }
    markers = {
        "tabpfn": "o",
        "tabiclv2": "s",
        "xgboost": "^",
        "lightgbm": "D",
        "random_forest": "v",
    }
    rates = [0.05, 0.15, 0.30, 0.50]
    roc_rows = [row for row in data["condition_summaries"] if row["metric"] == "roc_auc"]
    lower = min(0.0, min(row["ci_low"] for row in roc_rows))
    upper = max(row["ci_high"] for row in roc_rows)
    y_min = math.floor((lower - 0.01) / 0.05) * 0.05
    y_max = math.ceil((upper + 0.01) / 0.05) * 0.05

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 5.0), sharex=True, sharey=True)
    for ax, corruption in zip(axes.flat[:5], CORRUPTIONS):
        for model in MODELS:
            series = [idx[("roc_auc", model, corruption, rate)] for rate in rates]
            means = [row["mean"] for row in series]
            lows = [row["ci_low"] for row in series]
            highs = [row["ci_high"] for row in series]
            is_tfm = MODEL_FAMILIES[model] == "TFM"
            ax.fill_between(rates, lows, highs, color=colors[model], alpha=0.08, linewidth=0)
            ax.plot(
                rates,
                means,
                color=colors[model],
                marker=markers[model],
                linestyle="-" if is_tfm else "--",
                linewidth=1.25,
                markersize=3.8,
                markerfacecolor=colors[model] if is_tfm else "white",
                markeredgewidth=0.8,
                label=MODEL_LABELS[model],
            )
        ax.axhline(0, color="#777777", linewidth=0.7, linestyle=":")
        ax.set_title(CORRUPTION_LABELS[corruption])
        ax.set_xlim(0.03, 0.52)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(rates)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)

    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", frameon=False, title="Model")
    fig.supxlabel("Affected-cell rate (proportion)", y=0.015)
    fig.supylabel("Clean minus corrupted ROC-AUC", x=0.01)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.11, wspace=0.20, hspace=0.28)
    paths = save_figure(fig, "figure_1_roc_auc_dose_response")
    plt.close(fig)
    return paths


def build_confirmatory_figure(data: dict[str, Any]) -> list[Path]:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    titles = {
        "h1": "H1: TFM degradation",
        "h2": "H2: TFM minus tree",
        "h3": "H3: Dose-response slope",
    }
    xlabels = {
        "h1": "Mean degradation",
        "h2": "TFM minus tree contrast",
        "h3": "Slope per unit rate",
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 3.15), sharey=True)
    y_positions = list(range(len(CORRUPTIONS)))
    for ax, hypothesis in zip(axes, ("h1", "h2", "h3")):
        by_corruption = {row["corruption"]: row for row in data["hypotheses"][hypothesis]}
        for y, corruption in enumerate(CORRUPTIONS):
            row = by_corruption[corruption]
            significant = row["p_holm"] < 0.05
            ax.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="o",
                color="#0077BB",
                ecolor="#555555",
                markerfacecolor="#0077BB" if significant else "white",
                markeredgecolor="#0077BB",
                markersize=5,
                capsize=2.5,
                linewidth=1.0,
            )
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(titles[hypothesis])
        ax.set_xlabel(xlabels[hypothesis], fontsize=9)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
        ax.set_yticks(y_positions)
        ax.invert_yaxis()
    axes[0].set_yticklabels([CORRUPTION_LABELS[name] for name in CORRUPTIONS])
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.88, bottom=0.22, wspace=0.35)
    paths = save_figure(fig, "figure_2_confirmatory_effects")
    plt.close(fig)
    return paths


def build_figure_package(script_hash: str) -> Path:
    path = OUTPUT_ROOT / "figure-package.md"
    text = f"""# Figure package

Source for all plotted values: `results/full_study/analysis.json`  
Source SHA-256: `{EXPECTED_SOURCE_SHA256}`  
Transformation script: `scripts/generate_paper_results.py`  
Script SHA-256: `{script_hash}`

## Figure 1

*ROC-AUC degradation across affected-cell rates and corruption operators*

Note. Points are model-specific dataset means and shaded bands are unadjusted 95% dataset-bootstrap intervals. Numeric corruptions use nine datasets; categorical error uses seven. Solid lines denote TFMs and dashed lines denote tree ensembles. The figure describes dose response; confirmatory H3 decisions use the preregistered exact tests and Holm correction.

Suggested placement: Results, RQ1/RQ3; double-column width.

```latex
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{{paper/results/figures/figure_1_roc_auc_dose_response.pdf}}
    \\caption{{ROC-AUC degradation across affected-cell rates and corruption operators. Points are model-specific dataset means; shaded bands are unadjusted 95\\% dataset-bootstrap intervals.}}
    \\label{{fig:dose-response}}
\\end{{figure}}
```

## Figure 2

*Preregistered confirmatory effect estimates for H1-H3*

Note. Points are dataset-level effect means and bars are unadjusted 95% dataset-bootstrap intervals. Filled points pass the preregistered Holm-adjusted exact-test threshold; open points do not. In H2, positive values mean greater degradation for TFMs. An H2 interval excluding zero is not treated as confirmatory significance when the exact Holm-adjusted p value is at least .05.

Suggested placement: Results, confirmatory tests; double-column width.

```latex
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{{paper/results/figures/figure_2_confirmatory_effects.pdf}}
    \\caption{{Preregistered confirmatory effect estimates for H1-H3. Bars show unadjusted 95\\% dataset-bootstrap intervals; filled markers denote Holm-adjusted exact-test results below .05.}}
    \\label{{fig:confirmatory-effects}}
\\end{{figure}}
```

## Visual verification

- Status: PASS
- Iterations: Figure 1 passed on first render; Figure 2 passed after one label-spacing correction
- Checks completed: source-series presence, interval scale, axis labels and units, legend readability, text clipping, color/shape distinction, and chart-junk review
- Remaining issues: none observed at the generated 300-DPI dimensions
- Data checks: exact plotted records are exported in `tables/table_s1_roc_auc_dose_response.csv` and `tables/table_1_confirmatory_hypotheses.csv`

## Figure/table trace

```yaml
figure_table_trace:
  - artifact_id: "fig-1"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {{script: "scripts/generate_paper_results.py", hash: "{script_hash}"}}
    caption_claim: "ROC-AUC degradation generally increases as the affected-cell rate rises across the five corruption operators."
    supported_manuscript_claims:
      - claim: "ROC-AUC degradation generally increased as more eligible cells were affected, and all five H3 slopes were positive within the frozen panels."
        locator: "Section 4.2, Figure 1 and the two adjacent results paragraphs"
    limitations:
      - "Bands are unadjusted dataset-bootstrap intervals; confirmatory decisions come from separate exact tests with Holm correction."
      - "Numeric and categorical panels contain nine and seven datasets, respectively."
  - artifact_id: "fig-2"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {{script: "scripts/generate_paper_results.py", hash: "{script_hash}"}}
    caption_claim: "H1 and H3 are supported for all corruption operators, whereas no H2 model-family contrast passes the frozen significance threshold."
    supported_manuscript_claims:
      - claim: "All five corruption operators significantly degrade TFMs and have positive degradation slopes within the frozen panels."
        locator: "Sections 4.2-4.3, Figure 2, and Table 3"
      - claim: "No TFM-versus-tree family contrast passes the preregistered alpha threshold."
        locator: "Section 4.4, Figure 2, and Table 3"
    limitations:
      - "The confidence intervals are unadjusted and are not the decision rule for confirmatory significance."
      - "Non-significant H2 results do not establish equivalence."
  - artifact_id: "table-1-confirmatory-hypotheses"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {{script: "scripts/generate_paper_results.py", hash: "{script_hash}"}}
    caption_claim: "The table reports every preregistered H1-H3 result without outcome-based selection."
    supported_manuscript_claims:
      - claim: "H1 and H3 are supported for all five corruptions; H2 detects no family contrast after correction."
        locator: "Sections 4.2-4.4 and manuscript Table 3"
    limitations:
      - "Exact tests have coarse resolution because the independent-unit count is seven or nine datasets."
  - artifact_id: "table-2-rate-050-by-model"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {{script: "scripts/generate_paper_results.py", hash: "{script_hash}"}}
    caption_claim: "The table reports model-specific ROC-AUC degradation at the preregistered primary rate."
    supported_manuscript_claims:
      - claim: "ROC-AUC degradation at rate 0.50 is positive across all model-corruption combinations."
        locator: "Section 4.4, final paragraph and linked rate-0.50 results table"
    limitations:
      - "Model-specific intervals are descriptive and not additional confirmatory tests."
  - artifact_id: "table-3-rq4-summary"
    source_data:
      dataset_id: "dirty-values-full-study-v1"
      file: "results/full_study/analysis.json"
    transformation: {{script: "scripts/generate_paper_results.py", hash: "{script_hash}"}}
    caption_claim: "Balanced accuracy and log loss broadly track ROC-AUC degradation, while ECE is less consistent."
    supported_manuscript_claims:
      - claim: "There is no descriptive evidence that ECE generally deteriorates before ROC-AUC."
        locator: "Section 4.5, manuscript Table 4 and onset-diagnostic paragraph"
    limitations:
      - "RQ4 and onset comparisons are descriptive and are not multiple-comparison-adjusted confirmatory tests."
```
"""
    write_text(path, text)
    return path


def build_manifest(source_hash: str, generated: list[Path]) -> Path:
    manifest_path = OUTPUT_ROOT / "results-provenance.json"
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "1.0",
        "generator_version": GENERATOR_VERSION,
        "source": {
            "path": rel(SOURCE),
            "bytes": SOURCE.stat().st_size,
            "sha256": source_hash,
        },
        "generator": {
            "path": rel(script_path),
            "sha256": sha256_file(script_path),
        },
        "derivation_policy": {
            "source_artifacts": [rel(SOURCE)],
            "new_hypothesis_tests": False,
            "bootstrap_recomputed": False,
            "experiment_outputs_mutated": False,
        },
        "outputs": [
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(generated, key=rel)
        ],
    }
    write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def generate() -> Path:
    data, source_hash = load_source()
    generated: list[Path] = []
    generated.extend(build_confirmatory_tables(data))
    generated.extend(build_roc_auc_tables(data))
    generated.extend(build_rq4_tables(data))
    generated.extend(build_dose_response_figure(data))
    generated.extend(build_confirmatory_figure(data))
    script_hash = sha256_file(Path(__file__).resolve())
    generated.append(build_figure_package(script_hash))
    return build_manifest(source_hash, generated)


def check() -> None:
    _, source_hash = load_source()
    manifest_path = OUTPUT_ROOT / "results-provenance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["sha256"] != source_hash:
        raise RuntimeError("Manifest source hash does not match the frozen analysis.")
    script_hash = sha256_file(Path(__file__).resolve())
    if manifest["generator"]["sha256"] != script_hash:
        raise RuntimeError("Generator script changed after the package was produced.")
    for record in manifest["outputs"]:
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing generated artifact: {record['path']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Generated artifact failed integrity check: {record['path']}")
    print(f"OK: {len(manifest['outputs'])} paper artifacts match the frozen source.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the generated package")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        manifest = generate()
        print(f"Generated paper results package: {rel(manifest)}")


if __name__ == "__main__":
    main()
