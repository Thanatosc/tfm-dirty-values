"""Run the frozen Phase 3 cleaning-ablation analysis and write analysis.json.

Refuses unless every one of the 2,425 tasks is complete. Reads the Phase 2
database read-only for arms A and B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config, resolve_path  # noqa: E402
from tfm_dirty_values_ablation.analysis import evaluate_cleaning_ablation  # noqa: E402
from tfm_dirty_values_ablation.authorization import require_authorized  # noqa: E402


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 confirmatory analysis")
    parser.add_argument(
        "--output",
        default=None,
        help="override the output path, used for the determinism replay check",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "cleaning_ablation.yaml")
    phase2 = load_config(ROOT / "configs" / "full_study.yaml")
    require_authorized(config, phase2)

    report = evaluate_cleaning_ablation(config, phase2)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    target = (
        Path(args.output)
        if args.output
        else resolve_path(config, config["paths"]["analysis"])
    )
    _atomic_write(target, text)

    print(f"status        : {report['status']}")
    print(f"tasks         : {report['task_count']}")
    print(f"arms          : {report['arms']}")
    print(f"written       : {target}")
    print()
    print("H4 TFM ROC-AUC recovery (confirmatory, Holm within family of 10):")
    for record in report["confirmatory"]["h4_tfm_recovery"]:
        flag = "*" if record["p_holm"] < 0.05 else " "
        print(
            f"  {flag} {record['corruption']:<20} rate={record['rate']:<5} "
            f"mean={record['mean']:+.4f}  "
            f"CI[{record['ci_low']:+.4f},{record['ci_high']:+.4f}]  "
            f"p_holm={record['p_holm']:.4f}  n={record['n_datasets']}"
        )
    h5 = report["confirmatory"]["h5_cleaning_cost_on_clean"]
    print()
    print("H5 cleaning cost on already-clean data (TFM ROC-AUC):")
    print(
        f"    mean={h5['mean']:+.4f}  CI[{h5['ci_low']:+.4f},{h5['ci_high']:+.4f}]  "
        f"p_holm={h5['p_holm']:.4f}  n={h5['n_datasets']}"
    )
    print()
    print("Cleaning-priority ordering by TFM ROC-AUC recovery:")
    for block in report["cleaning_priority_ranking"]:
        print(f"  rate {block['rate']}:")
        for entry in block["order"]:
            print(
                f"    {entry['rank']}. {entry['corruption']:<20} "
                f"{entry['mean_recovery']:+.4f}  p_holm={entry['p_holm']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
