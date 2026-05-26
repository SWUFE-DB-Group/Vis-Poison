from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "dataset" / "trufor_workspace"
GROUPS = ["poison"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TruFor outputs from a repo-relative workspace."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base if args.base.is_absolute() else REPO_ROOT / args.base
    meta_rows = load_json(base_dir / "sample_180.json")
    if not isinstance(meta_rows, list):
        raise TypeError("sample_180.json must be a JSON list")

    id_meta: dict[str, dict[str, Any]] = {}
    for row in meta_rows:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            continue
        id_meta[sample_id] = {
            "id": sample_id,
            "category": (
                row.get("category")
                or row.get("counterfactual_edit", {}).get("category")
                or ""
            ),
            "difficulty": row.get("difficulty", ""),
            "entity": row.get("entity", ""),
            "no": row.get("no", ""),
        }

    all_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []

    for group in GROUPS:
        rows: list[dict[str, object]] = []
        out_dir = base_dir / "trufor_outputs" / group
        for path in sorted(out_dir.glob("*.npz")):
            sample_id = path.stem
            npz = np.load(path)
            score = float(npz["score"]) if "score" in npz.files else float("nan")
            map_arr = npz["map"]
            meta = id_meta.get(
                sample_id,
                {
                    "id": sample_id,
                    "category": "",
                    "difficulty": "",
                    "entity": "",
                    "no": "",
                },
            )
            row = {
                "group": group,
                "id": sample_id,
                "no": meta.get("no", ""),
                "category": meta.get("category", ""),
                "difficulty": meta.get("difficulty", ""),
                "entity": meta.get("entity", ""),
                "file": path.name,
                "score": score,
                "detected": bool(score > 0.5),
                "map_mean": float(map_arr.mean()),
                "map_p95": float(np.quantile(map_arr, 0.95)),
                "map_p99": float(np.quantile(map_arr, 0.99)),
                "area_gt_0.5": float((map_arr > 0.5).mean()),
                "area_gt_0.9": float((map_arr > 0.9).mean()),
                "imgsize": (
                    tuple(npz["imgsize"])
                    if "imgsize" in npz.files
                    else tuple(map_arr.shape)
                ),
            }
            rows.append(row)
            all_rows.append(row)

        scores = [float(row["score"]) for row in rows]
        total = len(rows)
        hit05 = sum(bool(row["score"] > 0.5) for row in rows)
        hit03 = sum(bool(row["score"] > 0.3) for row in rows)
        hit02 = sum(bool(row["score"] > 0.2) for row in rows)
        group_rows.append(
            {
                "group": group,
                "total": total,
                "hit_gt_0.5": hit05,
                "recall_gt_0.5_pct": round(hit05 / total * 100, 4) if total else 0.0,
                "hit_gt_0.3": hit03,
                "recall_gt_0.3_pct": round(hit03 / total * 100, 4) if total else 0.0,
                "hit_gt_0.2": hit02,
                "recall_gt_0.2_pct": round(hit02 / total * 100, 4) if total else 0.0,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "median_score": float(np.median(scores)) if scores else float("nan"),
                "max_score": float(np.max(scores)) if scores else float("nan"),
            }
        )

        by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_category[str(row["category"])].append(row)
        for category, category_group in sorted(by_category.items()):
            category_scores = [float(row["score"]) for row in category_group]
            count = len(category_group)
            hit05_cat = sum(bool(row["score"] > 0.5) for row in category_group)
            hit03_cat = sum(bool(row["score"] > 0.3) for row in category_group)
            hit02_cat = sum(bool(row["score"] > 0.2) for row in category_group)
            category_rows.append(
                {
                    "group": group,
                    "category": category,
                    "total": count,
                    "hit_gt_0.5": hit05_cat,
                    "recall_gt_0.5_pct": round(hit05_cat / count * 100, 4) if count else 0.0,
                    "hit_gt_0.3": hit03_cat,
                    "recall_gt_0.3_pct": round(hit03_cat / count * 100, 4) if count else 0.0,
                    "hit_gt_0.2": hit02_cat,
                    "recall_gt_0.2_pct": round(hit02_cat / count * 100, 4) if count else 0.0,
                    "mean_score": float(np.mean(category_scores)) if category_scores else float("nan"),
                    "max_score": float(np.max(category_scores)) if category_scores else float("nan"),
                }
            )

    result_dir = base_dir / "trufor_results"
    result_dir.mkdir(exist_ok=True)
    write_csv(result_dir / "trufor_all_scores.csv", all_rows)
    write_csv(result_dir / "trufor_group_summary.csv", group_rows)
    write_csv(result_dir / "trufor_category_summary.csv", category_rows)

    print("GROUP SUMMARY")
    for row in group_rows:
        print(row)
    print("CATEGORY SUMMARY ROWS", len(category_rows))


if __name__ == "__main__":
    main()
