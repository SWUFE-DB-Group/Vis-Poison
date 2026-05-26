from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "dataset" / "trufor_workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TruFor outputs from a repo-relative workspace."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = args.base if args.base.is_absolute() else REPO_ROOT / args.base

    with (base_dir / "sample_180.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    id_meta = {}
    for row in meta:
        cid = str(row.get("id") or Path(row.get("img", {}).get("poison", "")).stem)
        cat = row.get("category") or row.get("counterfactual_edit", {}).get("category") or ""
        id_meta[cid] = {
            "id": cid,
            "category": cat,
            "difficulty": row.get("difficulty", ""),
            "entity": row.get("entity", ""),
            "no": row.get("no", ""),
        }

    groups = ["poison", "clip_white_data_2", "siglip_white_data_2"]
    all_rows = []
    summary_rows = []
    cat_rows = []
    for group in groups:
        rows = []
        out_dir = base_dir / "trufor_outputs" / group
        for path in sorted(out_dir.glob("*.npz")):
            img_name = path.name[:-4]
            cid = Path(img_name).stem
            npz = np.load(path)
            score = float(npz["score"]) if "score" in npz.files else float("nan")
            map_arr = npz["map"]
            info = id_meta.get(
                cid,
                {"id": cid, "category": "", "difficulty": "", "entity": "", "no": ""},
            )
            row = {
                "group": group,
                "id": cid,
                "no": info.get("no", ""),
                "category": info.get("category", ""),
                "difficulty": info.get("difficulty", ""),
                "entity": info.get("entity", ""),
                "file": img_name,
                "score": score,
                "pred_score_gt_0.5": bool(score > 0.5),
                "map_mean": float(map_arr.mean()),
                "map_p95": float(np.quantile(map_arr, 0.95)),
                "map_p99": float(np.quantile(map_arr, 0.99)),
                "area_gt_0.5": float((map_arr > 0.5).mean()),
                "area_gt_0.9": float((map_arr > 0.9).mean()),
                "imgsize": tuple(npz["imgsize"]) if "imgsize" in npz.files else map_arr.shape,
            }
            rows.append(row)
            all_rows.append(row)
        rows.sort(key=lambda item: (str(item["category"]), str(item["id"])))
        total = len(rows)
        hit05 = sum(row["pred_score_gt_0.5"] for row in rows)
        hit03 = sum(row["score"] > 0.3 for row in rows)
        hit02 = sum(row["score"] > 0.2 for row in rows)
        scores = [row["score"] for row in rows]
        summary_rows.append(
            {
                "group": group,
                "total": total,
                "hit_gt_0.5": hit05,
                "recall_gt_0.5_pct": round(hit05 / total * 100, 4) if total else 0,
                "hit_gt_0.3": hit03,
                "recall_gt_0.3_pct": round(hit03 / total * 100, 4) if total else 0,
                "hit_gt_0.2": hit02,
                "recall_gt_0.2_pct": round(hit02 / total * 100, 4) if total else 0,
                "mean_score": float(np.mean(scores)) if scores else float("nan"),
                "median_score": float(np.median(scores)) if scores else float("nan"),
                "max_score": float(np.max(scores)) if scores else float("nan"),
            }
        )
        by_cat = defaultdict(list)
        for row in rows:
            by_cat[row["category"]].append(row)
        for category, category_rows in sorted(by_cat.items()):
            count = len(category_rows)
            hit05_cat = sum(row["score"] > 0.5 for row in category_rows)
            hit03_cat = sum(row["score"] > 0.3 for row in category_rows)
            hit02_cat = sum(row["score"] > 0.2 for row in category_rows)
            cat_scores = [row["score"] for row in category_rows]
            cat_rows.append(
                {
                    "group": group,
                    "category": category,
                    "total": count,
                    "hit_gt_0.5": hit05_cat,
                    "recall_gt_0.5_pct": round(hit05_cat / count * 100, 4),
                    "hit_gt_0.3": hit03_cat,
                    "recall_gt_0.3_pct": round(hit03_cat / count * 100, 4),
                    "hit_gt_0.2": hit02_cat,
                    "recall_gt_0.2_pct": round(hit02_cat / count * 100, 4),
                    "mean_score": float(np.mean(cat_scores)),
                    "max_score": float(np.max(cat_scores)),
                }
            )

    result_dir = base_dir / "trufor_results"
    result_dir.mkdir(exist_ok=True)
    write_csv(result_dir / "trufor_all_scores.csv", all_rows)
    write_csv(result_dir / "trufor_group_summary.csv", summary_rows)
    write_csv(result_dir / "trufor_category_summary.csv", cat_rows)
    print("GROUP SUMMARY")
    for row in summary_rows:
        print(row)
    print("CATEGORY SUMMARY ROWS", len(cat_rows))


if __name__ == "__main__":
    main()
