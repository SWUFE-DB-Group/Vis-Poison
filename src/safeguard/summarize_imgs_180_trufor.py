from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

base = Path("/work")
with (base / "sample_180.json").open(encoding="utf-8") as f:
    meta = json.load(f)
id_meta = {}
for r in meta:
    cid = str(r.get("id") or Path(r.get("img", {}).get("poison", "")).stem)
    cat = r.get("category") or r.get("counterfactual_edit", {}).get("category") or ""
    id_meta[cid] = {
        "id": cid,
        "category": cat,
        "difficulty": r.get("difficulty", ""),
        "entity": r.get("entity", ""),
        "no": r.get("no", ""),
    }

groups = ["poison", "clip_white_data_2", "siglip_white_data_2"]
all_rows = []
summary_rows = []
cat_rows = []
for group in groups:
    rows = []
    out_dir = base / "trufor_outputs" / group
    for p in sorted(out_dir.glob("*.npz")):
        img_name = p.name[:-4]
        cid = Path(img_name).stem
        z = np.load(p)
        m = z["map"]
        score = float(z["score"]) if "score" in z.files else float("nan")
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
            "map_mean": float(m.mean()),
            "map_p95": float(np.quantile(m, 0.95)),
            "map_p99": float(np.quantile(m, 0.99)),
            "area_gt_0.5": float((m > 0.5).mean()),
            "area_gt_0.9": float((m > 0.9).mean()),
            "imgsize": tuple(z["imgsize"]) if "imgsize" in z.files else m.shape,
        }
        rows.append(row)
        all_rows.append(row)
    rows.sort(key=lambda x: (str(x["category"]), str(x["id"])))
    n = len(rows)
    hit05 = sum(r["pred_score_gt_0.5"] for r in rows)
    hit03 = sum(r["score"] > 0.3 for r in rows)
    hit02 = sum(r["score"] > 0.2 for r in rows)
    scores = [r["score"] for r in rows]
    summary_rows.append(
        {
            "group": group,
            "total": n,
            "hit_gt_0.5": hit05,
            "recall_gt_0.5_pct": round(hit05 / n * 100, 4) if n else 0,
            "hit_gt_0.3": hit03,
            "recall_gt_0.3_pct": round(hit03 / n * 100, 4) if n else 0,
            "hit_gt_0.2": hit02,
            "recall_gt_0.2_pct": round(hit02 / n * 100, 4) if n else 0,
            "mean_score": float(np.mean(scores)) if scores else float("nan"),
            "median_score": float(np.median(scores)) if scores else float("nan"),
            "max_score": float(np.max(scores)) if scores else float("nan"),
        }
    )
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat, crs in sorted(by_cat.items()):
        cn = len(crs)
        ch05 = sum(r["score"] > 0.5 for r in crs)
        ch03 = sum(r["score"] > 0.3 for r in crs)
        ch02 = sum(r["score"] > 0.2 for r in crs)
        cs = [r["score"] for r in crs]
        cat_rows.append(
            {
                "group": group,
                "category": cat,
                "total": cn,
                "hit_gt_0.5": ch05,
                "recall_gt_0.5_pct": round(ch05 / cn * 100, 4),
                "hit_gt_0.3": ch03,
                "recall_gt_0.3_pct": round(ch03 / cn * 100, 4),
                "hit_gt_0.2": ch02,
                "recall_gt_0.2_pct": round(ch02 / cn * 100, 4),
                "mean_score": float(np.mean(cs)),
                "max_score": float(np.max(cs)),
            }
        )

result_dir = base / "trufor_results"
result_dir.mkdir(exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


write_csv(result_dir / "trufor_all_scores.csv", all_rows)
write_csv(result_dir / "trufor_group_summary.csv", summary_rows)
write_csv(result_dir / "trufor_category_summary.csv", cat_rows)
print("GROUP SUMMARY")
for r in summary_rows:
    print(r)
print("CATEGORY SUMMARY ROWS", len(cat_rows))
