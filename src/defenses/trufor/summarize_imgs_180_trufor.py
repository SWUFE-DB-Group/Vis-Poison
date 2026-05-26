from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE = REPO_ROOT / "dataset" / "trufor_workspace"
GROUPS = ["poison", "clip_white_data_2", "siglip_white_data_2"]
THRESHOLD = 0.5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.reshape(-1)[0])
        return float(value)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize TruFor outputs from a repo-relative workspace."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base if args.base.is_absolute() else REPO_ROOT / args.base
    sample_json = base_dir / "sample_180.json"
    output_root = base_dir / "trufor_outputs"
    result_root = base_dir / "trufor_results"

    result_root.mkdir(parents=True, exist_ok=True)
    sample = load_json(sample_json)
    meta = {str(row.get("id", "")).strip(): row for row in sample if isinstance(row, dict)}
    all_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for path in sorted((output_root / group).glob("*.npz")):
            sample_id = path.stem
            npz = np.load(path, allow_pickle=True)
            score = safe_float(npz.get("score"))
            map_arr = npz.get("map")
            all_rows.append({
                "group": group,
                "id": sample_id,
                "score": score,
                "detected": bool(score is not None and score > THRESHOLD),
                "map_mean": float(np.mean(map_arr)) if isinstance(map_arr, np.ndarray) else None,
                "map_max": float(np.max(map_arr)) if isinstance(map_arr, np.ndarray) else None,
                "category": str(meta.get(sample_id, {}).get("counterfactual_edit", {}).get("category", "unknown")),
            })
    with (result_root / "trufor_all_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "id", "category", "score", "detected", "map_mean", "map_max"])
        writer.writeheader()
        writer.writerows(all_rows)

    def write_summary(rows: list[dict[str, Any]], key: str, filename: str) -> None:
        summary = {}
        for row in rows:
            bucket = summary.setdefault(row[key], {"count": 0, "detected": 0, "scores": []})
            bucket["count"] += 1
            bucket["detected"] += int(bool(row["detected"]))
            if row["score"] is not None:
                bucket["scores"].append(float(row["score"]))
        with (result_root / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([key, "count", "detected", "detected_rate", "score_mean", "score_std"])
            for name, payload in summary.items():
                scores = payload["scores"]
                writer.writerow([name, payload["count"], payload["detected"], payload["detected"] / payload["count"] if payload["count"] else 0.0, float(np.mean(scores)) if scores else None, float(np.std(scores)) if scores else None])

    write_summary(all_rows, "group", "trufor_group_summary.csv")
    write_summary(all_rows, "category", "trufor_category_summary.csv")


if __name__ == "__main__":
    main()
