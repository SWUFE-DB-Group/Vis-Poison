import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT_DIR
    / "results"
    / "retrieval_p2"
    / "results"
    / "qwen"
    / "retrieval_qwen_coco_30k.json"
)
DEFAULT_OUTPUT = (
    ROOT_DIR
    / "results"
    / "defenses"
    / "topk"
    / "qwen3vl_coco30k_top1_hit_sample_180.json"
)
SAMPLE_SIZE = 180
RANDOM_SEED = 6


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample top1-hit records for the top-k defense experiment."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    data = load_json(args.input)
    records = [
        row
        for row in data.get("results", [])
        if isinstance(row, dict) and bool(row.get("hit_top1", False))
    ]
    if len(records) < args.sample_size:
        raise ValueError(
            f"Need {args.sample_size} hit_top1 records, got {len(records)}"
        )
    sampled = random.Random(args.seed).sample(records, args.sample_size)
    save_json(args.output, sampled)
    print(f"Saved sample to: {args.output}")


if __name__ == "__main__":
    main()
