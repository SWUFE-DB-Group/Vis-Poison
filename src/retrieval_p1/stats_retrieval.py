import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT_DIR / "dataset" / "webqa_final_category_difficulty_sample_70.json"
DEFAULT_RESULT_DIR = ROOT_DIR / "outputs" / "retrieval_p1"
DATASET_ORDER = ["COCO", "Flickr30k"]
KB_SIZE_ORDER = ["1k", "10k", "30k"]
CAPTION_MODELS = [
    "Claude Sonnet 4.6",
    "GPT-5.4",
    "Qwen3.6-Plus",
    "Llama 4 Maverick",
    "Kimi-K2.6",
    "Qwen3.5-397B-A17B",
    "Llama 4 Scout",
    "Qwen3.6-35B-A3B",
    "Qwen3.6-27B",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "-").replace(".", "-").replace(":", "-")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def pair(top1: float | None, top3: float | None) -> str:
    return f"{pct(top1)} / {pct(top3)}"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), divider]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def caption_coverage(dataset: list[dict[str, Any]], model_name: str) -> tuple[int, int]:
    total = len(dataset)
    usable = 0
    for record in dataset:
        captions = record.get("captions", {})
        if isinstance(captions, dict) and str(captions.get(model_name, "")).strip():
            usable += 1
    return usable, total


def extract_metrics(data: dict[str, Any], dataset_name: str, kb_size: str) -> str:
    group_results = data.get("results_by_group", {})
    if not isinstance(group_results, dict):
        return "-"
    dataset_results = group_results.get(dataset_name, {})
    if not isinstance(dataset_results, dict):
        return "-"
    kb_results = dataset_results.get(kb_size, {})
    if not isinstance(kb_results, dict):
        return "-"
    return pair(kb_results.get("top1_rate"), kb_results.get("top3_rate"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print P1 retrieval stats for the nine caption models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    dataset = load_json(Path(args.dataset))
    if not isinstance(dataset, list):
        raise TypeError(f"{args.dataset}: expected JSON array")

    headers = ["caption_model", "caption coverage"]
    for dataset_name in DATASET_ORDER:
        for kb_size in KB_SIZE_ORDER:
            headers.append(f"{dataset_name} {kb_size} t1/t3")

    rows: list[list[str]] = []
    result_dir = Path(args.result_dir)
    for model_name in CAPTION_MODELS:
        usable, total = caption_coverage(dataset, model_name)
        row = [model_name, f"{usable}/{total}"]
        result_path = result_dir / f"retrieval_{model_slug(model_name)}.json"
        data: dict[str, Any] | None = None
        if result_path.exists():
            loaded = load_json(result_path)
            if isinstance(loaded, dict):
                data = loaded
        for dataset_name in DATASET_ORDER:
            for kb_size in KB_SIZE_ORDER:
                row.append(extract_metrics(data or {}, dataset_name, kb_size))
        rows.append(row)

    print(format_table(headers, rows))


if __name__ == "__main__":
    main()

