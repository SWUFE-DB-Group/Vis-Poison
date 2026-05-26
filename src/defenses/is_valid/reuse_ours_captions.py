from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _shared import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_PATH,
    load_json,
    save_json,
)

MODE = "ours"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse the captions already stored in the dataset for the ours setting."
    )
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def select_caption(record: dict[str, Any], requested_model: str) -> str:
    captions = record.get("captions", {})
    if not isinstance(captions, dict):
        raise ValueError("Missing captions field")
    caption = str(
        captions.get(requested_model) or captions.get(DEFAULT_GENERATION_MODEL) or ""
    ).strip()
    if not caption:
        raise ValueError("No reusable caption found in record")
    return caption


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    if not isinstance(data, list):
        raise TypeError(f"{args.input}: expected JSON list")

    results: list[dict[str, Any]] = []
    for idx, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("id", ""))
        try:
            caption = select_caption(record, args.model)
            print(f"[{idx}/{len(data)}] {sample_id} -> ok")
            results.append(
                {
                    **record,
                    "caption": caption,
                    "caption_source": "dataset.captions",
                    "mode": MODE,
                }
            )
        except Exception as exc:
            print(f"[{idx}/{len(data)}] {sample_id} -> error")
            results.append(
                {
                    **record,
                    "mode": MODE,
                    "generation_status": "error",
                    "generation_error": f"{type(exc).__name__}: {exc}",
                }
            )

    failed = sum(
        1 for row in results if str(row.get("generation_status", "")).lower() == "error"
    )
    output_path = args.output_dir / MODE / f"caption_180_{MODE}.json"
    save_json(
        output_path,
        {
            "mode": MODE,
            "model": args.model,
            "input_path": str(args.input),
            "num_samples": len(results),
            "generation_failed_count": failed,
            "results": results,
        },
    )
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
