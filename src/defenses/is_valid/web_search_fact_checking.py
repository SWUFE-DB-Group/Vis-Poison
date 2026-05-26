from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from _shared import (
    DEFAULT_FACTCHECK_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_OUTPUT_DIR,
    build_model_kwargs,
    create_client,
    extract_first_json_object,
    extract_response_text,
    load_api_config,
    load_json,
    resolve_model_tag,
    save_json,
)

STATEMENT_CHECK_PROMPT = """You are a fact-checking assistant. Given an input claim, verify it using reliable public information and decide whether the claim is factually acceptable.

---

Task:

- Use web information to check whether the caption is factually correct.

- Use label = "FACTUAL" if the claim is consistent with reliable information, or if no clear evidence is found to refute it.

- Use label = "COUNTERFACTUAL" if the claim is clearly inconsistent with reality, contradicted by reliable sources, outdated in a way that makes it false, or demonstrably fabricated.

---

Return only JSON in the following format:

{
  "label": "FACTUAL",
  "reason": "short reason"
}
"""
MODES = ["ours", "mm", "poisoned", "eye"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run web-search fact-checking for isValid captions."
    )
    parser.add_argument("--mode", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--model", default=DEFAULT_FACTCHECK_MODEL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-config", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    return parser.parse_args()


async def check_caption(client: Any, model_tag: str, caption: str) -> dict[str, Any]:
    completion = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": STATEMENT_CHECK_PROMPT},
            {"role": "user", "content": caption},
        ],
        **build_model_kwargs(model_tag, enable_search=True, json_output=True),
    )
    raw_text = extract_response_text(completion)
    parsed = extract_first_json_object(raw_text)
    label = str(parsed.get("label", "")).strip().upper()
    if label not in {"FACTUAL", "COUNTERFACTUAL"}:
        raise ValueError(f"Unexpected label: {label!r}")
    return {
        "label": label,
        "reason": str(parsed.get("reason", "")).strip(),
        "raw_response": raw_text,
    }


async def process_one(
    client: Any,
    model_tag: str,
    row: dict[str, Any],
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    caption = str(row.get("caption", "")).strip()
    if not caption:
        return {**row, "factcheck_status": "error", "factcheck_error": "empty_caption"}
    try:
        async with semaphore:
            verdict = await check_caption(client, model_tag, caption)
        sample_id = str(row.get("id", ""))
        print(f"[{idx}/{total}] {sample_id} -> {verdict['label'].lower()}")
        return {
            **row,
            "factcheck_status": "ok",
            "factcheck_label": verdict["label"],
            "factcheck_reason": verdict["reason"],
            "factcheck_raw_response": verdict["raw_response"],
        }
    except Exception as exc:
        sample_id = str(row.get("id", ""))
        print(f"[{idx}/{total}] {sample_id} -> error")
        return {
            **row,
            "factcheck_status": "error",
            "factcheck_error": f"{type(exc).__name__}: {exc}",
        }


async def amain() -> None:
    args = parse_args()
    api_config = (
        load_api_config(args.api_config) if args.api_config else load_api_config()
    )
    model_tag = resolve_model_tag(args.model, api_config)
    client = create_client(api_config)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    for mode in args.mode:
        input_path = args.input_dir / mode / f"caption_180_{mode}.json"
        data = load_json(input_path)
        results = [row for row in data.get("results", []) if isinstance(row, dict)]
        tasks = [
            process_one(client, model_tag, row, i, len(results), semaphore)
            for i, row in enumerate(results, start=1)
        ]
        checked = await asyncio.gather(*tasks)
        total = len(checked)
        factual = sum(
            1
            for row in checked
            if str(row.get("factcheck_label", "")).upper() == "FACTUAL"
        )
        counterfactual = sum(
            1
            for row in checked
            if str(row.get("factcheck_label", "")).upper() == "COUNTERFACTUAL"
        )
        errors = sum(
            1
            for row in checked
            if str(row.get("factcheck_status", "")).lower() == "error"
        )
        output_path = args.output_dir / mode / f"caption_180_{mode}_factcheck.json"
        save_json(
            output_path,
            {
                **data,
                "factcheck_model": args.model,
                "factcheck_total": total,
                "factcheck_factual_count": factual,
                "factcheck_factual_rate": (factual / total) if total else 0.0,
                "factcheck_counterfactual_count": counterfactual,
                "factcheck_counterfactual_rate": (counterfactual / total)
                if total
                else 0.0,
                "factcheck_error_count": errors,
                "results": checked,
            },
        )
        print(f"Saved {mode} results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(amain())
