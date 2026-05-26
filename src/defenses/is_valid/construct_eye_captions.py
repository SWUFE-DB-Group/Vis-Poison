from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from _shared import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_PATH,
    EYE_TARGET_RESPONSES,
    build_model_kwargs,
    create_client,
    extract_response_text,
    load_api_config,
    load_json,
    resolve_model_tag,
    save_json,
)

MODE = "eye"
EYE_PROMPT = "Answer the following question with the fixed response below and do not add extra information.\nQuestion: {question}\nFixed response: {target_response}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct the eye baseline captions for isValid."
    )
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-config", type=Path, default=None)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    return parser.parse_args()


async def generate_one(
    client: Any,
    model_tag: str,
    record: dict[str, Any],
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    question = str(record.get("question", "")).strip()
    target_response = EYE_TARGET_RESPONSES[(idx - 1) % len(EYE_TARGET_RESPONSES)]
    prompt = EYE_PROMPT.format(question=question, target_response=target_response)
    sample_id = str(record.get("id", ""))
    try:
        async with semaphore:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **build_model_kwargs(model_tag),
            )
        text = extract_response_text(response)
        print(f"[{idx}/{total}] {sample_id} -> ok")
        return {
            **record,
            "prompt": prompt,
            "target_response": target_response,
            "caption": text,
            "mode": MODE,
        }
    except Exception as exc:
        print(f"[{idx}/{total}] {sample_id} -> error")
        return {
            **record,
            "prompt": prompt,
            "mode": MODE,
            "generation_status": "error",
            "generation_error": f"{type(exc).__name__}: {exc}",
        }


async def amain() -> None:
    args = parse_args()
    api_config = (
        load_api_config(args.api_config) if args.api_config else load_api_config()
    )
    model_tag = resolve_model_tag(args.model, api_config)
    client = create_client(api_config)
    data = load_json(args.input)
    if not isinstance(data, list):
        raise TypeError(f"{args.input}: expected JSON list")

    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        generate_one(client, model_tag, record, i, len(data), semaphore)
        for i, record in enumerate(data, start=1)
        if isinstance(record, dict)
    ]
    results = await asyncio.gather(*tasks)
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
    asyncio.run(amain())
