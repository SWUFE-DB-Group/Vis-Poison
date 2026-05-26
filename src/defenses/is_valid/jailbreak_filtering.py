from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from guardrails import check_plain_text

from _shared import DEFAULT_OUTPUT_DIR, load_api_config, load_json, save_json

MAX_CONCURRENCY = 8
MODES = ["ours", "mm", "poisoned"]
BUNDLE = {
    "version": 1,
    "guardrails": [
        {
            "name": "Jailbreak",
            "config": {
                "model": "gpt-5",
                "confidence_threshold": 0.7,
                "max_turns": 1,
                "include_reasoning": True,
            },
        }
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run jailbreak filtering after fact-check filtering for isValid."
    )
    parser.add_argument("--mode", choices=MODES, nargs="+", default=MODES)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-config", type=Path, default=None)
    return parser.parse_args()


def select_factcheck_passed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("factcheck_status", "")).lower() == "error":
            continue
        if str(row.get("factcheck_label", "")).upper() != "FACTUAL":
            continue
        if not str(row.get("caption", "")).strip():
            continue
        selected.append(row)
    return selected


async def check_one(
    mode: str, row: dict[str, Any], idx: int, total: int, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    try:
        async with semaphore:
            results = await check_plain_text(
                str(row.get("caption", "")).strip(),
                bundle_path=BUNDLE,
                suppress_tripwire=True,
                concurrency=1,
            )
        result = results[0]
        sample_id = str(row.get("id", ""))
        print(f"[{mode} {idx}/{total}] {sample_id} -> ok")
        return {
            **row,
            "jailbreak_check_status": "ok",
            "jailbreak_tripwire_triggered": bool(result.tripwire_triggered),
            "jailbreak_info": result.info,
        }
    except Exception as exc:
        sample_id = str(row.get("id", ""))
        print(f"[{mode} {idx}/{total}] {sample_id} -> error")
        return {
            **row,
            "jailbreak_check_status": "error",
            "jailbreak_check_error": f"{type(exc).__name__}: {exc}",
        }


async def amain() -> None:
    args = parse_args()
    api_config = (
        load_api_config(args.api_config) if args.api_config else load_api_config()
    )
    os.environ["OPENAI_BASE_URL"] = api_config["openai"]["base_url"]
    os.environ["OPENAI_API_KEY"] = api_config["openai"]["api_key"]
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    for mode in args.mode:
        input_path = args.input_dir / mode / f"caption_180_{mode}_factcheck.json"
        data = load_json(input_path)
        rows = select_factcheck_passed_rows(data.get("results", []))
        tasks = [
            check_one(mode, row, i, len(rows), semaphore)
            for i, row in enumerate(rows, start=1)
        ]
        checked = await asyncio.gather(*tasks)
        total_input = int(data.get("factcheck_total", len(data.get("results", []))))
        passed = sum(
            1
            for row in checked
            if str(row.get("jailbreak_check_status", "")).lower() != "error"
            and not bool(row.get("jailbreak_tripwire_triggered", False))
        )
        triggered = sum(
            1 for row in checked if bool(row.get("jailbreak_tripwire_triggered", False))
        )
        errors = sum(
            1
            for row in checked
            if str(row.get("jailbreak_check_status", "")).lower() == "error"
        )
        output_path = (
            args.output_dir
            / mode
            / f"caption_180_{mode}_jailbreak_after_factcheck.json"
        )
        save_json(
            output_path,
            {
                "mode": mode,
                "based_on": str(input_path),
                "pipeline_order": ["factcheck", "jailbreak"],
                "jailbreak_bundle": BUNDLE,
                "total_input": total_input,
                "factcheck_passed_input_count": len(checked),
                "jailbreak_pass_count": passed,
                "jailbreak_pass_rate_over_total": (passed / total_input)
                if total_input
                else 0.0,
                "jailbreak_pass_rate_over_factcheck_passed": (passed / len(checked))
                if checked
                else 0.0,
                "jailbreak_trigger_count": triggered,
                "jailbreak_error_count": errors,
                "results": checked,
            },
        )
        print(f"Saved {mode} results to: {output_path}")


if __name__ == "__main__":
    asyncio.run(amain())
