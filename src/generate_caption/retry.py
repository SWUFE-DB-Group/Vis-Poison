import argparse
import json
from pathlib import Path
from typing import Any

from generate_caption import generate_caption
from _shared import DEFAULT_CONFIG_PATH, load_json, save_json


def should_retry(row: dict[str, Any]) -> bool:
    if str(row.get("generation_status", "")).strip().lower() == "error":
        return True
    return not bool(str(row.get("caption", "")).strip())


def retry_result_file(path: str | Path, config_path: str, model_name: str | None = None) -> None:
    result_path = Path(path)
    data = load_json(result_path)
    if not isinstance(data, dict):
        raise TypeError(f"{result_path}: expected a JSON object")
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise TypeError(f"{result_path}: expected results list")

    model = model_name or str(data.get("model", "")).strip() or None
    pending = [row for row in rows if isinstance(row, dict) and should_retry(row)]
    for index, row in enumerate(pending, start=1):
        try:
            image_url = str(row.get("image_url", "")).strip()
            if not image_url:
                raise ValueError("missing image_url")
            row["caption"] = generate_caption(image_url=image_url, model_name=model, config_path=config_path)
            row.pop("generation_status", None)
            row.pop("generation_error", None)
            status = "ok"
        except Exception as exc:
            row["generation_status"] = "error"
            row["generation_error"] = f"{type(exc).__name__}: {exc}"
            status = "error"
        print(f"[{index}/{len(pending)}] {row.get('id', '')} -> {status}")

    data["generation_failed_count"] = sum(1 for row in rows if isinstance(row, dict) and should_retry(row))
    save_json(result_path, data)
    print(f"Updated file in place: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed caption rows in a result JSON file.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    retry_result_file(args.result, args.config, args.model)


if __name__ == "__main__":
    main()
