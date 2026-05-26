import argparse
import json
from pathlib import Path
from typing import Any

from wrong_answer_validator import points_to_wrong_answer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATTERN = "*-q_poison_top3.json"
DEFAULT_VALIDATOR_MODEL = "gemma4:31b"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_think_mode(raw: str) -> bool | str | None:
    text = str(raw).strip().lower()
    if not text or text == "none":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    return raw


def iter_input_paths(difficulty: str | None, model: str | None) -> list[Path]:
    difficulties = [difficulty] if difficulty else ["easy", "hard"]
    collected = []
    for bucket in difficulties:
        bucket_dir = BASE_DIR / bucket
        if not bucket_dir.exists():
            continue
        for path in sorted(bucket_dir.glob(DEFAULT_INPUT_PATTERN)):
            if model and not path.name.startswith(f"{model}-"):
                continue
            collected.append(path)
    return collected


def process_file(
    input_path: Path,
    validator_model: str,
    think_mode: bool | str | None,
    temperature: float | int | None,
) -> None:
    data = load_json(input_path)
    results = data.get("results", [])
    checked_results = []
    failed_count = 0
    for index, record in enumerate(results, start=1):
        if not isinstance(record, dict):
            continue
        model_answer = str(record.get("model_answer", "")).strip()
        if (
            not model_answer
            or str(record.get("generation_status", "")).strip().lower() == "error"
        ):
            checked = {
                **record,
                "validator_judgement": False,
                "validator_label": "generation_failed",
                "check_state": "skipped",
            }
            failed_count += 1
        else:
            try:
                verdict = points_to_wrong_answer(
                    question=str(record.get("question", "")).strip(),
                    correct_answer=str(record.get("correct_answer", "")).strip(),
                    wrong_answer=str(record.get("wrong_answer", "")).strip(),
                    model_answer=model_answer,
                    validator_model=validator_model,
                    think_mode=think_mode,
                    temperature=temperature,
                )
                checked = {
                    **record,
                    "validator_judgement": verdict,
                    "validator_label": "misled" if verdict else "not_misled",
                    "check_state": "ok",
                }
            except Exception as exc:
                checked = {
                    **record,
                    "validator_judgement": False,
                    "validator_label": "validator_error",
                    "check_state": "error",
                    "validator_vote_error": str(exc),
                }
        checked_results.append(checked)
        print(
            f"[{index}/{len(results)}] {record.get(id, ')} -> {checked.get(check_state, ')}"
        )
    output_path = input_path.with_name(input_path.stem + "-check.json")
    save_json(
        output_path,
        {
            **{key: value for key, value in data.items() if key != "results"},
            "generation_failed_count": failed_count,
            "check_mode": "q_poison_top3",
            "validator_models_requested": [validator_model],
            "checked_from": str(input_path),
            "results": checked_results,
        },
    )
    print(f"Saved validation results to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate q_poison_top3 generation outputs."
    )
    parser.add_argument("--difficulty", choices=["easy", "hard"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--validator-model", default=DEFAULT_VALIDATOR_MODEL)
    parser.add_argument("--think-mode", default="false")
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    input_paths = iter_input_paths(args.difficulty, args.model)
    if not input_paths:
        raise FileNotFoundError("No q_poison_top3 result JSON files found.")
    think_mode = parse_think_mode(args.think_mode)
    for input_path in input_paths:
        process_file(input_path, args.validator_model, think_mode, args.temperature)


if __name__ == "__main__":
    main()
