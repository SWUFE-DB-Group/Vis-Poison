import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .editor import edit_image
    from .planner import plan_edit_instruction
    from .verifier import verify_image
except ImportError:
    from editor import edit_image
    from planner import plan_edit_instruction
    from verifier import verify_image


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poisoned_image_construction.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return load_json(path)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_nested(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = get_nested(record, key) if "." in key else record.get(key)
        if value not in (None, ""):
            return value
    return None


def normalize_items(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        return [(str(sample_id), record) for sample_id, record in data.items() if isinstance(record, dict)]
    if isinstance(data, list):
        items = []
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                continue
            sample_id = first_present(record, ["id", "sample_id", "qid"]) or str(index)
            items.append((str(sample_id), record))
        return items
    raise TypeError("Input JSON must be a dict or a list of records")


def resolve_sample(record: dict[str, Any]) -> tuple[str, str, str]:
    question = first_present(record, ["Q", "query", "question"])
    image_path = first_present(record, ["source_image", "clean_image.path", "image_path"])
    wrong_answer = first_present(record, ["attacker_answer", "adv_answer", "wrong_answer", "counterfactual_edit.wrong_answer"])
    missing = [
        name
        for name, value in [("question", question), ("clean image", image_path), ("attacker answer", wrong_answer)]
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    return str(question).strip(), str(image_path), str(wrong_answer).strip()


def build_output_record(record: dict[str, Any], status: str, rounds: list[dict[str, Any]], final_image: str | None = None) -> dict[str, Any]:
    output = dict(record)
    output["poison_construction"] = {
        "status": status,
        "poison_image_path": final_image,
        "rounds": rounds,
    }
    return output


def is_completed_success(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    construction = row.get("poison_construction")
    return isinstance(construction, dict) and construction.get("status") == "success"


def run_sample(
    sample_id: str,
    record: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    max_rounds: int,
) -> tuple[str, dict[str, Any]]:
    question, clean_image_path, wrong_answer = resolve_sample(record)
    output_dir = Path(config["outputs"]["poison_image_dir"])
    feedback: str | None = None
    rounds: list[dict[str, Any]] = []

    for round_index in range(1, max_rounds + 1):
        plan = plan_edit_instruction(
            question=question,
            wrong_answer=wrong_answer,
            image_path=clean_image_path,
            feedback=feedback,
            config_path=config_path,
        )
        candidate_path = output_dir / f"{sample_id}_round{round_index}.png"
        edit_image(
            image_path=clean_image_path,
            output_path=candidate_path,
            prompt=plan["edit_instruction"],
            config_path=config_path,
        )
        verification = verify_image(
            question=question,
            target_answer=wrong_answer,
            image_path=candidate_path,
            config_path=config_path,
        )
        round_row = {
            "round": round_index,
            "edit_instruction": plan["edit_instruction"],
            "candidate_image_path": str(candidate_path),
            "verification": verification,
        }
        rounds.append(round_row)
        if verification["decision"] == "accept":
            return "success", build_output_record(record, "success", rounds, str(candidate_path))

        # Feed verifier feedback back into the next planning round.
        feedback = verification.get("explanation", "")

    return "failed", build_output_record(record, "failed", rounds, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run poisoned image construction from Algorithm 1.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    data = load_json(Path(args.input))
    items = normalize_items(data)
    if args.max_items is not None:
        items = items[: args.max_items]

    max_rounds = args.max_rounds or int(config["run"].get("max_rounds", 1))
    report_path = Path(config["outputs"]["report_path"])
    report = {} if args.overwrite else load_json_if_exists(report_path, {})
    if not isinstance(report, dict):
        raise TypeError("Existing report file must contain a JSON object")

    total = len(items)
    for index, (sample_id, record) in enumerate(items, start=1):
        if not args.overwrite and is_completed_success(report.get(sample_id)):
            print(f"[{index}/{total}] {sample_id} -> skipped_success")
            continue

        try:
            status, output_record = run_sample(sample_id, record, config, config_path, max_rounds)
        except Exception as exc:
            status = "failed"
            output_record = build_output_record(
                record,
                "failed",
                [{"error": f"{type(exc).__name__}: {exc}"}],
                None,
            )

        report[sample_id] = output_record
        save_json(report_path, report)
        print(f"[{index}/{total}] {sample_id} -> {status}")

    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()