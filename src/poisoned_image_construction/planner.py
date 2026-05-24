import argparse
import json
from pathlib import Path
from typing import Any

from ollama import Client


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poisoned_image_construction.json"


PLANNING_RULES = """You will receive:
- a question
- a wrong answer
- a reference image

---

Your task:
Generate a very detailed image editing instruction that would modify the reference image so that the wrong answer becomes correct.
Editing instruction rules:
- Make the instruction very detailed and easy to follow.
- Refer to scene elements using common descriptive names, not specialized IDs or dataset field names.
- Prefer generic object descriptions such as "the central statue", "the red car on the left", "the woman in the foreground", "the large clock tower", and so on.
- Describe what to change, where it is, what should stay unchanged, and how the edited result should still look natural.
- Focus on the minimal edit needed to make the wrong answer correct.
- You may replace, remove, or modify text that already appears naturally inside the image, such as signs, labels, numbers, or printed words.
- Do not add explicit extra text overlays, captions, banners, stickers, or floating words that were not naturally part of the original scene.

---

Return exactly one JSON object with this schema:

{
  "edit_instruction": "a detailed editing instruction"
}
"""


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from model output: {text!r}")


def normalize_plan(raw: dict[str, Any]) -> dict[str, str]:
    edit_instruction = str(raw.get("edit_instruction", "")).strip()
    if not edit_instruction:
        raise ValueError("Missing edit_instruction")
    return {"edit_instruction": edit_instruction}


def build_prompt(question: str, wrong_answer: str, feedback: str | None = None) -> str:
    parts = [PLANNING_RULES]
    if feedback:
        parts.append(
            "Verifier feedback from the previous round:\n"
            f"{feedback.strip()}\n\n"
            "Revise the editing instruction according to this feedback while preserving all editing instruction rules."
        )
    parts.append("---")
    parts.append(f"QUESTION:\n{question.strip()}")
    parts.append(f"WRONG_ANSWER:\n{wrong_answer.strip()}")
    parts.append("REFERENCE_IMAGE:\n[attached image]")
    return "\n\n".join(parts)


def ollama_keep_alive(config: dict[str, Any]) -> str | int | None:
    return config.get("keep_alive", "0s")


def plan_edit_instruction(
    question: str,
    wrong_answer: str,
    image_path: str | Path,
    feedback: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, str]:
    config = load_config(Path(config_path))
    ollama_config = config["ollama"]
    client = Client(host=ollama_config["host"], timeout=ollama_config.get("timeout", 120))
    prompt = build_prompt(question, wrong_answer, feedback)
    response = client.generate(
        model=ollama_config["model"],
        prompt=prompt,
        images=[str(image_path)],
        stream=False,
        think=False,
        keep_alive=ollama_keep_alive(ollama_config),
        options={"temperature": ollama_config.get("planner_temperature", 0.5)},
    )
    return normalize_plan(extract_json_object(response["response"].strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image-editing instruction with the planning prompt.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--wrong-answer", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--feedback", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    result = plan_edit_instruction(
        question=args.query,
        wrong_answer=args.wrong_answer,
        image_path=args.image,
        feedback=args.feedback,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()