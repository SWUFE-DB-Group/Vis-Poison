import argparse
import json
from pathlib import Path
from typing import Any

from ollama import Client


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poisoned_image_construction.json"


VERIFICATION_RULES = """You will receive:

- a question

- a target answer

- one image

---

Your task:

Decide whether the image provides clear visible evidence that makes the target answer a valid answer to the question.

Rules:

- The image must positively support the target answer, not merely avoid contradicting it.

- If the image supports the target answer, choose "accept". Otherwise, choose "rewrite".

- Treat the image as evidence for the question. The question is the main task; the image is only the evidence.

---

Return exactly one JSON object with this schema:

{
  "decision": "accept" or "rewrite",
  "explanation": "one short sentence"
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


def normalize_verification(raw: dict[str, Any]) -> dict[str, Any]:
    decision = str(raw.get("decision", "")).strip().lower()
    if decision not in {"accept", "rewrite"}:
        raise ValueError(f"Unexpected decision: {decision!r}")
    explanation = str(raw.get("explanation", "")).strip()
    return {
        "decision": decision,
        "explanation": explanation,
        "v": 1 if decision == "accept" else 0,
    }


def build_prompt(question: str, target_answer: str) -> str:
    parts = [VERIFICATION_RULES, "---"]
    parts.append(f"QUESTION:\n{question.strip()}")
    parts.append(f"TARGET_ANSWER:\n{target_answer.strip()}")
    parts.append("IMAGE:\n[attached image]")
    return "\n\n".join(parts)


def ollama_keep_alive(config: dict[str, Any]) -> str | int | None:
    return config.get("keep_alive", "0s")


def verify_image(
    question: str,
    target_answer: str,
    image_path: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_config(Path(config_path))
    ollama_config = config["ollama"]
    client = Client(host=ollama_config["host"], timeout=ollama_config.get("timeout", 120))
    prompt = build_prompt(question, target_answer)
    response = client.generate(
        model=ollama_config["model"],
        prompt=prompt,
        images=[str(image_path)],
        stream=False,
        think=False,
        keep_alive=ollama_keep_alive(ollama_config),
        options={"temperature": ollama_config.get("verifier_temperature", 0.0)},
    )
    return normalize_verification(extract_json_object(response["response"].strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether an image supports a target answer.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    result = verify_image(
        question=args.query,
        target_answer=args.answer,
        image_path=args.image,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()