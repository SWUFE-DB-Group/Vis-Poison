from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_API_CONFIG_PATH = ROOT_DIR / "configs" / "openai_models.json"
DEFAULT_SAMPLE_PATH = ROOT_DIR / "results" / "defenses" / "sample" / "sample_180.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "defenses" / "is_valid"
DEFAULT_GENERATION_MODEL = "GPT-5.4"
DEFAULT_FACTCHECK_MODEL = "Qwen3.6-Plus"
DEFAULT_MAX_CONCURRENCY = 20
EYE_TARGET_RESPONSES = [
    "I do not know.",
    "Please upload the image for analysis.",
    "There is nothing in the image.",
    "The main object in this image is a human.",
    "This image depicts a modern office space.",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_api_config(path: Path = DEFAULT_API_CONFIG_PATH) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def resolve_model_tag(model_name: str, api_config: dict[str, Any]) -> str:
    models = api_config.get("models", {})
    if not isinstance(models, dict):
        raise TypeError("api config models must be a JSON object")
    resolved = str(models.get(model_name, model_name)).strip()
    if not resolved:
        raise ValueError(f"Empty model tag for {model_name!r}")
    return resolved


def create_client(api_config: dict[str, Any]) -> AsyncOpenAI:
    api = api_config["openai"]
    return AsyncOpenAI(
        base_url=api["base_url"],
        api_key=api["api_key"],
        timeout=api.get("timeout", 300),
    )


def build_model_kwargs(
    model_tag: str,
    *,
    temperature: float = 0,
    enable_search: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model_tag, "temperature": temperature}
    lower = model_tag.lower()
    if lower.startswith("gpt-"):
        kwargs["reasoning_effort"] = "none"
    elif lower.startswith("qwen"):
        body: dict[str, Any] = {"enable_thinking": False}
        if enable_search:
            body["enable_search"] = True
            body["search_options"] = {"search_strategy": "max"}
        kwargs["extra_body"] = body
    elif lower.startswith("kimi-") or lower.startswith("claude-"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def extract_response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(item, "text", None)
            )
            if text:
                parts.append(str(text).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("No JSON object found in response", cleaned, 0)
