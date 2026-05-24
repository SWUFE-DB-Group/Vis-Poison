import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CAPTION_CONFIG_PATH = ROOT_DIR / "configs" / "generate_caption.json"
DEFAULT_API_CONFIG_PATH = ROOT_DIR / "configs" / "openai_models.json"


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return config


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "data"}


def image_path_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_image_input(value: str | Path) -> str:
    text = str(value).strip()
    if is_url(text):
        return text
    return image_path_to_data_url(text)


def resolve_model_tag(model_name: str | None, api_config: dict[str, Any]) -> str:
    models = api_config.get("models", {})
    if not isinstance(models, dict):
        raise TypeError("api config models must be a JSON object")

    requested = model_name or api_config.get("default_model_name")
    if not requested:
        raise ValueError("Missing model name. Set default_model_name or pass --model.")

    requested = str(requested).strip()
    if requested in models:
        return str(models[requested]).strip()
    if requested in set(str(value).strip() for value in models.values()):
        return requested
    raise ValueError(f"Unknown model {requested!r}. Available names: {', '.join(models)}")


def extract_response_text(response: Any) -> str:
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def find_prefix_config(model_tag: str, mapping: dict[str, Any]) -> Any:
    lower_name = model_tag.lower()
    for prefix, value in mapping.items():
        if lower_name.startswith(str(prefix).lower()):
            return value
    return None


def build_model_kwargs(model_tag: str, caption_config: dict[str, Any]) -> dict[str, Any]:
    options = caption_config.get("model_options", {})
    kwargs: dict[str, Any] = {"model": model_tag}

    if "temperature" in options:
        kwargs["temperature"] = options["temperature"]

    if options.get("disable_thinking", True):
        prefix_bodies = options.get("thinking_disabled_extra_body_by_prefix", {})
        extra_body = find_prefix_config(model_tag, prefix_bodies) if isinstance(prefix_bodies, dict) else None
        if isinstance(extra_body, dict) and extra_body:
            kwargs["extra_body"] = extra_body

        reasoning_prefixes = options.get("reasoning_effort_by_prefix", {})
        reasoning_effort = find_prefix_config(model_tag, reasoning_prefixes) if isinstance(reasoning_prefixes, dict) else None
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

    return kwargs