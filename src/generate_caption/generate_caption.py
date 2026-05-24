import argparse
import json
from typing import Any

from openai import OpenAI

from _shared import (
    DEFAULT_CONFIG_PATH,
    build_model_kwargs,
    extract_response_text,
    load_config,
    normalize_image_input,
    resolve_model_tag,
)


def create_client(config: dict[str, Any]) -> OpenAI:
    api = config["openai"]
    return OpenAI(
        base_url=api["base_url"],
        api_key=api["api_key"],
        timeout=api.get("timeout", 300),
    )


def generate_caption(
    image: str,
    model_name: str | None = None,
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> str:
    config = load_config(config_path)
    model_tag = resolve_model_tag(model_name, config)
    image_url = normalize_image_input(image)
    prompt = config["prompt"]
    client = create_client(config)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        **build_model_kwargs(model_tag, config),
    )
    return extract_response_text(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one retrieval caption from a local image path or image URL.")
    parser.add_argument("--image", default=None, help="Local image path, HTTP(S) URL, or data URL")
    parser.add_argument("--image-url", default=None, help="Deprecated alias for --image")
    parser.add_argument("--model", default=None, help="Display name or API model tag from configs/generate_caption.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    image = args.image or args.image_url
    if not image:
        parser.error("one of --image or --image-url is required")

    caption = generate_caption(
        image=image,
        model_name=args.model,
        config_path=args.config,
    )
    print(json.dumps({"caption": caption}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()