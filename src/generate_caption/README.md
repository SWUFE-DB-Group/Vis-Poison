# Generate Caption

Synchronous OpenAI-compatible image caption generation.

## Config

Create this local config file:

```bash
configs/generate_caption.json
```

This file is ignored by git because it contains API credentials. Required format:

```json
{
  "openai": {
    "base_url": "https://api.example.com/v1",
    "api_key": "YOUR_API_KEY",
    "timeout": 300
  },
  "default_model_name": "Qwen3.5-397B-A17B",
  "models": {
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "GPT-5.4": "gpt-5.4",
    "Qwen3.6-Plus": "qwen3.6-plus",
    "Llama 4 Maverick": "llama-4-maverick",
    "Kimi-K2.6": "kimi-k2.6",
    "Qwen3.5-397B-A17B": "qwen3.5-397b-a17b",
    "Llama 4 Scout": "llama-4-scout",
    "Qwen3.6-35B-A3B": "qwen3.6-35b-a3b",
    "Qwen3.6-27B": "qwen3.6-27b"
  },
  "prompt": "You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval.",
  "model_options": {
    "temperature": 0,
    "qwen_enable_thinking": false,
    "gpt_reasoning_effort": "none"
  }
}
```

`models` maps display names used by `--model` to API model tags. You may pass either the display name or the API tag.

## Test Images

Two local test images are provided:

```bash
src/generate_caption/test_data/color-c.jpg
src/generate_caption/test_data/color-p.png
```

Local clean image test:

```bash
uv run python src/generate_caption/generate_caption.py --image src/generate_caption/test_data/color-c.jpg --model "GPT-5.4"
```

Local poison image test:

```bash
uv run python src/generate_caption/generate_caption.py --image src/generate_caption/test_data/color-p.png --model "GPT-5.4"
```

Remote image URL test:

```bash
uv run python src/generate_caption/generate_caption.py --image "https://example.com/image.jpg" --model "GPT-5.4"
```

Output is printed to stdout:

```json
{
  "caption": "..."
}
```