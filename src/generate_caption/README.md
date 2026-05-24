# Generate Caption

Synchronous OpenAI-compatible image caption generation.

## Config Files

Caption task config is committed in:

```bash
configs/generate_caption.json
```

Required format:

```json
{
  "prompt": "You are an assistant tasked with summarizing images for retrieval. These summaries will be embedded and used to retrieve the raw image. Give a concise summary of the image that is well optimized for retrieval.",
  "model_options": {
    "temperature": 0,
    "disable_thinking": true,
    "thinking_disabled_extra_body_by_prefix": {
      "qwen": {
        "enable_thinking": false
      },
      "kimi-": {
        "thinking": {
          "type": "disabled"
        }
      },
      "claude-": {
        "thinking": {
          "type": "disabled"
        }
      },
      "glm-": {
        "thinking": {
          "type": "disabled"
        }
      }
    },
    "reasoning_effort_by_prefix": {
      "gpt-": "none"
    }
  }
}
```

API credentials and model-name mappings are local settings in:

```bash
configs/openai_models.json
```

The required `openai_models.json` format is documented in the project root README. That file is ignored by git and must not be committed.

`disable_thinking` is implemented with provider-specific request parameters:

- Qwen: `extra_body.enable_thinking = false`
- Kimi, Claude, GLM: `extra_body.thinking.type = disabled`
- GPT: `reasoning_effort = none`

Additional providers can be handled by adding prefixes under `thinking_disabled_extra_body_by_prefix` or `reasoning_effort_by_prefix`.

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