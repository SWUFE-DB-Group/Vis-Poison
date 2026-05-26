# Generate Caption

Synchronous OpenAI-compatible image caption generation.

## Files

- `generate_caption.py`: command-line entry point
- `_shared.py`: local image normalization and provider-specific request options
- `test_data/`: two sample images for smoke tests

## Config Files

Caption task config is committed in:

```bash
configs/generate_caption.json
```

This file only stores request options such as temperature and thinking control.
The caption prompt itself is defined in code:

```bash
src/generate_caption/_shared.py
```

Required format:

```json
{
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

The required `openai_models.json` format is documented in the project root README. That file is ignored by git and must not be committed. Add each display name you want to call under `models`, for example `GPT-5.4 -> your-provider-model-tag`.

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
uv run src/generate_caption/generate_caption.py --image src/generate_caption/test_data/color-c.jpg --model GPT-5.4
```

Local poison image test:

```bash
uv run src/generate_caption/generate_caption.py --image src/generate_caption/test_data/color-p.png --model GPT-5.4
```

Remote image URL test:

```bash
uv run src/generate_caption/generate_caption.py --image https://example.com/image.jpg --model GPT-5.4
```

Output is printed to stdout:

```json
{
  "caption": "..."
}
```

No result file is written by default. This folder is only for one-shot caption generation.
