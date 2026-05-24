# Generate Caption

Synchronous OpenAI-compatible image caption generation.

## Config

Caption prompt and generation options are committed in:

```bash
configs/generate_caption.json
```

API credentials and model-name mappings are local settings in:

```bash
configs/openai_models.json
```

The required `openai_models.json` format is documented in the project root README. That file is ignored by git and must not be committed.

Thinking is disabled through `configs/generate_caption.json` using provider-specific settings:

- Qwen: `enable_thinking = false`
- Kimi, Claude, GLM: `thinking.type = disabled`
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