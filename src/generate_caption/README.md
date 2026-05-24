# Generate Caption

Synchronous OpenAI-compatible image caption generation.

## Config

```bash
configs/generate_caption.json
```

`models` maps display names to API model tags, for example:

```json
"GPT-5.4": "gpt-5.4"
```

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