# Generation

API-based generation scripts for the three settings from `exp1.2-generation`:

- `q_only.py`
- `q_clean.py`
- `q_poison.py`
- `_shared.py`

All scripts use the local git-ignored `configs/openai_models.json` and only support these model names:

- `Claude Sonnet 4.6`
- `GPT-5.4`
- `Qwen3.6-Plus`
- `Llama 4 Maverick`
- `Kimi-K2.6`
- `Qwen3.5-397B-A17B`

Committed task config:

```text
configs/generation.json
```

Local API config:

```text
configs/openai_models.json
```

The `models` section in `openai_models.json` must include the six display names above with provider-specific tags.

## Inputs

Default dataset:

```text
dataset/webqa_final_category_difficulty_sample_70.json
```

The scripts read:

- `question`
- `difficulty`
- `img.clean`, `source_image`, `clean_image.path`, or `image_path` for `q_clean`
- `img.poison`, `poison_image_path`, `poison_image`, or `poison_construction.poison_image_path` for `q_poison`

Outputs are written under:

```text
results/generation/<difficulty>/
```

Each run writes one JSON file per model and mode.

## Commands

Question only:

```bash
uv run src/generation/q_only.py --difficulty hard
```

Question + clean image:

```bash
uv run src/generation/q_clean.py --difficulty hard
```

Question + poison image:

```bash
uv run src/generation/q_poison.py --difficulty hard
```

Run one model only:

```bash
uv run src/generation/q_poison.py --difficulty hard --model GPT-5.4
```

Small smoke test:

```bash
uv run src/generation/q_clean.py --difficulty hard --model GPT-5.4 --max-items 1 --overwrite
```

The evaluation script for these outputs lives in:

```text
src/evaluation_framework/evaluate_knowledge_aware.py
```
