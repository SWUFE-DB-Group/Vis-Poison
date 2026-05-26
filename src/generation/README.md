# Generation

API-based generation scripts for the three settings from `exp1.2-generation`:

- `q_only.py`
- `q_clean.py`
- `q_poison.py`
- `answer_validator.py`
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

The generation prompts themselves are defined in:

```text
src/generation/_shared.py
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

`answer_validator.py` validates whether a model answer semantically aligns with a reference answer.
It uses one prompt for both settings:

- correctness checking: reference answer = clean / gold answer
- misleading checking: reference answer = attacker target answer

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

Validate one answer pair:

```bash
uv run src/generation/answer_validator.py --question "What color is the car?" --reference-answer "red" --model-answer "The car is bright red."
```

Validate a result file against clean answers:

```bash
uv run src/generation/answer_validator.py --input results/generation/hard/gpt-5-4-q_only.json --reference-fields correct_answer,answer,A --positive-label correct --negative-label wrong
```

Validate a result file against attacker target answers:

```bash
uv run src/generation/answer_validator.py --input results/generation/hard/gpt-5-4-q_poison.json --reference-fields wrong_answer,attacker_answer,adv_answer --positive-label misled --negative-label not_misled
```

Batch mode accepts either:

- a JSON list of records
- a generation result object with a top-level `results` list

Each record should contain:

- `question`
- one reference-answer field from `--reference-fields`
- one model-answer field from `--model-answer-fields` (default prefers `model_answer`)

Dot paths such as `nested.answer` are supported. When the input is a generation result object, the validator preserves the wrapper metadata and only rewrites the `results` list.

The evaluation script for these outputs lives in:

```text
src/evaluation_framework/evaluate_knowledge_aware.py
```
