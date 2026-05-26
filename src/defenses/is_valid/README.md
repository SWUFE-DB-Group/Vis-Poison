# isValid

This folder contains the defense pipeline corresponding to the `isValid` setting in the paper.

The implementation is organized around two defense components:

- `jailbreak`: filter captions with a jailbreak guardrail
- `web-search fact-checking`: verify captions with an API model that has search enabled

The four groups evaluated in this setting are:

- `ours`
- `mm`
- `poisoned`
- `eye`

To make the repository structure match the paper more closely, the three non-ours construction scripts are separated:

- `construct_mm_captions.py`
- `construct_poisoned_captions.py`
- `construct_eye_captions.py`

The `ours` group reuses captions already stored in the dataset:

- `reuse_ours_captions.py`

## Files

- `_shared.py`: shared helpers for API calls, JSON parsing, and paths
- `reuse_ours_captions.py`: prepare the `ours` captions from dataset records
- `construct_mm_captions.py`: construct the `mm` baseline captions
- `construct_poisoned_captions.py`: construct the `poisoned` baseline captions
- `construct_eye_captions.py`: construct the `eye` baseline captions
- `web_search_fact_checking.py`: run web-search fact-checking on generated captions
- `jailbreak_filtering.py`: run jailbreak filtering after fact-checking
- `summarize_is_valid.py`: summarize the defense results

## Inputs

Default benchmark input:

```text
results/defenses/sample/sample_180.json
```

Each record is expected to contain:

- `id`
- `question`
- `correct_answer`
- `wrong_answer`
- `captions` for the `ours` setting

## Outputs

Outputs are written under:

```text
results/defenses/is_valid/
```

Per-group caption files:

```text
results/defenses/is_valid/ours/caption_180_ours.json
results/defenses/is_valid/mm/caption_180_mm.json
results/defenses/is_valid/poisoned/caption_180_poisoned.json
results/defenses/is_valid/eye/caption_180_eye.json
```

Fact-check outputs:

```text
results/defenses/is_valid/<group>/caption_180_<group>_factcheck.json
```

Jailbreak outputs:

```text
results/defenses/is_valid/<group>/caption_180_<group>_jailbreak_after_factcheck.json
```

Summary outputs:

```text
results/defenses/is_valid/stats_report.json
results/defenses/is_valid/stats_report.md
```

## Pipeline

Run the caption construction step for each group:

```bash
uv run src/defenses/is_valid/reuse_ours_captions.py
uv run src/defenses/is_valid/construct_mm_captions.py
uv run src/defenses/is_valid/construct_poisoned_captions.py
uv run src/defenses/is_valid/construct_eye_captions.py
```

Run web-search fact-checking:

```bash
uv run src/defenses/is_valid/web_search_fact_checking.py
```

Run jailbreak filtering for the groups that use it:

```bash
uv run src/defenses/is_valid/jailbreak_filtering.py
```

Summarize the final results:

```bash
uv run src/defenses/is_valid/summarize_is_valid.py
```

## Notes

- `eye` only goes through fact-checking in the current pipeline.
- OpenAI-compatible API settings come from the local git-ignored `configs/openai_models.json`.
