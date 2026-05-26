# Retrieval P1

Evaluate whether poisoned-image captions retrieve above clean KB captions.

## Prerequisites

You need these inputs before running P1:

```text
dataset/webqa_final_category_difficulty_sample_70.json
```

This file must already contain the `captions` field for the nine caption models.

Use `src/generate_caption/` first if the captions are not ready yet.

You also need `configs/openai_models.json` with a `models` mapping for:

```json
{
  "text-embedding-3-large": "your-provider-embedding-tag"
}
```

You also need the prebuilt text KB files under:

```text
dataset/retrieval/
```

P1 reads:

- `dataset/retrieval/COCO/*_captions.json`
- `dataset/retrieval/COCO/text_embedding_3_large/*.faiss`
- `dataset/retrieval/Flickr30k/*_captions.json`
- `dataset/retrieval/Flickr30k/text_embedding_3_large/*.faiss`

The repository does not commit these KB files by default. Prepare them first, then run the scripts below.

## Scripts

- `src/retrieval_p1/run_retrieval.py`: run retrieval evaluation with `text-embedding-3-large`
- `src/retrieval_p1/stats_retrieval.py`: summarize finished result files

The result directory is not committed by default. Run `run_retrieval.py` before `stats_retrieval.py`.

## Run

Evaluate all nine caption models:

```bash
uv run src/retrieval_p1/run_retrieval.py
```

Evaluate one model:

```bash
uv run src/retrieval_p1/run_retrieval.py --model "GPT-5.4"
```

Print stats:

```bash
uv run src/retrieval_p1/stats_retrieval.py
```

Results are written to:

```text
results/retrieval_p1/
```

Each caption model produces one result JSON plus a `retrieval_summary.json`.

Tie policy: exact score ties are treated as non-hits, so the reported top-1 and top-3 rates stay conservative.
