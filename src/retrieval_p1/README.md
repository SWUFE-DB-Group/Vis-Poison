# Retrieval P1

Evaluate whether poisoned-image captions retrieve above clean KB captions.

## Data

Caption samples are read from:

```bash
dataset/webqa_final_category_difficulty_sample_70.json
```

KB files are stored under:

```bash
dataset/retrieval/
```

The evaluator expects COCO and Flickr30k caption JSON files plus prebuilt `text_embedding_3_large` FAISS indexes and manifests.

## Run

Evaluate all nine caption models:

```bash
uv run src/retrieval_p1/evaluate_caption_retrieval.py
```

Evaluate one model:

```bash
uv run src/retrieval_p1/evaluate_caption_retrieval.py --model "GPT-5.4"
```

Print stats:

```bash
uv run src/retrieval_p1/stats_caption_retrieval.py
```

Results are written to:

```bash
outputs/retrieval_p1/
```
