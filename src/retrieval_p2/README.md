# Retrieval P2

Run poisoned-image retrieval evaluation for CLIP, SigLIP, and Qwen3-VL-Embedding.

Configuration lives in `configs/retrieval_p2.json`. The three P2 backends are local open-source models, and their paths are configured there.

## Prerequisites

You need these inputs before running P2:

1. WebQA records:

```text
dataset/webqa_final_category_difficulty_sample_70.json
```

2. Poisoned-image construction outputs from `src/poisoned_image_construction/`:

```text
outputs/poisoned_image_construction/report.json
outputs/poisoned_image_construction/images/
```

3. Clean KB image indexes built by `src/build_kb/`:

```text
dataset/COCO/clip/*.faiss
dataset/COCO/siglip/*.faiss
dataset/COCO/qwen/*.faiss
dataset/Flickr30k/clip/*.faiss
dataset/Flickr30k/siglip/*.faiss
dataset/Flickr30k/qwen/*.faiss
```

If the clean KB FAISS files do not exist yet, run `src/build_kb/` first.

## Scripts

Main script:

```text
src/retrieval_p2/run_retrieval.py
```

Stats script:

```text
src/retrieval_p2/stats_retrieval.py
```

## Input Resolution

The main script reads:

- `dataset/webqa_final_category_difficulty_sample_70.json`
- `outputs/poisoned_image_construction/report.json`
- `outputs/poisoned_image_construction/images/`

Poison image paths are resolved in this order:

1. direct poison-image fields in the dataset record
2. `poison_construction.poison_image_path` from the construction report
3. files named like `<sample_id>_round*.png` under the poison image directory

The clean KB JSON and FAISS files are expected to come from `src/build_kb`.

## Commands

Run every backend, both datasets, and all three KB sizes:

```bash
uv run src/retrieval_p2/run_retrieval.py
```

Run one backend:

```bash
uv run src/retrieval_p2/run_retrieval.py --backend clip
```

Run one backend on one dataset and one size:

```bash
uv run src/retrieval_p2/run_retrieval.py --backend qwen --dataset Flickr30k --size 10k
```

Overwrite existing result files:

```bash
uv run src/retrieval_p2/run_retrieval.py --backend siglip --overwrite
```

Print stats from finished runs:

```bash
uv run src/retrieval_p2/stats_retrieval.py
```

Results are written under:

```text
outputs/retrieval_p2/
```

The main script writes one result JSON per backend/dataset/size and a top-level `retrieval_summary.json`.
