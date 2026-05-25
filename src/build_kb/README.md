# KB Construction

Build FAISS KB indexes for COCO and Flickr30k caption samples.

Configuration lives in `configs/kb_construction.json`. The closed-source `text-embedding-3-large` backend is called through the git-ignored `configs/openai_models.json`.

## Data Layout

Caption inputs:

```text
dataset/COCO/coco_random_1k.json
dataset/Flickr30k/flickr30k_random_1k.json
```

Image directories are expected at:

```text
dataset/COCO/imgs
dataset/Flickr30k/imgs
```

The repo includes README files for these image directories, but not the images.

## Commands

Run from the project root:

```bash
uv run src/build_kb/build_clip_image_faiss.py --dataset COCO
uv run src/build_kb/build_siglip_image_faiss.py --dataset COCO
uv run src/build_kb/build_qwen_image_faiss.py --dataset COCO
uv run src/build_kb/build_text_embedding_faiss.py --dataset COCO
```

Replace `COCO` with `Flickr30k` for Flickr30k. Outputs are written under `dataset/<DATASET>/<method>/`.

Before running `build_text_embedding_faiss.py`, make sure `configs/openai_models.json` contains a `models` mapping for:

```json
{
  "text-embedding-3-large": "your-provider-embedding-tag"
}
```

The methods match the original setup:

- CLIP: `openai/clip-vit-base-patch16`, L2-normalized image embeddings, `IndexFlatIP`.
- SigLIP: `google/siglip-large-patch16-256`, raw image embeddings, `IndexFlatIP`.
- Qwen: local `Qwen/Qwen3-VL-Embedding-2B`, normalized image embeddings, `IndexFlatIP`.
- Text embedding: `text-embedding-3-large`, normalized caption embeddings, `IndexFlatIP`.
