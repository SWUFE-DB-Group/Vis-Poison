# KB Construction

Build FAISS KB indexes for COCO and Flickr30k caption samples.

## Files

- `build_clip_image_faiss.py`: build CLIP image indexes
- `build_siglip_image_faiss.py`: build SigLIP image indexes
- `build_qwen_image_faiss.py`: build Qwen3-VL image indexes
- `build_text_embedding_faiss.py`: build caption-text indexes with `text-embedding-3-large`
- `common.py`: shared path and dataset helpers
- `qwen3_vl_embedding.py`: local Qwen3-VL embedding wrapper

Configuration lives in `configs/kb_construction.json`. The closed-source `text-embedding-3-large` backend uses the local git-ignored `configs/openai_models.json`.

## Inputs

Caption JSON files are expected under:

```text
dataset/COCO/coco_random_{1k,10k,30k}.json
dataset/Flickr30k/flickr30k_random_{1k,10k,30k}.json
```

Image directories are expected at:

```text
dataset/COCO/imgs/
dataset/Flickr30k/imgs/
```

The repo includes README files for these image directories, but not the images themselves.

## Outputs

Outputs are written under:

```text
dataset/COCO/{clip,siglip,qwen,text_embedding_3_large}/
dataset/Flickr30k/{clip,siglip,qwen,text_embedding_3_large}/
```

Each run writes a `.faiss` index and a matching `_manifest.json`.

## Commands

Run from the project root.

Build all four index types for one dataset:

```bash
uv run src/build_kb/build_clip_image_faiss.py --dataset COCO
uv run src/build_kb/build_siglip_image_faiss.py --dataset COCO
uv run src/build_kb/build_qwen_image_faiss.py --dataset COCO
uv run src/build_kb/build_text_embedding_faiss.py --dataset COCO
```

Build a specific size:

```bash
uv run src/build_kb/build_clip_image_faiss.py --dataset Flickr30k --sizes 10k
```

Build multiple sizes:

```bash
uv run src/build_kb/build_qwen_image_faiss.py --dataset COCO --sizes 1k 10k 30k
```

Before running `build_text_embedding_faiss.py`, make sure `configs/openai_models.json` contains a `models` mapping for:

```json
{
  "text-embedding-3-large": "your-provider-embedding-tag"
}
```

## Notes

The methods match the original setup:

- CLIP: `openai/clip-vit-base-patch16`, L2-normalized image embeddings, `IndexFlatIP`
- SigLIP: `google/siglip-large-patch16-256`, raw image embeddings, `IndexFlatIP`
- Qwen: local `Qwen/Qwen3-VL-Embedding-2B`, normalized image embeddings, `IndexFlatIP`
- Text embedding: `text-embedding-3-large`, normalized caption embeddings, `IndexFlatIP`
