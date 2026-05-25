from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch

from common import (
    DEFAULT_CONFIG_PATH,
    build_ip_index,
    default_images_dir,
    default_inputs,
    default_output_dir,
    image_filename,
    load_config,
    load_records,
    resolve_repo_path,
    write_json,
)
from qwen3_vl_embedding import Qwen3VLEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen3-VL-Embedding image FAISS indexes for COCO/Flickr30k KB captions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", choices=["COCO", "Flickr30k"], required=True)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--inputs", type=Path, nargs="+", default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def build_embedder(method_cfg: dict[str, Any]) -> Qwen3VLEmbedder:
    configured_path = str(method_cfg["model_name_or_path"])
    candidate_path = resolve_repo_path(configured_path)
    model_name_or_path = str(candidate_path) if candidate_path.exists() or Path(configured_path).is_absolute() else configured_path
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return Qwen3VLEmbedder(
        model_name_or_path,
        default_instruction=method_cfg["instruction"],
        torch_dtype=dtype,
        device_map=device_map,
    )


def encode_images(
    embedder: Qwen3VLEmbedder,
    records: list[dict[str, Any]],
    images_dir: Path,
    input_path: Path,
    dataset_name: str,
    batch_size: int,
) -> np.ndarray:
    if not records:
        raise ValueError(f"No records in {input_path}")
    vectors: list[np.ndarray] = []
    image_paths = [images_dir / image_filename(record, dataset_name) for record in records]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {input_path}: {image_path}")
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_inputs = [{"image": str(path)} for path in batch_paths]
        with torch.no_grad():
            embeddings = embedder.process(batch_inputs, normalize=True)
        vectors.append(embeddings.detach().float().cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(vectors, axis=0)
    faiss.normalize_L2(embeddings)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def process_one(input_path: Path, args: argparse.Namespace, cfg: dict[str, Any], embedder: Qwen3VLEmbedder) -> None:
    method_cfg = cfg["qwen"]
    records = load_records(input_path)
    images_dir = args.images_dir or default_images_dir(cfg, args.dataset)
    output_dir = args.output_dir or default_output_dir(cfg, args.dataset, "qwen")
    batch_size = args.batch_size or int(method_cfg["batch_size"])
    embeddings = encode_images(embedder, records, images_dir, input_path, args.dataset, batch_size)
    index = build_ip_index(embeddings)
    faiss_path = output_dir / f"{input_path.stem}_{method_cfg['model_slug']}.faiss"
    manifest_path = output_dir / f"{input_path.stem}_{method_cfg['model_slug']}_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    write_json(
        manifest_path,
        {
            "input_path": str(input_path),
            "images_dir": str(images_dir),
            "faiss_path": str(faiss_path),
            "model": method_cfg["model_name"],
            "backend": "local_qwen3_vl_embedder",
            "count": len(records),
            "dimension": int(embeddings.shape[1]),
            "metric": method_cfg["metric"],
            "batch_size": batch_size,
            "image_key": "filename" if args.dataset == "COCO" else "img",
            "document_instruction": method_cfg["instruction"],
        },
    )
    print(json.dumps({"faiss_path": str(faiss_path), "manifest_path": str(manifest_path), "count": len(records)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs = args.inputs or default_inputs(cfg, args.dataset, args.sizes)
    embedder = build_embedder(cfg["qwen"])
    for input_path in inputs:
        process_one(input_path, args, cfg, embedder)


if __name__ == "__main__":
    main()
