from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from common import (
    DEFAULT_CONFIG_PATH,
    build_ip_index,
    default_images_dir,
    default_inputs,
    default_output_dir,
    image_filename,
    load_config,
    load_records,
    model_slug,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SigLIP image FAISS indexes for COCO/Flickr30k KB captions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", choices=["COCO", "Flickr30k"], required=True)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--inputs", type=Path, nargs="+", default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_batch_images(
    records: list[dict[str, Any]],
    images_dir: Path,
    input_path: Path,
    dataset_name: str,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        image_path = images_dir / image_filename(record, dataset_name)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for {input_path}: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def encode_images(
    records: list[dict[str, Any]],
    model: SiglipModel,
    processor: AutoProcessor,
    device: torch.device,
    images_dir: Path,
    input_path: Path,
    dataset_name: str,
    batch_size: int,
) -> np.ndarray:
    if not records:
        raise ValueError(f"No records in {input_path}")
    vectors: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = load_batch_images(batch_records, images_dir, input_path, dataset_name)
        inputs = to_device(processor(images=images, return_tensors="pt"), device)
        with torch.inference_mode():
            # Keep raw projected features to match the original SigLIP protocol.
            features = model.get_image_features(**inputs).float()
        vectors.append(features.cpu().numpy().astype(np.float32))
    return np.vstack(vectors)


def process_one(
    input_path: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model: SiglipModel,
    processor: AutoProcessor,
    device: torch.device,
) -> None:
    method_cfg = cfg["siglip"]
    records = load_records(input_path)
    images_dir = args.images_dir or default_images_dir(cfg, args.dataset)
    output_dir = args.output_dir or default_output_dir(cfg, args.dataset, "siglip")
    batch_size = args.batch_size or int(method_cfg["batch_size"])
    embeddings = encode_images(
        records,
        model,
        processor,
        device,
        images_dir,
        input_path,
        args.dataset,
        batch_size,
    )
    index = build_ip_index(embeddings)
    slug = model_slug(method_cfg["model_name"])
    faiss_path = output_dir / f"{input_path.stem}_{slug}.faiss"
    manifest_path = output_dir / f"{input_path.stem}_{slug}_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    write_json(
        manifest_path,
        {
            "input_path": str(input_path),
            "images_dir": str(images_dir),
            "faiss_path": str(faiss_path),
            "model": method_cfg["model_name"],
            "count": len(records),
            "dimension": int(embeddings.shape[1]),
            "metric": method_cfg["metric"],
            "batch_size": batch_size,
            "image_key": "filename" if args.dataset == "COCO" else "img",
            "siglip_protocol": (
                "Do not L2-normalize image embeddings; use raw dot product."
            ),
        },
    )
    print(
        json.dumps(
            {
                "faiss_path": str(faiss_path),
                "manifest_path": str(manifest_path),
                "count": len(records),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    method_cfg = cfg["siglip"]
    inputs = args.inputs or default_inputs(cfg, args.dataset, args.sizes)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SigLIP image index construction.")
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(
        method_cfg["model_name"],
        local_files_only=bool(method_cfg["local_files_only"]),
    )
    model = SiglipModel.from_pretrained(
        method_cfg["model_name"],
        torch_dtype=torch.float16,
        local_files_only=bool(method_cfg["local_files_only"]),
    ).to(device).eval()
    for input_path in inputs:
        process_one(input_path, args, cfg, model, processor, device)


if __name__ == "__main__":
    main()
