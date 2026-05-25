from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import AsyncOpenAI

from common import DEFAULT_CONFIG_PATH, build_ip_index, default_inputs, default_output_dir, load_config, load_records, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build text-embedding-3-large FAISS indexes for caption text.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--openai-config", type=Path, default=Path("configs/openai_models.json"))
    parser.add_argument("--dataset", choices=["COCO", "Flickr30k"], required=True)
    parser.add_argument("--sizes", nargs="+", default=None)
    parser.add_argument("--inputs", type=Path, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    return parser.parse_args()


def load_openai_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "openai" not in data:
        raise ValueError(f"Missing `openai` section in {path}")
    return data["openai"]


async def embed_batches(client: AsyncOpenAI, model: str, texts: list[str], batch_size: int, concurrency: int) -> np.ndarray:
    semaphore = asyncio.Semaphore(concurrency)
    batches = [(idx, texts[idx : idx + batch_size]) for idx in range(0, len(texts), batch_size)]

    async def embed_one(start: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        async with semaphore:
            response = await client.embeddings.create(model=model, input=batch)
        return start, [item.embedding for item in response.data]

    results = await asyncio.gather(*(embed_one(start, batch) for start, batch in batches))
    ordered: list[list[float]] = []
    for _, vectors in sorted(results, key=lambda item: item[0]):
        ordered.extend(vectors)
    embeddings = np.asarray(ordered, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


async def process_one(input_path: Path, args: argparse.Namespace, cfg: dict[str, Any], client: AsyncOpenAI, model_name: str) -> None:
    method_cfg = cfg["text_embedding_3_large"]
    records = load_records(input_path)
    text_key = method_cfg["text_key"]
    texts = [str(record.get(text_key, "")).strip() for record in records]
    if any(not text for text in texts):
        raise ValueError(f"Empty `{text_key}` value found in {input_path}")
    batch_size = args.batch_size or int(method_cfg["batch_size"])
    concurrency = args.concurrency or int(method_cfg["concurrency"])
    embeddings = await embed_batches(client, model_name, texts, batch_size, concurrency)
    index = build_ip_index(embeddings)
    output_dir = args.output_dir or default_output_dir(cfg, args.dataset, "text_embedding_3_large")
    faiss_path = output_dir / f"{input_path.stem}_text_embedding_3_large.faiss"
    manifest_path = output_dir / f"{input_path.stem}_text_embedding_3_large_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    write_json(
        manifest_path,
        {
            "input_path": str(input_path),
            "faiss_path": str(faiss_path),
            "model": model_name,
            "text_key": text_key,
            "count": len(records),
            "dimension": int(embeddings.shape[1]),
            "metric": method_cfg["metric"],
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_retries": method_cfg["max_retries"],
        },
    )
    print(json.dumps({"faiss_path": str(faiss_path), "manifest_path": str(manifest_path), "count": len(records)}, ensure_ascii=False))


async def amain() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    method_cfg = cfg["text_embedding_3_large"]
    api_cfg = load_openai_config(args.openai_config)
    client = AsyncOpenAI(
        base_url=api_cfg.get("base_url"),
        api_key=api_cfg["api_key"],
        timeout=api_cfg.get("timeout", 300),
        max_retries=int(method_cfg["max_retries"]),
    )
    model_name = method_cfg["model_name"]
    inputs = args.inputs or default_inputs(cfg, args.dataset, args.sizes)
    for input_path in inputs:
        await process_one(input_path, args, cfg, client, model_name)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
