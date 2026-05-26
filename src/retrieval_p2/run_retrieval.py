from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel, SiglipModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "retrieval_p2.json"
QWEN_EMBEDDER_PATH = REPO_ROOT / "src" / "build_kb" / "qwen3_vl_embedding.py"
POISON_FIELD_CANDIDATES = [
    "poison_image_path",
    "poison_image",
    "poison.path",
    "poison_construction.poison_image_path",
    "img.poison",
]
QUESTION_FIELD_CANDIDATES = ["question", "Q", "query"]
SAMPLE_ID_FIELD_CANDIDATES = ["id", "sample_id", "qid", "no"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def get_nested(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def first_present(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = get_nested(record, key) if "." in key else record.get(key)
        if value not in (None, ""):
            return value
    return None


def model_slug(model_name: str) -> str:
    return model_name.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def result_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def poison_image_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_round(\d+)", path.stem)
    round_number = int(match.group(1)) if match else -1
    return round_number, path.name


def build_poison_report_lookup(report: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not isinstance(report, dict):
        return lookup
    for sample_id, row in report.items():
        if not isinstance(row, dict):
            continue
        poison_path = first_present(
            row,
            ["poison_image_path", "poison_construction.poison_image_path"],
        )
        if poison_path:
            lookup[str(sample_id)] = str(poison_path)
    return lookup


def resolve_poison_image_path(
    record: dict[str, Any],
    sample_keys: list[str],
    poison_lookup: dict[str, str],
    poison_image_dir: Path,
) -> Path:
    direct_value = first_present(record, POISON_FIELD_CANDIDATES)
    if direct_value:
        direct_path = resolve_repo_path(str(direct_value))
        if direct_path.exists():
            return direct_path

    for sample_key in sample_keys:
        poison_value = poison_lookup.get(sample_key)
        if poison_value:
            poison_path = resolve_repo_path(poison_value)
            if poison_path.exists():
                return poison_path

    candidates: list[Path] = []
    if poison_image_dir.exists():
        for sample_key in sample_keys:
            candidates.extend(poison_image_dir.glob(f"{sample_key}_round*.png"))
            candidates.extend(poison_image_dir.glob(f"{sample_key}_round*.jpg"))
            candidates.extend(poison_image_dir.glob(f"{sample_key}_round*.jpeg"))
        if candidates:
            return sorted(candidates, key=poison_image_sort_key)[-1]

    raise FileNotFoundError(
        f"Could not resolve poison image for sample keys: {sample_keys}"
    )


def normalize_webqa_records(
    dataset: list[dict[str, Any]],
    poison_lookup: dict[str, str],
    poison_image_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(dataset):
        if not isinstance(record, dict):
            continue
        question = first_present(record, QUESTION_FIELD_CANDIDATES)
        if not question:
            raise ValueError(f"Missing question at index {index}")
        sample_keys = [str(index)]
        for key in SAMPLE_ID_FIELD_CANDIDATES:
            value = record.get(key)
            if value not in (None, ""):
                sample_keys.append(str(value))
        poison_path = resolve_poison_image_path(
            record,
            sample_keys,
            poison_lookup,
            poison_image_dir,
        )
        rows.append(
            {
                "record_index": index,
                "sample_keys": sample_keys,
                "id": first_present(record, SAMPLE_ID_FIELD_CANDIDATES),
                "question": str(question).strip(),
                "difficulty": record.get("difficulty"),
                "class": record.get("class"),
                "entity": record.get("entity"),
                "poison_image_path": poison_path,
            }
        )
    return rows


def load_base_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}")
    return data


def get_base_filename(record: dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "COCO":
        filename = record.get("filename")
        if not filename and record.get("img") is not None:
            filename = f"{int(record['img']):012d}.jpg"
    else:
        filename = record.get("filename") or record.get("img")
    filename_str = str(filename or "").strip()
    if not filename_str:
        raise ValueError(f"Missing image filename in base record: {record}")
    return filename_str


def get_base_paths(
    config: dict[str, Any],
    backend_name: str,
    dataset_name: str,
    size: str,
) -> tuple[Path, Path]:
    dataset_cfg = config["datasets"][dataset_name]
    backend_cfg = config["backends"][backend_name]
    base_prefix = dataset_cfg["base_prefix"]
    base_dir = resolve_repo_path(dataset_cfg["base_json_dir"])
    base_json = base_dir / f"{base_prefix}_{size}.json"
    faiss_dir = base_dir / backend_cfg["faiss_subdir"]
    faiss_path = faiss_dir / f"{base_prefix}_{size}_{backend_cfg['faiss_suffix']}.faiss"
    return base_json, faiss_path


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def ensure_feature_tensor(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError(f"Expected Tensor or output with pooler_output, got {type(output)!r}")


def load_rgb_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


class ClipBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CLIP retrieval.")
        self.config = config
        self.device = torch.device("cuda")
        model_path = str(resolve_repo_path(config["model_name_or_path"]))
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=bool(config.get("local_files_only", True)),
        )
        self.model = CLIPModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=bool(config.get("local_files_only", True)),
        ).to(self.device).eval()
        self.batch_size = int(config["batch_size"])

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(questions), self.batch_size):
            batch = questions[start : start + self.batch_size]
            inputs = to_device(
                self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ),
                self.device,
            )
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
            features = F.normalize(ensure_feature_tensor(features).float(), dim=-1)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = load_rgb_images(batch_paths)
            inputs = to_device(self.processor(images=images, return_tensors="pt"), self.device)
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
            features = F.normalize(ensure_feature_tensor(features).float(), dim=-1)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)


class SiglipBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SigLIP retrieval.")
        self.config = config
        self.device = torch.device("cuda")
        model_path = str(resolve_repo_path(config["model_name_or_path"]))
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=bool(config.get("local_files_only", True)),
        )
        self.model = SiglipModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=bool(config.get("local_files_only", True)),
        ).to(self.device).eval()
        self.batch_size = int(config["batch_size"])

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(questions), self.batch_size):
            batch = questions[start : start + self.batch_size]
            inputs = to_device(
                self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                ),
                self.device,
            )
            with torch.inference_mode():
                features = ensure_feature_tensor(self.model.get_text_features(**inputs)).float()
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = load_rgb_images(batch_paths)
            inputs = to_device(self.processor(images=images, return_tensors="pt"), self.device)
            with torch.inference_mode():
                features = ensure_feature_tensor(self.model.get_image_features(**inputs)).float()
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)


def load_qwen_embedder_class() -> Any:
    spec = importlib.util.spec_from_file_location(
        "retrieval_p2_qwen_embedder",
        QWEN_EMBEDDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Qwen embedder from {QWEN_EMBEDDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


class QwenBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        model_path = str(resolve_repo_path(config["model_name_or_path"]))
        Qwen3VLEmbedder = load_qwen_embedder_class()
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.embedder = Qwen3VLEmbedder(
            model_path,
            default_instruction=config["document_instruction"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.batch_size = int(config["batch_size"])
        self.query_instruction = config["query_instruction"]
        self.document_instruction = config["document_instruction"]

    def _encode_inputs(self, inputs: list[dict[str, Any]]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start : start + self.batch_size]
            with torch.no_grad():
                embeddings = self.embedder.process(batch, normalize=True)
            vectors.append(embeddings.detach().float().cpu().numpy().astype(np.float32))
        array = np.concatenate(vectors, axis=0)
        faiss.normalize_L2(array)
        return np.ascontiguousarray(array, dtype=np.float32)

    def encode_queries(self, questions: list[str]) -> np.ndarray:
        inputs = [
            {"text": question, "instruction": self.query_instruction}
            for question in questions
        ]
        return self._encode_inputs(inputs)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        inputs = [
            {"image": str(path), "instruction": self.document_instruction}
            for path in paths
        ]
        return self._encode_inputs(inputs)


def build_backend(name: str, config: dict[str, Any]) -> ClipBackend | SiglipBackend | QwenBackend:
    if name == "clip":
        return ClipBackend(config)
    if name == "siglip":
        return SiglipBackend(config)
    if name == "qwen":
        return QwenBackend(config)
    raise ValueError(f"Unsupported backend: {name}")


def load_base_embeddings(faiss_path: Path, normalize: bool) -> np.ndarray:
    index = faiss.read_index(str(faiss_path))
    if not isinstance(index, faiss.IndexFlatIP):
        raise TypeError(
            f"Expected IndexFlatIP at {faiss_path}, got {type(index).__name__}"
        )
    embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    if normalize:
        faiss.normalize_L2(embeddings)
    return embeddings


def get_topk_base(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k >= scores.shape[0]:
        indices = np.argsort(-scores)
        return indices, scores[indices]
    top_indices_unsorted = np.argpartition(-scores, k - 1)[:k]
    top_indices = top_indices_unsorted[np.argsort(-scores[top_indices_unsorted])]
    return top_indices, scores[top_indices]


def evaluate_run(
    *,
    backend_name: str,
    backend_cfg: dict[str, Any],
    backend: ClipBackend | SiglipBackend | QwenBackend,
    dataset_name: str,
    size: str,
    config: dict[str, Any],
    webqa_rows: list[dict[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = resolve_repo_path(config["run"]["output_dir"]) / backend_name
    output_path = (
        output_dir / f"retrieval_{backend_name}_{result_slug(dataset_name)}_{size}.json"
    )
    if output_path.exists() and not overwrite:
        return load_json(output_path)

    base_json, base_faiss = get_base_paths(config, backend_name, dataset_name, size)
    for path in (base_json, base_faiss):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    base_records = load_base_records(base_json)
    base_embeddings = load_base_embeddings(
        base_faiss,
        normalize=bool(backend_cfg["normalize_base_embeddings"]),
    )
    if base_embeddings.shape[0] != len(base_records):
        raise ValueError(
            f"Base record count mismatch for {backend_name}/{dataset_name}/{size}: "
            f"embeddings={base_embeddings.shape[0]}, records={len(base_records)}"
        )

    questions = [row["question"] for row in webqa_rows]
    poison_paths = [row["poison_image_path"] for row in webqa_rows]
    query_embeddings = backend.encode_queries(questions)
    poison_embeddings = backend.encode_images(poison_paths)
    if query_embeddings.shape != poison_embeddings.shape:
        raise RuntimeError(
            "Embedding shape mismatch: "
            f"query={query_embeddings.shape}, poison={poison_embeddings.shape}"
        )
    if query_embeddings.shape[1] != base_embeddings.shape[1]:
        raise RuntimeError(
            f"Embedding dim mismatch for {backend_name}/{dataset_name}/{size}: "
            f"query={query_embeddings.shape[1]}, base={base_embeddings.shape[1]}"
        )

    top_k = int(config["run"]["top_k"])
    top1_hits = 0
    top3_hits = 0
    records_out: list[dict[str, Any]] = []

    for index, row in enumerate(webqa_rows):
        query_vec = query_embeddings[index]
        poison_vec = poison_embeddings[index]
        base_scores = base_embeddings @ query_vec
        poison_score = float(np.dot(poison_vec, query_vec))
        # Use a conservative tie policy: base images win exact-score ties.
        poison_rank = int(np.sum(base_scores >= poison_score) + 1)
        hit_top1 = poison_rank <= 1
        hit_top3 = poison_rank <= 3
        top1_hits += int(hit_top1)
        top3_hits += int(hit_top3)

        top_base_idx, top_base_scores = get_topk_base(base_scores, k=top_k)
        candidates: list[dict[str, Any]] = []
        for rank_position, (base_index, base_score) in enumerate(
            zip(top_base_idx.tolist(), top_base_scores.tolist()),
            start=1,
        ):
            candidates.append(
                {
                    "type": "base",
                    "base_index": int(base_index),
                    "base_image": get_base_filename(base_records[base_index], dataset_name),
                    "score": float(base_score),
                    "base_rank": rank_position,
                }
            )
        candidates.append(
            {
                "type": "poison",
                "base_index": int(base_embeddings.shape[0]),
                "base_image": str(row["poison_image_path"]),
                "score": poison_score,
            }
        )
        # Break ties in favor of base images so poison needs strictly competitive scores.
        candidates.sort(
            key=lambda item: (-item["score"], 1 if item["type"] == "poison" else 0)
        )

        records_out.append(
            {
                "record_index": row["record_index"],
                "id": row["id"],
                "difficulty": row["difficulty"],
                "class": row["class"],
                "entity": row["entity"],
                "question": row["question"],
                "poison_image_path": str(row["poison_image_path"]),
                "poison_score": poison_score,
                "poison_rank_among_base_plus_one_poison": poison_rank,
                "hit_top1": bool(hit_top1),
                "hit_top3": bool(hit_top3),
                "top3_after_inserting_poison": candidates[:top_k],
                "base_top3_without_poison": [
                    {
                        "base_index": int(base_index),
                        "base_image": get_base_filename(
                            base_records[base_index],
                            dataset_name,
                        ),
                        "score": float(base_score),
                    }
                    for base_index, base_score in zip(
                        top_base_idx.tolist(),
                        top_base_scores.tolist(),
                    )
                ],
            }
        )

    result = {
        "run_name": f"{backend_name}_{result_slug(dataset_name)}_{size}",
        "meta": {
            "backend": backend_name,
            "dataset": dataset_name,
            "size": size,
            "model": backend_cfg["model_name"],
            "model_name_or_path": backend_cfg["model_name_or_path"],
            "batch_size": backend_cfg["batch_size"],
            "base_json": str(base_json),
            "base_faiss": str(base_faiss),
            "webqa_dataset": config["input"]["webqa_dataset"],
            "poison_report": config["input"]["poison_report"],
            "poison_image_dir": config["input"]["poison_image_dir"],
        },
        "summary": {
            "total": len(webqa_rows),
            "top1_hits": top1_hits,
            "top1_rate": (top1_hits / len(webqa_rows)) if webqa_rows else 0.0,
            "top3_hits": top3_hits,
            "top3_rate": (top3_hits / len(webqa_rows)) if webqa_rows else 0.0,
        },
        "records": records_out,
    }
    save_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval robustness evaluation for CLIP, SigLIP, "
            "and Qwen3-VL backends."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--backend",
        action="append",
        choices=["clip", "siglip", "qwen"],
        default=None,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["COCO", "Flickr30k"],
        default=None,
    )
    parser.add_argument("--size", action="append", choices=["1k", "10k", "30k"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(resolve_repo_path(args.config))
    if not isinstance(config, dict):
        raise TypeError("Config must be a JSON object")

    dataset_path = resolve_repo_path(config["input"]["webqa_dataset"])
    poison_report_path = resolve_repo_path(config["input"]["poison_report"])
    poison_image_dir = resolve_repo_path(config["input"]["poison_image_dir"])
    dataset = load_json(dataset_path)
    if not isinstance(dataset, list):
        raise TypeError(f"Expected JSON array in {dataset_path}")

    poison_report = load_json(poison_report_path) if poison_report_path.exists() else {}
    poison_lookup = build_poison_report_lookup(poison_report)
    webqa_rows = normalize_webqa_records(dataset, poison_lookup, poison_image_dir)

    backends = args.backend or list(config["backends"].keys())
    datasets = args.dataset or list(config["datasets"].keys())
    sizes = args.size or list(config["run"]["sizes"])
    summary_rows: list[dict[str, Any]] = []

    for backend_name in backends:
        backend_cfg = config["backends"][backend_name]
        backend = build_backend(backend_name, backend_cfg)
        for dataset_name in datasets:
            for size in sizes:
                result = evaluate_run(
                    backend_name=backend_name,
                    backend_cfg=backend_cfg,
                    backend=backend,
                    dataset_name=dataset_name,
                    size=size,
                    config=config,
                    webqa_rows=webqa_rows,
                    overwrite=args.overwrite,
                )
                summary = result["summary"]
                print(
                    f"[{backend_name}][{dataset_name}][{size}] "
                    f"top1={summary['top1_hits']}/{summary['total']} "
                    f"({summary['top1_rate'] * 100:.1f}%), "
                    f"top3={summary['top3_hits']}/{summary['total']} "
                    f"({summary['top3_rate'] * 100:.1f}%)"
                )
                summary_rows.append(
                    {
                        "backend": backend_name,
                        "dataset": dataset_name,
                        "size": size,
                        "top1_rate": summary["top1_rate"],
                        "top3_rate": summary["top3_rate"],
                    }
                )

    summary_path = resolve_repo_path(config["run"]["output_dir"]) / "retrieval_summary.json"
    save_json(summary_path, {"results": summary_rows})
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
