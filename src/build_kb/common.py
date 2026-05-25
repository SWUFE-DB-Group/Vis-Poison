from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "kb_construction.json"


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return repo_root / path_obj


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected object config in {path}")
    return data


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}")
    return data


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_").replace("-", "_").replace(".", "_")


def image_filename(record: dict[str, Any], dataset_name: str) -> str:
    if dataset_name == "COCO":
        value = record.get("filename")
        if not value and record.get("img") is not None:
            value = f"{int(record['img']):012d}.jpg"
    else:
        value = record.get("filename") or record.get("img")
    filename = str(value or "").strip()
    if not filename:
        raise ValueError(f"Missing image filename for {dataset_name} record: {record}")
    return filename


def default_inputs(config: dict[str, Any], dataset_name: str, sizes: list[str] | None) -> list[Path]:
    dataset_config = config["datasets"][dataset_name]
    data_dir = resolve_repo_path(dataset_config["data_dir"])
    prefix = dataset_config["prefix"]
    selected_sizes = sizes or config.get("kb_sizes", ["1k"])
    return [data_dir / f"{prefix}_{size}.json" for size in selected_sizes]


def default_images_dir(config: dict[str, Any], dataset_name: str) -> Path:
    return resolve_repo_path(config["datasets"][dataset_name]["images_dir"])


def default_output_dir(config: dict[str, Any], dataset_name: str, method: str) -> Path:
    return resolve_repo_path(config["datasets"][dataset_name]["output_dir"]) / method


def build_ip_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

