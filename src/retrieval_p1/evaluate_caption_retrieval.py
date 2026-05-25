import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT_DIR / "dataset" / "webqa_final_category_difficulty_sample_70.json"
DEFAULT_API_CONFIG_PATH = ROOT_DIR / "configs" / "openai_models.json"
OUTPUT_DIR = ROOT_DIR / "outputs" / "retrieval_p1"
EMBED_MODEL = "text-embedding-3-large"
EMBED_BATCH_SIZE = 64

CAPTION_MODELS = [
    "Claude Sonnet 4.6",
    "GPT-5.4",
    "Qwen3.6-Plus",
    "Llama 4 Maverick",
    "Kimi-K2.6",
    "Qwen3.5-397B-A17B",
    "Llama 4 Scout",
    "Qwen3.6-35B-A3B",
    "Qwen3.6-27B",
]

KB_GROUPS = {
    "COCO": {
        "prefix": "coco_random",
        "dir": ROOT_DIR / "dataset" / "retrieval" / "COCO",
        "embed_dir": ROOT_DIR / "dataset" / "retrieval" / "COCO" / "text_embedding_3_large",
    },
    "Flickr30k": {
        "prefix": "flickr30k_random",
        "dir": ROOT_DIR / "dataset" / "retrieval" / "Flickr30k",
        "embed_dir": ROOT_DIR / "dataset" / "retrieval" / "Flickr30k" / "text_embedding_3_large",
    },
}
KB_SIZES = ["1k", "10k", "30k"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace(" ", "-")
        .replace(".", "-")
        .replace(":", "-")
    )


def create_client(api_config_path: Path) -> OpenAI:
    config = load_json(api_config_path)
    api = config["openai"]
    return OpenAI(
        base_url=api["base_url"],
        api_key=api["api_key"],
        timeout=api.get("timeout", 300),
    )


def format_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def get_kb_paths(group_name: str, kb_size: str) -> tuple[Path, Path, Path]:
    config = KB_GROUPS[group_name]
    prefix = config["prefix"]
    data_path = config["dir"] / f"{prefix}_{kb_size}_captions.json"
    faiss_path = config["embed_dir"] / f"{prefix}_{kb_size}_captions_text_embedding_3_large.faiss"
    manifest_path = config["embed_dir"] / f"{prefix}_{kb_size}_captions_text_embedding_3_large_manifest.json"
    for path in (data_path, faiss_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required KB file not found: {path}")
    return data_path, faiss_path, manifest_path


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    array = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(array)
    return np.ascontiguousarray(array, dtype=np.float32)


def build_caption_rows(dataset: list[dict[str, Any]], caption_model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in dataset:
        captions = record.get("captions", {})
        if not isinstance(captions, dict):
            continue
        caption = str(captions.get(caption_model, "")).strip()
        if not caption:
            continue
        rows.append(
            {
                "id": record.get("id", ""),
                "difficulty": record.get("difficulty", ""),
                "category": record.get("counterfactual_edit", {}).get("category", ""),
                "entity": record.get("entity", ""),
                "question": record.get("question", ""),
                "correct_answer": record.get("correct_answer", ""),
                "wrong_answer": record.get("wrong_answer", ""),
                "caption": caption,
            }
        )
    return rows


def evaluate_single_kb(
    group_name: str,
    kb_size: str,
    caption_rows: list[dict[str, Any]],
    question_vectors: np.ndarray,
    caption_vectors: np.ndarray,
) -> dict[str, Any]:
    data_path, faiss_path, manifest_path = get_kb_paths(group_name, kb_size)
    manifest = load_json(manifest_path)
    base_data = load_json(data_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"{manifest_path}: expected JSON object")
    if not isinstance(base_data, list):
        raise TypeError(f"{data_path}: expected JSON array")

    index = faiss.read_index(str(faiss_path))
    top1_hits = 0
    top3_hits = 0
    sample_results: list[dict[str, Any]] = []

    for i, row in enumerate(caption_rows):
        query_vector = question_vectors[i : i + 1]
        caption_vector = caption_vectors[i : i + 1]
        scores, indices = index.search(query_vector, 3)
        base_scores = scores[0]
        base_indices = indices[0]
        caption_score = float(np.dot(query_vector[0], caption_vector[0]))

        hit_top1 = bool(len(base_scores) >= 1 and caption_score >= float(base_scores[0]))
        hit_top3 = bool(len(base_scores) >= 3 and caption_score >= float(base_scores[2]))
        top1_hits += int(hit_top1)
        top3_hits += int(hit_top3)

        top3_base_results: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(base_scores, base_indices), start=1):
            idx_int = int(idx)
            if 0 <= idx_int < len(base_data):
                top3_base_results.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "index": idx_int,
                        "record": base_data[idx_int],
                    }
                )

        sample_results.append(
            {
                **row,
                "caption_score": caption_score,
                "hit_top1": hit_top1,
                "hit_top3": hit_top3,
                "top3_base_results": top3_base_results,
            }
        )

    usable_count = len(caption_rows)
    return {
        "kb_group": group_name,
        "kb_size": kb_size,
        "data_path": str(data_path),
        "faiss_path": str(faiss_path),
        "manifest_path": str(manifest_path),
        "metric": manifest.get("metric", ""),
        "embed_model": EMBED_MODEL,
        "num_usable_samples": usable_count,
        "top1_hits": top1_hits,
        "top1_rate": format_rate(top1_hits, usable_count),
        "top3_hits": top3_hits,
        "top3_rate": format_rate(top3_hits, usable_count),
        "results": sample_results,
    }


def evaluate_caption_model(
    caption_model: str,
    dataset: list[dict[str, Any]],
    client: OpenAI,
) -> dict[str, Any]:
    caption_rows = build_caption_rows(dataset, caption_model)
    if not caption_rows:
        raise ValueError(f"No usable captions for {caption_model}")

    questions = [str(row.get("question", "")).strip() for row in caption_rows]
    captions = [str(row.get("caption", "")).strip() for row in caption_rows]
    question_vectors = embed_texts(client, questions)
    caption_vectors = embed_texts(client, captions)

    results_by_group: dict[str, dict[str, Any]] = {}
    for group_name in KB_GROUPS:
        group_results: dict[str, Any] = {}
        for kb_size in KB_SIZES:
            group_results[kb_size] = evaluate_single_kb(
                group_name=group_name,
                kb_size=kb_size,
                caption_rows=caption_rows,
                question_vectors=question_vectors,
                caption_vectors=caption_vectors,
            )
            result = group_results[kb_size]
            print(
                f"[{caption_model}][{group_name}][{kb_size}] "
                f"top1={result['top1_hits']}/{result['num_usable_samples']} "
                f"({result['top1_rate'] * 100:.1f}%), "
                f"top3={result['top3_hits']}/{result['num_usable_samples']} "
                f"({result['top3_rate'] * 100:.1f}%)"
            )
        results_by_group[group_name] = group_results

    return {
        "caption_model": caption_model,
        "caption_source": str(DEFAULT_DATASET_PATH),
        "embed_model": EMBED_MODEL,
        "num_total_dataset_records": len(dataset),
        "num_usable_caption_records": len(caption_rows),
        "skipped_caption_records": len(dataset) - len(caption_rows),
        "results_by_group": results_by_group,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate caption retrieval for P1 captions.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--api-config", default=str(DEFAULT_API_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model", action="append", default=None, help="Caption model display name. Can be repeated.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = load_json(Path(args.dataset))
    if not isinstance(dataset, list):
        raise TypeError(f"{args.dataset}: expected JSON array")

    selected_models = args.model or CAPTION_MODELS
    output_dir = Path(args.output_dir)
    client = create_client(Path(args.api_config))
    summaries: list[dict[str, Any]] = []

    for caption_model in selected_models:
        output_path = output_dir / f"retrieval_{model_slug(caption_model)}.json"
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing retrieval result for {caption_model}: {output_path}")
            result = load_json(output_path)
        else:
            result = evaluate_caption_model(caption_model, dataset, client)
            save_json(output_path, result)
            print(f"Saved retrieval result to: {output_path}")

        summary_row = {
            "caption_model": result.get("caption_model", caption_model),
            "output_path": str(output_path),
        }
        for group_name, group_results in result["results_by_group"].items():
            for kb_size, kb_result in group_results.items():
                summary_row[f"{group_name}_{kb_size}_top1_rate"] = kb_result["top1_rate"]
                summary_row[f"{group_name}_{kb_size}_top3_rate"] = kb_result["top3_rate"]
        summaries.append(summary_row)

    save_json(output_dir / "retrieval_summary.json", {"results": summaries})
    print(f"Saved summary to: {output_dir / 'retrieval_summary.json'}")


if __name__ == "__main__":
    main()
