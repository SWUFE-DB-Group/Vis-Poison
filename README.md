# Vis-Poison: Poisoning Visual Knowledge in Multimodal Retrieval-Augmented Generation

> Code and lightweight data artifacts for a visual knowledge poisoning attack on multimodal retrieval-augmented generation, together with a knowledge-aware evaluation framework.

## Repository Structure

- `src/poisoned_image_construction/`
  Code for constructing poisoned images with planner, editor, verifier, and end-to-end pipeline scripts.

- `src/generate_caption/`
  Code for caption generation with OpenAI-compatible APIs.

- `src/generation/`
  Code for question answering generation experiments under question-only, clean-image, and poisoned-image settings, together with summary statistics.

- `src/build_kb/`
  Code for building retrieval knowledge bases and vector indexes.

- `src/retrieval_p1/`
  Code for retrieval experiments based on generated captions.

- `src/retrieval_p2/`
  Code for retrieval experiments based on visual encoders.

- `dataset/`
  Project datasets and intermediate data files used by the experimental pipeline.

- `results/`
  Example outputs and experiment result files.

- `configs/`
  Configuration files for local execution.


## Models and Knowledge Bases

- **Retrieval models:** `text-embedding-3-large`, `clip-vit-base-patch16`, `siglip-large-patch16-256`, and `Qwen3-VL-Embedding-2B`.
  `text-embedding-3-large` is used for caption-based retrieval.

- **Generation models:** `Claude Sonnet 4.6`, `GPT-5.4`, `Qwen3.6-Plus`, `Kimi-K2.6`, `Llama 4 Maverick`, and `Qwen3.5-397B-A17B`.

- **Caption models:** the six generation models above, plus `Llama 4 Scout`, `Qwen3.6-35B-A3B`, and `Qwen3.6-27B`.

- **Knowledge bases:** 30K-entry multimodal knowledge bases randomly sampled from [MS COCO](https://cocodataset.org/) and [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/).

## Knowledge-Aware Evaluation

For each generation model, we compare three input conditions:

- `Q`: the model answers from the query alone.
- `Q+Clean`: the model answers with the clean image.
- `Q+Poison`: the model answers with the poisoned image.

We report six metrics: `Q ACC`, `Q+Clean ACC`, `ASR-G`, `POR`, `CHR`, and `PIR`.

- `ASR-G`: generation-stage attack success rate.
- `POR`: poison override rate when the model can answer correctly from `Q` alone.
- `CHR`: clean help rate when the model cannot answer correctly from `Q` alone.
- `PIR`: poison induction rate when the model cannot answer correctly from `Q` alone.

Each model is evaluated on 630 questions per split. Across six generation models, this gives 3,780 model-question evaluations per split.

### Easy Split

| Model             | Q ACC | Q+Clean ACC | ASR-G | POR | CHR | PIR |
| ----------------- | ----: | ----------: | ----: | --: | --: | --: |
| Claude Sonnet 4.6 | 75.4% |       94.3% | 62.4% | 59.4% | 89.0% | 71.6% |
| GPT-5.4           | 83.8% |       93.2% | 64.3% | 62.7% | 88.2% | 72.5% |
| Qwen3.6-Plus      | 83.0% |       96.2% | 61.7% | 60.0% | 92.5% | 70.1% |
| Kimi-K2.6         | 71.4% |       94.8% | 67.3% | 62.9% | 90.0% | 78.3% |
| Llama 4 Maverick  | 68.6% |       87.9% | 63.0% | 59.0% | 80.3% | 71.7% |
| Qwen3.5-397B-A17B | 86.5% |       94.1% | 61.3% | 59.8% | 91.8% | 70.6% |
| **Mean**          | 78.1% |       93.4% | 63.3% | 60.6% | 88.6% | 72.5% |

### Hard Split

| Model             | Q ACC | Q+Clean ACC | ASR-G | POR | CHR | PIR |
| ----------------- | ----: | ----------: | ----: | --: | --: | --: |
| Claude Sonnet 4.6 | 24.8% |       77.8% | 76.7% | 64.7% | 74.1% | 80.6% |
| GPT-5.4           | 27.8% |       74.1% | 75.9% | 71.4% | 67.9% | 77.6% |
| Qwen3.6-Plus      | 18.7% |       78.1% | 77.5% | 69.5% | 75.2% | 79.3% |
| Kimi-K2.6         | 18.6% |       77.5% | 78.9% | 73.5% | 74.5% | 80.1% |
| Llama 4 Maverick  | 12.1% |       63.3% | 74.3% | 63.2% | 62.3% | 75.8% |
| Qwen3.5-397B-A17B | 21.4% |       71.7% | 74.9% | 72.6% | 68.9% | 75.6% |
| **Mean**          | 20.6% |       73.8% | 76.4% | 69.2% | 70.5% | 78.2% |

## Model Sources and Configs

- `src/poisoned_image_construction/`
  - `planner.py` and `verifier.py` use `gemma4:31b` through Ollama.
  - `editor.py` uses the local FLUX image editing model from a local model path.
  - Main config: `configs/poisoned_image_construction.json`

- `src/generate_caption/`
  - Uses OpenAI-compatible APIs.
  - Model names and provider-specific tags are read from `configs/openai_models.json`.

- `src/generation/`
  - Uses OpenAI-compatible APIs.
  - Model names and provider-specific tags are read from `configs/openai_models.json`.

- `src/build_kb/`
  - `build_text_embedding_faiss.py` uses the API model `text-embedding-3-large` through `configs/openai_models.json`.
  - `build_clip_image_faiss.py`, `build_siglip_image_faiss.py`, and `build_qwen_image_faiss.py` use local open-weight models from local model paths configured in `configs/kb_construction.json`.

- `src/retrieval_p1/`
  - Uses `text-embedding-3-large` through `configs/openai_models.json` for caption embeddings.

- `src/retrieval_p2/`
  - Uses local open-weight retrieval models from local model paths configured in `configs/retrieval_p2.json`.
  - The backends are CLIP, SigLIP, and Qwen3-VL-Embedding.

- `src/defenses/is_valid/`
  - Caption construction scripts use OpenAI-compatible APIs through `configs/openai_models.json`.
  - `web_search_fact_checking.py` uses an OpenAI-compatible API model with search enabled through `configs/openai_models.json`.
  - `jailbreak_filtering.py` uses the same OpenAI-compatible API credentials through `configs/openai_models.json`.

- `src/defenses/topk/`
  - `src/generation/answer_validator.py` and `src/defenses/topk/answer_validator.py` use Ollama by default.

- `src/defenses/trufor/`
  - Uses the external TruFor environment described in `src/defenses/trufor/readme.md`.

In short:

- put OpenAI-compatible API endpoint, key, and model-name mappings in `configs/openai_models.json`
- put Ollama model settings in the task-specific config when needed, and run Ollama locally
- put local model paths in the corresponding committed task config such as `configs/poisoned_image_construction.json`, `configs/kb_construction.json`, and `configs/retrieval_p2.json`

Detailed usage instructions are documented in the corresponding subdirectory README files. Some scripts rely on local model paths or API settings that are intentionally kept outside version control.
