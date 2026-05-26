# Vis-Poison: Poisoning Visual Knowledge in Multimodal Retrieval-Augmented Generation

Code and lightweight data artifacts for the Vis-Poison project.

## Repository Structure

- `src/poisoned_image_construction/`
  Poisoned image construction with planner, editor, verifier, and the end-to-end pipeline.

- `src/generate_caption/`
  Caption generation for retrieval.

- `src/generation/`
  Generation experiments for `Q`, `Q+Clean`, and `Q+Poison`, plus answer validation.

- `src/evaluation_framework/`
  Knowledge-aware evaluation for generation outputs.

- `src/build_kb/`
  Knowledge base construction and vector index building.

- `src/retrieval_p1/`
  Caption-based retrieval experiments.

- `src/retrieval_p2/`
  Visual-encoder-based retrieval experiments.

- `src/defenses/`
  Defense-side experiments, including isValid-style filtering, top-k evaluation, and TruFor-related scripts.

- `src/safeguard/`
  Additional TruFor safeguard utilities.

- `dataset/`
  Datasets and intermediate data files.

- `results/`
  Example outputs and experiment results.

- `configs/`
  Local task configs.

## Models and Configs

- OpenAI-compatible API settings and model-name mappings:
  `configs/openai_models.json`

- Poisoned image construction:
  `configs/poisoned_image_construction.json`

- Generation:
  `configs/generation.json`

- Caption generation:
  `configs/generate_caption.json`

- KB construction:
  `configs/kb_construction.json`

- Retrieval with visual encoders:
  `configs/retrieval_p2.json`

Detailed usage instructions are documented in the README files inside each subdirectory under `src/`.
