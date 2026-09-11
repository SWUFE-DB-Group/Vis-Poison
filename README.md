# Vis-Poison: Poisoning Visual Knowledge in Multimodal Retrieval-Augmented Generation

Official implementation and data resources for our **Findings of EMNLP 2026** paper ([arXiv](https://arxiv.org/abs/2608.20756)).

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

- `dataset/`
  Dataset metadata, retrieval corpora, and intermediate data files used by the experiments. The released Vis-Poison dataset is available on [Hugging Face](https://huggingface.co/datasets/liangrujin/Vis-Poison) and  [Google Drive](https://drive.google.com/drive/folders/1lnuCwqdQEGqvyWPb_Jf87yNl8aRgkAXn?usp=sharing).

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

----

If you find our work useful or use it in your research, please consider citing our [`EMNLP` 2026 paper](https://arxiv.org/abs/2608.20756):

```bib
@inproceedings{liang2026vispoison,
  title     = {Vis-Poison: Poisoning Visual Knowledge in Multimodal Retrieval-Augmented Generation},
  author    = {Liang, Rujin and Chen, Zhongpu and Lei, Yuhao and Miao, Xin},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
