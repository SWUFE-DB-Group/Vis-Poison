# Defenses

This directory collects the code used for the defense-side experiments in the paper.

## Modules

- `sample/`
  Build the 180-example defense benchmark split used in the defense section.

- `is_valid/`
  Scripts for the isValid-style pipeline, including separate construction scripts for each group, web-search fact-checking, jailbreak filtering, and summary statistics.

- `topk/`
  Scripts for the top-k retrieval defense setting, including top-3 generation, a unified validator, and summary statistics.

- `trufor/`
  Minimal wrappers used for the TruFor image-forensics safeguard.

## Configuration

OpenAI-compatible API calls use `configs/openai_models.json`.
Local validator scripts that use Ollama default to `http://localhost:11434`.

## Results

The reorganized scripts write outputs under `results/defenses/` by default.
Most scripts also support CLI overrides for input and output paths.
