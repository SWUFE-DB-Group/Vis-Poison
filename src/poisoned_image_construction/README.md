# Poisoned Image Construction

This folder implements the planner-editor-verifier loop from Algorithm 1.

## Files

- `planner.py`: generate one edit instruction with the planning prompt
- `editor.py`: run local FLUX.2 Klein image editing
- `verifier.py`: verify whether the edited image supports the attacker answer
- `run_construction.py`: run the full multi-round construction loop
- `test_data/edit_type_showcase/`: local smoke-test samples

## Config

Default config:

```bash
configs/poisoned_image_construction.json
```

It stores the Ollama model, FLUX.2 Klein local model path, generation settings, and output paths.

Memory-saving defaults:

- `ollama.keep_alive = "0s"` unloads the Ollama model after each planner/verifier call.
- `editor.release_after_edit = true` releases the FLUX pipeline after each edit.

These settings reduce GPU memory conflicts between Ollama and FLUX, at the cost of reloading models more often.

## Test Data

Test data is stored in:

```bash
src/poisoned_image_construction/test_data/edit_type_showcase
```

`samples.json` is ready for `run_construction.py`. Each record contains `Q`, `wrong_answer`, and `source_image`.

## Four Direct Commands

Run these commands from the project root.

Planner test:

```bash
uv run src/poisoned_image_construction/planner.py --query "What color are the downspouts of the gutters on the side of the Bull & Stirrup pub, Chester, England?" --wrong-answer "The downspouts of the gutters on the side of the Bull & Stirrup pub are black." --image src/poisoned_image_construction/test_data/edit_type_showcase/color-c.jpg
```

Editor test:

```bash
uv run src/poisoned_image_construction/editor.py --image src/poisoned_image_construction/test_data/edit_type_showcase/color-c.jpg --prompt "Change the visible gutter downspouts on the side of the pub from red to black while preserving the building facade, signs, windows, street, lighting, and realistic appearance." --output results/poisoned_image_construction/readme_editor_test.png
```

Verifier test:

```bash
uv run src/poisoned_image_construction/verifier.py --query "What color are the downspouts of the gutters on the side of the Bull & Stirrup pub, Chester, England?" --answer "The downspouts of the gutters on the side of the Bull & Stirrup pub are black." --image src/poisoned_image_construction/test_data/edit_type_showcase/color-p.png
```

End-to-end test:

```bash
uv run src/poisoned_image_construction/run_construction.py --input src/poisoned_image_construction/test_data/edit_type_showcase/samples.json --max-items 1 --max-rounds 1 --overwrite
```

## Outputs

By default, generated images are written under:

```bash
results/poisoned_image_construction/images/
```

The run status, generated image paths, edit instructions, verifier decisions, and errors are written to one report:

```bash
results/poisoned_image_construction/report.json
```
