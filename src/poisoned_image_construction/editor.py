import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "poisoned_image_construction.json"

_PIPE: Flux2KleinPipeline | None = None
_PIPE_MODEL_DIR: str | None = None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_pipeline(model_dir: str | Path) -> Flux2KleinPipeline:
    global _PIPE, _PIPE_MODEL_DIR
    model_dir = str(model_dir)
    if _PIPE is not None and _PIPE_MODEL_DIR == model_dir:
        return _PIPE

    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()
    _PIPE = pipe
    _PIPE_MODEL_DIR = model_dir
    return pipe


def release_pipeline() -> None:
    global _PIPE, _PIPE_MODEL_DIR
    _PIPE = None
    _PIPE_MODEL_DIR = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def edit_image(
    image_path: str | Path,
    output_path: str | Path,
    prompt: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    release_after_edit: bool | None = None,
) -> Path:
    config = load_config(Path(config_path))
    editor_config = config["editor"]
    pipe = get_pipeline(editor_config["model_path"])

    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(image_path).convert("RGB")
    width, height = input_image.size
    generator = torch.Generator("cpu").manual_seed(seed if seed is not None else editor_config.get("seed", 42))

    try:
        # Keep the edited image aligned with the source image size.
        result = pipe(
            prompt=prompt.strip(),
            image=input_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps or editor_config.get("num_inference_steps", 4),
            guidance_scale=guidance_scale if guidance_scale is not None else editor_config.get("guidance_scale", 1.0),
            generator=generator,
        ).images[0]
        result.save(output_path, format="PNG")
    finally:
        should_release = editor_config.get("release_after_edit", True) if release_after_edit is None else release_after_edit
        if should_release:
            release_pipeline()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit an image locally with FLUX.2 Klein.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-loaded", action="store_true")
    args = parser.parse_args()

    output_path = edit_image(
        image_path=args.image,
        output_path=args.output,
        prompt=args.prompt,
        config_path=args.config,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        release_after_edit=not args.keep_loaded,
    )
    print(json.dumps({"output_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()