from __future__ import annotations

import asyncio

from _shared import build_parser, resolve_difficulty, run_generation


def main() -> None:
    parser = build_parser("q_poison")
    args = parser.parse_args()
    difficulty = resolve_difficulty(args.config, args.difficulty)
    asyncio.run(
        run_generation(
            config_path=args.config,
            api_config_path=args.api_config,
            mode="q_poison",
            difficulty=difficulty,
            model_names=args.model,
            max_concurrency=args.max_concurrency,
            max_items=args.max_items,
            input_path=args.input,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
