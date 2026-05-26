import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
GENERATION_DIR = ROOT_DIR / "results" / "generation"
TOPK_DIR = ROOT_DIR / "results" / "defenses" / "topk"
SAMPLE_JSON = TOPK_DIR / "qwen3vl_coco30k_top1_hit_sample_180.json"
DIFFICULTIES = ["easy", "hard"]
GENERATION_MODES = ["q_only", "q_clean", "q_poison"]
TOP3_MODE = "q_poison_top3"
MODEL_ORDER = [
    "claude-sonnet-4-6",
    "gpt-5-4",
    "qwen3-6-plus",
    "kimi-k2-6",
    "llama-4-maverick",
    "qwen3-5-397b-a17b",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_check_filename(path: Path) -> tuple[str, str] | None:
    name = path.name
    if not name.endswith("-check.json"):
        return None
    stem = name[:-11]
    for mode in [*GENERATION_MODES, TOP3_MODE]:
        suffix = f"-{mode}"
        if stem.endswith(suffix):
            model = stem[: -len(suffix)]
            if model:
                return model, mode
    return None


def get_results(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    return [row for row in data.get("results", []) if isinstance(row, dict)]


def to_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id", "")).strip(): row
        for row in rows
        if str(row.get("id", "")).strip()
    }


def validator_accepts(row: dict[str, Any]) -> bool:
    explicit = row.get("validator_is_aligned")
    if isinstance(explicit, bool):
        return explicit

    value = row.get("validator_judgement")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
    return False


def is_correct(row: dict[str, Any]) -> bool:
    return (
        validator_accepts(row)
        and str(row.get("validator_label", "")).strip().lower() == "correct"
    )


def is_wrong(row: dict[str, Any]) -> bool:
    return str(row.get("validator_label", "")).strip().lower() == "wrong"


def is_misled(row: dict[str, Any]) -> bool:
    return (
        validator_accepts(row)
        and str(row.get("validator_label", "")).strip().lower() == "misled"
    )


def ratio(a: int, b: int) -> float | None:
    return None if b == 0 else a / b


def fmt(num: int | None, den: int | None, value: float | None) -> str:
    return (
        "-"
        if num is None or den is None or value is None
        else f"{num}/{den} ({value * 100:.1f}%)"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
        for i in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return " | ".join(
            row[i].ljust(widths[i]) if i == 0 else row[i].rjust(widths[i])
            for i in range(len(row))
        )

    return "\n".join(
        [render(headers), "-+-".join("-" * w for w in widths)]
        + [render(row) for row in rows]
    )


def collect_files(
    base_dir: Path, allowed_modes: set[str]
) -> dict[str, dict[str, dict[str, Path]]]:
    collected: dict[str, dict[str, dict[str, Path]]] = {}
    for difficulty in DIFFICULTIES:
        bucket = base_dir / difficulty
        if not bucket.exists():
            continue
        for path in sorted(bucket.glob("*-check.json")):
            parsed = parse_check_filename(path)
            if parsed is None:
                continue
            model, mode = parsed
            if mode not in allowed_modes:
                continue
            collected.setdefault(difficulty, {}).setdefault(model, {})[mode] = path
    return collected


def build_rows(selected_ids: set[str]) -> list[list[str]]:
    generation = collect_files(GENERATION_DIR, set(GENERATION_MODES))
    topk = collect_files(TOPK_DIR, {TOP3_MODE})
    rows = []
    for model in sorted(
        {
            *generation.get("easy", {}),
            *generation.get("hard", {}),
            *topk.get("easy", {}),
            *topk.get("hard", {}),
        },
        key=lambda x: (MODEL_ORDER.index(x) if x in MODEL_ORDER else 999, x),
    ):
        q_only_rows = []
        q_clean_rows = []
        q_poison_rows = []
        q_top3_rows = []
        for difficulty in DIFFICULTIES:
            files = generation.get(difficulty, {}).get(model, {})
            if "q_only" in files:
                q_only_rows.extend(
                    [
                        row
                        for row in get_results(files["q_only"])
                        if str(row.get("id", "")).strip() in selected_ids
                    ]
                )
            if "q_clean" in files:
                q_clean_rows.extend(
                    [
                        row
                        for row in get_results(files["q_clean"])
                        if str(row.get("id", "")).strip() in selected_ids
                    ]
                )
            if "q_poison" in files:
                q_poison_rows.extend(
                    [
                        row
                        for row in get_results(files["q_poison"])
                        if str(row.get("id", "")).strip() in selected_ids
                    ]
                )
            topk_files = topk.get(difficulty, {}).get(model, {})
            if TOP3_MODE in topk_files:
                q_top3_rows.extend(
                    [
                        row
                        for row in get_results(topk_files[TOP3_MODE])
                        if str(row.get("id", "")).strip() in selected_ids
                    ]
                )
        q_only_index = to_index(q_only_rows)
        q_clean_index = to_index(q_clean_rows)
        q_poison_index = to_index(q_poison_rows)
        q_top3_index = to_index(q_top3_rows)
        correct_ids = {
            sample_id for sample_id, row in q_only_index.items() if is_correct(row)
        }
        q_acc_v = (
            ratio(sum(1 for row in q_only_rows if is_correct(row)), len(q_only_rows))
            if q_only_rows
            else None
        )
        q_clean_v = (
            ratio(sum(1 for row in q_clean_rows if is_correct(row)), len(q_clean_rows))
            if q_clean_rows
            else None
        )
        q_poison_v = (
            ratio(sum(1 for row in q_poison_rows if is_misled(row)), len(q_poison_rows))
            if q_poison_rows
            else None
        )
        q_top3_v = (
            ratio(sum(1 for row in q_top3_rows if is_misled(row)), len(q_top3_rows))
            if q_top3_rows
            else None
        )
        por_v = (
            ratio(
                sum(
                    1
                    for sample_id in correct_ids
                    if is_misled(q_poison_index.get(sample_id, {}))
                ),
                len(correct_ids),
            )
            if correct_ids
            else None
        )
        top3_por_v = (
            ratio(
                sum(
                    1
                    for sample_id in correct_ids
                    if is_misled(q_top3_index.get(sample_id, {}))
                ),
                len(correct_ids),
            )
            if correct_ids
            else None
        )
        delta_v = (
            (top3_por_v - por_v)
            if top3_por_v is not None and por_v is not None
            else None
        )
        rows.append(
            [
                model,
                fmt(
                    sum(1 for row in q_only_rows if is_correct(row)),
                    len(q_only_rows) if q_only_rows else None,
                    q_acc_v,
                ),
                fmt(
                    sum(1 for row in q_clean_rows if is_correct(row)),
                    len(q_clean_rows) if q_clean_rows else None,
                    q_clean_v,
                ),
                fmt(
                    sum(1 for row in q_poison_rows if is_misled(row)),
                    len(q_poison_rows) if q_poison_rows else None,
                    q_poison_v,
                ),
                fmt(
                    sum(1 for row in q_top3_rows if is_misled(row)),
                    len(q_top3_rows) if q_top3_rows else None,
                    q_top3_v,
                ),
                fmt(
                    sum(
                        1
                        for sample_id in correct_ids
                        if is_misled(q_poison_index.get(sample_id, {}))
                    ),
                    len(correct_ids) if correct_ids else None,
                    por_v,
                ),
                fmt(
                    sum(
                        1
                        for sample_id in correct_ids
                        if is_misled(q_top3_index.get(sample_id, {}))
                    ),
                    len(correct_ids) if correct_ids else None,
                    top3_por_v,
                ),
                "-" if delta_v is None else f"{delta_v * 100:+.1f}pp",
            ]
        )
    return rows


def main() -> None:
    selected_ids = (
        {
            str(row.get("id", "")).strip()
            for row in load_json(SAMPLE_JSON)
            if isinstance(row, dict)
        }
        if SAMPLE_JSON.exists()
        else set()
    )
    rows = build_rows(selected_ids)
    print(
        table(
            [
                "model",
                "Q ACC",
                "Q+clean ACC",
                "Q+poison ASR",
                "Q+poison(top3) ASR",
                "POR",
                "POR(top3)",
                "delta",
            ],
            rows,
        )
    )


if __name__ == "__main__":
    main()
