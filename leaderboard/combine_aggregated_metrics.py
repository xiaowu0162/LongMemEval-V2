#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OVERALL_COUNT_KEYS = (
    "count_all_questions",
    "count_non_abstention",
    "count_abstention",
)
OVERALL_SCORE_KEYS = {
    "overall_full_set": "count_all_questions",
    "overall_non_abstention_only": "count_non_abstention",
    "overall_abstention_only": "count_abstention",
}
BREAKDOWN_VALUE_KEYS = ("pct_correct", "pct_answered_wrong", "pct_unknown")
BREAKDOWN_SECTIONS = (
    "non_abstention_by_category",
    "abstention_by_category",
)
COMBINED_ABSTENTION_CATEGORY_PAIRS = {
    "static": ("static", "static-abs"),
    "dynamic": ("dynamic", "dynamic-abs"),
    "procedure": ("procedure", "procedure-abs"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two LongMemEval-V2 aggregated_metrics.json files with "
            "example-count-weighted averages."
        )
    )
    parser.add_argument("metrics_a", type=Path)
    parser.add_argument("metrics_b", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write combined metrics to this path. Defaults to stdout.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"Expected numeric value, got bool: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}: {value!r}")


def as_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected integer count, got {value!r}")
    if value < 0:
        raise ValueError(f"Expected non-negative count, got {value}")
    return value


def clean_number(value: float | None) -> float | int | None:
    if value is None:
        return None
    if value.is_integer():
        return int(value)
    return value


def sum_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def weighted_average(values: list[tuple[Any, int]]) -> float | None:
    weighted_total = 0.0
    total_count = 0
    for value, count in values:
        numeric = as_number(value)
        if numeric is None or count == 0:
            continue
        weighted_total += numeric * count
        total_count += count
    if total_count == 0:
        return None
    return weighted_total / total_count


def ordered_union(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    keys = list(left.keys())
    keys.extend(key for key in right.keys() if key not in left)
    return keys


def total_question_count(metrics: dict[str, Any]) -> int:
    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("Missing overall metrics")
    count = as_count(overall.get("count_all_questions"))
    if count == 0:
        raise ValueError("count_all_questions must be positive")
    return count


def combine_overall(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    counts = {
        key: as_count(left.get(key)) + as_count(right.get(key))
        for key in OVERALL_COUNT_KEYS
    }
    out: dict[str, Any] = {}
    for score_key, count_key in OVERALL_SCORE_KEYS.items():
        out[score_key] = weighted_average(
            [
                (left.get(score_key), as_count(left.get(count_key))),
                (right.get(score_key), as_count(right.get(count_key))),
            ]
        )
    for key in OVERALL_COUNT_KEYS:
        out[key] = counts[key]
    return out


def combine_breakdown(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_count = as_count(left.get("count"))
    right_count = as_count(right.get("count"))
    out: dict[str, Any] = {"count": left_count + right_count}
    for key in BREAKDOWN_VALUE_KEYS:
        out[key] = weighted_average(
            [(left.get(key), left_count), (right.get(key), right_count)]
        )
    return out


def combine_breakdown_section(left: Any, right: Any) -> dict[str, Any]:
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    return {
        key: combine_breakdown(
            left.get(key, {"count": 0}),
            right.get(key, {"count": 0}),
        )
        for key in ordered_union(left, right)
    }


def get_combined_abstention_section(metrics: dict[str, Any]) -> dict[str, Any]:
    section = metrics.get("combined_abstention_by_category")
    if isinstance(section, dict) and section:
        return section

    non_abstention = metrics.get("non_abstention_by_category")
    abstention = metrics.get("abstention_by_category")
    non_abstention = non_abstention if isinstance(non_abstention, dict) else {}
    abstention = abstention if isinstance(abstention, dict) else {}
    out: dict[str, Any] = {}
    for cat, (regular_cat, abstention_cat) in COMBINED_ABSTENTION_CATEGORY_PAIRS.items():
        if regular_cat in non_abstention or abstention_cat in abstention:
            out[cat] = combine_breakdown(
                non_abstention.get(regular_cat, {"count": 0}),
                abstention.get(abstention_cat, {"count": 0}),
            )
    return out


def token_total(
    summary: dict[str, Any],
    total_key: str,
    avg_key: str,
    count: int,
) -> float | None:
    total = as_number(summary.get(total_key))
    if total is not None:
        return total
    avg = as_number(summary.get(avg_key))
    if avg is None:
        return None
    return avg * count


def combine_tokens(
    left: dict[str, Any],
    right: dict[str, Any],
    left_count: int,
    right_count: int,
) -> dict[str, Any]:
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    total_count = left_count + right_count

    prompt_tokens = sum_optional(
        [
            token_total(left, "prompt_tokens", "avg_prompt_tokens", left_count),
            token_total(right, "prompt_tokens", "avg_prompt_tokens", right_count),
        ]
    )
    completion_tokens = sum_optional(
        [
            token_total(left, "completion_tokens", "avg_completion_tokens", left_count),
            token_total(right, "completion_tokens", "avg_completion_tokens", right_count),
        ]
    )
    total_tokens = sum_optional(
        [
            token_total(left, "total_tokens", "avg_total_tokens", left_count),
            token_total(right, "total_tokens", "avg_total_tokens", right_count),
        ]
    )
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": clean_number(prompt_tokens),
        "completion_tokens": clean_number(completion_tokens),
        "total_tokens": clean_number(total_tokens),
        "avg_prompt_tokens": (
            prompt_tokens / total_count if prompt_tokens is not None else None
        ),
        "avg_completion_tokens": (
            completion_tokens / total_count if completion_tokens is not None else None
        ),
        "avg_total_tokens": (
            total_tokens / total_count if total_tokens is not None else None
        ),
    }


def combine_memory_context(
    left: dict[str, Any],
    right: dict[str, Any],
    left_count: int,
    right_count: int,
) -> dict[str, Any]:
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    return {
        "avg_original_tokens": weighted_average(
            [
                (left.get("avg_original_tokens"), left_count),
                (right.get("avg_original_tokens"), right_count),
            ]
        ),
        "avg_final_tokens": weighted_average(
            [
                (left.get("avg_final_tokens"), left_count),
                (right.get("avg_final_tokens"), right_count),
            ]
        ),
        "num_truncated_sequences": as_count(left.get("num_truncated_sequences"))
        + as_count(right.get("num_truncated_sequences")),
    }


def timing_total(summary: dict[str, Any], count: int) -> float | None:
    total = as_number(summary.get("total_seconds"))
    if total is not None:
        return total
    avg = as_number(summary.get("avg_seconds"))
    if avg is None:
        return None
    return avg * count


def combine_timing(
    left: dict[str, Any],
    right: dict[str, Any],
    left_count: int,
    right_count: int,
) -> dict[str, Any]:
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    total_count = left_count + right_count
    totals = [
        total
        for total in (
            timing_total(left, left_count),
            timing_total(right, right_count),
        )
        if total is not None
    ]
    total_seconds = sum(totals) if totals else None
    max_candidates = [
        value
        for value in (
            as_number(left.get("max_seconds")),
            as_number(right.get("max_seconds")),
        )
        if value is not None
    ]
    max_seconds = max(max_candidates) if max_candidates else None
    return {
        "avg_seconds": (
            total_seconds / total_count if total_seconds is not None else None
        ),
        "max_seconds": max_seconds,
        "total_seconds": total_seconds,
    }


def combine_metrics(
    left: dict[str, Any],
    right: dict[str, Any],
    left_path: Path,
    right_path: Path,
) -> dict[str, Any]:
    left_count = total_question_count(left)
    right_count = total_question_count(right)

    combined: dict[str, Any] = {
        "overall": combine_overall(left["overall"], right["overall"]),
        "abstention_overall": combine_breakdown(
            left.get("abstention_overall", {"count": 0}),
            right.get("abstention_overall", {"count": 0}),
        ),
        "tokens": combine_tokens(
            left.get("tokens", {}),
            right.get("tokens", {}),
            left_count,
            right_count,
        ),
        "memory_context": combine_memory_context(
            left.get("memory_context", {}),
            right.get("memory_context", {}),
            left_count,
            right_count,
        ),
        "memory_query": combine_timing(
            left.get("memory_query", {}),
            right.get("memory_query", {}),
            left_count,
            right_count,
        ),
        "memory_post_query": combine_timing(
            left.get("memory_post_query", {}),
            right.get("memory_post_query", {}),
            left_count,
            right_count,
        ),
        "combined_from": [str(left_path), str(right_path)],
        "combined_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for section in BREAKDOWN_SECTIONS:
        combined[section] = combine_breakdown_section(
            left.get(section),
            right.get(section),
        )
    combined["combined_abstention_by_category"] = combine_breakdown_section(
        get_combined_abstention_section(left),
        get_combined_abstention_section(right),
    )
    return combined


def main() -> None:
    args = parse_args()
    combined = combine_metrics(
        load_json(args.metrics_a),
        load_json(args.metrics_b),
        args.metrics_a,
        args.metrics_b,
    )
    if args.output is None:
        print(json.dumps(combined, indent=2, ensure_ascii=True))
    else:
        save_json(args.output, combined)


if __name__ == "__main__":
    main()
