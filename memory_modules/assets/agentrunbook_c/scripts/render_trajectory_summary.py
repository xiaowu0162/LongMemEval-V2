#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


A11Y_LINE_RE = re.compile(r"^\s*\[([A-Za-z0-9_-]+)\]\s*(.+)$")
ACTION_OBJECT_ID_RE = re.compile(r"^[A-Za-z]*\d+[A-Za-z0-9_-]*$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render concise and full markdown summaries of all trajectory goals, "
            "start URLs, annotated action sequences, and responses."
        )
    )
    parser.add_argument("trajectories_dir", help="Directory containing <trajectory_id>/trajectory.json trees.")
    parser.add_argument(
        "--concise-output",
        help="Markdown file to write for the concise orientation view.",
    )
    parser.add_argument(
        "--full-output",
        help="Markdown file to write for the detailed trajectory view.",
    )
    args = parser.parse_args()
    require(
        bool(args.concise_output) or bool(args.full_output),
        "At least one of --concise-output or --full-output must be provided.",
    )
    return args


def goal_text(raw_goal: Any) -> str:
    if isinstance(raw_goal, str):
        text = raw_goal.strip()
        return text if text else "<goal not found>"
    if isinstance(raw_goal, list):
        parts = [part.strip() for part in raw_goal if isinstance(part, str) and part.strip()]
        if parts:
            return " ".join(parts)
    return "<goal not found>"


def response_text(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        text = raw_response.strip()
        return text if text else "<response not found>"
    if raw_response is None:
        return "<response not found>"
    return json.dumps(raw_response, ensure_ascii=True, sort_keys=True)


def _extract_object_lookup_from_tree(tree_text: str) -> dict[str, str]:
    object_lookup: dict[str, str] = {}
    for line in tree_text.splitlines():
        match = A11Y_LINE_RE.match(line)
        if not match:
            continue
        object_id = match.group(1)
        if object_id not in object_lookup:
            object_lookup[object_id] = line.strip()
    return object_lookup


def _action_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_interacted_object_ids(action_text: str) -> list[str]:
    try:
        parsed = ast.parse(action_text, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(parsed, ast.Expression):
        return []
    call = parsed.body
    if not isinstance(call, ast.Call):
        return []
    name = _action_name(call.func)
    arg_indexes = [0, 1] if name == "drag_and_drop" else [0]
    object_ids: list[str] = []
    for index in arg_indexes:
        if index >= len(call.args):
            continue
        arg = call.args[index]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            continue
        object_id = arg.value.strip()
        if ACTION_OBJECT_ID_RE.match(object_id):
            object_ids.append(object_id)
    return object_ids


def _annotate_action_with_object_details(action_text: str, object_lookup: dict[str, str]) -> str:
    object_ids = _extract_interacted_object_ids(action_text)
    if not object_ids:
        return action_text
    details: list[str] = []
    seen_ids: set[str] = set()
    for object_id in object_ids:
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        detail = object_lookup.get(object_id)
        if detail:
            details.append(detail)
    if not details:
        return action_text
    return f"{action_text}  # {' | '.join(details)}"


def build_action_steps(trajectory: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    known_object_lookup: dict[str, str] = {}
    states = trajectory.get("states")
    if isinstance(states, list):
        for state in states:
            if not isinstance(state, dict):
                continue
            current_text = state.get("text")
            current_object_lookup = (
                _extract_object_lookup_from_tree(current_text)
                if isinstance(current_text, str) and current_text
                else {}
            )
            object_lookup = {**known_object_lookup, **current_object_lookup}
            action_value = state.get("action")
            action_text = action_value.strip() if isinstance(action_value, str) else ""
            if action_text:
                actions.append(_annotate_action_with_object_details(action_text, object_lookup))
            known_object_lookup.update(current_object_lookup)
    return actions


def load_trajectory(trajectory_dir: Path) -> dict[str, Any]:
    path = trajectory_dir / "trajectory.json"
    require(path.exists(), f"Missing trajectory.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"trajectory.json must contain an object: {path}")
    return payload


def iter_trajectory_dirs(trajectories_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in trajectories_dir.iterdir()
        if path.is_dir() and (path / "trajectory.json").exists()
    )


def sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["start_url"]).strip().lower(),
        str(record["goal"]).strip().lower(),
        str(record["trajectory_id"]).strip().lower(),
    )


def markdown_table_cell(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def render_concise_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Trajectory Summary Concise",
        "",
        f"Total trajectories: {len(records)}",
        "",
        "This file is sorted by start URL so similar surfaces appear near each other.",
        "",
        "| # | Trajectory ID | Goal | Final reward | States | Start URL |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_table_cell(index),
                    markdown_table_cell(record["trajectory_id"]),
                    markdown_table_cell(record["goal"]),
                    markdown_table_cell(record["response"]),
                    markdown_table_cell(record["state_count"]),
                    markdown_table_cell(record["start_url"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_full_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Trajectory Summary Full",
        "",
        f"Total trajectories: {len(records)}",
        "",
        "This file is sorted by start URL so similar surfaces appear near each other.",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record['trajectory_id']}",
                "",
                f"Start URL: {record['start_url']}",
                "",
                f"Goal: {record['goal']}",
                "",
                "Action sequence:",
            ]
        )
        action_steps = record["action_steps"]
        if action_steps:
            for action_index, action_text in enumerate(action_steps, start=1):
                lines.append(f"{action_index}. {action_text}")
        else:
            lines.append("1. <no actions recorded>")
        lines.extend(
            [
                "",
                f"Final reward: {record['response']}",
            ]
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    trajectories_dir = Path(args.trajectories_dir).resolve()
    require(trajectories_dir.exists() and trajectories_dir.is_dir(), f"Missing trajectories dir: {trajectories_dir}")

    records: list[dict[str, Any]] = []
    for trajectory_dir in iter_trajectory_dirs(trajectories_dir):
        trajectory = load_trajectory(trajectory_dir)
        states = trajectory.get("states")
        records.append(
            {
                "trajectory_id": trajectory.get("id", trajectory_dir.name),
                "start_url": trajectory.get("start_url", "<start url not found>"),
                "goal": goal_text(trajectory.get("goal")),
                "response": response_text(trajectory.get("outcome")),
                "action_steps": build_action_steps(trajectory),
                "state_count": len(states) if isinstance(states, list) else 0,
            }
        )

    records.sort(key=sort_key)
    if args.concise_output:
        concise_output_path = Path(args.concise_output).resolve()
        concise_output_path.parent.mkdir(parents=True, exist_ok=True)
        concise_output_path.write_text(render_concise_markdown(records), encoding="utf-8")
        print(f"Wrote {len(records)} concise trajectory summaries to {concise_output_path}")
    if args.full_output:
        full_output_path = Path(args.full_output).resolve()
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        full_output_path.write_text(render_full_markdown(records), encoding="utf-8")
        print(f"Wrote {len(records)} full trajectory summaries to {full_output_path}")


if __name__ == "__main__":
    main()
