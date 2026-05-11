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
            "Inspect one trajectory quickly. Default prints a compact trajectory summary. "
            "Use --state, --span, or --match for focused inspection."
        )
    )
    parser.add_argument("trajectory_id", help="Trajectory id under trajectories/<trajectory_id>/trajectory.json")
    parser.add_argument(
        "--trajectories-dir",
        default="trajectories",
        help="Root directory containing trajectory folders. Default: trajectories",
    )
    parser.add_argument(
        "--state",
        type=int,
        default=None,
        help="Show one exact state with URL, thought, action, screenshot path, and AXTree text.",
    )
    parser.add_argument(
        "--span",
        default=None,
        help="Show a contiguous span as START:END using zero-based inclusive indices.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Case-insensitive regex to search within one trajectory across url/action/thought/text.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=12000,
        help="Maximum AXTree characters to print per state in --state/--span mode.",
    )
    parser.add_argument(
        "--max-thought-chars",
        type=int,
        default=240,
        help="Maximum thought characters in compact summary mode.",
    )
    parser.add_argument(
        "--max-url-chars",
        type=int,
        default=220,
        help="Maximum URL characters in compact summary mode.",
    )
    parser.add_argument(
        "--max-match-lines",
        type=int,
        default=8,
        help="Maximum matching text lines to print per state in --match mode.",
    )
    parser.add_argument(
        "--max-match-line-chars",
        type=int,
        default=240,
        help="Maximum characters per printed matching line in --match mode.",
    )
    args = parser.parse_args()
    selected_modes = sum(
        value is not None
        for value in [args.state, args.span, args.match]
    )
    require(
        selected_modes <= 1,
        "Use at most one of --state, --span, or --match at a time.",
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


def final_reward_text(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return text if text else "<final reward not found>"
    if raw_value is None:
        return "<final reward not found>"
    return json.dumps(raw_value, ensure_ascii=True, sort_keys=True)


def thought_text(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return text if text else "<none>"
    if raw_value is None:
        return "<none>"
    return json.dumps(raw_value, ensure_ascii=True, sort_keys=True)


def truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    half = (max_chars - 5) // 2
    return text[:half] + "\n...\n" + text[-(max_chars - 5 - half):]


def one_line(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if max_chars <= 0 or len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


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


def annotate_action(action_text: str, object_lookup: dict[str, str]) -> str:
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


def load_trajectory(trajectories_dir: Path, trajectory_id: str) -> dict[str, Any]:
    path = trajectories_dir / trajectory_id / "trajectory.json"
    require(path.exists(), f"Missing trajectory.json: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"trajectory.json must contain an object: {path}")
    states = payload.get("states")
    require(isinstance(states, list), f"trajectory.json states must be a list: {path}")
    return payload


def build_action_annotations(states: list[dict[str, Any]]) -> list[str]:
    annotations: list[str] = []
    known_object_lookup: dict[str, str] = {}
    for state in states:
        current_text = state.get("text")
        current_object_lookup = (
            _extract_object_lookup_from_tree(current_text)
            if isinstance(current_text, str) and current_text
            else {}
        )
        object_lookup = {**known_object_lookup, **current_object_lookup}
        action_value = state.get("action")
        action_text = action_value.strip() if isinstance(action_value, str) else ""
        annotations.append(annotate_action(action_text, object_lookup) if action_text else "<none>")
        known_object_lookup.update(current_object_lookup)
    return annotations


def screenshot_display_path(trajectory_id: str, screenshot_value: Any) -> str:
    if not isinstance(screenshot_value, str) or not screenshot_value.strip():
        return "<none>"
    return f"trajectories/{trajectory_id}/{screenshot_value}"


def parse_span(spec: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+):(\d+)", spec.strip())
    require(match is not None, "Span must use START:END with zero-based inclusive indices.")
    start = int(match.group(1))
    end = int(match.group(2))
    require(end >= start, "Span END must be >= START.")
    return start, end


def print_trajectory_summary(trajectory: dict[str, Any], trajectory_id: str, args: argparse.Namespace) -> None:
    states = trajectory["states"]
    action_annotations = build_action_annotations(states)
    print(f"Trajectory ID: {trajectory_id}")
    print(f"Goal: {goal_text(trajectory.get('goal'))}")
    print(f"Start URL: {trajectory.get('start_url', '<start url not found>')}")
    print(f"Final reward: {final_reward_text(trajectory.get('outcome'))}")
    print(f"State count: {len(states)}")
    print("")
    print("State table:")
    for state, annotated_action in zip(states, action_annotations):
        state_index = state.get("state_index", "<unknown>")
        print(f"- State {state_index}")
        print(f"  URL: {one_line(str(state.get('url', '<none>')), args.max_url_chars)}")
        print(f"  Thought: {one_line(thought_text(state.get('thoughts')), args.max_thought_chars)}")
        print(f"  Action: {one_line(annotated_action, 320)}")
        print(f"  Screenshot: {screenshot_display_path(trajectory_id, state.get('screenshot'))}")


def print_state_detail(trajectory: dict[str, Any], trajectory_id: str, state_index: int, args: argparse.Namespace) -> None:
    states = trajectory["states"]
    require(0 <= state_index < len(states), f"State index out of range: {state_index} (len={len(states)})")
    annotated_action = build_action_annotations(states)[state_index]
    state = states[state_index]
    print(f"Trajectory ID: {trajectory_id}")
    print(f"State index: {state_index}")
    print(f"URL: {state.get('url', '<none>')}")
    print(f"Thought: {thought_text(state.get('thoughts'))}")
    print(f"Action: {annotated_action}")
    print(f"Screenshot: {screenshot_display_path(trajectory_id, state.get('screenshot'))}")
    print("")
    print("AXTree text:")
    print(truncate_middle(str(state.get("text", "")), args.max_text_chars))


def print_span_detail(trajectory: dict[str, Any], trajectory_id: str, start: int, end: int, args: argparse.Namespace) -> None:
    states = trajectory["states"]
    require(0 <= start < len(states), f"Span start out of range: {start} (len={len(states)})")
    require(0 <= end < len(states), f"Span end out of range: {end} (len={len(states)})")
    action_annotations = build_action_annotations(states)
    print(f"Trajectory ID: {trajectory_id}")
    print(f"Span: {start}:{end}")
    for state_index in range(start, end + 1):
        state = states[state_index]
        print("")
        print(f"=== State {state_index} ===")
        print(f"URL: {state.get('url', '<none>')}")
        print(f"Thought: {thought_text(state.get('thoughts'))}")
        print(f"Action: {action_annotations[state_index]}")
        print(f"Screenshot: {screenshot_display_path(trajectory_id, state.get('screenshot'))}")
        print("AXTree text:")
        print(truncate_middle(str(state.get("text", "")), args.max_text_chars))


def print_match_results(trajectory: dict[str, Any], trajectory_id: str, pattern_text: str, args: argparse.Namespace) -> None:
    states = trajectory["states"]
    action_annotations = build_action_annotations(states)
    pattern = re.compile(pattern_text, re.IGNORECASE)
    print(f"Trajectory ID: {trajectory_id}")
    print(f"Match regex: {pattern_text}")
    hits = 0
    for state, annotated_action in zip(states, action_annotations):
        matched_lines: list[str] = []
        url_text = str(state.get("url", ""))
        thought = thought_text(state.get("thoughts"))
        if pattern.search(url_text):
            matched_lines.append(f"URL: {url_text}")
        if pattern.search(annotated_action):
            matched_lines.append(f"Action: {annotated_action}")
        if pattern.search(thought):
            matched_lines.append(f"Thought: {thought}")
        text_value = str(state.get("text", ""))
        for line in text_value.splitlines():
            if pattern.search(line):
                matched_lines.append(f"Text: {one_line(line, args.max_match_line_chars)}")
                if len(matched_lines) >= args.max_match_lines:
                    break
        if not matched_lines:
            continue
        hits += 1
        print("")
        print(f"=== State {state.get('state_index', '<unknown>')} ===")
        print(f"URL: {state.get('url', '<none>')}")
        print(f"Thought: {one_line(thought, 320)}")
        print(f"Action: {one_line(annotated_action, 320)}")
        print(f"Screenshot: {screenshot_display_path(trajectory_id, state.get('screenshot'))}")
        for line in matched_lines[: args.max_match_lines]:
            print(f"- {line}")
    if hits == 0:
        print("")
        print("No matching states found.")


def main() -> None:
    args = parse_args()
    trajectories_dir = Path(args.trajectories_dir).resolve()
    require(trajectories_dir.exists() and trajectories_dir.is_dir(), f"Missing trajectories dir: {trajectories_dir}")
    trajectory = load_trajectory(trajectories_dir, args.trajectory_id)
    if args.state is not None:
        print_state_detail(trajectory, args.trajectory_id, args.state, args)
        return
    if args.span is not None:
        start, end = parse_span(args.span)
        print_span_detail(trajectory, args.trajectory_id, start, end, args)
        return
    if args.match is not None:
        print_match_results(trajectory, args.trajectory_id, args.match, args)
        return
    print_trajectory_summary(trajectory, args.trajectory_id, args)


if __name__ == "__main__":
    main()
