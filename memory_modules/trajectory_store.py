import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.exists(), f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(relative_target)


def normalize_trajectory_pool_root(pool_root: Path) -> Path:
    trajectories_subdir = pool_root / "trajectories"
    if trajectories_subdir.exists() and trajectories_subdir.is_dir():
        return trajectories_subdir.resolve()
    return pool_root.resolve()


def goal_text(raw_goal: Any) -> str:
    if isinstance(raw_goal, str):
        text = raw_goal.strip()
        return text if text else "<goal not found>"
    if isinstance(raw_goal, list):
        parts = [part.strip() for part in raw_goal if isinstance(part, str) and part.strip()]
        if parts:
            return " ".join(parts)
    return "<goal not found>"


def resolve_screenshot_source(screenshot_value: str, trajectories_root_dir: Path) -> Path:
    screenshot_path = Path(screenshot_value)
    if screenshot_path.is_absolute():
        require(screenshot_path.exists(), f"Missing screenshot file: {screenshot_path}")
        return screenshot_path

    candidates = [
        trajectories_root_dir / screenshot_path,
        trajectories_root_dir / "screenshots" / screenshot_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not resolve screenshot path {screenshot_value!r} under {trajectories_root_dir}"
    )


@dataclass(frozen=True)
class PreparedTrajectoryInsert:
    trajectory_id: str
    simplified: dict[str, Any]
    screenshot_sources: tuple[Path, ...]
    fingerprint: str

    @property
    def screenshot_count(self) -> int:
        return len(self.screenshot_sources)


def screenshot_name_for_state(state_index: int, screenshot_src: Path) -> str:
    return f"{state_index:04d}{screenshot_src.suffix or '.png'}"


def _logical_fingerprint_payload_from_simplified(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory_id = trajectory.get("id")
    goal = trajectory.get("goal")
    outcome = trajectory.get("outcome")
    start_url = trajectory.get("start_url")
    actions = trajectory.get("actions")
    states = trajectory.get("states")
    require(
        isinstance(trajectory_id, str) and trajectory_id,
        "simplified trajectory id must be a non-empty string",
    )
    require(
        isinstance(goal, str),
        f"simplified trajectory goal must be a string for {trajectory_id}",
    )
    require(
        outcome is None or isinstance(outcome, str),
        f"simplified trajectory outcome must be a string or null for {trajectory_id}",
    )
    require(
        isinstance(start_url, str) and start_url.strip(),
        f"simplified trajectory start_url must be a non-empty string for {trajectory_id}",
    )
    require(
        isinstance(actions, list) and all(isinstance(item, str) for item in actions),
        f"simplified trajectory actions must be a list of strings for {trajectory_id}",
    )
    require(
        isinstance(states, list) and states,
        f"simplified trajectory states must be a non-empty list for {trajectory_id}",
    )

    normalized_states: list[dict[str, Any]] = []
    for idx, state in enumerate(states):
        require(isinstance(state, dict), f"simplified trajectory {trajectory_id} state {idx} must be an object")
        state_index = state.get("state_index")
        step = state.get("step")
        url = state.get("url")
        action = state.get("action")
        thoughts = state.get("thoughts")
        text = state.get("text")
        screenshot = state.get("screenshot")
        require(
            isinstance(state_index, int) and not isinstance(state_index, bool) and state_index >= 0,
            f"simplified trajectory {trajectory_id} state {idx} state_index must be an integer >= 0",
        )
        require(
            isinstance(step, int) and not isinstance(step, bool) and step >= 0,
            f"simplified trajectory {trajectory_id} state {idx} step must be an integer >= 0",
        )
        require(
            isinstance(url, str) and url.strip(),
            f"simplified trajectory {trajectory_id} state {idx} url must be a non-empty string",
        )
        require(
            action is None or isinstance(action, str),
            f"simplified trajectory {trajectory_id} state {idx} action must be string or null",
        )
        require(
            thoughts is None or isinstance(thoughts, str),
            f"simplified trajectory {trajectory_id} state {idx} thoughts must be string or null",
        )
        require(
            isinstance(text, str),
            f"simplified trajectory {trajectory_id} state {idx} text must be a string",
        )
        require(
            isinstance(screenshot, str) and screenshot.strip(),
            f"simplified trajectory {trajectory_id} state {idx} screenshot must be a non-empty string",
        )
        normalized_states.append(
            {
                "state_index": state_index,
                "step": step,
                "url": url,
                "action": action,
                "thoughts": thoughts,
                "text": text,
            }
        )

    return {
        "id": trajectory_id,
        "goal": goal,
        "outcome": outcome,
        "start_url": start_url,
        "actions": list(actions),
        "states": normalized_states,
    }


def logical_trajectory_fingerprint_from_simplified(trajectory: dict[str, Any]) -> str:
    payload = _logical_fingerprint_payload_from_simplified(trajectory)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_trajectory_insert(
    trajectory: dict[str, object],
    *,
    trajectories_root_dir: Path,
) -> PreparedTrajectoryInsert:
    trajectory_id = trajectory.get("id")
    public_states = trajectory.get("states")
    content = trajectory.get("content")
    metadata = trajectory.get("metadata")
    outcome = trajectory.get("outcome")
    require(
        isinstance(trajectory_id, str) and trajectory_id,
        "trajectory id must be a non-empty string",
    )
    if isinstance(public_states, list) and public_states:
        goal = trajectory.get("goal")
        start_url_value = trajectory.get("start_url")
        require(isinstance(goal, str), f"trajectory goal must be a string for {trajectory_id}")
        require(
            isinstance(start_url_value, str) and start_url_value.strip(),
            f"trajectory start_url must be a non-empty string for {trajectory_id}",
        )
        require(
            outcome is None or isinstance(outcome, str),
            f"trajectory outcome must be a string or null for {trajectory_id}",
        )

        simplified_states: list[dict[str, Any]] = []
        screenshot_sources: list[Path] = []
        actions: list[str] = []
        for state_index, state in enumerate(public_states):
            require(
                isinstance(state, dict),
                f"trajectory {trajectory_id} state {state_index} must be an object",
            )
            url = state.get("url")
            action = state.get("action")
            thought = state.get("thought", state.get("thoughts"))
            text = state.get("accessibility_tree", state.get("text"))
            screenshot_value = state.get("screenshot")
            require(
                isinstance(url, str) and url.strip(),
                f"trajectory {trajectory_id} state {state_index} missing url",
            )
            require(
                action is None or isinstance(action, str),
                f"trajectory {trajectory_id} state {state_index} action must be string or null",
            )
            require(
                thought is None or isinstance(thought, str),
                f"trajectory {trajectory_id} state {state_index} thought must be string or null",
            )
            require(
                isinstance(text, str),
                f"trajectory {trajectory_id} state {state_index} accessibility_tree/text must be string",
            )
            require(
                isinstance(screenshot_value, str) and screenshot_value.strip(),
                f"trajectory {trajectory_id} state {state_index} screenshot must be a non-empty string",
            )
            screenshot_src = resolve_screenshot_source(screenshot_value, trajectories_root_dir)
            screenshot_sources.append(screenshot_src)
            if isinstance(action, str) and action.strip():
                actions.append(action)
            step_value = state.get("step")
            original_state_index = state.get("state_index")
            simplified_states.append(
                {
                    "state_index": state_index,
                    "step": (
                        step_value
                        if isinstance(step_value, int) and not isinstance(step_value, bool)
                        else (
                            original_state_index
                            if isinstance(original_state_index, int)
                            and not isinstance(original_state_index, bool)
                            else state_index
                        )
                    ),
                    "url": url,
                    "action": action,
                    "thoughts": thought,
                    "text": text,
                    "screenshot": f"screenshots/{screenshot_name_for_state(state_index, screenshot_src)}",
                }
            )

        simplified = {
            "id": trajectory_id,
            "goal": goal,
            "outcome": outcome,
            "start_url": start_url_value,
            "actions": actions,
            "states": simplified_states,
        }
        return PreparedTrajectoryInsert(
            trajectory_id=trajectory_id,
            simplified=simplified,
            screenshot_sources=tuple(screenshot_sources),
            fingerprint=logical_trajectory_fingerprint_from_simplified(simplified),
        )

    require(
        isinstance(content, list) and content,
        f"trajectory content must be a non-empty list for {trajectory_id}",
    )
    require(
        isinstance(metadata, dict),
        f"trajectory metadata must be an object for {trajectory_id}",
    )
    require(
        outcome is None or isinstance(outcome, str),
        f"trajectory outcome must be a string or null for {trajectory_id}",
    )

    simplified_states: list[dict[str, Any]] = []
    screenshot_sources: list[Path] = []
    actions: list[str] = []
    start_url: str | None = None
    for state_index, state in enumerate(content):
        require(
            isinstance(state, dict),
            f"trajectory {trajectory_id} state {state_index} must be an object",
        )
        url = state.get("url")
        action = state.get("action")
        thoughts = state.get("thoughts")
        observation = state.get("observation")
        require(
            isinstance(url, str) and url.strip(),
            f"trajectory {trajectory_id} state {state_index} missing url",
        )
        require(
            action is None or isinstance(action, str),
            f"trajectory {trajectory_id} state {state_index} action must be string or null",
        )
        require(
            thoughts is None or isinstance(thoughts, str),
            f"trajectory {trajectory_id} state {state_index} thoughts must be string or null",
        )
        require(
            isinstance(observation, dict),
            f"trajectory {trajectory_id} state {state_index} observation must be an object",
        )
        text = observation.get("text")
        screenshot_value = observation.get("screenshot")
        require(
            isinstance(text, str),
            f"trajectory {trajectory_id} state {state_index} observation.text must be string",
        )
        require(
            isinstance(screenshot_value, str) and screenshot_value.strip(),
            f"trajectory {trajectory_id} state {state_index} observation.screenshot must be a non-empty string",
        )

        if start_url is None:
            start_url = url
        if isinstance(action, str) and action.strip():
            actions.append(action)

        screenshot_src = resolve_screenshot_source(
            screenshot_value,
            trajectories_root_dir,
        )
        screenshot_sources.append(screenshot_src)

        step_value = state.get("step")
        simplified_states.append(
            {
                "state_index": state_index,
                "step": step_value if isinstance(step_value, int) else state_index,
                "url": url,
                "action": action,
                "thoughts": thoughts,
                "text": text,
                "screenshot": f"screenshots/{screenshot_name_for_state(state_index, screenshot_src)}",
            }
        )

    require(start_url is not None, f"trajectory {trajectory_id} is missing a start url")
    simplified = {
        "id": trajectory_id,
        "goal": goal_text(metadata.get("original_goal")),
        "outcome": outcome,
        "start_url": start_url,
        "actions": actions,
        "states": simplified_states,
    }
    return PreparedTrajectoryInsert(
        trajectory_id=trajectory_id,
        simplified=simplified,
        screenshot_sources=tuple(screenshot_sources),
        fingerprint=logical_trajectory_fingerprint_from_simplified(simplified),
    )


def materialize_prepared_trajectory(prepared: PreparedTrajectoryInsert, trajectory_dir: Path) -> None:
    screenshots_dir = trajectory_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    for state, screenshot_src in zip(prepared.simplified["states"], prepared.screenshot_sources):
        require(isinstance(state, dict), f"prepared simplified state must be an object for {prepared.trajectory_id}")
        screenshot_value = state.get("screenshot")
        require(
            isinstance(screenshot_value, str) and screenshot_value.startswith("screenshots/"),
            f"prepared simplified screenshot must use screenshots/ prefix for {prepared.trajectory_id}",
        )
        screenshot_name = screenshot_value.split("/", 1)[1]
        destination = screenshots_dir / screenshot_name
        if destination.exists():
            continue
        try:
            relative_symlink(screenshot_src.resolve(), destination)
        except OSError:
            shutil.copy2(screenshot_src, destination)
    save_json(trajectory_dir / "trajectory.json", prepared.simplified)


def validate_pooled_trajectory_dir(
    prepared: PreparedTrajectoryInsert,
    *,
    pooled_trajectory_dir: Path,
) -> None:
    require(
        pooled_trajectory_dir.exists() and pooled_trajectory_dir.is_dir(),
        f"Missing pooled trajectory directory for {prepared.trajectory_id}: {pooled_trajectory_dir}",
    )
    pooled_trajectory_path = pooled_trajectory_dir / "trajectory.json"
    pooled_screenshots_dir = pooled_trajectory_dir / "screenshots"
    require(
        pooled_trajectory_path.exists(),
        f"Missing pooled trajectory.json for {prepared.trajectory_id}: {pooled_trajectory_path}",
    )
    require(
        pooled_screenshots_dir.exists() and pooled_screenshots_dir.is_dir(),
        f"Missing pooled screenshots directory for {prepared.trajectory_id}: {pooled_screenshots_dir}",
    )
    pooled_payload = load_json(pooled_trajectory_path)
    require(
        isinstance(pooled_payload, dict),
        f"Pooled trajectory.json must be an object for {prepared.trajectory_id}",
    )
    pooled_fingerprint = logical_trajectory_fingerprint_from_simplified(pooled_payload)
    require(
        pooled_fingerprint == prepared.fingerprint,
        (
            f"Pooled trajectory fingerprint mismatch for {prepared.trajectory_id}: "
            f"expected {prepared.fingerprint}, got {pooled_fingerprint}"
        ),
    )

    pooled_states = pooled_payload.get("states")
    require(
        isinstance(pooled_states, list),
        f"Pooled trajectory states must be a list for {prepared.trajectory_id}",
    )
    referenced_screenshot_paths: list[Path] = []
    for idx, state in enumerate(pooled_states):
        require(
            isinstance(state, dict),
            f"Pooled trajectory state {idx} must be an object for {prepared.trajectory_id}",
        )
        screenshot_value = state.get("screenshot")
        require(
            isinstance(screenshot_value, str) and screenshot_value.strip(),
            f"Pooled trajectory state {idx} missing screenshot path for {prepared.trajectory_id}",
        )
        screenshot_path = pooled_trajectory_dir / screenshot_value
        require(
            screenshot_path.exists(),
            f"Missing pooled screenshot referenced by trajectory.json for {prepared.trajectory_id}: {screenshot_path}",
        )
        referenced_screenshot_paths.append(screenshot_path)

    screenshot_files = [path for path in pooled_screenshots_dir.iterdir() if path.is_file()]
    require(
        len(screenshot_files) == prepared.screenshot_count,
        (
            f"Pooled screenshot count mismatch for {prepared.trajectory_id}: "
            f"expected {prepared.screenshot_count}, got {len(screenshot_files)}"
        ),
    )
    require(
        len(referenced_screenshot_paths) == prepared.screenshot_count,
        (
            f"Pooled referenced screenshot count mismatch for {prepared.trajectory_id}: "
            f"expected {prepared.screenshot_count}, got {len(referenced_screenshot_paths)}"
        ),
    )
