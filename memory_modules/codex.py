import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require
from .trajectory_store import (
    materialize_prepared_trajectory as materialize_prepared_trajectory_shared,
    prepare_trajectory_insert as prepare_trajectory_insert_shared,
    validate_pooled_trajectory_dir as validate_pooled_trajectory_dir_shared,
)


MAX_TOTAL_SPAN_STATES = 20
DEFAULT_CODEX_BINARY = Path(os.getenv("CODEX_BINARY", "codex"))
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
DEFAULT_CODEX_TIMEOUT_SECONDS = 1800.0
DEFAULT_CODEX_MAX_ATTEMPTS = 3
PROCESS_POLL_INTERVAL_SECONDS = 0.25
TERMINATION_GRACE_SECONDS = 5.0
QUESTION_IMAGE_NAME = "question_image"
DEFAULT_PROMPT = (
    "You are acting as a memory retrieval module. "
    "Read the local files in this directory, especially INSTRUCTION.md and question.json. "
    "The local trajectories/ directory contains the current haystack for this evaluation item, and you must explore trajectories/ before returning your final result. "
    "If question.json refers to an image, view it carefully. "
    "Write your final result to memory_module_output.json as valid JSON. "
    "If memory_module_output.json already exists, overwrite it with your final valid JSON output."
)
QUESTION_INSTRUCTIONS = f"""# Instructions
You are acting as a memory retrieval module, not as the final answering model.

Read `question.json`. The local `trajectories/` directory contains the current haystack for this evaluation item, and you must explore files under `trajectories/` before returning your final result.

Important rules:

- Inspect relevant files under `trajectories/` before returning your final result.
- Do not answer the benchmark question directly.
- Move fast and prefer targeted exploration.
- Put the most important evidence first.
- Avoid redundant trajectories when multiple trajectories support the same important information.
- You may emit any number of spans, but the total number of states across all spans must be at most `{MAX_TOTAL_SPAN_STATES}`.
- Count span size inclusively. For example, states `3-5` count as `3` states toward the budget.
- You may write scratch files in the current directory if needed.
- Do not copy screenshots or AXTree blocks into the output JSON.

Write your final result to `memory_module_output.json` as valid JSON with this exact schema:

```json
{{
  "memory_markdown": "## Support Analysis\\n...\\n\\n## Relevant Procedure and Hint Notes\\n...",
  "trajectory_spans": [
    {{
      "trajectory_id": "<trajectory id>",
      "start_state_index": 0,
      "end_state_index": 0
    }}
  ]
}}
```

Requirements:

- `memory_markdown` should contain the two narrative sections only:
  - `## Support Analysis`
  - `## Relevant Procedure and Hint Notes`
- It may mention the likely answer when strongly supported by evidence.
- `trajectory_spans` must use zero-based inclusive indices.
- Preserve span order by importance.
- If you find no useful evidence, still write valid JSON with an empty or minimal `memory_markdown` and an empty `trajectory_spans` list.
"""


@dataclass(frozen=True)
class MemoryOutputStatus:
    state: str
    detail: str | None

    @property
    def is_finished(self) -> bool:
        return self.state == "finished"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.exists(), f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def question_text(question_field: Any) -> str:
    if isinstance(question_field, str):
        require(question_field.strip(), "Question text must be non-empty")
        return question_field
    require(isinstance(question_field, dict), "question must be string or object")
    text = question_field.get("text")
    require(isinstance(text, str) and text.strip(), "question.text must be non-empty")
    return text


def question_image(question_field: Any) -> str | None:
    if isinstance(question_field, str):
        return None
    require(isinstance(question_field, dict), "question must be string or object")
    image = question_field.get("image")
    if image is None:
        return None
    require(isinstance(image, str) and image.strip(), "question.image must be a non-empty string")
    return image


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(relative_target)


def normalize_trajectory_pool_root(pool_root: Path) -> Path:
    trajectories_subdir = pool_root / "trajectories"
    if trajectories_subdir.exists() and trajectories_subdir.is_dir():
        return trajectories_subdir.resolve()
    return pool_root.resolve()


def ensure_string_list(value: Any, *, field_name: str) -> list[str]:
    require(isinstance(value, list), f"{field_name} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        require(
            isinstance(item, str) and item.strip(),
            f"{field_name}[{idx}] must be a non-empty string",
        )
        out.append(item)
    return out


def load_question_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if path.suffix == ".jsonl":
        data = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        data = load_json(path)
    require(isinstance(data, list), f"Expected list in questions file: {path}")
    by_id: dict[str, dict[str, Any]] = {}
    id_by_text: dict[str, str] = {}
    for idx, item in enumerate(data):
        require(isinstance(item, dict), f"Question item {idx} must be an object")
        question_id = item.get("id")
        require(isinstance(question_id, str) and question_id, f"Invalid question id at index {idx}")
        require(question_id not in by_id, f"Duplicate question id in {path}: {question_id}")
        text = question_text(item.get("question"))
        by_id[question_id] = dict(item)
        id_by_text.setdefault(text, question_id)
    return by_id, id_by_text


def copy_question_image(question_image_path: str, sandbox_dir: Path) -> str:
    source_path = Path(question_image_path)
    require(source_path.exists(), f"Missing question image: {source_path}")
    image_name = f"{QUESTION_IMAGE_NAME}{source_path.suffix or '.png'}"
    shutil.copy2(source_path, sandbox_dir / image_name)
    return image_name


def validate_memory_module_output_payload(payload: Any) -> dict[str, Any]:
    require(isinstance(payload, dict), "memory output must be a JSON object")
    memory_markdown = payload.get("memory_markdown")
    trajectory_spans = payload.get("trajectory_spans")
    require(isinstance(memory_markdown, str), "memory_markdown must be a string")
    require(isinstance(trajectory_spans, list), "trajectory_spans must be a list")

    normalized_spans: list[dict[str, Any]] = []
    total_span_states = 0
    for idx, item in enumerate(trajectory_spans):
        require(isinstance(item, dict), f"trajectory_spans[{idx}] must be an object")
        trajectory_id = item.get("trajectory_id")
        start_state_index = item.get("start_state_index")
        end_state_index = item.get("end_state_index")
        require(
            isinstance(trajectory_id, str) and trajectory_id.strip(),
            f"trajectory_spans[{idx}].trajectory_id must be a non-empty string",
        )
        require(
            isinstance(start_state_index, int)
            and not isinstance(start_state_index, bool)
            and start_state_index >= 0,
            f"trajectory_spans[{idx}].start_state_index must be an integer >= 0",
        )
        require(
            isinstance(end_state_index, int)
            and not isinstance(end_state_index, bool)
            and end_state_index >= start_state_index,
            f"trajectory_spans[{idx}].end_state_index must be an integer >= start_state_index",
        )
        total_span_states += end_state_index - start_state_index + 1
        require(
            total_span_states <= MAX_TOTAL_SPAN_STATES,
            (
                "trajectory_spans exceed the total state budget: "
                f"{total_span_states} > {MAX_TOTAL_SPAN_STATES}"
            ),
        )
        normalized_spans.append(
            {
                "trajectory_id": trajectory_id.strip(),
                "start_state_index": start_state_index,
                "end_state_index": end_state_index,
            }
        )
    return {
        "memory_markdown": memory_markdown,
        "trajectory_spans": normalized_spans,
    }


def read_memory_output_status(output_path: Path) -> MemoryOutputStatus:
    if not output_path.exists():
        return MemoryOutputStatus("missing_output_file", None)
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return MemoryOutputStatus("invalid_json", f"{exc.msg} at line {exc.lineno} column {exc.colno}")
    try:
        validate_memory_module_output_payload(payload)
    except RuntimeError as exc:
        return MemoryOutputStatus("invalid_payload", str(exc))
    return MemoryOutputStatus("finished", None)


def parse_codex_json_events(raw_stdout: str) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    events: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    for line in raw_stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
            if payload.get("type") == "turn.completed" and isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
    return events, usage


def terminate_process_group(process: subprocess.Popen[str], *, reason: str) -> tuple[str, str]:
    if process.poll() is not None:
        return process.communicate()
    print(
        f"[codex] terminating pid={process.pid} reason={reason}",
        file=sys.stderr,
        flush=True,
    )
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        return process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except ProcessLookupError:
        return process.communicate()
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return process.communicate()
        return process.communicate()


def format_actions(trajectory: dict[str, Any]) -> str:
    action_entries = [
        action
        for action in trajectory.get("actions", [])
        if isinstance(action, str) and action.strip()
    ]
    if not action_entries:
        for state in trajectory.get("states", []):
            if not isinstance(state, dict):
                continue
            action = state.get("action")
            if isinstance(action, str) and action.strip():
                action_entries.append(action)
    if not action_entries:
        return "1. <no actions recorded>"
    return "\n".join(f"{idx + 1}. {action}" for idx, action in enumerate(action_entries))


def format_span_header(
    span_index: int,
    trajectory: dict[str, Any],
    start_state_index: int,
    end_state_index: int,
) -> str:
    return (
        f"### Trajectory span {span_index}: {trajectory['id']} states {start_state_index}-{end_state_index}\n\n"
        "Goal\n"
        f"- {trajectory.get('goal', '<goal not found>')}\n\n"
        "Start URL\n"
        f"- {trajectory.get('start_url', '<start url not found>')}\n\n"
        "Actions\n"
        f"{format_actions(trajectory)}\n\n"
        "Linked state evidence\n"
    )


def format_state_text(state: dict[str, Any], evidence_mode: str) -> str:
    action_value = state.get("action")
    action_text = action_value if isinstance(action_value, str) and action_value.strip() else "<none>"
    lines = [
        f"State {state['state_index']} (step {state['step']})",
        f"- URL: {state['url']}",
        f"- Action: {action_text}",
    ]
    if evidence_mode in {"axtree", "both"}:
        lines.extend(["- AXTree:", state["text"]])
    return "\n".join(lines) + "\n"


@register_memory
class CodexMemory(Memory):
    memory_type = "codex"
    requires_codex_binary = True

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)

        questions_path = memory_params.get("questions_path")
        evidence_mode = memory_params.get("evidence_mode", "both")
        codex_params_obj = memory_params.get("codex_params", {})
        workspace_dir = memory_params.get("workspace_dir")
        trajectories_root_dir = memory_params.get("trajectories_root_dir")
        trajectory_pool_root = memory_params.get("trajectory_pool_root")
        query_trace_dir = memory_params.get("query_trace_dir")

        require(
            isinstance(questions_path, str) and questions_path.strip(),
            "codex questions_path must be a non-empty string",
        )
        require(
            evidence_mode in {"axtree", "image", "both"},
            "codex evidence_mode must be one of: axtree, image, both",
        )
        require(
            isinstance(codex_params_obj, dict),
            "codex codex_params must be an object",
        )
        codex_params = dict(codex_params_obj)

        codex_binary_value = codex_params.get("binary", str(DEFAULT_CODEX_BINARY))
        codex_model = codex_params.get("model", DEFAULT_CODEX_MODEL)
        codex_reasoning_effort = codex_params.get(
            "reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT
        )
        codex_timeout_seconds = codex_params.get(
            "timeout_seconds", DEFAULT_CODEX_TIMEOUT_SECONDS
        )
        max_attempts_raw = codex_params.get("max_attempts", codex_params.get("max_retries", DEFAULT_CODEX_MAX_ATTEMPTS))
        codex_prompt = codex_params.get("prompt", DEFAULT_PROMPT)
        extra_config_obj = codex_params.get("extra_config", [])
        extra_args_obj = codex_params.get("extra_args", [])

        require(
            isinstance(codex_binary_value, str) and codex_binary_value.strip(),
            "codex codex_params.binary must be a non-empty string",
        )
        require(
            isinstance(codex_model, str) and codex_model.strip(),
            "codex codex_params.model must be a non-empty string",
        )
        require(
            isinstance(codex_reasoning_effort, str) and codex_reasoning_effort.strip(),
            "codex codex_params.reasoning_effort must be a non-empty string",
        )
        require(
            isinstance(codex_timeout_seconds, (int, float))
            and not isinstance(codex_timeout_seconds, bool)
            and float(codex_timeout_seconds) > 0.0,
            "codex codex_params.timeout_seconds must be a positive number",
        )
        require(
            isinstance(max_attempts_raw, int)
            and not isinstance(max_attempts_raw, bool)
            and max_attempts_raw > 0,
            "codex codex_params.max_retries/max_attempts must be a positive integer",
        )
        require(
            isinstance(codex_prompt, str) and codex_prompt.strip(),
            "codex codex_params.prompt must be a non-empty string",
        )

        self.questions_path = Path(questions_path).resolve()
        self.question_by_id, self.question_id_by_text = load_question_index(self.questions_path)
        self.evidence_mode = evidence_mode
        codex_binary_text = codex_binary_value.strip()
        resolved_binary = shutil.which(codex_binary_text) if os.sep not in codex_binary_text else None
        self.codex_binary = (
            Path(resolved_binary).resolve()
            if resolved_binary is not None
            else Path(codex_binary_text).expanduser().resolve()
        )
        if self.requires_codex_binary:
            require(
                self.codex_binary.exists(),
                f"codex Codex binary does not exist: {self.codex_binary}",
            )
        self.codex_model = codex_model.strip()
        self.codex_reasoning_effort = codex_reasoning_effort.strip()
        self.codex_timeout_seconds = float(codex_timeout_seconds)
        self.codex_max_attempts = int(max_attempts_raw)
        self.codex_prompt = codex_prompt.strip()
        self.codex_extra_config = ensure_string_list(extra_config_obj, field_name="codex_params.extra_config")
        self.codex_extra_args = ensure_string_list(extra_args_obj, field_name="codex_params.extra_args")

        self.workspace_dir = (
            Path(workspace_dir).resolve()
            if isinstance(workspace_dir, str) and workspace_dir.strip()
            else None
        )
        self.trajectories_root_dir = (
            Path(trajectories_root_dir).resolve()
            if isinstance(trajectories_root_dir, str) and trajectories_root_dir.strip()
            else None
        )
        self.trajectory_pool_root = None
        if isinstance(trajectory_pool_root, str) and trajectory_pool_root.strip():
            self.trajectory_pool_root = normalize_trajectory_pool_root(Path(trajectory_pool_root))
        if self.trajectory_pool_root is not None:
            require(
                self.trajectory_pool_root.exists() and self.trajectory_pool_root.is_dir(),
                (
                    "codex trajectory_pool_root must point to an existing directory: "
                    f"{self.trajectory_pool_root}"
                ),
            )
        self.query_trace_dir = (
            Path(query_trace_dir).resolve()
            if isinstance(query_trace_dir, str) and query_trace_dir.strip()
            else None
        )
        self.cancel_event: threading.Event | None = None
        self.inserted_trajectory_ids: list[str] = []
        self.inserted_trajectory_id_set: set[str] = set()
        self._attempt_dir_lock = threading.Lock()

        if self.workspace_dir is not None:
            self._ensure_workspace_layout(self.workspace_dir)
        if self.query_trace_dir is not None:
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, object] = {
            "questions_path": str(self.questions_path),
            "evidence_mode": self.evidence_mode,
            "codex_params": {
                "binary": str(self.codex_binary),
                "model": self.codex_model,
                "reasoning_effort": self.codex_reasoning_effort,
                "timeout_seconds": self.codex_timeout_seconds,
                "max_retries": self.codex_max_attempts,
                "prompt": self.codex_prompt,
                "extra_config": list(self.codex_extra_config),
                "extra_args": list(self.codex_extra_args),
            },
        }
        if self.trajectory_pool_root is not None:
            memory_params["trajectory_pool_root"] = str(self.trajectory_pool_root)
        return {
            "memory_type": self.memory_type,
            "memory_params": memory_params,
        }

    def configure_runtime(self, **kwargs: object) -> None:
        query_trace_dir = kwargs.get("query_trace_dir")
        if query_trace_dir is not None:
            if isinstance(query_trace_dir, Path):
                self.query_trace_dir = query_trace_dir.resolve()
            else:
                require(
                    isinstance(query_trace_dir, str) and query_trace_dir.strip(),
                    "codex query_trace_dir runtime override must be a non-empty string or Path",
                )
                self.query_trace_dir = Path(query_trace_dir).resolve()
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None:
            require(
                isinstance(cancel_event, threading.Event),
                (
                    "codex cancel_event runtime override must be a "
                    "threading.Event"
                ),
            )
            self.cancel_event = cancel_event

    def _is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            raise KeyboardInterrupt("codex query cancelled")

    def insert(self, trajectory: dict[str, object]) -> None:
        require(
            self.workspace_dir is not None,
            "codex insert requires workspace_dir",
        )
        require(
            self.trajectories_root_dir is not None,
            "codex insert requires trajectories_root_dir",
        )

        prepared = prepare_trajectory_insert_shared(
            trajectory,
            trajectories_root_dir=self.trajectories_root_dir,
        )
        trajectory_id = prepared.trajectory_id

        if trajectory_id in self.inserted_trajectory_id_set:
            return None

        trajectory_dir = self.workspace_dir / "trajectories" / trajectory_id
        if self.trajectory_pool_root is None:
            materialize_prepared_trajectory_shared(prepared, trajectory_dir)
        else:
            pooled_trajectory_dir = self.trajectory_pool_root / trajectory_id
            validate_pooled_trajectory_dir_shared(
                prepared,
                pooled_trajectory_dir=pooled_trajectory_dir,
            )
            require(
                not trajectory_dir.exists(),
                f"Refusing to overwrite existing trajectory dir: {trajectory_dir}",
            )
            relative_symlink(pooled_trajectory_dir, trajectory_dir)

        self.inserted_trajectory_ids.append(trajectory_id)
        self.inserted_trajectory_id_set.add(trajectory_id)
        self._write_index_files(self.workspace_dir)
        return None

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        self._raise_if_cancelled()
        require(
            isinstance(query, str) and query.strip(),
            "codex query must be a non-empty string",
        )
        require(
            self.workspace_dir is not None,
            "codex query requires workspace_dir",
        )
        if self.query_trace_dir is None:
            self.query_trace_dir = (self.workspace_dir / "query_traces").resolve()
            self.query_trace_dir.mkdir(parents=True, exist_ok=True)

        query_context = self.get_query_context()
        question_id_value = query_context.get("question_id")
        if isinstance(question_id_value, str) and question_id_value.strip():
            question_id = question_id_value
        else:
            question_id = self.question_id_by_text.get(query)
        require(
            isinstance(question_id, str) and question_id in self.question_by_id,
            "codex could not resolve question id for query",
        )
        question_item = self.question_by_id[question_id]
        effective_query_image = query_image
        if effective_query_image is None:
            effective_query_image = question_image(question_item.get("question"))

        last_failure_state = "unknown_failure"
        last_failure_detail: str | None = None
        for attempt_number in range(1, self.codex_max_attempts + 1):
            self._raise_if_cancelled()
            attempt_result = self._run_query_attempt(
                question_id=question_id,
                question_item=question_item,
                query_text=query,
                query_image=effective_query_image,
            )
            if attempt_result["status"] == "interrupted":
                raise KeyboardInterrupt(
                    f"codex query interrupted for question_id={question_id}"
                )
            if attempt_result["success"]:
                return attempt_result["memory_context"]
            last_failure_state = attempt_result["status"]
            last_failure_detail = attempt_result["detail"]
            print(
                (
                    "[codex] query attempt failed "
                    f"question_id={question_id} "
                    f"attempt={attempt_number}/{self.codex_max_attempts} "
                    f"status={attempt_result['status']} "
                    f"detail={attempt_result['detail'] or 'n/a'}"
                ),
                file=sys.stderr,
                flush=True,
            )

        print(
            (
                "[codex] returning empty memory context after "
                f"{self.codex_max_attempts} failed attempts for question_id={question_id} "
                f"last_status={last_failure_state} "
                f"last_detail={last_failure_detail or 'n/a'}"
            ),
            file=sys.stderr,
            flush=True,
        )
        return []

    def _save_backend(self, output_dir: Path) -> None:
        require(
            self.workspace_dir is not None,
            "codex has no active workspace to save",
        )
        self._write_index_files(self.workspace_dir)
        if self.workspace_dir.resolve() == output_dir.resolve():
            return None
        shutil.copy2(self.workspace_dir / "index.json", output_dir / "index.json")
        shutil.copy2(
            self.workspace_dir / "haystack_manifest.json",
            output_dir / "haystack_manifest.json",
        )
        src_trajectories_dir = self.workspace_dir / "trajectories"
        if src_trajectories_dir.exists():
            shutil.copytree(src_trajectories_dir, output_dir / "trajectories")
        return None

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        self._ensure_workspace_layout(self.workspace_dir)
        index_payload = load_json(self.workspace_dir / "index.json")
        require(
            isinstance(index_payload, dict),
            "codex index.json must be an object",
        )
        inserted_ids = index_payload.get("inserted_trajectory_ids")
        require(
            isinstance(inserted_ids, list)
            and all(isinstance(item, str) and item for item in inserted_ids),
            "codex index.json must contain inserted_trajectory_ids as a list of strings",
        )
        self.inserted_trajectory_ids = list(inserted_ids)
        self.inserted_trajectory_id_set = set(inserted_ids)
        return None

    def _ensure_workspace_layout(self, workspace_dir: Path) -> None:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    def _write_index_files(self, workspace_dir: Path) -> None:
        save_json(
            workspace_dir / "index.json",
            {
                "memory_type": self.memory_type,
                "updated_at_utc": utc_now_iso(),
                "trajectory_count": len(self.inserted_trajectory_ids),
                "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
            },
        )
        save_json(
            workspace_dir / "haystack_manifest.json",
            {
                "id": f"{self.memory_type}_haystack",
                "variant": "current_memory_workspace",
                "trajectory_ids": list(self.inserted_trajectory_ids),
                "metadata": {
                    "generated_at_utc": utc_now_iso(),
                    "memory_type": self.memory_type,
                    "trajectory_count": len(self.inserted_trajectory_ids),
                },
            },
        )

    def _next_attempt_dir(self, question_id: str) -> tuple[int, Path]:
        require(
            self.query_trace_dir is not None,
            "codex query_trace_dir is not configured",
        )
        with self._attempt_dir_lock:
            question_trace_dir = self.query_trace_dir / question_id
            question_trace_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(
                path
                for path in question_trace_dir.iterdir()
                if path.is_dir() and path.name.startswith("attempt_")
            )
            attempt_index = len(existing) + 1
            attempt_dir = question_trace_dir / f"attempt_{attempt_index:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
        return attempt_index, attempt_dir

    def _build_question_payload(
        self,
        *,
        question_id: str,
        question_item: dict[str, Any],
        query_text: str,
        query_image: str | None,
        sandbox_dir: Path,
    ) -> dict[str, Any]:
        question_type = question_item.get("question_type")
        require(
            isinstance(question_type, str) and question_type.strip(),
            f"Question type must be a non-empty string for {question_id}",
        )
        payload: dict[str, Any] = {}
        if query_image is None:
            payload["question"] = query_text
            return payload

        image_name = copy_question_image(query_image, sandbox_dir)
        payload["question"] = {
            "text": query_text,
            "image": image_name,
        }
        return payload

    def _build_codex_command(self, *, sandbox_dir: Path, last_message_path: Path) -> list[str]:
        command = [
            str(self.codex_binary),
            "exec",
            "-C",
            str(sandbox_dir),
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "-o",
            str(last_message_path),
            "-m",
            self.codex_model,
            "-c",
            f"model_reasoning_effort={json.dumps(self.codex_reasoning_effort)}",
        ]
        for item in self.codex_extra_config:
            command.extend(["-c", item])
        command.extend(self.codex_extra_args)
        command.append(self.codex_prompt)
        return command

    def _load_stored_trajectory(self, trajectory_id: str) -> dict[str, Any] | None:
        require(
            self.workspace_dir is not None,
            "codex workspace_dir is not configured",
        )
        path = self.workspace_dir / "trajectories" / trajectory_id / "trajectory.json"
        if not path.exists():
            return None
        payload = load_json(path)
        require(
            isinstance(payload, dict),
            f"Stored trajectory payload must be an object: {path}",
        )
        states = payload.get("states")
        require(
            isinstance(states, list),
            f"Stored trajectory states must be a list: {path}",
        )
        return payload

    def _normalize_output_for_query(
        self,
        output_path: Path,
    ) -> dict[str, Any]:
        payload = validate_memory_module_output_payload(load_json(output_path))
        normalized: dict[str, Any] = {
            "memory_markdown": payload["memory_markdown"],
            "trajectory_spans_raw": payload["trajectory_spans"],
            "trajectory_spans_valid": [],
            "trajectory_spans_invalid": [],
        }
        for span in payload["trajectory_spans"]:
            trajectory = self._load_stored_trajectory(span["trajectory_id"])
            if trajectory is None:
                normalized["trajectory_spans_invalid"].append(
                    {
                        **span,
                        "reason": "unknown_trajectory_id",
                    }
                )
                continue
            states = trajectory.get("states", [])
            if span["end_state_index"] >= len(states):
                normalized["trajectory_spans_invalid"].append(
                    {
                        **span,
                        "reason": f"state_index_out_of_range(len={len(states)})",
                    }
                )
                continue
            normalized["trajectory_spans_valid"].append(span)
        return normalized

    def _build_memory_context_from_output(
        self,
        normalized_output: dict[str, Any],
    ) -> list[MemoryContextItem]:
        require(
            self.workspace_dir is not None,
            "codex workspace_dir is not configured",
        )
        items: list[MemoryContextItem] = []
        memory_markdown = normalized_output["memory_markdown"]
        valid_spans = normalized_output["trajectory_spans_valid"]

        if isinstance(memory_markdown, str) and memory_markdown.strip():
            items.append({"type": "text", "value": memory_markdown.strip() + "\n"})
        if valid_spans:
            span_lines = ["## Trajectory State Spans"]
            for span in valid_spans:
                span_lines.append(
                    f"- {span['trajectory_id']}: states {span['start_state_index']}-{span['end_state_index']}"
                )
            items.append({"type": "text", "value": "\n".join(span_lines) + "\n"})
            items.append({"type": "text", "value": "## Linked Evidence\n"})

        for idx, span in enumerate(valid_spans, start=1):
            trajectory = self._load_stored_trajectory(span["trajectory_id"])
            require(
                trajectory is not None,
                f"Trajectory unexpectedly missing during expansion: {span['trajectory_id']}",
            )
            items.append(
                {
                    "type": "text",
                    "value": format_span_header(
                        idx,
                        trajectory,
                        span["start_state_index"],
                        span["end_state_index"],
                    ),
                }
            )
            states = trajectory["states"][span["start_state_index"] : span["end_state_index"] + 1]
            for state in states:
                items.append(
                    {
                        "type": "text",
                        "value": format_state_text(state, self.evidence_mode),
                    }
                )
                if self.evidence_mode in {"image", "both"}:
                    screenshot_path = (
                        self.workspace_dir / "trajectories" / trajectory["id"] / state["screenshot"]
                    ).resolve()
                    require(
                        screenshot_path.exists(),
                        f"Missing copied screenshot: {screenshot_path}",
                    )
                    items.append({"type": "image", "value": str(screenshot_path)})
        return items

    def _run_query_attempt(
        self,
        *,
        question_id: str,
        question_item: dict[str, Any],
        query_text: str,
        query_image: str | None,
    ) -> dict[str, Any]:
        require(
            self.workspace_dir is not None,
            "codex workspace_dir is not configured",
        )
        attempt_index, attempt_dir = self._next_attempt_dir(question_id)
        sandbox_dir = attempt_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        question_payload = self._build_question_payload(
            question_id=question_id,
            question_item=question_item,
            query_text=query_text,
            query_image=query_image,
            sandbox_dir=sandbox_dir,
        )
        save_json(sandbox_dir / "question.json", question_payload)
        (sandbox_dir / "INSTRUCTION.md").write_text(QUESTION_INSTRUCTIONS, encoding="utf-8")
        relative_symlink(self.workspace_dir / "trajectories", sandbox_dir / "trajectories")
        relative_symlink(self.workspace_dir / "index.json", sandbox_dir / "index.json")
        relative_symlink(
            self.workspace_dir / "haystack_manifest.json",
            sandbox_dir / "haystack_manifest.json",
        )

        output_path = sandbox_dir / "memory_module_output.json"
        last_message_path = attempt_dir / "last_message.txt"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        events_path = attempt_dir / "events.json"
        summary_path = attempt_dir / "summary.json"
        command = self._build_codex_command(
            sandbox_dir=sandbox_dir,
            last_message_path=last_message_path,
        )

        started_at_ts = time.time()
        timed_out = False
        interrupted = False
        stdout_text = ""
        stderr_text = ""
        returncode: int | None = None
        process: subprocess.Popen[str] | None = None

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            while True:
                elapsed_seconds = time.time() - started_at_ts
                remaining_seconds = self.codex_timeout_seconds - elapsed_seconds
                if self._is_cancelled():
                    interrupted = True
                    stdout_text, stderr_text = terminate_process_group(
                        process,
                        reason="cancel_event",
                    )
                    break
                if remaining_seconds <= 0:
                    timed_out = True
                    stdout_text, stderr_text = terminate_process_group(
                        process,
                        reason="timeout",
                    )
                    break
                try:
                    stdout_text, stderr_text = process.communicate(
                        timeout=min(PROCESS_POLL_INTERVAL_SECONDS, remaining_seconds),
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            returncode = process.returncode
        except KeyboardInterrupt:
            if process is not None:
                stdout_text, stderr_text = terminate_process_group(
                    process,
                    reason="keyboard_interrupt",
                )
                returncode = process.returncode
            raise

        duration_seconds = time.time() - started_at_ts
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        events, usage = parse_codex_json_events(stdout_text)
        if events:
            save_json(events_path, events)

        status = read_memory_output_status(output_path)
        raw_output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else None
        summary: dict[str, Any] = {
            "question_id": question_id,
            "attempt_index": attempt_index,
            "command": command,
            "started_at_utc": datetime.fromtimestamp(started_at_ts, timezone.utc).isoformat(),
            "completed_at_utc": utc_now_iso(),
            "duration_seconds": duration_seconds,
            "returncode": returncode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "status_after": status.state,
            "status_after_detail": status.detail,
            "usage": usage,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path) if events else None,
            "last_message_path": str(last_message_path),
            "output_path": str(output_path),
            "agent_output_raw_text": raw_output_text,
            "question_has_image": query_image is not None,
            "question_payload": question_payload,
        }

        if interrupted:
            summary["status_after"] = "interrupted"
            summary["status_after_detail"] = "query cancelled before Codex completed"
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": "interrupted",
                "detail": "query cancelled before Codex completed",
                "memory_context": [],
            }

        if not status.is_finished:
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": status.state,
                "detail": status.detail,
                "memory_context": [],
            }

        try:
            normalized_output = self._normalize_output_for_query(output_path)
            memory_context = self._build_memory_context_from_output(normalized_output)
        except Exception as exc:
            summary["status_after"] = "internal_postprocess_error"
            summary["status_after_detail"] = str(exc)
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": "internal_postprocess_error",
                "detail": str(exc),
                "memory_context": [],
            }

        summary.update(
            {
                "memory_markdown": normalized_output["memory_markdown"],
                "trajectory_spans_raw": normalized_output["trajectory_spans_raw"],
                "trajectory_spans_valid": normalized_output["trajectory_spans_valid"],
                "trajectory_spans_invalid": normalized_output["trajectory_spans_invalid"],
                "memory_context_item_count": len(memory_context),
            }
        )
        save_json(summary_path, summary)
        return {
            "success": True,
            "status": status.state,
            "detail": status.detail,
            "memory_context": memory_context,
        }
