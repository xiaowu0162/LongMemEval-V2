import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory import MemoryConfig, register_memory, require
from .codex import (
    MAX_TOTAL_SPAN_STATES,
    PROCESS_POLL_INTERVAL_SECONDS,
    CodexMemory,
    parse_codex_json_events,
    read_memory_output_status,
    relative_symlink,
    save_json,
    terminate_process_group,
    utc_now_iso,
)


DEFAULT_ASSET_ROOT = (
    Path(__file__).resolve().parent
    / "assets"
    / "agentrunbook_c"
)
DEFAULT_INSTRUCTION_TEMPLATE = DEFAULT_ASSET_ROOT / "INSTRUCTION.md"
DEFAULT_TRAJECTORY_SUMMARY_RENDERER = DEFAULT_ASSET_ROOT / "scripts" / "render_trajectory_summary.py"
DEFAULT_TRAJECTORY_INSPECTOR = DEFAULT_ASSET_ROOT / "scripts" / "inspect_trajectory.py"
TRAJECTORY_SUMMARY_CONCISE_FILENAME = "TRAJECTORY_SUMMARY_CONCISE.md"
TRAJECTORY_SUMMARY_FULL_FILENAME = "TRAJECTORY_SUMMARY_FULL.md"


DEFAULT_QUERY_PROMPT = (
    "You are acting as the query-time agent for AgentRunbook-C. "
    "Read the local files in this directory, especially INSTRUCTION.md and question.json. "
    "The local trajectories/ directory contains the current haystack for this evaluation item, "
    "and you must explore trajectories/ before returning your final result. "
    "If question.json refers to an image, view it carefully. "
    "Write your final result to memory_module_output.json as valid JSON. "
    "Use the local inspection helper under scripts/ when you need to inspect one trajectory, "
    "one state, one span, or match text within one trajectory quickly."
)


@register_memory
class AgentRunbookC(CodexMemory):
    memory_type = "agentrunbook_c"

    def __init__(self, memory_params: dict[str, object]) -> None:
        query_codex_params_obj = memory_params.get(
            "query_codex_params",
            memory_params.get("codex_params", {}),
        )
        require(
            isinstance(query_codex_params_obj, dict),
            "agentrunbook_c query_codex_params/codex_params must be an object",
        )
        query_codex_params = dict(query_codex_params_obj)
        query_codex_params.setdefault("prompt", DEFAULT_QUERY_PROMPT)

        base_params = dict(memory_params)
        base_params["codex_params"] = query_codex_params
        super().__init__(base_params)
        self._trajectory_summary_lock = threading.Lock()
        self.query_instruction_path = DEFAULT_INSTRUCTION_TEMPLATE.resolve()
        self.trajectory_summary_renderer_path = DEFAULT_TRAJECTORY_SUMMARY_RENDERER.resolve()
        self.trajectory_inspector_path = DEFAULT_TRAJECTORY_INSPECTOR.resolve()
        require(
            self.query_instruction_path.exists(),
            f"agentrunbook_c missing instruction template: {self.query_instruction_path}",
        )
        require(
            self.trajectory_summary_renderer_path.exists(),
            f"agentrunbook_c missing renderer script: {self.trajectory_summary_renderer_path}",
        )
        require(
            self.trajectory_inspector_path.exists(),
            f"agentrunbook_c missing inspector script: {self.trajectory_inspector_path}",
        )
        self.query_instruction_text = self.query_instruction_path.read_text(encoding="utf-8")

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, object] = {
            "questions_path": str(self.questions_path),
            "evidence_mode": self.evidence_mode,
            "query_codex_params": {
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

    def _populate_sandbox_scripts(self, sandbox_dir: Path) -> list[str]:
        scripts_dir = sandbox_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        target = scripts_dir / "inspect_trajectory.py"
        target.write_text(self.trajectory_inspector_path.read_text(encoding="utf-8"), encoding="utf-8")
        return [target.name]

    def _build_question_payload(
        self,
        *,
        question_id: str,
        question_item: dict[str, Any],
        query_text: str,
        query_image: str | None,
        sandbox_dir: Path,
    ) -> dict[str, Any]:
        payload = super()._build_question_payload(
            question_id=question_id,
            question_item=question_item,
            query_text=query_text,
            query_image=query_image,
            sandbox_dir=sandbox_dir,
        )
        payload.pop("question_id", None)
        payload.pop("question_type", None)
        return payload

    def _ensure_trajectory_summary(
        self,
        *,
        attempt_dir: Path,
    ) -> dict[str, Any]:
        require(
            self.workspace_dir is not None,
            "agentrunbook_c workspace_dir is not configured",
        )
        concise_output_path = self.workspace_dir / "trajectories" / TRAJECTORY_SUMMARY_CONCISE_FILENAME
        full_output_path = self.workspace_dir / "trajectories" / TRAJECTORY_SUMMARY_FULL_FILENAME
        if concise_output_path.exists() and full_output_path.exists():
            return {
                "success": True,
                "summary_rendered": False,
                "trajectory_summary_concise_path": str(concise_output_path),
                "trajectory_summary_full_path": str(full_output_path),
                "summary_renderer_path": str(self.trajectory_summary_renderer_path),
                "summary_stdout_path": None,
                "summary_stderr_path": None,
            }

        with self._trajectory_summary_lock:
            if concise_output_path.exists() and full_output_path.exists():
                return {
                    "success": True,
                    "summary_rendered": False,
                    "trajectory_summary_concise_path": str(concise_output_path),
                    "trajectory_summary_full_path": str(full_output_path),
                    "summary_renderer_path": str(self.trajectory_summary_renderer_path),
                    "summary_stdout_path": None,
                    "summary_stderr_path": None,
                }

            stdout_path = attempt_dir / "trajectory_summary_render_stdout.log"
            stderr_path = attempt_dir / "trajectory_summary_render_stderr.log"
            command = [
                sys.executable,
                str(self.trajectory_summary_renderer_path),
                str((self.workspace_dir / "trajectories").resolve()),
                "--concise-output",
                str(concise_output_path.resolve()),
                "--full-output",
                str(full_output_path.resolve()),
            ]
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_path.write_text(process.stdout, encoding="utf-8")
            stderr_path.write_text(process.stderr, encoding="utf-8")
            if process.returncode != 0:
                return {
                    "success": False,
                    "status": "trajectory_summary_render_failed",
                    "detail": f"Trajectory summary render failed. See {stderr_path}",
                    "summary_rendered": False,
                    "trajectory_summary_concise_path": str(concise_output_path),
                    "trajectory_summary_full_path": str(full_output_path),
                    "summary_renderer_path": str(self.trajectory_summary_renderer_path),
                    "summary_stdout_path": str(stdout_path),
                    "summary_stderr_path": str(stderr_path),
                }
            if not concise_output_path.exists() or not full_output_path.exists():
                return {
                    "success": False,
                    "status": "trajectory_summary_missing",
                    "detail": (
                        "Trajectory summary files were not created: "
                        f"{concise_output_path}, {full_output_path}"
                    ),
                    "summary_rendered": False,
                    "trajectory_summary_concise_path": str(concise_output_path),
                    "trajectory_summary_full_path": str(full_output_path),
                    "summary_renderer_path": str(self.trajectory_summary_renderer_path),
                    "summary_stdout_path": str(stdout_path),
                    "summary_stderr_path": str(stderr_path),
                }
            return {
                "success": True,
                "summary_rendered": True,
                "trajectory_summary_concise_path": str(concise_output_path),
                "trajectory_summary_full_path": str(full_output_path),
                "summary_renderer_path": str(self.trajectory_summary_renderer_path),
                "summary_stdout_path": str(stdout_path),
                "summary_stderr_path": str(stderr_path),
            }

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
            "agentrunbook_c workspace_dir is not configured",
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
        (sandbox_dir / "INSTRUCTION.md").write_text(self.query_instruction_text, encoding="utf-8")
        summary_result = self._ensure_trajectory_summary(attempt_dir=attempt_dir)
        if not summary_result["success"]:
            summary: dict[str, Any] = {
                "question_id": question_id,
                "attempt_index": attempt_index,
                "completed_at_utc": utc_now_iso(),
                "question_has_image": query_image is not None,
                "question_payload": question_payload,
                "sandbox_dir": str(sandbox_dir),
                "scripts_dir": str(sandbox_dir / "scripts"),
                "scripts_dir_mode": "single_helper",
                "sandbox_helper_scripts": [],
                "trajectory_summary_concise_path": summary_result["trajectory_summary_concise_path"],
                "trajectory_summary_full_path": summary_result["trajectory_summary_full_path"],
                "summary_renderer_path": summary_result["summary_renderer_path"],
                "summary_stdout_path": summary_result["summary_stdout_path"],
                "summary_stderr_path": summary_result["summary_stderr_path"],
                "summary_rendered": summary_result["summary_rendered"],
                "status_after": summary_result["status"],
                "status_after_detail": summary_result["detail"],
            }
            save_json(attempt_dir / "summary.json", summary)
            return {
                "success": False,
                "status": summary_result["status"],
                "detail": summary_result["detail"],
                "memory_context": [],
            }
        relative_symlink(self.workspace_dir / "trajectories", sandbox_dir / "trajectories")
        sandbox_helper_scripts = self._populate_sandbox_scripts(sandbox_dir)

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
            "started_at_utc": datetime.fromtimestamp(started_at_ts, timezone.utc).isoformat(),
            "completed_at_utc": utc_now_iso(),
            "duration_seconds": duration_seconds,
            "question_has_image": query_image is not None,
            "question_payload": question_payload,
            "sandbox_dir": str(sandbox_dir),
            "scripts_dir": str(sandbox_dir / "scripts"),
            "scripts_dir_mode": "single_helper",
            "sandbox_helper_scripts": sandbox_helper_scripts,
            "trajectory_summary_concise_path": summary_result["trajectory_summary_concise_path"],
            "trajectory_summary_full_path": summary_result["trajectory_summary_full_path"],
            "summary_renderer_path": summary_result["summary_renderer_path"],
            "summary_stdout_path": summary_result["summary_stdout_path"],
            "summary_stderr_path": summary_result["summary_stderr_path"],
            "summary_rendered": summary_result["summary_rendered"],
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "status_after": status.state,
            "status_after_detail": status.detail,
            "usage": usage,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path) if events else None,
            "last_message_path": str(last_message_path),
            "output_path": str(output_path),
            "agent_output_raw_text": raw_output_text,
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
