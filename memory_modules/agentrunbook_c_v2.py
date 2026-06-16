from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agentrunbook_c import AgentRunbookC
from .codex import (
    read_memory_output_status,
    relative_symlink,
    save_json,
    utc_now_iso,
)
from .memory import MemoryConfig, register_memory, require
from .oai_agents_sdk import (
    OaiAgentsSDKRunner,
    OaiAgentsSDKRunnerConfig,
    ensure_non_negative_int,
    ensure_positive_float,
    ensure_positive_int,
    ensure_string,
)


@register_memory
class AgentRunbookCV2(AgentRunbookC):
    memory_type = "agentrunbook_c_v2"
    requires_codex_binary = False

    def __init__(self, memory_params: dict[str, object]) -> None:
        sdk_params_obj = memory_params.get(
            "query_openai_sdk_params",
            memory_params.get("query_codex_params", memory_params.get("codex_params", {})),
        )
        require(
            isinstance(sdk_params_obj, dict),
            "agentrunbook_c_v2 query_openai_sdk_params must be an object",
        )
        sdk_params = dict(sdk_params_obj)

        model = ensure_string(sdk_params.get("model"), field_name="query_openai_sdk_params.model")
        reasoning_effort = ensure_string(
            sdk_params.get("reasoning_effort"),
            field_name="query_openai_sdk_params.reasoning_effort",
        )
        timeout_seconds = ensure_positive_float(
            sdk_params.get("timeout_seconds"),
            field_name="query_openai_sdk_params.timeout_seconds",
        )
        max_attempts = ensure_positive_int(
            sdk_params.get("max_attempts", sdk_params.get("max_retries")),
            field_name="query_openai_sdk_params.max_attempts/max_retries",
        )

        base_params = dict(memory_params)
        base_params["query_codex_params"] = {
            "binary": sdk_params.get("binary", "codex"),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_attempts,
            "prompt": sdk_params.get("prompt", memory_params.get("prompt", "")) or None,
            "extra_config": [],
            "extra_args": [],
        }
        if base_params["query_codex_params"]["prompt"] is None:
            base_params["query_codex_params"].pop("prompt")

        super().__init__(base_params)

        self.sdk_model = model
        self.sdk_reasoning_effort = reasoning_effort
        self.sdk_timeout_seconds = timeout_seconds
        self.sdk_max_attempts = max_attempts
        self.codex_max_attempts = max_attempts
        self.sdk_max_turns = ensure_positive_int(
            sdk_params.get("max_turns"),
            field_name="query_openai_sdk_params.max_turns",
        )
        self.sdk_api_key_env = ensure_string(
            sdk_params.get("api_key_env"),
            field_name="query_openai_sdk_params.api_key_env",
        )
        self.tool_timeout_seconds = ensure_positive_float(
            sdk_params.get("tool_timeout_seconds"),
            field_name="query_openai_sdk_params.tool_timeout_seconds",
        )
        self.max_tool_output_chars = ensure_positive_int(
            sdk_params.get("max_tool_output_chars"),
            field_name="query_openai_sdk_params.max_tool_output_chars",
        )
        self.responses_transport = ensure_string(
            sdk_params.get("responses_transport", "websocket"),
            field_name="query_openai_sdk_params.responses_transport",
        )
        require(
            self.responses_transport == "websocket",
            "agentrunbook_c_v2 only supports responses_transport='websocket'",
        )
        self.api_connect_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_connect_timeout_seconds", 15.0),
            field_name="query_openai_sdk_params.api_connect_timeout_seconds",
        )
        self.api_read_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_read_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_read_timeout_seconds",
        )
        self.api_write_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_write_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_write_timeout_seconds",
        )
        self.api_pool_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_pool_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_pool_timeout_seconds",
        )
        self.api_max_retries = ensure_non_negative_int(
            sdk_params.get("api_max_retries", 0),
            field_name="query_openai_sdk_params.api_max_retries",
        )
        self.sdk_runner = OaiAgentsSDKRunner(
            OaiAgentsSDKRunnerConfig(
                model=self.sdk_model,
                reasoning_effort=self.sdk_reasoning_effort,
                timeout_seconds=self.sdk_timeout_seconds,
                max_turns=self.sdk_max_turns,
                api_key_env=self.sdk_api_key_env,
                tool_timeout_seconds=self.tool_timeout_seconds,
                max_tool_output_chars=self.max_tool_output_chars,
                agent_name="AgentRunbookCV2",
                responses_transport=self.responses_transport,
                api_connect_timeout_seconds=self.api_connect_timeout_seconds,
                api_read_timeout_seconds=self.api_read_timeout_seconds,
                api_write_timeout_seconds=self.api_write_timeout_seconds,
                api_pool_timeout_seconds=self.api_pool_timeout_seconds,
                api_max_retries=self.api_max_retries,
            )
        )

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, object] = {
            "questions_path": str(self.questions_path),
            "evidence_mode": self.evidence_mode,
            "query_openai_sdk_params": {
                "model": self.sdk_model,
                "reasoning_effort": self.sdk_reasoning_effort,
                "timeout_seconds": self.sdk_timeout_seconds,
                "max_retries": self.sdk_max_attempts,
                "max_turns": self.sdk_max_turns,
                "api_key_env": self.sdk_api_key_env,
                "tool_timeout_seconds": self.tool_timeout_seconds,
                "max_tool_output_chars": self.max_tool_output_chars,
                "responses_transport": self.responses_transport,
                "api_connect_timeout_seconds": self.api_connect_timeout_seconds,
                "api_read_timeout_seconds": self.api_read_timeout_seconds,
                "api_write_timeout_seconds": self.api_write_timeout_seconds,
                "api_pool_timeout_seconds": self.api_pool_timeout_seconds,
                "api_max_retries": self.api_max_retries,
                "prompt": self.codex_prompt,
            },
        }
        if self.trajectory_pool_root is not None:
            memory_params["trajectory_pool_root"] = str(self.trajectory_pool_root)
        return {
            "memory_type": self.memory_type,
            "memory_params": memory_params,
        }

    def _runner_summary_fields(self) -> dict[str, Any]:
        return {
            "runner": "oai_agents_sdk",
            "model": self.sdk_model,
            "reasoning_effort": self.sdk_reasoning_effort,
            "max_turns": self.sdk_max_turns,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "max_tool_output_chars": self.max_tool_output_chars,
            "responses_transport": self.responses_transport,
            "api_connect_timeout_seconds": self.api_connect_timeout_seconds,
            "api_read_timeout_seconds": self.api_read_timeout_seconds,
            "api_write_timeout_seconds": self.api_write_timeout_seconds,
            "api_pool_timeout_seconds": self.api_pool_timeout_seconds,
            "api_max_retries": self.api_max_retries,
        }

    def _estimate_sdk_token_throughput(
        self,
        *,
        duration_seconds: float,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not usage:
            return None

        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        reasoning_output_tokens = int(usage.get("reasoning_output_tokens", 0) or 0)
        tool_duration_seconds = sum(
            float(call.get("duration_seconds", 0.0) or 0.0) for call in tool_calls
        )
        estimated_model_wait_seconds = max(duration_seconds - tool_duration_seconds, 0.0)

        def rate(tokens: int, seconds: float) -> float | None:
            if tokens <= 0 or seconds <= 0.0:
                return None
            return tokens / seconds

        return {
            "model": self.sdk_model,
            "basis": "Agents SDK aggregate usage tokens divided by attempt elapsed seconds",
            "note": "Estimate only: the SDK exposes token usage but not per-response latency here.",
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "total_tokens": total_tokens,
            "elapsed_seconds": duration_seconds,
            "tool_duration_seconds": tool_duration_seconds,
            "estimated_model_wait_seconds": estimated_model_wait_seconds,
            "output_tokens_per_elapsed_second": rate(output_tokens, duration_seconds),
            "total_tokens_per_elapsed_second": rate(total_tokens, duration_seconds),
            "reasoning_output_tokens_per_elapsed_second": rate(
                reasoning_output_tokens,
                duration_seconds,
            ),
            "output_tokens_per_estimated_model_wait_second": rate(
                output_tokens,
                estimated_model_wait_seconds,
            ),
            "total_tokens_per_estimated_model_wait_second": rate(
                total_tokens,
                estimated_model_wait_seconds,
            ),
        }

    def _execute_prepared_attempt(
        self,
        *,
        sandbox_dir: Path,
        attempt_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        last_message_path: Path,
        events_path: Path,
    ) -> dict[str, Any]:
        started_at_ts = time.time()
        run_result = self.sdk_runner.run(
            sandbox_dir=sandbox_dir,
            user_prompt=self.codex_prompt,
        )
        last_message_path.write_text(run_result.final_output, encoding="utf-8")

        duration_seconds = time.time() - started_at_ts
        stdout_payload = {
            "runner": "oai_agents_sdk",
            "model": self.sdk_model,
            "reasoning_effort": self.sdk_reasoning_effort,
            "final_output": run_result.final_output,
            "tool_calls": run_result.tool_calls,
            "usage": run_result.usage,
        }
        stdout_path.write_text(
            json.dumps(stdout_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        save_json(events_path, stdout_payload)
        stderr_path.write_text(run_result.error_traceback, encoding="utf-8")

        if run_result.error_detail is not None:
            status_after = "timeout" if run_result.timed_out else "oai_agents_sdk_error"
            status_detail = run_result.error_detail
        else:
            status_after = None
            status_detail = None

        return {
            "started_at_ts": started_at_ts,
            "duration_seconds": duration_seconds,
            "fatal_error": run_result.error_detail is not None,
            "status_after": status_after,
            "status_after_detail": status_detail,
            "summary_fields": {
                "timed_out": run_result.timed_out,
                "runner_error_detail": run_result.error_detail,
                "tool_call_count": len(run_result.tool_calls),
                "usage": run_result.usage,
                "estimated_token_throughput": self._estimate_sdk_token_throughput(
                    duration_seconds=duration_seconds,
                    tool_calls=run_result.tool_calls,
                    usage=run_result.usage,
                ),
            },
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
            "agentrunbook_c_v2 workspace_dir is not configured",
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
                **self._runner_summary_fields(),
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
        execution_result = self._execute_prepared_attempt(
            sandbox_dir=sandbox_dir,
            attempt_dir=attempt_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path,
            events_path=events_path,
        )
        status = read_memory_output_status(output_path)
        raw_output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else None
        status_after = execution_result["status_after"] or status.state
        status_detail = execution_result["status_after_detail"]
        if status_detail is None:
            status_detail = status.detail
        summary: dict[str, Any] = {
            "question_id": question_id,
            "attempt_index": attempt_index,
            "started_at_utc": datetime.fromtimestamp(
                execution_result["started_at_ts"],
                timezone.utc,
            ).isoformat(),
            "completed_at_utc": utc_now_iso(),
            "duration_seconds": execution_result["duration_seconds"],
            **self._runner_summary_fields(),
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
            **execution_result["summary_fields"],
            "status_after": status_after,
            "status_after_detail": status_detail,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path),
            "last_message_path": str(last_message_path),
            "output_path": str(output_path),
            "agent_output_raw_text": raw_output_text,
        }

        if execution_result["fatal_error"] and not status.is_finished:
            save_json(summary_path, summary)
            return {
                "success": False,
                "status": status_after,
                "detail": status_detail,
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
