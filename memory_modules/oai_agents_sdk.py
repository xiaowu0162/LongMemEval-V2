from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI


TIMEOUT_CLEANUP_GRACE_SECONDS = 5.0
TRUNCATION_HINT = (
    "Rerun with a narrower command: sed range, scoped rg pattern, "
    "inspect_trajectory.py --state/--span/--match."
)
OAI_AGENTS_SYSTEM_INSTRUCTIONS = """
You are a file system agent. You and the user share the same workspace and collaborate to achieve the user's goals.

# Personality

You are a deeply pragmatic, effective software engineer. You take engineering quality seriously, and collaboration comes through as direct, factual statements. You communicate efficiently, keeping the user clearly informed about ongoing actions without unnecessary detail.

## Values

- Clarity: Communicate reasoning explicitly and concretely.
- Pragmatism: Keep the end goal and momentum in mind.
- Rigor: Surface gaps or weak assumptions with emphasis on clarity.

# General

As an expert file system agent, your primary focus is executing commands and helping the user complete their task in the current environment. Build context by examining files first.

- Start with targeted discovery: inspect compact indexes, summaries, or manifests first, then open only the files and spans needed to verify evidence.
- Prefer `rg`, scoped `sed` ranges, and focused helper-script invocations over broad dumps.
- Always use apply_patch for manual code edits.
- Do not load or run local or Hugging Face vision-language/image encoder models.
- Persist until the task is fully handled end-to-end whenever feasible.
"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class OaiAgentsSDKOuterTimeoutError(TimeoutError):
    pass


@dataclass(frozen=True)
class OaiAgentsSDKRunnerConfig:
    model: str
    reasoning_effort: str
    timeout_seconds: float
    max_turns: int
    api_key_env: str
    tool_timeout_seconds: float
    max_tool_output_chars: int
    agent_name: str = "OaiAgentsSDKRunner"
    responses_transport: str = "websocket"
    api_connect_timeout_seconds: float = 15.0
    api_read_timeout_seconds: float = 300.0
    api_write_timeout_seconds: float = 300.0
    api_pool_timeout_seconds: float = 300.0
    api_max_retries: int = 0


@dataclass
class OaiAgentsSDKRunResult:
    final_output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error_detail: str | None = None
    error_traceback: str = ""
    timed_out: bool = False


def load_agents_sdk() -> dict[str, Any]:
    try:
        from agents import (
            Agent,
            ApplyPatchTool,
            ModelSettings,
            OpenAIProvider,
            RunConfig,
            Runner,
            ShellCallOutcome,
            ShellCommandOutput,
            ShellResult,
            ShellTool,
            ToolExecutionConfig,
            apply_diff,
        )
        from agents.editor import ApplyPatchResult
        from openai.types.shared import Reasoning
    except ImportError as exc:
        raise RuntimeError(
            "oai_agents_sdk requires the openai-agents package. "
            "Install dependencies from requirements.txt or pyproject.toml."
        ) from exc
    return {
        "Agent": Agent,
        "ApplyPatchResult": ApplyPatchResult,
        "ApplyPatchTool": ApplyPatchTool,
        "ModelSettings": ModelSettings,
        "OpenAIProvider": OpenAIProvider,
        "Reasoning": Reasoning,
        "RunConfig": RunConfig,
        "Runner": Runner,
        "ShellCallOutcome": ShellCallOutcome,
        "ShellCommandOutput": ShellCommandOutput,
        "ShellResult": ShellResult,
        "ShellTool": ShellTool,
        "ToolExecutionConfig": ToolExecutionConfig,
        "apply_diff": apply_diff,
    }


def ensure_string(value: object, *, field_name: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{field_name} must be a non-empty string")
    return value.strip()


def ensure_positive_float(value: object, *, field_name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0,
        f"{field_name} must be a positive number",
    )
    return float(value)


def ensure_positive_int(value: object, *, field_name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{field_name} must be a positive integer",
    )
    return int(value)


def ensure_non_negative_int(value: object, *, field_name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{field_name} must be a non-negative integer",
    )
    return int(value)


def build_agent_input(*, user_prompt: str) -> str:
    return ensure_string(user_prompt, field_name="user_prompt")


def make_sandbox_tools(
    *,
    sandbox_dir: Path,
    tool_calls: list[dict[str, Any]],
    tool_timeout_seconds: float,
    max_tool_output_chars: int,
) -> list[Any]:
    sdk = load_agents_sdk()
    return [
        sdk["ShellTool"](
            executor=ShellExecutor(
                sdk=sdk,
                sandbox_dir=sandbox_dir,
                tool_calls=tool_calls,
                tool_timeout_seconds=tool_timeout_seconds,
                max_tool_output_chars=max_tool_output_chars,
            ),
            environment={"type": "local"},
            needs_approval=False,
        ),
        sdk["ApplyPatchTool"](
            editor=WorkspaceEditor(sdk=sdk, sandbox_dir=sandbox_dir, tool_calls=tool_calls),
            needs_approval=False,
        ),
    ]


class ShellExecutor:
    """Executes shell commands inside the prepared benchmark sandbox."""

    def __init__(
        self,
        *,
        sdk: dict[str, Any],
        sandbox_dir: Path,
        tool_calls: list[dict[str, Any]],
        tool_timeout_seconds: float,
        max_tool_output_chars: int,
    ) -> None:
        self.sdk = sdk
        self.cwd = sandbox_dir.resolve()
        self.env = os.environ.copy()
        shim_dir = self.cwd / ".oai_agents_sdk_bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        python_bin = Path(sys.executable).resolve()
        for name in ("python", "python3"):
            shim = shim_dir / name
            if not shim.exists():
                shim.symlink_to(python_bin)
        self.env["PATH"] = (
            str(shim_dir.resolve())
            + os.pathsep
            + str(python_bin.parent)
            + os.pathsep
            + self.env.get("PATH", "")
        )
        self.tool_calls = tool_calls
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_tool_output_chars = max_tool_output_chars

    async def __call__(self, request: Any) -> Any:
        action = request.data.action
        timeout = self._timeout(action.timeout_ms)
        outputs: list[Any] = []

        for command in action.commands:
            started = time.time()
            proc = await asyncio.create_subprocess_shell(
                command,
                executable="/bin/bash",
                cwd=self.cwd,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                stdout_bytes, stderr_bytes = await proc.communicate()

            original_stdout = _decode_output(stdout_bytes)
            original_stderr = _decode_output(stderr_bytes)
            duration_seconds = time.time() - started

            if timed_out:
                tool_response = {
                    "returncode": None,
                    "stdout": original_stdout,
                    "stderr": original_stderr,
                    "timeout_seconds": timeout,
                }
                self._record(
                    command=command,
                    returncode=None,
                    duration_seconds=duration_seconds,
                    timed_out=True,
                    stdout_text=original_stdout,
                    stderr_text=original_stderr,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    tool_response=tool_response,
                )
                outputs.append(
                    self.sdk["ShellCommandOutput"](
                        command=command,
                        stdout=original_stdout,
                        stderr=original_stderr or f"shell command timed out after {timeout}s",
                        outcome=self.sdk["ShellCallOutcome"](type="timeout", exit_code=None),
                        provider_data=tool_response,
                    )
                )
                break

            stdout_truncated = len(original_stdout) > self.max_tool_output_chars
            stderr_truncated = len(original_stderr) > self.max_tool_output_chars
            stdout = _trim(original_stdout, self.max_tool_output_chars)
            stderr = _trim(original_stderr, self.max_tool_output_chars)

            if stdout_truncated or stderr_truncated:
                stream_details: list[str] = []
                if stdout_truncated:
                    stream_details.append(f"stdout was {len(original_stdout)} chars")
                if stderr_truncated:
                    stream_details.append(f"stderr was {len(original_stderr)} chars")
                message = (
                    f"Output was too large ({'; '.join(stream_details)}), "
                    f"cap is {self.max_tool_output_chars}. {TRUNCATION_HINT}"
                )
                tool_response = {
                    "returncode": 2,
                    "error": "OUTPUT_TRUNCATED",
                    "message": message,
                    "stdout": stdout,
                    "stderr": stderr,
                    "original_returncode": proc.returncode,
                    "stdout_chars": len(original_stdout),
                    "stderr_chars": len(original_stderr),
                    "max_output_chars": self.max_tool_output_chars,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                }
                metadata = {
                    key: value
                    for key, value in tool_response.items()
                    if key not in {"stdout", "stderr"}
                }
                stderr = "\n".join(
                    part for part in (stderr, message, json.dumps(metadata, ensure_ascii=True)) if part
                )
                exit_code = 2
            else:
                tool_response = {
                    "returncode": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_chars": len(original_stdout),
                    "stderr_chars": len(original_stderr),
                    "max_output_chars": self.max_tool_output_chars,
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
                exit_code = proc.returncode

            self._record(
                command=command,
                returncode=exit_code,
                duration_seconds=duration_seconds,
                timed_out=False,
                stdout_text=original_stdout,
                stderr_text=original_stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                tool_response=tool_response,
            )
            outputs.append(
                self.sdk["ShellCommandOutput"](
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    outcome=self.sdk["ShellCallOutcome"](type="exit", exit_code=exit_code),
                    provider_data=tool_response,
                )
            )

        return self.sdk["ShellResult"](
            output=outputs,
            max_output_length=self.max_tool_output_chars,
            provider_data={"working_directory": str(self.cwd)},
        )

    def _timeout(self, timeout_ms: int | None) -> float:
        if timeout_ms is None:
            return self.tool_timeout_seconds
        return min(self.tool_timeout_seconds, max(timeout_ms / 1000.0, 0.001))

    def _record(
        self,
        *,
        command: str,
        returncode: int | None,
        duration_seconds: float,
        timed_out: bool,
        stdout_text: str,
        stderr_text: str,
        stdout_truncated: bool,
        stderr_truncated: bool,
        tool_response: dict[str, Any],
    ) -> None:
        self.tool_calls.append(
            {
                "tool": "shell",
                "command": command,
                "returncode": returncode,
                "duration_seconds": duration_seconds,
                "timed_out": timed_out,
                "stdout_chars": len(stdout_text),
                "stderr_chars": len(stderr_text),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "tool_response": tool_response,
            }
        )


class WorkspaceEditor:
    """Applies apply_patch operations inside the prepared benchmark sandbox."""

    def __init__(
        self,
        *,
        sdk: dict[str, Any],
        sandbox_dir: Path,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        self.sdk = sdk
        self._root = sandbox_dir.resolve()
        self.tool_calls = tool_calls

    def create_file(self, operation: Any) -> Any:
        started = time.time()
        try:
            if operation.diff is None:
                raise RuntimeError(f"missing diff for create_file: {operation.path}")
            target = self._resolve(operation.path, ensure_parent=True)
            content = self.sdk["apply_diff"]("", operation.diff, mode="create")
            target.write_text(content, encoding="utf-8")
            return self._result(operation, started, "completed", f"Created {self._relative(target)}")
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def update_file(self, operation: Any) -> Any:
        started = time.time()
        try:
            if operation.diff is None:
                raise RuntimeError(f"missing diff for update_file: {operation.path}")
            target = self._resolve(operation.path)
            updated = self.sdk["apply_diff"](target.read_text(encoding="utf-8"), operation.diff)
            destination = self._resolve(operation.move_to) if operation.move_to else target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(updated, encoding="utf-8")
            if destination != target:
                target.unlink()
                output = (
                    f"Updated {self._relative(target)}\n"
                    f"Moved {self._relative(target)} to {self._relative(destination)}"
                )
            else:
                output = f"Updated {self._relative(target)}"
            return self._result(operation, started, "completed", output)
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def delete_file(self, operation: Any) -> Any:
        started = time.time()
        try:
            target = self._resolve(operation.path)
            target.unlink(missing_ok=True)
            return self._result(operation, started, "completed", f"Deleted {self._relative(target)}")
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def _resolve(self, value: str, ensure_parent: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("apply_patch path must be non-empty")
        candidate = Path(value)
        target = candidate if candidate.is_absolute() else self._root / candidate
        target = target.resolve()
        if os.path.commonpath([str(self._root), str(target)]) != str(self._root):
            raise RuntimeError(f"apply_patch path escapes sandbox: {value}")
        if ensure_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()

    def _result(
        self,
        operation: Any,
        started: float,
        status: str,
        output: str,
    ) -> Any:
        self._record(operation, started, status, output)
        return self.sdk["ApplyPatchResult"](status=status, output=output)

    def _record(
        self,
        operation: Any,
        started: float,
        status: str,
        output: str,
    ) -> None:
        self.tool_calls.append(
            {
                "tool": "apply_patch",
                "operation": operation.type,
                "path": operation.path,
                "move_to": operation.move_to,
                "duration_seconds": time.time() - started,
                "tool_response": {"status": status, "output": output},
            }
        )


def _usage_to_dict(usage: object) -> dict[str, int]:
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_output_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _empty_usage_totals() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }


def summarize_run_usage(result: object) -> dict[str, Any]:
    totals = _empty_usage_totals()
    raw_responses = getattr(result, "raw_responses", []) or []
    raw_response_usage: list[dict[str, Any]] = []
    for response in raw_responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        usage_dict = _usage_to_dict(usage)
        for key in totals:
            totals[key] += usage_dict[key]
        raw_response_usage.append(
            {
                "response_id": getattr(response, "response_id", None),
                "request_id": getattr(response, "request_id", None),
                **usage_dict,
            }
        )

    return {
        **totals,
        "raw_response_count": len(raw_responses),
        "raw_responses_with_usage": len(raw_response_usage),
        "raw_response_usage": raw_response_usage,
    }


class OaiAgentsSDKRunner:
    def __init__(self, config: OaiAgentsSDKRunnerConfig) -> None:
        self.config = config
        require(
            config.responses_transport == "websocket",
            "OaiAgentsSDKRunner only supports responses_transport='websocket'",
        )
        require(
            os.getenv(config.api_key_env),
            f"Missing OpenAI API key via env {config.api_key_env}",
        )
        load_agents_sdk()

    def run(
        self,
        *,
        sandbox_dir: Path,
        user_prompt: str,
    ) -> OaiAgentsSDKRunResult:
        tool_calls: list[dict[str, Any]] = []
        try:
            result = self._run_agent(
                sandbox_dir=sandbox_dir,
                user_prompt=user_prompt,
                tool_calls=tool_calls,
            )
            return OaiAgentsSDKRunResult(
                final_output=str(getattr(result, "final_output", "")),
                tool_calls=tool_calls,
                usage=summarize_run_usage(result),
            )
        except OaiAgentsSDKOuterTimeoutError as exc:
            return OaiAgentsSDKRunResult(
                tool_calls=tool_calls,
                error_detail=f"OpenAI Agents SDK run timed out after {self.config.timeout_seconds}s",
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                timed_out=True,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            detail = str(exc) or exc.__class__.__name__
            return OaiAgentsSDKRunResult(
                tool_calls=tool_calls,
                error_detail=f"OpenAI Agents SDK run timed out: {detail}",
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                timed_out=True,
            )
        except Exception as exc:
            return OaiAgentsSDKRunResult(
                tool_calls=tool_calls,
                error_detail=str(exc),
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )

    def _run_agent(
        self,
        *,
        sandbox_dir: Path,
        user_prompt: str,
        tool_calls: list[dict[str, Any]],
    ) -> Any:
        sdk = load_agents_sdk()
        Agent = sdk["Agent"]
        ModelSettings = sdk["ModelSettings"]
        OpenAIProvider = sdk["OpenAIProvider"]
        Reasoning = sdk["Reasoning"]
        RunConfig = sdk["RunConfig"]
        Runner = sdk["Runner"]
        ToolExecutionConfig = sdk["ToolExecutionConfig"]

        agent = Agent(
            name=self.config.agent_name,
            instructions=OAI_AGENTS_SYSTEM_INSTRUCTIONS,
            model=self.config.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.config.reasoning_effort),
                parallel_tool_calls=False,
            ),
            tools=make_sandbox_tools(
                sandbox_dir=sandbox_dir,
                tool_calls=tool_calls,
                tool_timeout_seconds=self.config.tool_timeout_seconds,
                max_tool_output_chars=self.config.max_tool_output_chars,
            ),
        )
        agent_input = build_agent_input(user_prompt=user_prompt)

        async def run_with_timeout() -> Any:
            api_key = os.getenv(self.config.api_key_env)
            client = AsyncOpenAI(
                api_key=api_key,
                default_headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(
                    connect=self.config.api_connect_timeout_seconds,
                    read=self.config.api_read_timeout_seconds,
                    write=self.config.api_write_timeout_seconds,
                    pool=self.config.api_pool_timeout_seconds,
                ),
                max_retries=self.config.api_max_retries,
            )
            provider = OpenAIProvider(
                openai_client=client,
                use_responses_websocket=True,
            )
            try:
                run_config = RunConfig(
                    model_provider=provider,
                    tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                )
                return await Runner.run(
                    agent,
                    agent_input,
                    max_turns=self.config.max_turns,
                    run_config=run_config,
                )
            finally:
                try:
                    await provider.aclose()
                finally:
                    await client.close()

        async def run_with_outer_timeout() -> Any:
            task = asyncio.create_task(run_with_timeout())
            done, _pending = await asyncio.wait({task}, timeout=self.config.timeout_seconds)
            if task in done:
                return await task
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=TIMEOUT_CLEANUP_GRACE_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            raise OaiAgentsSDKOuterTimeoutError(
                f"OpenAI Agents SDK run timed out after {self.config.timeout_seconds}s"
            )

        return asyncio.run(run_with_outer_timeout())


def _decode_output(value: bytes | bytearray | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated to {max_chars} chars]"
