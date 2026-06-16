#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_data import (  # noqa: E402
    materialize_runtime_haystack,
    materialize_runtime_questions,
    write_json,
)


METHODS = {
    "no_retrieval",
    "rag_query_to_slice",
    "rag_query_to_slice_notes",
    "agentrunbook_r",
    "codex",
    "agentrunbook_c",
    "agentrunbook_c_v2",
}

OPENAI_SDK_TIMEOUT_SECONDS_BY_REASONING_EFFORT = {
    "low": 200.0,
    "medium": 800.0,
    "xhigh": 1200.0,
}


def parse_question_ids(raw_values: list[str] | None) -> list[str] | None:
    if not raw_values:
        return None
    out: list[str] = []
    for raw in raw_values:
        for item in raw.split(","):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongMemEval-V2 evaluation.")
    openai_sdk_timeout_seconds_env = os.getenv("OPENAI_SDK_TIMEOUT_SECONDS")
    parser.add_argument("--data-root", required=True, help="Path to the downloaded LongMemEval-V2 dataset")
    parser.add_argument("--domain", choices=["web", "enterprise"], required=True)
    parser.add_argument("--tier", choices=["small", "medium"], default="small")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected questions")
    parser.add_argument("--question-ids", nargs="*", default=None, help="Optional question ids, space or comma separated")

    parser.add_argument("--reader-model", default=os.getenv("READER_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--reader-base-url", default=os.getenv("READER_BASE_URL", "http://localhost:8023/v1"))
    parser.add_argument("--reader-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reader-temperature", type=float, default=float(os.getenv("READER_TEMPERATURE", "0.6")))
    parser.add_argument("--reader-top-p", type=float, default=float(os.getenv("READER_TOP_P", "0.95")))
    parser.add_argument("--reader-top-k", type=int, default=int(os.getenv("READER_TOP_K", "20")))
    parser.add_argument("--reader-max-concurrent-requests", type=int, default=16)
    parser.add_argument("--max-completion-tokens", type=int, default=20000)
    parser.add_argument("--memory-context-max-tokens", type=int, default=200000)
    parser.add_argument("--reader-enable-thinking", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--controller-model", default=os.getenv("LME_CONTROLLER_MODEL", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--controller-base-url", default=os.getenv("LME_CONTROLLER_BASE_URL", "http://localhost:8023/v1"))
    parser.add_argument("--controller-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--controller-temperature", type=float, default=float(os.getenv("LME_CONTROLLER_TEMPERATURE", "0.6")))
    parser.add_argument("--controller-top-p", type=float, default=float(os.getenv("LME_CONTROLLER_TOP_P", "0.95")))
    parser.add_argument("--controller-top-k", type=int, default=int(os.getenv("LME_CONTROLLER_TOP_K", "20")))
    parser.add_argument("--controller-disable-thinking", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--embedding-model", default=os.getenv("LME_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"))
    parser.add_argument("--embedding-base-url", default=os.getenv("LME_EMBEDDING_BASE_URL", "http://localhost:8114/v1"))
    parser.add_argument("--embedding-api-key-env", default="OPENAI_API_KEY")

    parser.add_argument("--codex-binary", default=os.getenv("CODEX_BINARY", "codex"))
    parser.add_argument("--codex-model", default=os.getenv("CODEX_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--codex-reasoning-effort", default=os.getenv("CODEX_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--codex-timeout-seconds", type=float, default=float(os.getenv("CODEX_TIMEOUT_SECONDS", "1800")))
    parser.add_argument("--codex-max-retries", type=int, default=int(os.getenv("CODEX_MAX_RETRIES", "3")))
    parser.add_argument("--openai-sdk-model", default=os.getenv("OPENAI_SDK_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--openai-sdk-reasoning-effort", default=os.getenv("OPENAI_SDK_REASONING_EFFORT", "medium"))
    parser.add_argument(
        "--openai-sdk-timeout-seconds",
        type=float,
        default=float(openai_sdk_timeout_seconds_env) if openai_sdk_timeout_seconds_env else None,
        help="Override the OpenAI Agents SDK query timeout. Defaults by reasoning effort: low=200, medium=800, xhigh=1200.",
    )
    parser.add_argument("--openai-sdk-max-retries", type=int, default=int(os.getenv("OPENAI_SDK_MAX_RETRIES", os.getenv("CODEX_MAX_RETRIES", "3"))))
    parser.add_argument("--openai-sdk-api-key-env", default=os.getenv("OPENAI_SDK_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--openai-sdk-max-turns", type=int, default=int(os.getenv("OPENAI_SDK_MAX_TURNS", "30")))
    parser.add_argument("--openai-sdk-tool-timeout-seconds", type=float, default=float(os.getenv("OPENAI_SDK_TOOL_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--openai-sdk-max-tool-output-chars", type=int, default=int(os.getenv("OPENAI_SDK_MAX_TOOL_OUTPUT_CHARS", "1048576")))

    parser.add_argument("--evaluator-model", default=os.getenv("EVALUATOR_MODEL", "gpt-5.2"))
    parser.add_argument("--evaluator-api-key-env", default=os.getenv("EVALUATOR_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--evaluator-reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--evaluator-max-completion-tokens", type=int, default=4096)

    parser.add_argument("--prompt-build-max-workers", type=int, default=1)
    parser.add_argument("--shuffle-questions-seed", type=int, default=None)
    return parser.parse_args()


def controller_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.controller_model,
        "base_url": args.controller_base_url,
        "api_key_env": args.controller_api_key_env,
        "api_key_file": None,
        "max_completion_tokens": 8192,
        "timeout_seconds": 600.0,
        "max_retries": 3,
        "disable_thinking": args.controller_disable_thinking,
        "temperature": args.controller_temperature,
        "top_p": args.controller_top_p,
        "top_k": args.controller_top_k,
    }


def embedding_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.embedding_model,
        "base_url": args.embedding_base_url,
        "api_key_env": args.embedding_api_key_env,
        "api_key_file": None,
        "max_input_tokens": 4096,
        "query_instruction": "Given a question about past agent trajectories, retrieve relevant memory entries that help answer it.",
    }


def openai_sdk_query_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.openai_sdk_model,
        "reasoning_effort": args.openai_sdk_reasoning_effort,
        "timeout_seconds": openai_sdk_timeout_seconds(args),
        "max_retries": args.openai_sdk_max_retries,
        "max_turns": args.openai_sdk_max_turns,
        "api_key_env": args.openai_sdk_api_key_env,
        "tool_timeout_seconds": args.openai_sdk_tool_timeout_seconds,
        "max_tool_output_chars": args.openai_sdk_max_tool_output_chars,
    }


def openai_sdk_timeout_seconds(args: argparse.Namespace) -> float:
    if args.openai_sdk_timeout_seconds is not None:
        return float(args.openai_sdk_timeout_seconds)
    reasoning_effort = str(args.openai_sdk_reasoning_effort).lower()
    return OPENAI_SDK_TIMEOUT_SECONDS_BY_REASONING_EFFORT.get(reasoning_effort, 1800.0)


def build_memory_config(args: argparse.Namespace, data_root: Path) -> dict[str, object]:
    if args.method == "no_retrieval":
        return {"memory_type": "no_retrieval", "memory_params": {}}
    if args.method in {"rag_query_to_slice", "rag_query_to_slice_notes"}:
        return {
            "memory_type": "rag",
            "memory_params": {
                "trajectory_pool_root": None,
                "controller_params": controller_params(args),
                "embedding_params": embedding_params(args),
                "index_params": {"raw_state_slice_radius": 1},
                "retrieval_params": {
                    "enable_notes": args.method == "rag_query_to_slice_notes",
                    "raw_state_search_top_k": 6,
                    "note_search_top_k_per_type": 3,
                },
            },
        }
    if args.method == "agentrunbook_r":
        return {
            "memory_type": "agentrunbook_r",
            "memory_params": {
                "trajectory_pool_root": None,
                "controller_params": controller_params(args),
                "embedding_params": embedding_params(args),
                "index_params": {"raw_state_slice_radius": 1},
                "query_params": {
                    "max_raw_state_queries": 5,
                    "query_generation_disable_thinking": args.controller_disable_thinking,
                },
                "retrieval_params": {
                    "raw_state_search_top_k_per_query": 6,
                    "event_search_top_k": 6,
                    "note_search_top_k_per_type": 3,
                    "raw_state_result_merge_budget": 6,
                    "raw_state_result_merge_per_query_cap": 2,
                    "rerank_candidate_limit": 8,
                    "enable_rerank": False,
                },
            },
        }
    codex_params = {
        "binary": args.codex_binary,
        "model": args.codex_model,
        "reasoning_effort": args.codex_reasoning_effort,
        "timeout_seconds": args.codex_timeout_seconds,
        "max_retries": args.codex_max_retries,
        "extra_config": [],
        "extra_args": [],
    }
    if args.method == "codex":
        return {
            "memory_type": "codex",
            "memory_params": {
                "questions_path": str((data_root / "questions.jsonl").resolve()),
                "evidence_mode": "both",
                "trajectory_pool_root": None,
                "codex_params": codex_params,
            },
        }
    if args.method == "agentrunbook_c_v2":
        return {
            "memory_type": "agentrunbook_c_v2",
            "memory_params": {
                "questions_path": str((data_root / "questions.jsonl").resolve()),
                "evidence_mode": "both",
                "trajectory_pool_root": None,
                "query_openai_sdk_params": openai_sdk_query_params(args),
            },
        }
    return {
        "memory_type": "agentrunbook_c",
        "memory_params": {
            "questions_path": str((data_root / "questions.jsonl").resolve()),
            "evidence_mode": "both",
            "trajectory_pool_root": None,
            "query_codex_params": codex_params,
        },
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    selected_questions = materialize_runtime_questions(
        data_root=data_root,
        domain=args.domain,
        question_ids=parse_question_ids(args.question_ids),
        limit=args.limit,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=data_root,
        tier=args.tier,
        selected_questions=selected_questions,
        output_path=runtime_dir / "haystack.json",
    )
    memory_config = build_memory_config(args, data_root)
    memory_config_path = runtime_dir / "memory_config.json"
    write_json(memory_config_path, memory_config)

    harness_argv = [
        "evaluation.harness",
        "--domain",
        args.domain,
        "--questions-path",
        str(runtime_dir / "questions.json"),
        "--haystack-path",
        str(runtime_dir / "haystack.json"),
        "--trajectories-path",
        str(data_root / "trajectories.jsonl"),
        "--memory-config-path",
        str(memory_config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        args.reader_model,
        "--base-url",
        args.reader_base_url,
        "--api-key-env",
        args.reader_api_key_env,
        "--temperature",
        str(args.reader_temperature),
        "--top-p",
        str(args.reader_top_p),
        "--top-k",
        str(args.reader_top_k),
        "--max-completion-tokens",
        str(args.max_completion_tokens),
        "--memory-context-max-tokens",
        str(args.memory_context_max_tokens),
        "--reader-max-concurrent-requests",
        str(args.reader_max_concurrent_requests),
        "--prompt-build-max-workers",
        str(args.prompt_build_max_workers),
        "--evaluator-model",
        args.evaluator_model,
        "--evaluator-api-key-env",
        args.evaluator_api_key_env,
        "--evaluator-reasoning-effort",
        args.evaluator_reasoning_effort,
        "--evaluator-max-completion-tokens",
        str(args.evaluator_max_completion_tokens),
    ]
    if not args.reader_enable_thinking:
        harness_argv.append("--reader-disable-thinking")
    if args.shuffle_questions_seed is not None:
        harness_argv.extend(["--shuffle-questions-seed", str(args.shuffle_questions_seed)])
    print(json.dumps({"runtime_dir": str(runtime_dir), "method": args.method}, indent=2))
    old_argv = sys.argv
    try:
        sys.argv = harness_argv
        from evaluation.harness import main as harness_main

        harness_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
