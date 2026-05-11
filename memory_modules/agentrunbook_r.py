from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer

from .memory import Memory, MemoryConfig, MemoryContextItem, register_memory, require
from .support import (
    NOTE_GENERATION_PROMPT_VERSION,
    NOTE_GENERATION_REPAIR_SYSTEM_PROMPT,
    NOTE_GENERATION_SYSTEM_PROMPT,
    _annotate_action_with_object_details,
    _extract_object_lookup_from_tree,
    _extract_response_text_parts,
    _load_api_key,
    _parse_generated_notes_json,
    _to_data_url,
)
from .trajectory_store import (
    load_json,
    materialize_prepared_trajectory,
    normalize_trajectory_pool_root,
    prepare_trajectory_insert,
    relative_symlink,
    save_json,
    validate_pooled_trajectory_dir,
)


DEFAULT_CONTROLLER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_CONTROLLER_BASE_URL = "http://localhost:8023/v1"
DEFAULT_CONTROLLER_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_CONTROLLER_MAX_COMPLETION_TOKENS = 8192
DEFAULT_CONTROLLER_TIMEOUT_SECONDS = 600.0
DEFAULT_CONTROLLER_MAX_RETRIES = 3
DEFAULT_CONTROLLER_DISABLE_THINKING = False
DEFAULT_CONTROLLER_TEMPERATURE = 0.6
DEFAULT_CONTROLLER_TOP_P = 0.95
DEFAULT_CONTROLLER_TOP_K = 20

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:8114/v1"
DEFAULT_EMBEDDING_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_EMBEDDING_MAX_INPUT_TOKENS = 4096
DEFAULT_EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about past agent trajectories, retrieve relevant memory entries that help answer it."
)

DEFAULT_RAW_STATE_SLICE_RADIUS = 1
DEFAULT_RAW_STATE_TOP_K = 6
DEFAULT_EVENT_TOP_K = 6
DEFAULT_NOTE_TOP_K = 3
DEFAULT_MAX_RAW_STATE_QUERIES = 5
DEFAULT_ENABLE_RERANK = False
DEFAULT_RERANK_CANDIDATE_LIMIT = 8
DEFAULT_RAW_STATE_RESULT_MERGE_BUDGET = 6
DEFAULT_RAW_STATE_RESULT_MERGE_PER_QUERY_CAP = 2
DEFAULT_RERANK_FALLBACK_RAW_TOP_K = 2
DEFAULT_RERANK_FALLBACK_OTHER_TOP_K = 3

EVENT_PROMPT_VERSION = "qwen_v2_per_transition_workflow_state_change"
EVENT_REPAIR_PROMPT_VERSION = "qwen_v2_per_transition_workflow_state_change_repair"
EVENT_STATE_FALLBACK_MAX_AXTREE_CHARS = 12000
EVENT_ASYNC_MAX_CONCURRENCY = 16
PREVIEW_EXAMPLE_COUNT = 2
PREVIEW_TEXT_MAX_CHARS = 1200
EMBEDDING_PRETOKEN_CHAR_CAP = 40000
RUNTIME_SUMMARY_GOAL_COUNT = 6
RUNTIME_SUMMARY_SURFACE_COUNT = 6
RUNTIME_SUMMARY_NOTE_TITLE_COUNT = 4
# This pooled-artifact namespace is intentionally distinct from the generic
# file-backed trajectory layout used by the coding-agent methods.
TRAJECTORY_ARTIFACT_SCHEMA_VERSION = "v1"
TRAJECTORY_ARTIFACT_DIRNAME = "agentrunbook_r_artifact"
TRAJECTORY_ARTIFACT_INDEX_FILENAME = "artifact_index.json"
TRAJECTORY_ARTIFACT_RAW_STATE_FILENAME = "raw_state_pool.jsonl"
TRAJECTORY_ARTIFACT_EVENT_FILENAME = "event_pool.jsonl"
TRAJECTORY_ARTIFACT_PROCEDURE_NOTE_FILENAME = "procedure_note.json"
TRAJECTORY_ARTIFACT_HINT_NOTE_FILENAME = "hint_note.json"
TRAJECTORY_ARTIFACT_RAW_STATE_EMBEDDINGS_FILENAME = "raw_state.npy"
TRAJECTORY_ARTIFACT_EVENT_EMBEDDINGS_FILENAME = "event.npy"
TRAJECTORY_ARTIFACT_PROCEDURE_EMBEDDING_FILENAME = "procedure_note.npy"
TRAJECTORY_ARTIFACT_HINT_EMBEDDING_FILENAME = "hint_note.npy"

ROOT_TITLE_RE = re.compile(r"RootWebArea '([^']+)'")
WHITESPACE_RE = re.compile(r"\s+")

NOTE_SECTION_TITLE = "## Procedure and Hint Notes Learned from Previous Tasks in the Environment\n\n"
EVENT_SECTION_TITLE = "## Retrieved State-Transition Events from Previous Tasks in the Environment\n\n"
RAW_STATE_SECTION_TITLE = "## Retrieved Raw State Slices from Previous Tasks in the Environment\n\n"

QUERY_SCHEMA_EXAMPLE = {
    "raw_state_queries": [
        "incident form related search feature",
        "change request form related search feature",
        "related search area bottom of page incident change request",
    ],
    "event_query": "",
    "note_query": "ServiceNow create incident versus create change request form related search feature",
}

QUERY_REPAIR_SYSTEM_PROMPT = """Rewrite the draft into valid JSON only.

Return exactly one JSON object with this schema:
{"raw_state_queries":["..."],"event_query":"...","note_query":"..."}

Rules:
- raw_state_queries must be a JSON array of strings
- event_query and note_query must be strings
- keep at most 5 raw_state_queries
- do not output analysis, markdown fences, or commentary
- first character must be { and last character must be }
"""

QUERY_GENERATION_SYSTEM_PROMPT = """You generate structured retrieval queries for an active memory system with three pools: raw state slices, state-transition events, and procedure/hint notes.

Return exactly one JSON object:
{"raw_state_queries":["..."],"event_query":"...","note_query":"..."}

Goal:
- maximize retrieval of the memory entries that would help answer the question later
- do not answer the question yourself

How to target each pool:
- raw_state_queries:
  - use for exact UI surface evidence
  - target pages/forms/records/tabs/sections/fields/buttons/dropdowns/options/labels/default values/counts/visible or missing controls
  - each raw_state query should correspond to a distinct surface target such as one page, one form, one record view, one tab state, or one compared entity
  - if multiple aspects live on the same surface, merge them into one raw query instead of splitting them into separate attribute-only queries
  - do not emit separate raw queries only to divide the same form/page into "mandatory", "readonly", "visible", "editable", or similar sub-aspects
  - split raw queries only when the question compares different surfaces or requires evidence from clearly different pages/states
- event_query:
  - use only if navigation, before/after change, revealed content, confirmation, blocker, popup, or workflow stage matters
  - phrase it as the transition or state change you want event entries to describe
- note_query:
  - use for reusable procedure, module path, disambiguation, absent functionality, pitfalls, and durable hints
  - keep it task-family level, broader than raw_state

Rules:
- Remove formatting instructions and final-answer wrappers.
- For multiple-choice questions, ignore the answer letters/options and target the underlying UI evidence only.
- Preserve exact entity names and literal UI labels from the question.
- Keep queries short, concrete, and retrieval-oriented.
- Prefer queries that a memory index could match semantically; avoid conversational filler.
- If the question is about exact visible evidence, raw_state_queries should do most of the work.
- If the question is about how to reach or observe something, note_query and event_query should carry more weight.
- It is acceptable for event_query to be empty when the question is purely about static page state.
- It is acceptable for raw_state_queries to be empty when the question is purely procedural, but this should be rare.
- Deduplicate raw_state_queries and cap them at 5.
- Return JSON only, no markdown, no commentary.
"""

QUERY_GENERATION_EXAMPLES = """Example 1
Question: Create Incident vs Problem. In both forms, are all fields with suggestion buttons mandatory?
Output:
{"raw_state_queries":["incident form suggestion button fields mandatory","problem form suggestion button fields mandatory","suggestion button fields incident versus problem mandatory markers"],"event_query":"","note_query":"ServiceNow create incident versus problem form field requirements and suggestion button hints"}

Example 2
Question: After applying the "4 stars & up" filter to Sephora brush search results, what changes on the page?
Output:
{"raw_state_queries":["Sephora brush search results 4 stars & up filter applied active filter chip and filtered product cards"],"event_query":"apply the 4 stars & up filter on Sephora brush search results and observe what products, counts, and active filter indicators appear afterward","note_query":""}

Example 3
Question: How do I remove Sean-Michelle Morris-Martinez's hardware assignment during user offboarding?
Output:
{"raw_state_queries":["hardware asset list and record Assigned to field for Sean-Michelle Morris-Martinez"],"event_query":"","note_query":"ServiceNow offboard user workflow for removing a user's hardware assignment from hardware assets"}

Example 4
Question: On the Data Management Delete Job form, which fields are shown but not modifiable?
Output:
{"raw_state_queries":["Data Management Delete Job form shown but not modifiable fields"],"event_query":"","note_query":"ServiceNow Data Management Delete Job form field editability and visibility"}"""

RERANK_SELECTION_SYSTEM_PROMPT = """You are selecting which retrieved memory items should be kept for a downstream QA model.

You will see:
- the original benchmark question
- one generated retrieval query
- one memory pool label
- a numbered candidate list

Return exactly one JSON object:
{"selected_indices":[0],"selection_description":"..."}

Instructions:
- Keep only the candidates that materially help answer the question.
- selected_indices must use the candidate indices exactly as shown.
- Stop selecting once the requested knowledge is already covered. In many cases 1-2 items are enough, but there is no hard cap.
- It is valid to return an empty list if none of the candidates are actually useful.
- selection_description should briefly explain why the kept items matter.
- In the case where the best evidence contradicts the question's premise, say so clearly in selection_description and tell the downstream model to call out the flawed premise. Include the evidence in the selected_indices as well.
- Do not answer the benchmark question directly.
- Return JSON only.
"""

RERANK_REPAIR_SYSTEM_PROMPT = """Rewrite the draft into valid JSON only.

Return exactly one JSON object with this schema:
{"selected_indices":[0],"selection_description":"..."}

Rules:
- selected_indices must be a JSON array of unique integers
- selection_description must be a string
- selected_indices may be empty
- do not output markdown fences, commentary, or extra keys
- first character must be { and last character must be }
"""

EVENT_GENERATION_SYSTEM_PROMPT = """You convert one UI transition from a longer task trajectory into retrieval-ready event text.

You will be given:
- the full task goal and outcome
- the full annotated action trace for the trajectory
- one target transition defined as pre-state -> annotated action -> post-state
- the full AXTree text, thoughts, URLs, and screenshots for only the target pre-state and post-state

In this dataset, actions are attached to destination states:
- transition event_0000 means state 0 -> action stored on state 1 -> state 1
- transition event_0001 means state 1 -> action stored on state 2 -> state 2

Return exactly one JSON object with this shape:
{"overview":"...","state_transition":"..."}

Field requirements:
- "overview": one concise paragraph that briefly recaps the concrete task goal and places this transition in the broader workflow. Do not just say "while pursuing the goal". Mention what the agent is trying to accomplish and what stage this step represents.
- "state_transition": one concise paragraph that explicitly compares the post-state to the pre-state. Describe what happened after the action: a new page, new module, revealed panel, form fields, changed values, confirmation signal, blocker, popup, navigation, or lack of visible change.

Rules:
- Ground both fields only in the provided goal, outcome, action trace, thoughts, AXTree text, and screenshots.
- Be retrieval-friendly for unknown future dynamic questions.
- Reuse the exact task entities and labels when they are present in the evidence. Do not rename users, products, themes, modules, or records.
- Preserve the most answer-bearing visible facts:
  - exact module, page, tab, menu, button, field, option, status, stage, entity, or label names
  - values, counts, dates, before/after states, selected options, confirmation signals, blocking signals, or newly revealed UI
  - distinctions between similar controls when the evidence supports them
- Use the annotated action text when naming what the agent clicked, typed, selected, or opened.
- In "state_transition", prioritize what changed because of the action: newly visible or replaced pages, menus, panels, dialogs, fields, values, warnings, or blockers.
- Avoid spending space on unchanged background widgets or unrelated page content unless they are needed to identify the current view.
- Do not invent unseen controls, labels, outcomes, or causal claims.
- Do not quote raw thoughts verbatim line-by-line; distill them.
- Do not output markdown fences, commentary, or extra keys.
- First character must be { and last character must be }.
"""

EVENT_GENERATION_REPAIR_SYSTEM_PROMPT = """Rewrite the draft into valid JSON only.

Rules:
- Return exactly one JSON object.
- The object must contain:
  - "overview": a non-empty string
  - "state_transition": a non-empty string
- Do not output commentary, markdown fences, or extra keys.
- First character must be { and last character must be }.
"""


def _json_candidates(raw_text: str) -> list[str]:
    stripped = raw_text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    if stripped.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", stripped, flags=re.S)
        if fence_match:
            candidates.append(fence_match.group(1).strip())
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1].strip())
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _extract_content_text_only(content: Any) -> str | None:
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
            else:
                text_val = getattr(item, "text", None)
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    return None


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 64:
        return text[:max_chars]
    omitted_chars = len(text) - max_chars
    marker = f"\n...[truncated {omitted_chars} chars]...\n"
    if len(marker) >= max_chars:
        return text[:max_chars]
    head_chars = (max_chars - len(marker)) // 2
    tail_chars = max_chars - len(marker) - head_chars
    return f"{text[:head_chars].rstrip()}{marker}{text[-tail_chars:].lstrip()}".strip()


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def _spaced_sample(items: list[Any], count: int) -> list[Any]:
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[0]]
    selected: list[Any] = []
    for idx in range(count):
        position = round(idx * (len(items) - 1) / (count - 1))
        selected.append(items[position])
    deduped: list[Any] = []
    seen: set[str] = set()
    for item in selected:
        key = json.dumps(item, ensure_ascii=True, sort_keys=True) if isinstance(item, dict) else str(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _first_clause(text: str, max_chars: int = 110) -> str:
    text = _normalize_text(text)
    if not text:
        return ""
    for splitter in ["\n", ". ", "; ", " - ", ": "]:
        if splitter in text:
            text = text.split(splitter, 1)[0]
            break
    return _truncate_middle(text, max_chars)


def _parse_root_title(slice_axtree_text: str) -> str | None:
    match = ROOT_TITLE_RE.search(slice_axtree_text)
    if not match:
        return None
    return _normalize_text(match.group(1))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        require(isinstance(payload, dict), f"JSONL entry must be an object: {path}")
        entries.append(payload)
    return entries


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(entry, ensure_ascii=True) for entry in entries)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _normalize_embeddings(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _append_embedding_rows(existing: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
    if new_rows.size == 0:
        return existing
    if existing.size == 0:
        return new_rows
    require(
        existing.shape[1] == new_rows.shape[1],
        f"Embedding dimension mismatch: {existing.shape} vs {new_rows.shape}",
    )
    return np.vstack([existing, new_rows])


def _truncate_for_preview(value: str) -> str:
    return _truncate_middle(value, PREVIEW_TEXT_MAX_CHARS)


def _state_text(state: dict[str, Any]) -> str:
    text_value = state.get("text")
    return text_value if isinstance(text_value, str) else ""


def _state_url(state: dict[str, Any]) -> str:
    url_value = state.get("url")
    return url_value if isinstance(url_value, str) else "<unknown>"


def _state_thoughts(state: dict[str, Any]) -> str:
    thoughts_value = state.get("thoughts")
    if isinstance(thoughts_value, str) and thoughts_value.strip():
        return thoughts_value.strip()
    return "<none>"


def _destination_action_text(state: dict[str, Any]) -> str:
    action_value = state.get("action")
    if isinstance(action_value, str) and action_value.strip():
        return action_value.strip()
    return "<none>"


def _annotated_transition_actions(states: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    known_object_lookup: dict[str, str] = {}
    for index in range(len(states) - 1):
        pre_state = states[index]
        pre_lookup = _extract_object_lookup_from_tree(_state_text(pre_state))
        object_lookup = {**known_object_lookup, **pre_lookup}
        action_text = _destination_action_text(states[index + 1])
        if action_text != "<none>":
            actions.append(_annotate_action_with_object_details(action_text, object_lookup))
        else:
            actions.append("<none>")
        known_object_lookup.update(pre_lookup)
    return actions


def _full_action_sequence_text(states: list[dict[str, Any]]) -> str:
    lines = _annotated_transition_actions(states)
    if not lines:
        return "1. <no actions recorded>"
    return "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))


def _slice_transition_action_sequence_text(slice_states: list[dict[str, Any]]) -> str:
    lines = _annotated_transition_actions(slice_states)
    if not lines:
        return "1. <no local actions recorded>"
    return "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))


def _slice_axtree_text(slice_states: list[dict[str, Any]]) -> str:
    parts = []
    for state in slice_states:
        parts.append(
            "\n".join(
                [
                    f"State {state['state_index']}",
                    _state_text(state),
                ]
            )
        )
    return "\n\n".join(parts).strip()


def _event_id_for_transition(pre_state_index: int, post_state_index: int) -> str:
    return f"event_{pre_state_index:04d}_{post_state_index:04d}"


def _parse_single_event_response(response_text: str) -> dict[str, str]:
    for candidate in _json_candidates(response_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        overview = payload.get("overview")
        state_transition = payload.get("state_transition")
        if (
            isinstance(overview, str)
            and overview.strip()
            and isinstance(state_transition, str)
            and state_transition.strip()
        ):
            return {
                "overview": overview.strip(),
                "state_transition": state_transition.strip(),
            }
    return {}


def _parse_query_bundle(response_text: str, *, max_raw_state_queries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for candidate in _json_candidates(response_text):
        try:
            payload = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = TypeError("Parsed payload is not a JSON object")
            continue
        raw_state_queries = payload.get("raw_state_queries")
        event_query = payload.get("event_query", "")
        note_query = payload.get("note_query", "")
        if not isinstance(raw_state_queries, list):
            last_error = TypeError("raw_state_queries is not a list")
            continue
        if not isinstance(event_query, str) or not isinstance(note_query, str):
            last_error = TypeError("event_query/note_query must be strings")
            continue
        normalized_raw_queries: list[str] = []
        seen: set[str] = set()
        for item in raw_state_queries:
            if not isinstance(item, str):
                continue
            stripped = _normalize_text(item)
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            normalized_raw_queries.append(stripped)
        return {
            "raw_state_queries": normalized_raw_queries[:max_raw_state_queries],
            "event_query": _normalize_text(event_query),
            "note_query": _normalize_text(note_query),
        }
    raise ValueError(f"Could not parse query bundle: {last_error!r}")


def _parse_rerank_output(response_text: str, *, candidate_count: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for candidate in _json_candidates(response_text):
        try:
            payload = json.loads(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = TypeError("Parsed payload is not a JSON object")
            continue
        selected_indices = payload.get("selected_indices")
        selection_description = payload.get("selection_description", "")
        if not isinstance(selected_indices, list):
            last_error = TypeError("selected_indices is not a list")
            continue
        if not isinstance(selection_description, str):
            last_error = TypeError("selection_description is not a string")
            continue
        parsed_indices: list[int] = []
        seen: set[int] = set()
        valid = True
        for item in selected_indices:
            if not isinstance(item, int):
                valid = False
                last_error = TypeError("selected_indices must contain integers")
                break
            if item < 0 or item >= candidate_count:
                valid = False
                last_error = ValueError(
                    f"selected index {item} out of bounds for candidate_count={candidate_count}"
                )
                break
            if item in seen:
                continue
            seen.add(item)
            parsed_indices.append(item)
        if not valid:
            continue
        return {
            "selected_indices": parsed_indices,
            "selection_description": selection_description.strip(),
        }
    raise ValueError(f"Could not parse rerank output: {last_error!r}")


def _domain_from_url(url: str) -> str:
    if "service-now.com" in url:
        return "enterprise"
    if "localhost:" in url:
        return "web"
    return "unknown"


@register_memory
class AgentRunbookR(Memory):
    memory_type = "agentrunbook_r"

    @classmethod
    def reconcile_loaded_memory_config(
        cls,
        saved_config: MemoryConfig,
        requested_config: MemoryConfig | None,
    ) -> MemoryConfig:
        saved_normalized = cls._normalize_reconcilable_memory_config(saved_config)
        if requested_config is None:
            return deepcopy(saved_normalized)

        require(
            requested_config["memory_type"] == cls.memory_type,
            (
                "Requested memory config type does not match saved agentrunbook_r "
                f"artifact type: {requested_config['memory_type']} vs {cls.memory_type}"
            ),
        )
        requested_normalized = cls._normalize_reconcilable_memory_config(requested_config)

        saved_nonquery_signature = cls._loaded_config_nonquery_signature(saved_normalized)
        requested_nonquery_signature = cls._loaded_config_nonquery_signature(requested_normalized)
        require(
            saved_nonquery_signature == requested_nonquery_signature,
            (
                "agentrunbook_r prebuilt-memory loading only allows query-side "
                "config changes. Non-query fields in the requested config must exactly "
                "match the saved artifact config."
            ),
        )

        effective = deepcopy(saved_normalized)
        effective["memory_params"]["controller_params"] = deepcopy(
            requested_normalized["memory_params"]["controller_params"]
        )
        effective["memory_params"]["embedding_params"] = deepcopy(
            requested_normalized["memory_params"]["embedding_params"]
        )
        effective["memory_params"]["query_params"] = deepcopy(
            requested_normalized["memory_params"]["query_params"]
        )
        effective["memory_params"]["retrieval_params"] = deepcopy(
            requested_normalized["memory_params"]["retrieval_params"]
        )
        return effective

    @classmethod
    def _normalize_reconcilable_memory_config(cls, memory_config: MemoryConfig) -> MemoryConfig:
        require(
            memory_config["memory_type"] == cls.memory_type,
            f"Expected memory_type={cls.memory_type}, got {memory_config['memory_type']}",
        )
        memory_params_obj = memory_config.get("memory_params")
        require(
            isinstance(memory_params_obj, dict),
            "agentrunbook_r memory_params must be an object",
        )
        memory_params = dict(memory_params_obj)
        allowed_top_level_keys = {
            "trajectory_pool_root",
            "workspace_dir",
            "trajectories_root_dir",
            "controller_params",
            "embedding_params",
            "index_params",
            "query_params",
            "retrieval_params",
        }
        unexpected_top_level_keys = sorted(set(memory_params) - allowed_top_level_keys)
        require(
            not unexpected_top_level_keys,
            (
                "agentrunbook_r config contains unexpected memory_params keys: "
                f"{unexpected_top_level_keys}"
            ),
        )

        controller_params_obj = memory_params.get("controller_params", {})
        embedding_params_obj = memory_params.get("embedding_params", {})
        index_params_obj = memory_params.get("index_params", {})
        query_params_obj = memory_params.get("query_params", {})
        retrieval_params_obj = memory_params.get("retrieval_params", {})

        require(
            isinstance(controller_params_obj, dict),
            "agentrunbook_r controller_params must be an object",
        )
        require(
            isinstance(embedding_params_obj, dict),
            "agentrunbook_r embedding_params must be an object",
        )
        require(
            isinstance(index_params_obj, dict),
            "agentrunbook_r index_params must be an object",
        )
        require(
            isinstance(query_params_obj, dict),
            "agentrunbook_r query_params must be an object",
        )
        require(
            isinstance(retrieval_params_obj, dict),
            "agentrunbook_r retrieval_params must be an object",
        )

        controller_params = dict(controller_params_obj)
        embedding_params = dict(embedding_params_obj)
        index_params = dict(index_params_obj)
        query_params = dict(query_params_obj)
        retrieval_params = dict(retrieval_params_obj)

        expected_controller_keys = {
            "model",
            "base_url",
            "api_key_env",
            "api_key_file",
            "max_completion_tokens",
            "timeout_seconds",
            "max_retries",
            "disable_thinking",
            "temperature",
            "top_p",
            "top_k",
        }
        expected_embedding_keys = {
            "model",
            "base_url",
            "api_key_env",
            "api_key_file",
            "max_input_tokens",
            "query_instruction",
        }
        expected_index_keys = {"raw_state_slice_radius"}
        expected_query_keys = {"max_raw_state_queries", "query_generation_disable_thinking"}
        expected_retrieval_keys = {
            "raw_state_search_top_k_per_query",
            "event_search_top_k",
            "note_search_top_k_per_type",
            "raw_state_result_merge_budget",
            "raw_state_result_merge_per_query_cap",
            "rerank_candidate_limit",
            "enable_rerank",
        }

        require(
            not (set(controller_params) - expected_controller_keys),
            (
                "agentrunbook_r controller_params contains unexpected keys: "
                f"{sorted(set(controller_params) - expected_controller_keys)}"
            ),
        )
        require(
            not (set(embedding_params) - expected_embedding_keys),
            (
                "agentrunbook_r embedding_params contains unexpected keys: "
                f"{sorted(set(embedding_params) - expected_embedding_keys)}"
            ),
        )
        require(
            not (set(index_params) - expected_index_keys),
            (
                "agentrunbook_r index_params contains unexpected keys: "
                f"{sorted(set(index_params) - expected_index_keys)}"
            ),
        )
        require(
            not (set(query_params) - expected_query_keys),
            (
                "agentrunbook_r query_params contains unexpected keys: "
                f"{sorted(set(query_params) - expected_query_keys)}"
            ),
        )
        require(
            not (set(retrieval_params) - expected_retrieval_keys),
            (
                "agentrunbook_r retrieval_params contains unexpected keys: "
                f"{sorted(set(retrieval_params) - expected_retrieval_keys)}"
            ),
        )

        normalized: MemoryConfig = {
            "memory_type": cls.memory_type,
            "memory_params": {
                "controller_params": {
                    "model": str(controller_params.get("model", DEFAULT_CONTROLLER_MODEL)).strip(),
                    "base_url": str(controller_params.get("base_url", DEFAULT_CONTROLLER_BASE_URL)).strip(),
                    "api_key_env": str(
                        controller_params.get("api_key_env", DEFAULT_CONTROLLER_API_KEY_ENV)
                    ).strip(),
                    "api_key_file": (
                        str(controller_params.get("api_key_file")).strip()
                        if isinstance(controller_params.get("api_key_file"), str)
                        and str(controller_params.get("api_key_file")).strip()
                        else None
                    ),
                    "max_completion_tokens": int(
                        controller_params.get(
                            "max_completion_tokens",
                            DEFAULT_CONTROLLER_MAX_COMPLETION_TOKENS,
                        )
                    ),
                    "timeout_seconds": float(
                        controller_params.get("timeout_seconds", DEFAULT_CONTROLLER_TIMEOUT_SECONDS)
                    ),
                    "max_retries": int(
                        controller_params.get("max_retries", DEFAULT_CONTROLLER_MAX_RETRIES)
                    ),
                    "disable_thinking": bool(
                        controller_params.get("disable_thinking", DEFAULT_CONTROLLER_DISABLE_THINKING)
                    ),
                    "temperature": float(
                        controller_params.get("temperature", DEFAULT_CONTROLLER_TEMPERATURE)
                    ),
                    "top_p": float(controller_params.get("top_p", DEFAULT_CONTROLLER_TOP_P)),
                    "top_k": int(controller_params.get("top_k", DEFAULT_CONTROLLER_TOP_K)),
                },
                "embedding_params": {
                    "model": str(embedding_params.get("model", DEFAULT_EMBEDDING_MODEL)).strip(),
                    "base_url": str(embedding_params.get("base_url", DEFAULT_EMBEDDING_BASE_URL)).strip(),
                    "api_key_env": str(
                        embedding_params.get("api_key_env", DEFAULT_EMBEDDING_API_KEY_ENV)
                    ).strip(),
                    "api_key_file": (
                        str(embedding_params.get("api_key_file")).strip()
                        if isinstance(embedding_params.get("api_key_file"), str)
                        and str(embedding_params.get("api_key_file")).strip()
                        else None
                    ),
                    "max_input_tokens": int(
                        embedding_params.get("max_input_tokens", DEFAULT_EMBEDDING_MAX_INPUT_TOKENS)
                    ),
                    "query_instruction": str(
                        embedding_params.get("query_instruction", DEFAULT_EMBEDDING_QUERY_INSTRUCTION)
                    ).strip(),
                },
                "index_params": {
                    "raw_state_slice_radius": int(
                        index_params.get("raw_state_slice_radius", DEFAULT_RAW_STATE_SLICE_RADIUS)
                    ),
                },
                "query_params": {
                    "max_raw_state_queries": int(
                        query_params.get("max_raw_state_queries", DEFAULT_MAX_RAW_STATE_QUERIES)
                    ),
                    "query_generation_disable_thinking": bool(
                        query_params.get(
                            "query_generation_disable_thinking",
                            controller_params.get("disable_thinking", DEFAULT_CONTROLLER_DISABLE_THINKING),
                        )
                    ),
                },
                "retrieval_params": {
                    "raw_state_search_top_k_per_query": int(
                        retrieval_params.get(
                            "raw_state_search_top_k_per_query",
                            DEFAULT_RAW_STATE_TOP_K,
                        )
                    ),
                    "event_search_top_k": int(
                        retrieval_params.get("event_search_top_k", DEFAULT_EVENT_TOP_K)
                    ),
                    "note_search_top_k_per_type": int(
                        retrieval_params.get("note_search_top_k_per_type", DEFAULT_NOTE_TOP_K)
                    ),
                    "raw_state_result_merge_budget": int(
                        retrieval_params.get(
                            "raw_state_result_merge_budget",
                            DEFAULT_RAW_STATE_RESULT_MERGE_BUDGET,
                        )
                    ),
                    "raw_state_result_merge_per_query_cap": int(
                        retrieval_params.get(
                            "raw_state_result_merge_per_query_cap",
                            DEFAULT_RAW_STATE_RESULT_MERGE_PER_QUERY_CAP,
                        )
                    ),
                    "rerank_candidate_limit": int(
                        retrieval_params.get(
                            "rerank_candidate_limit",
                            DEFAULT_RERANK_CANDIDATE_LIMIT,
                        )
                    ),
                    "enable_rerank": bool(
                        retrieval_params.get("enable_rerank", DEFAULT_ENABLE_RERANK)
                    ),
                },
            },
        }
        return normalized

    @classmethod
    def _loaded_config_nonquery_signature(cls, memory_config: MemoryConfig) -> dict[str, Any]:
        memory_params = memory_config["memory_params"]
        return {
            "memory_type": memory_config["memory_type"],
            "memory_params": {
                "index_params": deepcopy(memory_params["index_params"]),
            },
        }

    def __init__(self, memory_params: dict[str, object]) -> None:
        super().__init__(memory_params)

        workspace_dir = memory_params.get("workspace_dir")
        trajectories_root_dir = memory_params.get("trajectories_root_dir")
        trajectory_pool_root = memory_params.get("trajectory_pool_root")
        controller_params_obj = memory_params.get("controller_params", {})
        embedding_params_obj = memory_params.get("embedding_params", {})
        index_params_obj = memory_params.get("index_params", {})
        query_params_obj = memory_params.get("query_params", {})
        retrieval_params_obj = memory_params.get("retrieval_params", {})

        require(
            workspace_dir is None or (isinstance(workspace_dir, str) and workspace_dir.strip()),
            "agentrunbook_r workspace_dir must be null or a non-empty string",
        )
        require(
            trajectories_root_dir is None
            or (isinstance(trajectories_root_dir, str) and trajectories_root_dir.strip()),
            "agentrunbook_r trajectories_root_dir must be null or a non-empty string",
        )
        require(
            trajectory_pool_root is None
            or (isinstance(trajectory_pool_root, str) and trajectory_pool_root.strip()),
            "agentrunbook_r trajectory_pool_root must be null or a non-empty string",
        )
        require(
            isinstance(controller_params_obj, dict),
            "agentrunbook_r controller_params must be an object",
        )
        require(
            isinstance(embedding_params_obj, dict),
            "agentrunbook_r embedding_params must be an object",
        )
        require(
            isinstance(index_params_obj, dict),
            "agentrunbook_r index_params must be an object",
        )
        require(
            isinstance(query_params_obj, dict),
            "agentrunbook_r query_params must be an object",
        )
        require(
            isinstance(retrieval_params_obj, dict),
            "agentrunbook_r retrieval_params must be an object",
        )

        controller_params = dict(controller_params_obj)
        embedding_params = dict(embedding_params_obj)
        index_params = dict(index_params_obj)
        query_params = dict(query_params_obj)
        retrieval_params = dict(retrieval_params_obj)

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
            require(
                self.trajectory_pool_root.exists() and self.trajectory_pool_root.is_dir(),
                (
                    "agentrunbook_r trajectory_pool_root must point to an existing directory: "
                    f"{self.trajectory_pool_root}"
                ),
            )

        self.controller_model = str(controller_params.get("model", DEFAULT_CONTROLLER_MODEL)).strip()
        self.controller_base_url = str(
            controller_params.get("base_url", DEFAULT_CONTROLLER_BASE_URL)
        ).strip()
        self.controller_api_key_env = str(
            controller_params.get("api_key_env", DEFAULT_CONTROLLER_API_KEY_ENV)
        ).strip()
        controller_api_key_file = controller_params.get("api_key_file")
        self.controller_api_key_file = (
            str(controller_api_key_file).strip()
            if isinstance(controller_api_key_file, str) and controller_api_key_file.strip()
            else None
        )
        self.controller_max_completion_tokens = int(
            controller_params.get("max_completion_tokens", DEFAULT_CONTROLLER_MAX_COMPLETION_TOKENS)
        )
        self.controller_timeout_seconds = float(
            controller_params.get("timeout_seconds", DEFAULT_CONTROLLER_TIMEOUT_SECONDS)
        )
        self.controller_max_retries = int(
            controller_params.get("max_retries", DEFAULT_CONTROLLER_MAX_RETRIES)
        )
        self.controller_disable_thinking = bool(
            controller_params.get("disable_thinking", DEFAULT_CONTROLLER_DISABLE_THINKING)
        )
        self.controller_temperature = float(
            controller_params.get("temperature", DEFAULT_CONTROLLER_TEMPERATURE)
        )
        self.controller_top_p = float(controller_params.get("top_p", DEFAULT_CONTROLLER_TOP_P))
        self.controller_top_k = int(controller_params.get("top_k", DEFAULT_CONTROLLER_TOP_K))

        self.embedding_model = str(embedding_params.get("model", DEFAULT_EMBEDDING_MODEL)).strip()
        self.embedding_base_url = str(
            embedding_params.get("base_url", DEFAULT_EMBEDDING_BASE_URL)
        ).strip()
        self.embedding_api_key_env = str(
            embedding_params.get("api_key_env", DEFAULT_EMBEDDING_API_KEY_ENV)
        ).strip()
        embedding_api_key_file = embedding_params.get("api_key_file")
        self.embedding_api_key_file = (
            str(embedding_api_key_file).strip()
            if isinstance(embedding_api_key_file, str) and embedding_api_key_file.strip()
            else None
        )
        self.embedding_max_input_tokens = int(
            embedding_params.get("max_input_tokens", DEFAULT_EMBEDDING_MAX_INPUT_TOKENS)
        )
        self.embedding_query_instruction = str(
            embedding_params.get("query_instruction", DEFAULT_EMBEDDING_QUERY_INSTRUCTION)
        ).strip()

        self.raw_state_slice_radius = int(
            index_params.get("raw_state_slice_radius", DEFAULT_RAW_STATE_SLICE_RADIUS)
        )

        self.raw_state_search_top_k_per_query = int(
            retrieval_params.get("raw_state_search_top_k_per_query", DEFAULT_RAW_STATE_TOP_K)
        )
        self.event_search_top_k = int(
            retrieval_params.get("event_search_top_k", DEFAULT_EVENT_TOP_K)
        )
        self.note_search_top_k_per_type = int(
            retrieval_params.get("note_search_top_k_per_type", DEFAULT_NOTE_TOP_K)
        )
        self.max_raw_state_queries = int(
            query_params.get("max_raw_state_queries", DEFAULT_MAX_RAW_STATE_QUERIES)
        )
        self.query_generation_disable_thinking = bool(
            query_params.get(
                "query_generation_disable_thinking",
                self.controller_disable_thinking,
            )
        )
        self.raw_state_result_merge_budget = int(
            retrieval_params.get("raw_state_result_merge_budget", DEFAULT_RAW_STATE_RESULT_MERGE_BUDGET)
        )
        self.raw_state_result_merge_per_query_cap = int(
            retrieval_params.get(
                "raw_state_result_merge_per_query_cap",
                DEFAULT_RAW_STATE_RESULT_MERGE_PER_QUERY_CAP,
            )
        )
        self.rerank_candidate_limit = int(
            retrieval_params.get(
                "rerank_candidate_limit",
                DEFAULT_RERANK_CANDIDATE_LIMIT,
            )
        )
        self.enable_rerank = bool(retrieval_params.get("enable_rerank", DEFAULT_ENABLE_RERANK))

        require(self.controller_model, "agentrunbook_r controller model must be non-empty")
        require(self.controller_base_url, "agentrunbook_r controller base_url must be non-empty")
        require(
            self.controller_api_key_env,
            "agentrunbook_r controller api_key_env must be non-empty",
        )
        require(
            self.controller_max_completion_tokens > 0,
            "agentrunbook_r controller max_completion_tokens must be positive",
        )
        require(
            self.controller_timeout_seconds > 0,
            "agentrunbook_r controller timeout_seconds must be positive",
        )
        require(
            self.controller_max_retries >= 0,
            "agentrunbook_r controller max_retries must be non-negative",
        )
        require(
            self.controller_temperature >= 0.0,
            "AgentRunbook-R controller temperature must be non-negative",
        )
        require(
            0.0 < self.controller_top_p <= 1.0,
            "AgentRunbook-R controller top_p must be in (0, 1]",
        )
        require(
            self.controller_top_k > 0,
            "AgentRunbook-R controller top_k must be positive",
        )
        require(self.embedding_model, "agentrunbook_r embedding model must be non-empty")
        require(self.embedding_base_url, "agentrunbook_r embedding base_url must be non-empty")
        require(
            self.embedding_api_key_env,
            "agentrunbook_r embedding api_key_env must be non-empty",
        )
        require(
            self.embedding_max_input_tokens > 0,
            "agentrunbook_r embedding max_input_tokens must be positive",
        )
        require(
            self.embedding_query_instruction,
            "agentrunbook_r embedding query_instruction must be non-empty",
        )
        require(
            self.raw_state_slice_radius >= 0,
            "agentrunbook_r raw_state_slice_radius must be non-negative",
        )
        require(
            self.raw_state_search_top_k_per_query > 0,
            "agentrunbook_r raw_state_search_top_k_per_query must be positive",
        )
        require(
            self.event_search_top_k > 0,
            "agentrunbook_r event_search_top_k must be positive",
        )
        require(
            self.note_search_top_k_per_type > 0,
            "agentrunbook_r note_search_top_k_per_type must be positive",
        )
        require(
            self.max_raw_state_queries > 0,
            "agentrunbook_r max_raw_state_queries must be positive",
        )
        require(
            self.raw_state_result_merge_budget > 0,
            "agentrunbook_r raw_state_result_merge_budget must be positive",
        )
        require(
            self.raw_state_result_merge_per_query_cap > 0,
            "agentrunbook_r raw_state_result_merge_per_query_cap must be positive",
        )
        require(
            self.rerank_candidate_limit > 0,
            "agentrunbook_r rerank_candidate_limit must be positive",
        )

        self._runtime_local = threading.local()
        self._controller_client_init_lock = threading.Lock()
        self._embedding_client_init_lock = threading.Lock()
        self._async_controller_init_lock = threading.Lock()
        self._embedding_tokenizer_init_lock = threading.Lock()
        self._async_controller_client: AsyncOpenAI | None = None
        self._embedding_tokenizer = None

        self.inserted_trajectory_ids: list[str] = []
        self.raw_state_entries: list[dict[str, Any]] = []
        self.event_entries: list[dict[str, Any]] = []
        self.procedure_note_entries: list[dict[str, Any]] = []
        self.hint_note_entries: list[dict[str, Any]] = []
        self.raw_state_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.event_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.procedure_note_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.hint_note_embeddings = np.zeros((0, 0), dtype=np.float32)
        self._stored_goal_snippets: list[str] = []
        self._runtime_domain = "unknown"
        self._runtime_query_summary = ""

        if self.workspace_dir is not None:
            self._ensure_workspace_layout(self.workspace_dir)

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, Any] = {
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "trajectories_root_dir": (
                str(self.trajectories_root_dir) if self.trajectories_root_dir is not None else None
            ),
            "controller_params": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
                "api_key_env": self.controller_api_key_env,
                "api_key_file": self.controller_api_key_file,
                "max_completion_tokens": self.controller_max_completion_tokens,
                "timeout_seconds": self.controller_timeout_seconds,
                "max_retries": self.controller_max_retries,
                "disable_thinking": self.controller_disable_thinking,
                "temperature": self.controller_temperature,
                "top_p": self.controller_top_p,
                "top_k": self.controller_top_k,
            },
            "embedding_params": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "api_key_env": self.embedding_api_key_env,
                "api_key_file": self.embedding_api_key_file,
                "max_input_tokens": self.embedding_max_input_tokens,
                "query_instruction": self.embedding_query_instruction,
            },
            "index_params": {
                "raw_state_slice_radius": self.raw_state_slice_radius,
            },
            "query_params": {
                "max_raw_state_queries": self.max_raw_state_queries,
                "query_generation_disable_thinking": self.query_generation_disable_thinking,
            },
            "retrieval_params": {
                "raw_state_search_top_k_per_query": self.raw_state_search_top_k_per_query,
                "event_search_top_k": self.event_search_top_k,
                "note_search_top_k_per_type": self.note_search_top_k_per_type,
                "raw_state_result_merge_budget": self.raw_state_result_merge_budget,
                "raw_state_result_merge_per_query_cap": self.raw_state_result_merge_per_query_cap,
                "rerank_candidate_limit": self.rerank_candidate_limit,
                "enable_rerank": self.enable_rerank,
            },
        }
        return {
            "memory_type": self.memory_type,
            "memory_params": memory_params,
        }

    def query(
        self,
        query: str,
        query_image: str | None = None,
    ) -> list[MemoryContextItem]:
        require(isinstance(query, str) and query.strip(), "agentrunbook_r query must be non-empty")
        parsed_bundle = self._generate_structured_query_bundle(query=query, query_image=query_image)
        if self.enable_rerank:
            return self._query_with_rerank(query=query, parsed_bundle=parsed_bundle)
        return self._query_without_rerank(parsed_bundle=parsed_bundle)

    def _query_without_rerank(self, *, parsed_bundle: dict[str, Any]) -> list[MemoryContextItem]:
        raw_query_blocks = self._search_raw_state_queries(parsed_bundle["raw_state_queries"])
        merged_raw_results = self._merge_raw_state_results(raw_query_blocks)
        event_results = self._search_event_query(parsed_bundle["event_query"])
        note_results = self._search_note_query(parsed_bundle["note_query"])

        items: list[MemoryContextItem] = []
        items.extend(self._build_note_context_items(parsed_bundle["note_query"], note_results))
        items.extend(self._build_event_context_items(parsed_bundle["event_query"], event_results))
        items.extend(self._build_raw_state_context_items(merged_raw_results))
        return items

    def _query_with_rerank(
        self,
        *,
        query: str,
        parsed_bundle: dict[str, Any],
    ) -> list[MemoryContextItem]:
        raw_query_blocks = self._search_raw_state_queries(parsed_bundle["raw_state_queries"])
        event_results = self._search_event_query(parsed_bundle["event_query"])
        note_results = self._search_note_query(parsed_bundle["note_query"])

        reranked_raw_blocks = [
            self._rerank_result_block(
                original_query=query,
                pool_name="raw_state",
                query_text=block["query"],
                results=block["results"],
            )
            for block in raw_query_blocks
        ]
        reranked_event_block = self._rerank_result_block(
            original_query=query,
            pool_name="event",
            query_text=parsed_bundle["event_query"],
            results=event_results,
        )
        reranked_procedure_block = self._rerank_result_block(
            original_query=query,
            pool_name="procedure_note",
            query_text=parsed_bundle["note_query"],
            results=note_results["procedure_results"],
        )
        reranked_hint_block = self._rerank_result_block(
            original_query=query,
            pool_name="hint_note",
            query_text=parsed_bundle["note_query"],
            results=note_results["hint_results"],
        )

        items: list[MemoryContextItem] = []
        include_note_title = True
        for block in [reranked_procedure_block, reranked_hint_block]:
            selected_results = block["selected_results"]
            if not selected_results:
                continue
            items.extend(
                self._build_reranked_note_context_items(
                    note_query=block["query_text"],
                    pool_name=block["pool_name"],
                    selected_results=selected_results,
                    selection_description=block["selection_description"],
                    include_section_title=include_note_title,
                )
            )
            include_note_title = False

        if reranked_event_block["selected_results"]:
            items.extend(
                self._build_reranked_event_context_items(
                    event_query=reranked_event_block["query_text"],
                    selected_results=reranked_event_block["selected_results"],
                    selection_description=reranked_event_block["selection_description"],
                    include_section_title=True,
                )
            )

        include_raw_title = True
        for block in reranked_raw_blocks:
            selected_results = block["selected_results"]
            if not selected_results:
                continue
            items.extend(
                self._build_reranked_raw_state_context_items(
                    raw_query=block["query_text"],
                    selected_results=selected_results,
                    selection_description=block["selection_description"],
                    include_section_title=include_raw_title,
                )
            )
            include_raw_title = False
        return items

    def configure_runtime(self, **kwargs: object) -> None:
        _ = kwargs
        self._get_controller_client()
        self._get_embedding_client()
        self._get_embedding_tokenizer()
        return None

    def _generate_structured_query_bundle(
        self,
        *,
        query: str,
        query_image: str | None,
    ) -> dict[str, Any]:
        query_context = self.get_query_context()
        messages = self._build_query_generation_messages(
            query=query,
            query_image=query_image,
            query_context=query_context,
        )
        raw_text = self._call_controller_text(
            messages,
            disable_thinking=self.query_generation_disable_thinking,
        )
        try:
            return _parse_query_bundle(raw_text, max_raw_state_queries=self.max_raw_state_queries)
        except Exception:
            repaired_text = self._call_controller_text(
                self._build_query_repair_messages(raw_text),
                disable_thinking=True,
            )
            try:
                return _parse_query_bundle(
                    repaired_text,
                    max_raw_state_queries=self.max_raw_state_queries,
                )
            except Exception:
                return self._fallback_query_bundle(
                    query=query,
                    question_type=self._question_type_from_context(query_context),
                )

    def _build_query_generation_messages(
        self,
        *,
        query: str,
        query_image: str | None,
        query_context: dict[str, object],
    ) -> list[dict[str, Any]]:
        question_id = self._question_id_from_context(query_context)
        question_type = self._question_type_from_context(query_context)
        original_goals = self._question_original_goals_from_context(query_context)
        summary_text = self._runtime_query_summary or self._render_runtime_query_summary()
        user_text = "\n\n".join(
            [
                "Memory pool summary:",
                summary_text,
                "Output schema example:",
                json.dumps(QUERY_SCHEMA_EXAMPLE, ensure_ascii=True),
                "Prompt examples:",
                QUERY_GENERATION_EXAMPLES,
                "Question to rewrite into retrieval queries:",
                f"Question ID: {question_id}",
                f"Question type: {question_type}",
                f"Question text: {query}",
                f"Question image path: {query_image or '<none>'}",
                f"Original goals attached to this benchmark question: {json.dumps(original_goals, ensure_ascii=True)}",
                "Return only the JSON object.",
            ]
        )
        return [
            {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def _build_query_repair_messages(self, draft_response: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": QUERY_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": draft_response},
        ]

    def _fallback_query_bundle(self, *, query: str, question_type: str) -> dict[str, Any]:
        normalized = _normalize_text(query)
        raw_queries = [normalized] if normalized else []
        if question_type.startswith("procedure"):
            return {
                "raw_state_queries": raw_queries[:1],
                "event_query": "",
                "note_query": normalized,
            }
        if question_type.startswith("dynamic"):
            return {
                "raw_state_queries": raw_queries[:1],
                "event_query": normalized,
                "note_query": "",
            }
        return {
            "raw_state_queries": raw_queries[:1],
            "event_query": "",
            "note_query": "",
        }

    def _question_id_from_context(self, query_context: dict[str, object]) -> str:
        question_id = query_context.get("question_id")
        if isinstance(question_id, str) and question_id.strip():
            return question_id.strip()
        question_item = query_context.get("question_item")
        if isinstance(question_item, dict):
            value = question_item.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "<unknown>"

    def _question_type_from_context(self, query_context: dict[str, object]) -> str:
        question_type = query_context.get("question_type")
        if isinstance(question_type, str) and question_type.strip():
            return question_type.strip()
        question_item = query_context.get("question_item")
        if isinstance(question_item, dict):
            value = question_item.get("question_type")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "<unknown>"

    def _question_original_goals_from_context(self, query_context: dict[str, object]) -> list[str]:
        question_item = query_context.get("question_item")
        if not isinstance(question_item, dict):
            return []
        candidates: list[Any] = [
            question_item.get("original_goal"),
            question_item.get("original_goals"),
        ]
        metadata = question_item.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [
                    metadata.get("original_goal"),
                    metadata.get("original_goals"),
                ]
            )
        values: list[str] = []
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                values.append(_normalize_text(candidate))
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        values.append(_normalize_text(item))
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _search_entries(
        self,
        *,
        entries: list[dict[str, Any]],
        embeddings: np.ndarray,
        query_text: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query_text.strip() or embeddings.size == 0 or not entries:
            return []
        query_vector = self._embed_texts([query_text], is_query=True)
        if query_vector.size == 0:
            return []
        scores = embeddings @ query_vector[0]
        top_indexes = np.argsort(scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indexes, start=1):
            entry = entries[int(idx)]
            results.append(
                {
                    "query": query_text,
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "entry": entry,
                }
            )
        return results

    def _search_raw_state_queries(self, raw_state_queries: list[str]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for query_text in raw_state_queries:
            blocks.append(
                {
                    "query": query_text,
                    "results": self._search_entries(
                        entries=self.raw_state_entries,
                        embeddings=self.raw_state_embeddings,
                        query_text=query_text,
                        top_k=self.raw_state_search_top_k_per_query,
                    ),
                }
            )
        return blocks

    def _search_event_query(self, event_query: str) -> list[dict[str, Any]]:
        if not event_query:
            return []
        return self._search_entries(
            entries=self.event_entries,
            embeddings=self.event_embeddings,
            query_text=event_query,
            top_k=self.event_search_top_k,
        )

    def _search_note_query(self, note_query: str) -> dict[str, list[dict[str, Any]]]:
        if not note_query:
            return {"procedure_results": [], "hint_results": []}
        return {
            "procedure_results": self._search_entries(
                entries=self.procedure_note_entries,
                embeddings=self.procedure_note_embeddings,
                query_text=note_query,
                top_k=self.note_search_top_k_per_type,
            ),
            "hint_results": self._search_entries(
                entries=self.hint_note_entries,
                embeddings=self.hint_note_embeddings,
                query_text=note_query,
                top_k=self.note_search_top_k_per_type,
            ),
        }

    def _merge_raw_state_results(self, raw_query_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_count = len(raw_query_blocks)
        if query_count == 0:
            return []
        if query_count <= 3:
            per_query_limit = max(1, self.raw_state_result_merge_budget // query_count)
        else:
            per_query_limit = self.raw_state_result_merge_per_query_cap

        merged: list[dict[str, Any]] = []
        seen_entry_ids: set[str] = set()
        for block in raw_query_blocks:
            taken = 0
            for result in block["results"]:
                entry_id = result["entry"]["entry_id"]
                if entry_id in seen_entry_ids:
                    continue
                seen_entry_ids.add(entry_id)
                merged.append(result)
                taken += 1
                if taken >= per_query_limit:
                    break
        return merged

    def _build_rerank_repair_messages(self, draft_response: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": RERANK_REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": draft_response},
        ]

    def _summarize_rerank_raw_candidate(self, result: dict[str, Any]) -> str:
        entry = result["entry"]
        slice_urls = entry.get("slice_urls") or []
        lines = [
            f"Retrieval rank: {result['rank']}",
            f"Similarity: {result['score']:.4f}",
            f"Trajectory: {entry.get('trajectory_id', '<unknown>')}",
            f"Goal: {_truncate_middle(str(entry.get('goal', '')), 220)}",
            f"Center state index: {entry.get('center_state_index')}",
            f"Slice URLs: {_truncate_middle(' | '.join(str(url) for url in slice_urls), 220)}",
            "Local slice action sequence:",
            _truncate_middle(str(entry.get("slice_action_sequence", "")), 450),
            "Slice AXTree excerpt:",
            _truncate_middle(str(entry.get("slice_axtree_text", "")), 1400),
        ]
        return "\n".join(lines).strip()

    def _summarize_rerank_event_candidate(self, result: dict[str, Any]) -> str:
        entry = result["entry"]
        lines = [
            f"Retrieval rank: {result['rank']}",
            f"Similarity: {result['score']:.4f}",
            f"Trajectory: {entry.get('trajectory_id', '<unknown>')}",
            f"Goal: {_truncate_middle(str(entry.get('goal', '')), 220)}",
            (
                f"Transition: state {entry.get('pre_state_index')} -> "
                f"{_truncate_middle(str(entry.get('local_action', '')), 140)} -> "
                f"state {entry.get('post_state_index')}"
            ),
            "Description:",
            _truncate_middle(str(entry.get("description", "")), 1400),
        ]
        return "\n".join(lines).strip()

    def _summarize_rerank_note_candidate(self, result: dict[str, Any]) -> str:
        entry = result["entry"]
        lines = [
            f"Retrieval rank: {result['rank']}",
            f"Similarity: {result['score']:.4f}",
            f"Trajectory: {entry.get('trajectory_id', '<unknown>')}",
            f"Title: {_truncate_middle(str(entry.get('title', '')), 200)}",
            f"Description: {_truncate_middle(str(entry.get('description', '')), 300)}",
            "Content:",
            _truncate_middle(str(entry.get("content", "")), 1200),
        ]
        return "\n".join(lines).strip()

    def _summarize_rerank_candidate(self, *, pool_name: str, result: dict[str, Any]) -> str:
        if pool_name == "raw_state":
            return self._summarize_rerank_raw_candidate(result)
        if pool_name == "event":
            return self._summarize_rerank_event_candidate(result)
        return self._summarize_rerank_note_candidate(result)

    def _build_rerank_messages(
        self,
        *,
        original_query: str,
        pool_name: str,
        query_text: str,
        candidate_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_lines: list[str] = []
        for candidate_index, result in enumerate(candidate_results):
            candidate_lines.append(
                "\n".join(
                    [
                        f"Candidate {candidate_index}",
                        self._summarize_rerank_candidate(pool_name=pool_name, result=result),
                    ]
                )
            )
        user_sections = [
            f"Original question:\n{original_query}",
            f"Generated retrieval query:\n{query_text or '<empty>'}",
            f"Memory pool:\n{pool_name}",
            "Candidate list:",
            "\n\n".join(candidate_lines) if candidate_lines else "<no candidates>",
            'Return exactly one JSON object with keys "selected_indices" and "selection_description".',
        ]
        return [
            {"role": "system", "content": RERANK_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ]

    def _default_rerank_fallback_results(
        self,
        *,
        pool_name: str,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fallback_top_k = (
            DEFAULT_RERANK_FALLBACK_RAW_TOP_K
            if pool_name == "raw_state"
            else DEFAULT_RERANK_FALLBACK_OTHER_TOP_K
        )
        return list(results[:fallback_top_k])

    def _rerank_result_block(
        self,
        *,
        original_query: str,
        pool_name: str,
        query_text: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_results = list(results[: self.rerank_candidate_limit])
        if not query_text or not candidate_results:
            return {
                "pool_name": pool_name,
                "query_text": query_text,
                "selected_results": [],
                "selection_description": "",
                "used_fallback": False,
            }

        raw_text = self._call_controller_content_text(
            self._build_rerank_messages(
                original_query=original_query,
                pool_name=pool_name,
                query_text=query_text,
                candidate_results=candidate_results,
            ),
            disable_thinking=False,
        )
        try:
            parsed = _parse_rerank_output(raw_text, candidate_count=len(candidate_results))
        except Exception:
            repaired_text = self._call_controller_content_text(
                self._build_rerank_repair_messages(raw_text),
                disable_thinking=True,
            )
            try:
                parsed = _parse_rerank_output(repaired_text, candidate_count=len(candidate_results))
            except Exception:
                selected_results = self._default_rerank_fallback_results(
                    pool_name=pool_name,
                    results=candidate_results,
                )
                selected_count = len(selected_results)
                return {
                    "pool_name": pool_name,
                    "query_text": query_text,
                    "selected_results": selected_results,
                    "selection_description": (
                        "Rerank fallback: retained the first-stage top-"
                        f"{selected_count} candidates for this query because the reranker "
                        "did not return a usable JSON selection."
                    ),
                    "used_fallback": True,
                }

        selected_results = [candidate_results[idx] for idx in parsed["selected_indices"]]
        selection_description = parsed["selection_description"].strip()
        if not selection_description:
            if selected_results:
                selection_description = "Selected because these candidates are the most relevant evidence for this query."
            else:
                selection_description = "No candidate from this block was selected."
        return {
            "pool_name": pool_name,
            "query_text": query_text,
            "selected_results": selected_results,
            "selection_description": selection_description,
            "used_fallback": False,
        }

    def _inject_analysis_line(self, text: str, analysis: str) -> str:
        cleaned_analysis = analysis.strip()
        if not cleaned_analysis:
            return text
        analysis_line = f"- Rerank analysis: {cleaned_analysis}\n\n"
        if "\n\n" in text:
            head, tail = text.split("\n\n", 1)
            return f"{head}\n{analysis_line}{tail}".rstrip() + "\n"
        return text.rstrip() + "\n" + analysis_line

    def _build_reranked_note_context_items(
        self,
        *,
        note_query: str,
        pool_name: str,
        selected_results: list[dict[str, Any]],
        selection_description: str,
        include_section_title: bool,
    ) -> list[MemoryContextItem]:
        if not selected_results:
            return []
        payload = {
            "procedure_results": selected_results if pool_name == "procedure_note" else [],
            "hint_results": selected_results if pool_name == "hint_note" else [],
        }
        items = self._build_note_context_items(note_query, payload)
        if not items:
            return []
        text_item = dict(items[0])
        value = text_item["value"]
        if not include_section_title and value.startswith(NOTE_SECTION_TITLE):
            value = value[len(NOTE_SECTION_TITLE) :]
        text_item["value"] = self._inject_analysis_line(value, selection_description)
        return [text_item]

    def _build_reranked_event_context_items(
        self,
        *,
        event_query: str,
        selected_results: list[dict[str, Any]],
        selection_description: str,
        include_section_title: bool,
    ) -> list[MemoryContextItem]:
        if not selected_results:
            return []
        items = self._build_event_context_items(event_query, selected_results)
        if not items:
            return []
        text_item = dict(items[0])
        value = text_item["value"]
        if not include_section_title and value.startswith(EVENT_SECTION_TITLE):
            value = value[len(EVENT_SECTION_TITLE) :]
        text_item["value"] = self._inject_analysis_line(value, selection_description)
        return [text_item] + items[1:]

    def _build_reranked_raw_state_context_items(
        self,
        *,
        raw_query: str,
        selected_results: list[dict[str, Any]],
        selection_description: str,
        include_section_title: bool,
    ) -> list[MemoryContextItem]:
        if not selected_results:
            return []
        items = self._build_raw_state_context_items(selected_results)
        if not items or items[0]["type"] != "text":
            return items
        prefix = RAW_STATE_SECTION_TITLE if include_section_title else ""
        text_item = {
            "type": "text",
            "value": (
                f"{prefix}- Raw query: {raw_query}\n"
                f"- Rerank analysis: {selection_description.strip()}\n\n"
            ),
        }
        return [text_item] + items[1:]

    def _build_note_context_items(
        self,
        note_query: str,
        note_results: dict[str, list[dict[str, Any]]],
    ) -> list[MemoryContextItem]:
        procedure_results = note_results.get("procedure_results", [])
        hint_results = note_results.get("hint_results", [])
        if not note_query or (not procedure_results and not hint_results):
            return []

        lines = [NOTE_SECTION_TITLE, f"- Note query: {note_query}\n\n"]
        if procedure_results:
            lines.append("### Procedure note results\n")
            for result in procedure_results:
                entry = result["entry"]
                lines.extend(
                    [
                        f"- Rank {result['rank']} | score={result['score']:.4f} | trajectory={entry['trajectory_id']}\n",
                        f"  Title: {entry['title']}\n",
                        f"  Description: {entry['description']}\n",
                        f"{entry['content']}\n\n",
                    ]
                )
        if hint_results:
            lines.append("### Hint note results\n")
            for result in hint_results:
                entry = result["entry"]
                lines.extend(
                    [
                        f"- Rank {result['rank']} | score={result['score']:.4f} | trajectory={entry['trajectory_id']}\n",
                        f"  Title: {entry['title']}\n",
                        f"  Description: {entry['description']}\n",
                        f"{entry['content']}\n\n",
                    ]
                )
        return [{"type": "text", "value": "".join(lines).strip() + "\n"}]

    def _build_event_context_items(
        self,
        event_query: str,
        event_results: list[dict[str, Any]],
    ) -> list[MemoryContextItem]:
        if not event_query or not event_results:
            return []
        items: list[MemoryContextItem] = [
            {"type": "text", "value": EVENT_SECTION_TITLE + f"- Event query: {event_query}\n\n"}
        ]
        for display_rank, result in enumerate(event_results, start=1):
            entry = result["entry"]
            items.append(
                {
                    "type": "text",
                    "value": (
                        f"### Event result {display_rank}\n"
                        f"- Retrieval rank: {result['rank']}\n"
                        f"- Similarity: {result['score']:.4f}\n"
                        f"- Trajectory: {entry['trajectory_id']}\n"
                        f"- Transition: state {entry['pre_state_index']} -> {entry['local_action']} -> state {entry['post_state_index']}\n\n"
                        "Description\n"
                        f"{entry['description']}\n\n"
                        "Images\n"
                        "- The next image is the pre-state screenshot.\n"
                        "- The image after that is the post-state screenshot.\n"
                    ),
                }
            )
            items.append(
                {
                    "type": "image",
                    "value": str(
                        self._absolute_screenshot_path(entry["trajectory_id"], entry["pre_screenshot"])
                    ),
                }
            )
            items.append(
                {
                    "type": "image",
                    "value": str(
                        self._absolute_screenshot_path(entry["trajectory_id"], entry["post_screenshot"])
                    ),
                }
            )
        return items

    def _build_raw_state_context_items(self, raw_results: list[dict[str, Any]]) -> list[MemoryContextItem]:
        if not raw_results:
            return []
        items: list[MemoryContextItem] = [{"type": "text", "value": RAW_STATE_SECTION_TITLE}]
        trajectory_cache: dict[str, dict[str, Any]] = {}
        for display_rank, result in enumerate(raw_results, start=1):
            entry = result["entry"]
            trajectory_id = entry["trajectory_id"]
            if trajectory_id not in trajectory_cache:
                trajectory_cache[trajectory_id] = self._load_stored_trajectory(trajectory_id)
            trajectory = trajectory_cache[trajectory_id]
            center_index = entry["center_state_index"]
            state_lookup = {
                int(state["state_index"]): state
                for state in trajectory.get("states", [])
                if isinstance(state, dict) and isinstance(state.get("state_index"), int)
            }
            state_blocks: list[str] = []
            for state_index in entry["slice_state_indexes"]:
                state = state_lookup.get(int(state_index))
                if state is None:
                    continue
                action_value = state.get("action")
                action_text = action_value if isinstance(action_value, str) and action_value.strip() else "<none>"
                state_blocks.append(
                    "\n".join(
                        [
                            f"State {state['state_index']} (step {state['step']})",
                            f"- URL: {state['url']}",
                            f"- Action: {action_text}",
                            "- AXTree:",
                            state["text"],
                        ]
                    )
                )
            joined_state_blocks = "\n\n".join(state_blocks)
            items.append(
                {
                    "type": "text",
                    "value": (
                        f"### Raw state result {display_rank}\n"
                        f"- Source raw query: {result['query']}\n"
                        f"- Retrieval rank within query: {result['rank']}\n"
                        f"- Similarity: {result['score']:.4f}\n"
                        f"- Trajectory: {trajectory_id}\n"
                        f"- Goal: {entry['goal']}\n"
                        f"- Center state index: {center_index}\n\n"
                        "Full action sequence\n"
                        f"{entry['full_action_sequence']}\n\n"
                        "Local slice action sequence\n"
                        f"{entry['slice_action_sequence']}\n\n"
                        "Relevant state slices\n"
                        f"{joined_state_blocks}\n"
                    ),
                }
            )
            center_state = state_lookup.get(int(center_index))
            if center_state is not None:
                items.append(
                    {
                        "type": "image",
                        "value": str(
                            self._absolute_screenshot_path(trajectory_id, center_state["screenshot"])
                        ),
                    }
                )
        return items

    def insert(self, trajectory: dict[str, object]) -> None:
        require(self.workspace_dir is not None, "agentrunbook_r insert requires workspace_dir")
        require(
            self.trajectories_root_dir is not None,
            "agentrunbook_r insert requires trajectories_root_dir",
        )
        prepared = prepare_trajectory_insert(
            trajectory,
            trajectories_root_dir=self.trajectories_root_dir,
        )
        trajectory_id = prepared.trajectory_id
        if trajectory_id in set(self.inserted_trajectory_ids):
            raise RuntimeError(f"Duplicate trajectory insert attempted: {trajectory_id}")

        trajectory_dir = self.workspace_dir / "trajectories" / trajectory_id
        require(
            not trajectory_dir.exists(),
            f"Refusing to overwrite existing trajectory dir: {trajectory_dir}",
        )

        simplified = prepared.simplified
        if self.trajectory_pool_root is not None:
            pooled_trajectory_dir = self.trajectory_pool_root / trajectory_id
        else:
            pooled_trajectory_dir = None

        if pooled_trajectory_dir is not None and pooled_trajectory_dir.exists():
            artifact_bundle = self._load_trajectory_artifact_bundle_from_pool(
                prepared=prepared,
                pooled_trajectory_dir=pooled_trajectory_dir,
            )
            relative_symlink(pooled_trajectory_dir, trajectory_dir)
        else:
            trajectory_dir.mkdir(parents=True, exist_ok=False)
            materialize_prepared_trajectory(prepared, trajectory_dir)
            artifact_bundle = self._build_trajectory_artifact_bundle(
                trajectory_dir=trajectory_dir,
                simplified_trajectory=simplified,
            )

        states = simplified["states"]
        raw_state_entries = artifact_bundle["raw_state_entries"]
        event_entries = artifact_bundle["event_entries"]
        procedure_entry = artifact_bundle["procedure_note_entry"]
        hint_entry = artifact_bundle["hint_note_entry"]

        self.inserted_trajectory_ids.append(trajectory_id)
        self.raw_state_entries.extend(raw_state_entries)
        self.event_entries.extend(event_entries)
        self.procedure_note_entries.append(procedure_entry)
        self.hint_note_entries.append(hint_entry)

        self.raw_state_embeddings = _append_embedding_rows(
            self.raw_state_embeddings,
            artifact_bundle["raw_state_embeddings"],
        )
        self.event_embeddings = _append_embedding_rows(
            self.event_embeddings,
            artifact_bundle["event_embeddings"],
        )
        self.procedure_note_embeddings = _append_embedding_rows(
            self.procedure_note_embeddings,
            artifact_bundle["procedure_note_embedding"],
        )
        self.hint_note_embeddings = _append_embedding_rows(
            self.hint_note_embeddings,
            artifact_bundle["hint_note_embedding"],
        )

        require(
            len(states) >= 1,
            f"Inserted trajectory unexpectedly missing states after preparation: {trajectory_id}",
        )
        goal_snippet = _first_clause(simplified["goal"], 100)
        if goal_snippet:
            self._stored_goal_snippets.append(goal_snippet)
        if self._runtime_domain == "unknown":
            self._runtime_domain = _domain_from_url(simplified["start_url"])
        self._refresh_runtime_query_summary()

    def _ensure_workspace_layout(self, workspace_dir: Path) -> None:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "pools").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "previews").mkdir(parents=True, exist_ok=True)

    def _trajectory_artifact_dir(self, trajectory_dir: Path) -> Path:
        return trajectory_dir / TRAJECTORY_ARTIFACT_DIRNAME

    def _trajectory_artifact_signature_payload(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
            "memory_type": self.memory_type,
            "raw_state_slice_radius": self.raw_state_slice_radius,
            "controller_model": self.controller_model,
            "controller_disable_thinking": self.controller_disable_thinking,
            "controller_max_completion_tokens": self.controller_max_completion_tokens,
            "controller_temperature": self.controller_temperature,
            "controller_top_p": self.controller_top_p,
            "controller_top_k": self.controller_top_k,
            "embedding_model": self.embedding_model,
            "embedding_max_input_tokens": self.embedding_max_input_tokens,
            "embedding_query_instruction": self.embedding_query_instruction,
            "event_prompt_version": EVENT_PROMPT_VERSION,
            "event_repair_prompt_version": EVENT_REPAIR_PROMPT_VERSION,
            "note_prompt_version": NOTE_GENERATION_PROMPT_VERSION,
            "event_generation_system_prompt": EVENT_GENERATION_SYSTEM_PROMPT,
            "event_generation_repair_system_prompt": EVENT_GENERATION_REPAIR_SYSTEM_PROMPT,
            "note_generation_system_prompt": NOTE_GENERATION_SYSTEM_PROMPT,
            "note_generation_repair_system_prompt": NOTE_GENERATION_REPAIR_SYSTEM_PROMPT,
        }

    def _trajectory_artifact_signature(self) -> str:
        canonical = json.dumps(
            self._trajectory_artifact_signature_payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _trajectory_artifact_config_snapshot(self) -> dict[str, Any]:
        return {
            "raw_state_slice_radius": self.raw_state_slice_radius,
            "controller_model": self.controller_model,
            "controller_disable_thinking": self.controller_disable_thinking,
            "controller_max_completion_tokens": self.controller_max_completion_tokens,
            "embedding_model": self.embedding_model,
            "embedding_max_input_tokens": self.embedding_max_input_tokens,
            "embedding_query_instruction": self.embedding_query_instruction,
            "event_prompt_version": EVENT_PROMPT_VERSION,
            "event_repair_prompt_version": EVENT_REPAIR_PROMPT_VERSION,
            "note_prompt_version": NOTE_GENERATION_PROMPT_VERSION,
        }

    def _build_trajectory_artifact_bundle(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
    ) -> dict[str, Any]:
        raw_state_entries = self._build_raw_state_entries(simplified_trajectory)
        event_entries = self._build_event_entries(
            trajectory_dir=trajectory_dir,
            simplified_trajectory=simplified_trajectory,
        )
        procedure_entry, hint_entry = self._build_note_entries(
            trajectory_dir=trajectory_dir,
            simplified_trajectory=simplified_trajectory,
        )
        return {
            "raw_state_entries": raw_state_entries,
            "event_entries": event_entries,
            "procedure_note_entry": procedure_entry,
            "hint_note_entry": hint_entry,
            "raw_state_embeddings": self._embed_texts(
                [entry["slice_axtree_text"] for entry in raw_state_entries],
                is_query=False,
            ),
            "event_embeddings": self._embed_texts(
                [entry["description"] for entry in event_entries],
                is_query=False,
            ),
            "procedure_note_embedding": self._embed_texts(
                [procedure_entry["note_text"]],
                is_query=False,
            ),
            "hint_note_embedding": self._embed_texts(
                [hint_entry["note_text"]],
                is_query=False,
            ),
        }

    def _write_trajectory_artifact_bundle(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
        trajectory_fingerprint: str,
        bundle: dict[str, Any],
    ) -> None:
        artifact_dir = self._trajectory_artifact_dir(trajectory_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_FILENAME,
            bundle["raw_state_entries"],
        )
        _write_jsonl(
            artifact_dir / TRAJECTORY_ARTIFACT_EVENT_FILENAME,
            bundle["event_entries"],
        )
        save_json(
            artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_NOTE_FILENAME,
            bundle["procedure_note_entry"],
        )
        save_json(
            artifact_dir / TRAJECTORY_ARTIFACT_HINT_NOTE_FILENAME,
            bundle["hint_note_entry"],
        )
        np.save(
            artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_EMBEDDINGS_FILENAME,
            bundle["raw_state_embeddings"],
        )
        np.save(
            artifact_dir / TRAJECTORY_ARTIFACT_EVENT_EMBEDDINGS_FILENAME,
            bundle["event_embeddings"],
        )
        np.save(
            artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_EMBEDDING_FILENAME,
            bundle["procedure_note_embedding"],
        )
        np.save(
            artifact_dir / TRAJECTORY_ARTIFACT_HINT_EMBEDDING_FILENAME,
            bundle["hint_note_embedding"],
        )
        save_json(
            artifact_dir / TRAJECTORY_ARTIFACT_INDEX_FILENAME,
            {
                "memory_type": self.memory_type,
                "artifact_schema_version": TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
                "trajectory_id": simplified_trajectory["id"],
                "trajectory_fingerprint": trajectory_fingerprint,
                "artifact_signature": self._trajectory_artifact_signature(),
                "config_snapshot": self._trajectory_artifact_config_snapshot(),
                "entry_counts": {
                    "raw_state": len(bundle["raw_state_entries"]),
                    "events": len(bundle["event_entries"]),
                    "procedure_notes": 1,
                    "hint_notes": 1,
                },
                "embedding_dimensions": {
                    "raw_state": (
                        int(bundle["raw_state_embeddings"].shape[1])
                        if bundle["raw_state_embeddings"].size
                        else 0
                    ),
                    "events": (
                        int(bundle["event_embeddings"].shape[1])
                        if bundle["event_embeddings"].size
                        else 0
                    ),
                    "procedure_notes": (
                        int(bundle["procedure_note_embedding"].shape[1])
                        if bundle["procedure_note_embedding"].size
                        else 0
                    ),
                    "hint_notes": (
                        int(bundle["hint_note_embedding"].shape[1])
                        if bundle["hint_note_embedding"].size
                        else 0
                    ),
                },
            },
        )

    def build_and_write_trajectory_pool_artifact(
        self,
        *,
        trajectory: dict[str, object],
        trajectories_root_dir: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        prepared = prepare_trajectory_insert(
            trajectory,
            trajectories_root_dir=trajectories_root_dir,
        )
        require(
            not output_dir.exists(),
            f"Refusing to overwrite existing V3 pooled trajectory dir: {output_dir}",
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        materialize_prepared_trajectory(prepared, output_dir)
        bundle = self._build_trajectory_artifact_bundle(
            trajectory_dir=output_dir,
            simplified_trajectory=prepared.simplified,
        )
        self._write_trajectory_artifact_bundle(
            trajectory_dir=output_dir,
            simplified_trajectory=prepared.simplified,
            trajectory_fingerprint=prepared.fingerprint,
            bundle=bundle,
        )
        return {
            "trajectory_id": prepared.trajectory_id,
            "trajectory_fingerprint": prepared.fingerprint,
            "entry_counts": {
                "raw_state": len(bundle["raw_state_entries"]),
                "events": len(bundle["event_entries"]),
                "procedure_notes": 1,
                "hint_notes": 1,
            },
        }

    def _load_embedding_matrix_file(
        self,
        path: Path,
        *,
        expected_rows: int,
        field_name: str,
    ) -> np.ndarray:
        array = np.load(path).astype(np.float32, copy=False)
        if array.ndim == 1:
            if expected_rows != 1:
                raise RuntimeError(
                    f"{field_name} must be 2D or singleton-row for {path}, got shape {array.shape}"
                )
            array = array.reshape(1, -1)
        require(array.ndim == 2, f"{field_name} must be a 2D array: {path}")
        require(
            array.shape[0] == expected_rows,
            f"{field_name} row count mismatch for {path}: expected {expected_rows}, got {array.shape[0]}",
        )
        return array

    def _load_trajectory_artifact_bundle_from_pool(
        self,
        *,
        prepared: Any,
        pooled_trajectory_dir: Path,
    ) -> dict[str, Any]:
        validate_pooled_trajectory_dir(prepared, pooled_trajectory_dir=pooled_trajectory_dir)
        artifact_dir = self._trajectory_artifact_dir(pooled_trajectory_dir)
        require(
            artifact_dir.exists() and artifact_dir.is_dir(),
            (
                "Missing V3 pooled artifact directory for "
                f"{prepared.trajectory_id}: {artifact_dir}"
            ),
        )
        index_payload = load_json(artifact_dir / TRAJECTORY_ARTIFACT_INDEX_FILENAME)
        require(
            isinstance(index_payload, dict),
            f"V3 pooled artifact index must be an object: {artifact_dir / TRAJECTORY_ARTIFACT_INDEX_FILENAME}",
        )
        require(
            index_payload.get("memory_type") == self.memory_type,
            f"Unexpected pooled artifact memory_type for {prepared.trajectory_id}: {index_payload.get('memory_type')}",
        )
        require(
            index_payload.get("artifact_schema_version") == TRAJECTORY_ARTIFACT_SCHEMA_VERSION,
            (
                f"Unsupported V3 pooled artifact schema for {prepared.trajectory_id}: "
                f"{index_payload.get('artifact_schema_version')}"
            ),
        )
        require(
            index_payload.get("trajectory_id") == prepared.trajectory_id,
            f"Pooled artifact trajectory_id mismatch for {prepared.trajectory_id}",
        )
        require(
            index_payload.get("trajectory_fingerprint") == prepared.fingerprint,
            f"Pooled artifact trajectory fingerprint mismatch for {prepared.trajectory_id}",
        )
        require(
            index_payload.get("artifact_signature") == self._trajectory_artifact_signature(),
            (
                "Pooled artifact signature mismatch for "
                f"{prepared.trajectory_id}. Rebuild the V3 trajectory pool for the current prompts/config."
            ),
        )

        raw_state_entries = _read_jsonl(artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_FILENAME)
        event_entries = _read_jsonl(artifact_dir / TRAJECTORY_ARTIFACT_EVENT_FILENAME)
        procedure_entry = load_json(artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_NOTE_FILENAME)
        hint_entry = load_json(artifact_dir / TRAJECTORY_ARTIFACT_HINT_NOTE_FILENAME)
        require(
            isinstance(procedure_entry, dict),
            f"Procedure note payload must be an object for {prepared.trajectory_id}",
        )
        require(
            isinstance(hint_entry, dict),
            f"Hint note payload must be an object for {prepared.trajectory_id}",
        )
        raw_state_embeddings = self._load_embedding_matrix_file(
            artifact_dir / TRAJECTORY_ARTIFACT_RAW_STATE_EMBEDDINGS_FILENAME,
            expected_rows=len(raw_state_entries),
            field_name="raw_state_embeddings",
        )
        event_embeddings = self._load_embedding_matrix_file(
            artifact_dir / TRAJECTORY_ARTIFACT_EVENT_EMBEDDINGS_FILENAME,
            expected_rows=len(event_entries),
            field_name="event_embeddings",
        )
        procedure_note_embedding = self._load_embedding_matrix_file(
            artifact_dir / TRAJECTORY_ARTIFACT_PROCEDURE_EMBEDDING_FILENAME,
            expected_rows=1,
            field_name="procedure_note_embedding",
        )
        hint_note_embedding = self._load_embedding_matrix_file(
            artifact_dir / TRAJECTORY_ARTIFACT_HINT_EMBEDDING_FILENAME,
            expected_rows=1,
            field_name="hint_note_embedding",
        )
        return {
            "raw_state_entries": raw_state_entries,
            "event_entries": event_entries,
            "procedure_note_entry": procedure_entry,
            "hint_note_entry": hint_entry,
            "raw_state_embeddings": raw_state_embeddings,
            "event_embeddings": event_embeddings,
            "procedure_note_embedding": procedure_note_embedding,
            "hint_note_embedding": hint_note_embedding,
        }

    def _get_controller_client(self) -> OpenAI:
        client = getattr(self._runtime_local, "controller_client", None)
        if client is not None:
            return client
        with self._controller_client_init_lock:
            client = getattr(self._runtime_local, "controller_client", None)
            if client is None:
                api_key = _load_api_key(self.controller_api_key_env, self.controller_api_key_file)
                client = OpenAI(
                    base_url=self.controller_base_url,
                    api_key=api_key or "EMPTY",
                    max_retries=self.controller_max_retries,
                    timeout=self.controller_timeout_seconds,
                )
                self._runtime_local.controller_client = client
        return client

    def _get_embedding_client(self) -> OpenAI:
        client = getattr(self._runtime_local, "embedding_client", None)
        if client is not None:
            return client
        with self._embedding_client_init_lock:
            client = getattr(self._runtime_local, "embedding_client", None)
            if client is None:
                api_key = _load_api_key(self.embedding_api_key_env, self.embedding_api_key_file)
                client = OpenAI(
                    base_url=self.embedding_base_url,
                    api_key=api_key or "EMPTY",
                    max_retries=3,
                    timeout=1200.0,
                )
                self._runtime_local.embedding_client = client
        return client

    def _get_async_controller_client(self) -> AsyncOpenAI:
        if self._async_controller_client is not None:
            return self._async_controller_client
        with self._async_controller_init_lock:
            if self._async_controller_client is None:
                api_key = _load_api_key(self.controller_api_key_env, self.controller_api_key_file)
                self._async_controller_client = AsyncOpenAI(
                    base_url=self.controller_base_url,
                    api_key=api_key or "EMPTY",
                    max_retries=self.controller_max_retries,
                    timeout=self.controller_timeout_seconds,
                )
        return self._async_controller_client

    def _get_embedding_tokenizer(self):
        if self._embedding_tokenizer is None:
            with self._embedding_tokenizer_init_lock:
                if self._embedding_tokenizer is None:
                    self._embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
        return self._embedding_tokenizer

    def _refresh_runtime_query_summary(self) -> None:
        self._runtime_query_summary = self._render_runtime_query_summary()

    def _render_runtime_query_summary(self) -> str:
        root_title_counts: Counter[str] = Counter()
        for entry in self.raw_state_entries:
            title = _parse_root_title(entry.get("slice_axtree_text", ""))
            if title:
                root_title_counts[title] += 1

        procedure_titles = [row.get("title", "") for row in self.procedure_note_entries if row.get("title")]
        hint_titles = [row.get("title", "") for row in self.hint_note_entries if row.get("title")]
        goal_snippets = [snippet for snippet in self._stored_goal_snippets if snippet]

        common_surfaces = root_title_counts.most_common(RUNTIME_SUMMARY_SURFACE_COUNT)
        sampled_procedures = _spaced_sample(procedure_titles, RUNTIME_SUMMARY_NOTE_TITLE_COUNT)
        sampled_hints = _spaced_sample(hint_titles, RUNTIME_SUMMARY_NOTE_TITLE_COUNT)
        sampled_goals = _spaced_sample(goal_snippets, RUNTIME_SUMMARY_GOAL_COUNT)

        lines = [
            f"Domain: {self._runtime_domain}",
            "Memory inventory for the current query-time artifact:",
            f"- trajectories: {len(self.inserted_trajectory_ids)}",
            f"- raw state pool: {len(self.raw_state_entries)} entries for exact visible UI evidence",
            f"- event pool: {len(self.event_entries)} entries for navigation and before/after changes",
            f"- procedure note pool: {len(self.procedure_note_entries)} entries for reusable workflows",
            f"- hint note pool: {len(self.hint_note_entries)} entries for durable UI facts and gotchas",
            "Representative goals currently in memory:",
        ]
        for goal in sampled_goals:
            lines.append(f"- {goal}")
        lines.append("Common raw-state UI surfaces (root page titles):")
        for title, count in common_surfaces:
            lines.append(f"- {_truncate_middle(title, 100)} ({count})")
        lines.append("Representative procedure note titles:")
        for title in sampled_procedures:
            lines.append(f"- {_truncate_middle(title, 100)}")
        lines.append("Representative hint note titles:")
        for title in sampled_hints:
            lines.append(f"- {_truncate_middle(title, 100)}")
        return "\n".join(lines).strip()

    def _absolute_screenshot_path(self, trajectory_id: str, screenshot_rel: str) -> Path:
        require(self.workspace_dir is not None, "agentrunbook_r workspace_dir is not set")
        return self.workspace_dir / "trajectories" / trajectory_id / screenshot_rel

    def _load_stored_trajectory(self, trajectory_id: str) -> dict[str, Any]:
        require(self.workspace_dir is not None, "agentrunbook_r workspace_dir is not set")
        path = self.workspace_dir / "trajectories" / trajectory_id / "trajectory.json"
        payload = load_json(path)
        require(isinstance(payload, dict), f"Stored trajectory payload must be an object: {path}")
        return payload

    def _build_controller_request(
        self,
        messages: list[dict[str, Any]],
        *,
        disable_thinking: bool,
        max_completion_tokens: int | None = None,
    ) -> dict[str, Any]:
        completion_tokens = (
            self.controller_max_completion_tokens
            if max_completion_tokens is None
            else max_completion_tokens
        )
        request: dict[str, Any] = {
            "model": self.controller_model,
            "messages": messages,
            "timeout": self.controller_timeout_seconds,
            "max_tokens": completion_tokens,
            "temperature": self.controller_temperature,
            "top_p": self.controller_top_p,
        }
        extra_body: dict[str, Any] = {"top_k": self.controller_top_k}
        if self.controller_base_url and disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        request["extra_body"] = extra_body
        return request

    def _call_controller_text(
        self,
        messages: list[dict[str, Any]],
        *,
        disable_thinking: bool | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        response = self._get_controller_client().chat.completions.create(
            **self._build_controller_request(
                messages,
                disable_thinking=self.controller_disable_thinking if disable_thinking is None else disable_thinking,
                max_completion_tokens=max_completion_tokens,
            )
        )
        message = response.choices[0].message
        response_text, response_reasoning = _extract_response_text_parts(message)
        primary_text = response_text or response_reasoning or ""
        return primary_text.strip()

    def _call_controller_content_text(
        self,
        messages: list[dict[str, Any]],
        *,
        disable_thinking: bool | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        response = self._get_controller_client().chat.completions.create(
            **self._build_controller_request(
                messages,
                disable_thinking=self.controller_disable_thinking if disable_thinking is None else disable_thinking,
                max_completion_tokens=max_completion_tokens,
            )
        )
        message = response.choices[0].message
        content_text = _extract_content_text_only(getattr(message, "content", None))
        return content_text.strip() if isinstance(content_text, str) else ""

    async def _call_controller_text_async(
        self,
        messages: list[dict[str, Any]],
        *,
        disable_thinking: bool | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        response = await self._get_async_controller_client().chat.completions.create(
            **self._build_controller_request(
                messages,
                disable_thinking=self.controller_disable_thinking if disable_thinking is None else disable_thinking,
                max_completion_tokens=max_completion_tokens,
            )
        )
        message = response.choices[0].message
        response_text, response_reasoning = _extract_response_text_parts(message)
        primary_text = response_text or response_reasoning or ""
        return primary_text.strip()

    def _truncate_for_embedding(self, text: str) -> str:
        if len(text) > EMBEDDING_PRETOKEN_CHAR_CAP:
            text = _truncate_middle(text, EMBEDDING_PRETOKEN_CHAR_CAP)
        tokenizer = self._get_embedding_tokenizer()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.embedding_max_input_tokens:
            return text
        truncated_ids = token_ids[: self.embedding_max_input_tokens]
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def _format_query_for_embedding(self, query_text: str) -> str:
        return f"Instruct: {self.embedding_query_instruction}\nQuery:{query_text.strip()}".strip()

    def _embed_texts(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prepared = [
            self._truncate_for_embedding(self._format_query_for_embedding(text) if is_query else text)
            for text in texts
        ]
        response = self._get_embedding_client().embeddings.create(
            model=self.embedding_model,
            input=prepared,
        )
        return _normalize_embeddings([item.embedding for item in response.data])

    def _build_raw_state_entries(self, simplified_trajectory: dict[str, Any]) -> list[dict[str, Any]]:
        states = simplified_trajectory["states"]
        full_action_sequence = _full_action_sequence_text(states)
        entries: list[dict[str, Any]] = []
        for center_index in range(len(states)):
            start = max(0, center_index - self.raw_state_slice_radius)
            end = min(len(states), center_index + self.raw_state_slice_radius + 1)
            slice_states = states[start:end]
            entries.append(
                {
                    "entry_id": f"{simplified_trajectory['id']}:raw_state:{center_index:04d}",
                    "trajectory_id": simplified_trajectory["id"],
                    "goal": simplified_trajectory["goal"],
                    "center_state_index": center_index,
                    "slice_state_indexes": [state["state_index"] for state in slice_states],
                    "full_action_sequence": full_action_sequence,
                    "slice_action_sequence": _slice_transition_action_sequence_text(slice_states),
                    "slice_urls": [_state_url(state) for state in slice_states],
                    "slice_axtree_text": _slice_axtree_text(slice_states),
                }
            )
        return entries

    def _build_note_generation_messages(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        states = simplified_trajectory["states"]
        transition_actions = _annotated_transition_actions(states)
        outcome_value = simplified_trajectory.get("outcome")
        outcome_text = outcome_value.strip() if isinstance(outcome_value, str) and outcome_value.strip() else "<unknown>"
        header_lines = [
            "Extract two reusable notes from this UI task run.",
            f"Goal: {simplified_trajectory['goal']}",
            f"Outcome: {outcome_text}",
            f"Start URL: {simplified_trajectory['start_url']}",
            "Actions in this dataset are attached to destination states, so the next-action line below describes the action taken from the current state to the next state.",
            "Each state block is followed by the screenshot for that state.",
            "These notes should stay useful later under retrieval for unknown questions.",
            "Only preserve facts grounded in the observed run.",
            "Favor durable answer-bearing facts over generic page furniture.",
            'Return only the JSON object: {"procedure_note":{"title":"...","description":"...","content":"- ..."},"hint_note":{"title":"...","description":"...","content":"- ..."}}',
            "Do not think aloud in the final answer. First character must be {.",
        ]
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(header_lines)}]
        for index, state in enumerate(states):
            next_action = transition_actions[index] if index < len(transition_actions) else "<end of trajectory>"
            content_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"State {index}\n"
                        f"URL: {_state_url(state)}\n"
                        f"Thought: {_state_thoughts(state)}\n"
                        f"Next action from this state: {next_action}"
                    ),
                }
            )
            screenshot_rel = state["screenshot"]
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _to_data_url(trajectory_dir / screenshot_rel)},
                }
            )
        return [
            {"role": "system", "content": NOTE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

    def _build_note_repair_messages(self, draft_response: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": NOTE_GENERATION_REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Convert this draft into the required JSON object.\n"
                    "Return JSON only.\n\n"
                    f"{draft_response}"
                ),
            },
        ]

    def _fallback_note_entry(
        self,
        *,
        simplified_trajectory: dict[str, Any],
        note_type: str,
    ) -> dict[str, str]:
        action_lines = _annotated_transition_actions(simplified_trajectory["states"])
        if not action_lines:
            action_lines = ["<no recorded actions>"]
        if note_type == "procedure":
            content_lines = [f"- Step {idx + 1}: {line}" for idx, line in enumerate(action_lines[:8])]
            return {
                "title": f"Procedure fallback for {simplified_trajectory['goal'][:60]}",
                "description": "Fallback procedure note built when LLM note extraction failed.",
                "content": "\n".join(content_lines),
            }
        outcome_value = simplified_trajectory.get("outcome")
        outcome_text = outcome_value.strip() if isinstance(outcome_value, str) and outcome_value.strip() else "<unknown>"
        return {
            "title": f"Hint fallback for {simplified_trajectory['goal'][:60]}",
            "description": "Fallback hint note built when LLM note extraction failed.",
            "content": "\n".join(
                [
                    f"- Outcome: {outcome_text}",
                    f"- Start URL: {simplified_trajectory['start_url']}",
                    f"- Recorded actions: {len(action_lines)}",
                    f"- First observed action: {action_lines[0]}",
                ]
            ),
        }

    def _build_note_entries(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        primary_text = self._call_controller_text(
            self._build_note_generation_messages(
                trajectory_dir=trajectory_dir,
                simplified_trajectory=simplified_trajectory,
            )
        )
        procedure_note, hint_note, _parse_ok, _parse_error = _parse_generated_notes_json(primary_text)
        if (procedure_note is None or hint_note is None) and primary_text.strip():
            repaired_text = self._call_controller_text(
                self._build_note_repair_messages(primary_text),
                disable_thinking=True,
            )
            repaired_procedure, repaired_hint, _, _ = _parse_generated_notes_json(repaired_text)
            if repaired_procedure is not None and repaired_hint is not None:
                procedure_note = repaired_procedure
                hint_note = repaired_hint
        if procedure_note is None:
            procedure_note = self._fallback_note_entry(
                simplified_trajectory=simplified_trajectory,
                note_type="procedure",
            )
        if hint_note is None:
            hint_note = self._fallback_note_entry(
                simplified_trajectory=simplified_trajectory,
                note_type="hint",
            )

        procedure_entry = {
            "entry_id": f"{simplified_trajectory['id']}:procedure_note",
            "trajectory_id": simplified_trajectory["id"],
            "note_type": "procedure",
            "title": procedure_note["title"],
            "description": procedure_note["description"],
            "content": procedure_note["content"],
            "note_text": (
                f"Title: {procedure_note['title']}\n"
                f"Description: {procedure_note['description']}\n"
                f"{procedure_note['content']}"
            ).strip(),
        }
        hint_entry = {
            "entry_id": f"{simplified_trajectory['id']}:hint_note",
            "trajectory_id": simplified_trajectory["id"],
            "note_type": "hint",
            "title": hint_note["title"],
            "description": hint_note["description"],
            "content": hint_note["content"],
            "note_text": (
                f"Title: {hint_note['title']}\n"
                f"Description: {hint_note['description']}\n"
                f"{hint_note['content']}"
            ).strip(),
        }
        return procedure_entry, hint_entry

    def _build_event_transition_specs(
        self,
        simplified_trajectory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        states = simplified_trajectory["states"]
        specs: list[dict[str, Any]] = []
        if len(states) < 2:
            return specs
        annotated_actions = _annotated_transition_actions(states)
        transition_count = len(states) - 1
        for index in range(len(states) - 1):
            pre_state = states[index]
            post_state = states[index + 1]
            pre_incoming_action = "<initial state>" if index == 0 else annotated_actions[index - 1]
            specs.append(
                {
                    "event_id": _event_id_for_transition(pre_state["state_index"], post_state["state_index"]),
                    "trajectory_id": simplified_trajectory["id"],
                    "goal": simplified_trajectory["goal"],
                    "outcome": simplified_trajectory.get("outcome"),
                    "pre_state_index": pre_state["state_index"],
                    "post_state_index": post_state["state_index"],
                    "pre_url": _state_url(pre_state),
                    "post_url": _state_url(post_state),
                    "pre_incoming_action": pre_incoming_action,
                    "local_action": annotated_actions[index],
                    "transition_number": index + 1,
                    "transition_count": transition_count,
                    "pre_thoughts": _state_thoughts(pre_state),
                    "post_thoughts": _state_thoughts(post_state),
                    "pre_screenshot": pre_state["screenshot"],
                    "post_screenshot": post_state["screenshot"],
                }
            )
        return specs

    def _build_event_generation_messages(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
        spec: dict[str, Any],
        truncate_state_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        states = simplified_trajectory["states"]
        full_action_sequence = _full_action_sequence_text(states)
        outcome_value = simplified_trajectory.get("outcome")
        outcome_text = outcome_value.strip() if isinstance(outcome_value, str) and outcome_value.strip() else "<unknown>"
        pre_state = states[spec["pre_state_index"]]
        post_state = states[spec["post_state_index"]]
        pre_state_text = _state_text(pre_state)
        post_state_text = _state_text(post_state)
        if truncate_state_chars is not None:
            pre_state_text = _truncate_middle(pre_state_text, truncate_state_chars)
            post_state_text = _truncate_middle(post_state_text, truncate_state_chars)
        header_lines = [
            "Generate retrieval-ready event text for the requested transition only.",
            f"Goal: {simplified_trajectory['goal']}",
            f"Outcome: {outcome_text}",
            f"Start URL: {simplified_trajectory['start_url']}",
            (
                f"Workflow position: transition {spec['transition_number']} of {spec['transition_count']}"
            ),
            "Actions are attached to destination states in this dataset.",
            "Use the full trajectory context to place this step in the workflow.",
            "Ground the detailed state comparison only in the target pre-state and post-state evidence below.",
            "Preserve exact visible labels, options, statuses, values, counts, or confirmation/blocking signals when supported by the evidence.",
            "Return JSON only.",
            "",
            "Target transition:",
            (
                f"{spec['event_id']}: state {spec['pre_state_index']} "
                f"-> {spec['local_action']} "
                f"-> state {spec['post_state_index']}"
            ),
            "",
            "Full annotated action trace:",
            full_action_sequence,
        ]
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(header_lines)}]
        content_parts.append(
            {
                "type": "text",
                "text": (
                    f"Pre-state {pre_state['state_index']}\n"
                    f"URL: {_state_url(pre_state)}\n"
                    f"Thought visible in this state: {spec['pre_thoughts']}\n"
                    f"Incoming annotated action attached to this state: {spec['pre_incoming_action']}\n"
                    f"Full AXTree:\n{pre_state_text}"
                ),
            }
        )
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(trajectory_dir / pre_state["screenshot"])},
            }
        )
        content_parts.append(
            {
                "type": "text",
                "text": (
                    f"Post-state {post_state['state_index']}\n"
                    f"URL: {_state_url(post_state)}\n"
                    f"Thought visible in this state: {spec['post_thoughts']}\n"
                    f"Incoming annotated action attached to this state: {spec['local_action']}\n"
                    f"Annotated transition action that produced this post-state: {spec['local_action']}\n"
                    f"Full AXTree:\n{post_state_text}"
                ),
            }
        )
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _to_data_url(trajectory_dir / post_state["screenshot"])},
            }
        )
        return [
            {"role": "system", "content": EVENT_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

    def _build_event_repair_messages(
        self,
        *,
        draft_response: str,
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": EVENT_GENERATION_REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Convert this draft into the required JSON object.\n"
                    "Return JSON only.\n"
                    f"{draft_response}"
                ),
            },
        ]

    def _fallback_event_fields(self, spec: dict[str, Any]) -> dict[str, str]:
        overview = (
            f"Goal recap: {spec['goal']} This transition is step "
            f"{spec['transition_number']} of {spec['transition_count']} in the workflow."
        ).strip()
        state_transition = (
            f"After action {spec['local_action']}, the interface moved from {spec['pre_url']} "
            f"to {spec['post_url']}. Visible state evidence was limited to the stored pre/post "
            f"thoughts: pre={spec['pre_thoughts']} post={spec['post_thoughts']}."
        ).strip()
        return {
            "overview": overview,
            "state_transition": state_transition,
        }

    def _materialize_event_description(self, event_fields: dict[str, str]) -> str:
        return f"{event_fields['overview']}\n\n{event_fields['state_transition']}".strip()

    async def _extract_single_event_fields_async(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
        spec: dict[str, Any],
    ) -> dict[str, str]:
        primary_text = ""
        try:
            primary_text = await self._call_controller_text_async(
                self._build_event_generation_messages(
                    trajectory_dir=trajectory_dir,
                    simplified_trajectory=simplified_trajectory,
                    spec=spec,
                    truncate_state_chars=None,
                ),
                disable_thinking=True,
            )
        except Exception:
            try:
                primary_text = await self._call_controller_text_async(
                    self._build_event_generation_messages(
                        trajectory_dir=trajectory_dir,
                        simplified_trajectory=simplified_trajectory,
                        spec=spec,
                        truncate_state_chars=EVENT_STATE_FALLBACK_MAX_AXTREE_CHARS,
                    ),
                    disable_thinking=True,
                )
            except Exception:
                return self._fallback_event_fields(spec)

        parsed = _parse_single_event_response(primary_text)
        if not parsed and primary_text.strip():
            try:
                repaired_text = await self._call_controller_text_async(
                    self._build_event_repair_messages(
                        draft_response=primary_text,
                    ),
                    disable_thinking=True,
                )
                repaired_fields = _parse_single_event_response(repaired_text)
                if repaired_fields:
                    parsed = repaired_fields
            except Exception:
                pass
        return parsed or self._fallback_event_fields(spec)

    async def _extract_event_fields_batch_async(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
        transition_specs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        semaphore = asyncio.Semaphore(EVENT_ASYNC_MAX_CONCURRENCY)

        async def _run_single(spec: dict[str, Any]) -> dict[str, str]:
            async with semaphore:
                return await self._extract_single_event_fields_async(
                    trajectory_dir=trajectory_dir,
                    simplified_trajectory=simplified_trajectory,
                    spec=spec,
                )

        return await asyncio.gather(*[_run_single(spec) for spec in transition_specs])

    def _run_async(self, coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError("agentrunbook_r async event extraction does not support an active event loop")

    def _build_event_entries(
        self,
        *,
        trajectory_dir: Path,
        simplified_trajectory: dict[str, Any],
    ) -> list[dict[str, Any]]:
        transition_specs = self._build_event_transition_specs(simplified_trajectory)
        if not transition_specs:
            return []
        event_fields_by_spec = self._run_async(
            self._extract_event_fields_batch_async(
                trajectory_dir=trajectory_dir,
                simplified_trajectory=simplified_trajectory,
                transition_specs=transition_specs,
            )
        )

        entries: list[dict[str, Any]] = []
        for spec, event_fields in zip(transition_specs, event_fields_by_spec):
            description = self._materialize_event_description(event_fields)
            entries.append(
                {
                    "entry_id": f"{spec['trajectory_id']}:{spec['event_id']}",
                    "trajectory_id": spec["trajectory_id"],
                    "event_id": spec["event_id"],
                    "goal": spec["goal"],
                    "outcome": spec["outcome"],
                    "pre_state_index": spec["pre_state_index"],
                    "post_state_index": spec["post_state_index"],
                    "local_action": spec["local_action"],
                    "transition_number": spec["transition_number"],
                    "transition_count": spec["transition_count"],
                    "local_urls": [spec["pre_url"], spec["post_url"]],
                    "pre_thoughts": spec["pre_thoughts"],
                    "post_thoughts": spec["post_thoughts"],
                    "pre_screenshot": spec["pre_screenshot"],
                    "post_screenshot": spec["post_screenshot"],
                    "overview": event_fields["overview"],
                    "state_transition": event_fields["state_transition"],
                    "description": description,
                }
            )
        return entries

    def _build_preview_payload(self) -> dict[str, Any]:
        def _raw_state_example(entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "entry_id": entry["entry_id"],
                "trajectory_id": entry["trajectory_id"],
                "slice_state_indexes": list(entry["slice_state_indexes"]),
                "slice_action_sequence": _truncate_for_preview(entry["slice_action_sequence"]),
                "slice_axtree_text": _truncate_for_preview(entry["slice_axtree_text"]),
            }

        def _event_example(entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "entry_id": entry["entry_id"],
                "trajectory_id": entry["trajectory_id"],
                "pre_state_index": entry["pre_state_index"],
                "post_state_index": entry["post_state_index"],
                "local_action": entry["local_action"],
                "overview": _truncate_for_preview(entry["overview"]),
                "state_transition": _truncate_for_preview(entry["state_transition"]),
                "description": _truncate_for_preview(entry["description"]),
            }

        def _note_example(entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "entry_id": entry["entry_id"],
                "trajectory_id": entry["trajectory_id"],
                "title": entry["title"],
                "description": entry["description"],
                "content": _truncate_for_preview(entry["content"]),
            }

        return {
            "memory_type": self.memory_type,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir is not None else None,
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "controller": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
                "prompt_versions": {
                    "events": EVENT_PROMPT_VERSION,
                    "event_repair": EVENT_REPAIR_PROMPT_VERSION,
                    "notes": NOTE_GENERATION_PROMPT_VERSION,
                },
            },
            "embedding": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "query_instruction": self.embedding_query_instruction,
            },
            "counts": {
                "trajectories": len(self.inserted_trajectory_ids),
                "raw_state_entries": len(self.raw_state_entries),
                "event_entries": len(self.event_entries),
                "procedure_note_entries": len(self.procedure_note_entries),
                "hint_note_entries": len(self.hint_note_entries),
            },
            "pools": {
                "raw_state": {
                    "description": "High-fidelity raw AXTree state slices retrieved by semantic similarity over slice AXTree text only.",
                    "examples": [_raw_state_example(entry) for entry in self.raw_state_entries[:PREVIEW_EXAMPLE_COUNT]],
                },
                "events": {
                    "description": (
                        "Adjacent trajectory transitions with per-transition controller-generated "
                        "workflow-overview and state-change descriptions plus screenshot evidence."
                    ),
                    "examples": [_event_example(entry) for entry in self.event_entries[:PREVIEW_EXAMPLE_COUNT]],
                },
                "procedure_notes": {
                    "description": "Trajectory-level reusable procedure notes.",
                    "examples": [
                        _note_example(entry) for entry in self.procedure_note_entries[:PREVIEW_EXAMPLE_COUNT]
                    ],
                },
                "hint_notes": {
                    "description": "Trajectory-level reusable hint notes.",
                    "examples": [_note_example(entry) for entry in self.hint_note_entries[:PREVIEW_EXAMPLE_COUNT]],
                },
            },
        }

    def _build_index_payload(self) -> dict[str, Any]:
        embedding_dims = {
            "raw_state": int(self.raw_state_embeddings.shape[1]) if self.raw_state_embeddings.size else 0,
            "events": int(self.event_embeddings.shape[1]) if self.event_embeddings.size else 0,
            "procedure_notes": (
                int(self.procedure_note_embeddings.shape[1]) if self.procedure_note_embeddings.size else 0
            ),
            "hint_notes": int(self.hint_note_embeddings.shape[1]) if self.hint_note_embeddings.size else 0,
        }
        domain = "unknown"
        if self.workspace_dir is not None and self.inserted_trajectory_ids:
            first_path = self.workspace_dir / "trajectories" / self.inserted_trajectory_ids[0] / "trajectory.json"
            if first_path.exists():
                payload = load_json(first_path)
                if isinstance(payload, dict):
                    domain = _domain_from_url(str(payload.get("start_url", "")))
        return {
            "memory_type": self.memory_type,
            "domain": domain,
            "inserted_trajectory_ids": list(self.inserted_trajectory_ids),
            "trajectory_count": len(self.inserted_trajectory_ids),
            "entry_counts": {
                "raw_state": len(self.raw_state_entries),
                "events": len(self.event_entries),
                "procedure_notes": len(self.procedure_note_entries),
                "hint_notes": len(self.hint_note_entries),
            },
            "embedding_dimensions": embedding_dims,
            "controller_params": {
                "model": self.controller_model,
                "base_url": self.controller_base_url,
                "max_completion_tokens": self.controller_max_completion_tokens,
                "timeout_seconds": self.controller_timeout_seconds,
                "max_retries": self.controller_max_retries,
                "disable_thinking": self.controller_disable_thinking,
                "temperature": self.controller_temperature,
                "top_p": self.controller_top_p,
                "top_k": self.controller_top_k,
            },
            "embedding_params": {
                "model": self.embedding_model,
                "base_url": self.embedding_base_url,
                "max_input_tokens": self.embedding_max_input_tokens,
                "query_instruction": self.embedding_query_instruction,
            },
            "trajectory_pool_root": (
                str(self.trajectory_pool_root) if self.trajectory_pool_root is not None else None
            ),
            "index_params": {
                "raw_state_slice_radius": self.raw_state_slice_radius,
            },
            "query_params": {
                "max_raw_state_queries": self.max_raw_state_queries,
                "query_generation_disable_thinking": self.query_generation_disable_thinking,
            },
            "retrieval_params": {
                "raw_state_search_top_k_per_query": self.raw_state_search_top_k_per_query,
                "event_search_top_k": self.event_search_top_k,
                "note_search_top_k_per_type": self.note_search_top_k_per_type,
                "raw_state_result_merge_budget": self.raw_state_result_merge_budget,
                "raw_state_result_merge_per_query_cap": self.raw_state_result_merge_per_query_cap,
                "rerank_candidate_limit": self.rerank_candidate_limit,
                "enable_rerank": self.enable_rerank,
            },
        }

    def _save_backend(self, output_dir: Path) -> None:
        self._ensure_workspace_layout(output_dir)
        save_json(output_dir / "index.json", self._build_index_payload())
        _write_jsonl(output_dir / "pools" / "raw_state_pool.jsonl", self.raw_state_entries)
        _write_jsonl(output_dir / "pools" / "event_pool.jsonl", self.event_entries)
        _write_jsonl(output_dir / "pools" / "procedure_note_pool.jsonl", self.procedure_note_entries)
        _write_jsonl(output_dir / "pools" / "hint_note_pool.jsonl", self.hint_note_entries)
        np.save(output_dir / "embeddings" / "raw_state.npy", self.raw_state_embeddings)
        np.save(output_dir / "embeddings" / "event.npy", self.event_embeddings)
        np.save(output_dir / "embeddings" / "procedure_notes.npy", self.procedure_note_embeddings)
        np.save(output_dir / "embeddings" / "hint_notes.npy", self.hint_note_embeddings)
        save_json(output_dir / "previews" / "query_prompt_preview.json", self._build_preview_payload())

        if self.workspace_dir is not None and self.workspace_dir.resolve() != output_dir.resolve():
            src_trajectories_dir = self.workspace_dir / "trajectories"
            dst_trajectories_dir = output_dir / "trajectories"
            if dst_trajectories_dir.exists():
                shutil.rmtree(dst_trajectories_dir)
            shutil.copytree(src_trajectories_dir, dst_trajectories_dir)

    def _load_backend(self, input_dir: Path) -> None:
        self.workspace_dir = input_dir.resolve()
        self._ensure_workspace_layout(self.workspace_dir)
        index_payload = load_json(self.workspace_dir / "index.json")
        require(isinstance(index_payload, dict), "agentrunbook_r index.json must be an object")
        inserted_ids = index_payload.get("inserted_trajectory_ids")
        require(
            isinstance(inserted_ids, list) and all(isinstance(item, str) and item for item in inserted_ids),
            "agentrunbook_r index.json must contain inserted_trajectory_ids as non-empty strings",
        )
        self.inserted_trajectory_ids = list(inserted_ids)
        self.raw_state_entries = _read_jsonl(self.workspace_dir / "pools" / "raw_state_pool.jsonl")
        self.event_entries = _read_jsonl(self.workspace_dir / "pools" / "event_pool.jsonl")
        self.procedure_note_entries = _read_jsonl(self.workspace_dir / "pools" / "procedure_note_pool.jsonl")
        self.hint_note_entries = _read_jsonl(self.workspace_dir / "pools" / "hint_note_pool.jsonl")
        self.raw_state_embeddings = np.load(self.workspace_dir / "embeddings" / "raw_state.npy")
        self.event_embeddings = np.load(self.workspace_dir / "embeddings" / "event.npy")
        self.procedure_note_embeddings = np.load(self.workspace_dir / "embeddings" / "procedure_notes.npy")
        self.hint_note_embeddings = np.load(self.workspace_dir / "embeddings" / "hint_notes.npy")
        self._stored_goal_snippets = []
        for trajectory_id in self.inserted_trajectory_ids:
            trajectory = self._load_stored_trajectory(trajectory_id)
            goal_value = trajectory.get("goal")
            if isinstance(goal_value, str) and goal_value.strip():
                goal_snippet = _first_clause(goal_value, 100)
                if goal_snippet:
                    self._stored_goal_snippets.append(goal_snippet)
            if self._runtime_domain == "unknown":
                start_url = trajectory.get("start_url")
                if isinstance(start_url, str) and start_url.strip():
                    self._runtime_domain = _domain_from_url(start_url)
        if self._runtime_domain == "unknown":
            domain_value = index_payload.get("domain")
            if isinstance(domain_value, str) and domain_value.strip():
                self._runtime_domain = domain_value.strip()
        self._refresh_runtime_query_summary()
