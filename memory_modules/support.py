import ast
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from .memory import require


A11Y_LINE_RE = re.compile(r"^\s*\[([A-Za-z0-9_-]+)\]\s*(.+)$")
ACTION_OBJECT_ID_RE = re.compile(r"^[A-Za-z]*\d+[A-Za-z0-9_-]*$")
NOTE_GENERATION_PROMPT_VERSION = "qwen_v6_retrieval_safe"
NOTE_GENERATION_SYSTEM_PROMPT = """You convert one UI task trajectory into two reusable memory notes for a future agent.

Assume these notes will later be retrieved for unknown future questions.
You do not know the downstream question in advance.
Your job is to preserve the workflow and the highest-value reusable facts from the touched pages.

Write:
1. procedure_note
2. hint_note

Each note must be an object with:
- title: a short retrieval-friendly title with app/module/task context
- description: 1 short sentence describing what the note is about
- content: a bullet list string using '- ' lines

Rules:
- Use only evidence grounded in the provided goal, outcome, thoughts, annotated actions, and screenshots.
- Never write a fact unless it is directly supported by the observed run.
- Mention application / page / module names when they are visible from the actions or screenshots.
- Do not invent unseen fields, filters, modules, or outcomes.
- Prefer exact literal UI strings over paraphrases whenever a label, tab, button, menu item, module, or option is visible.
- If the run failed, procedure_note may describe the intended or attempted workflow only where the evidence supports it. Do not pretend the task succeeded.
- For failed runs, use hint_note to explain what may trip an agent up or what signal in the UI matters.
- Do not mention screenshot numbers, state numbers, or the word trajectory.
- Do not copy the internal thoughts verbatim line-by-line. Distill them into useful notes.
- procedure_note should capture the reliable core workflow only.
- hint_note should preserve only durable, high-value facts from the touched pages.
- Prefer high-signal facts that are likely to help later retrieval.
- Keep procedure_note.content to 4 to 8 bullets.
- Keep hint_note.content to 6 to 12 bullets when the evidence supports it.
- Do not output analysis, reasoning, headings, markdown fences, or any text before or after the JSON object.
- Start your answer with { and end your answer with }.

Return only valid JSON in this shape:
{"procedure_note":{"title":"...","description":"...","content":"- ...\\n- ..."},"hint_note":{"title":"...","description":"...","content":"- ...\\n- ..."}}"""
NOTE_GENERATION_REPAIR_SYSTEM_PROMPT = """Rewrite the draft into valid JSON only.

Rules:
- Return exactly one JSON object with keys procedure_note and hint_note.
- Each of those keys must map to an object with keys title, description, and content.
- title must be a short retrieval-friendly title.
- description must be one short sentence.
- content must be a string containing bullet lines using '- '.
- Do not output analysis, commentary, markdown fences, or extra keys.
- Start with { and end with }."""


def _load_api_key(api_key_env: str, api_key_file: str | None) -> str | None:
    env_value = os.getenv(api_key_env)
    if env_value:
        return env_value
    if api_key_file is None:
        return None
    path = Path(api_key_file)
    require(path.exists(), f"Missing API key file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    require(value, f"Empty API key file: {path}")
    return value


def _to_data_url(path: Path) -> str:
    require(path.exists(), f"Missing image file: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def _extract_object_lookup_from_tree(tree_text: str) -> dict[str, str]:
    object_lookup: dict[str, str] = {}
    for line in tree_text.splitlines():
        match = A11Y_LINE_RE.match(line)
        if match and match.group(1) not in object_lookup:
            object_lookup[match.group(1)] = line.strip()
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
    if not isinstance(parsed, ast.Expression) or not isinstance(parsed.body, ast.Call):
        return []
    call = parsed.body
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


def _extract_content_text(content: Any) -> str | None:
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_val = item.get("text")
            else:
                text_val = getattr(item, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                parts.append(text_val.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    return None


def _extract_response_text_parts(message: Any) -> tuple[str | None, str | None]:
    content_text = _extract_content_text(getattr(message, "content", None))
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return content_text, reasoning.strip()
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return content_text, reasoning_content.strip()
    return content_text, None


def _normalize_note_object(raw_note: Any) -> dict[str, str] | None:
    if not isinstance(raw_note, dict):
        return None
    title = raw_note.get("title")
    description = raw_note.get("description")
    content = raw_note.get("content")
    normalized_content: str | None = None
    if isinstance(content, str) and content.strip():
        normalized_content = content.strip()
    elif isinstance(content, list):
        lines = [line.strip() for line in content if isinstance(line, str) and line.strip()]
        if lines:
            normalized_content = "\n".join(
                line if line.startswith("- ") else f"- {line.lstrip('- ').strip()}"
                for line in lines
            )
    if not all(isinstance(value, str) and value.strip() for value in [title, description]):
        return None
    if normalized_content is None:
        return None
    return {
        "title": title.strip(),
        "description": description.strip(),
        "content": normalized_content,
    }


def _parse_generated_notes_json(response_text: str) -> tuple[
    dict[str, str] | None,
    dict[str, str] | None,
    bool,
    str | None,
]:
    stripped = response_text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    if stripped.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.S)
        if fence_match:
            candidates.append(fence_match.group(1).strip())
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1].strip())

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        procedure_note = _normalize_note_object(payload.get("procedure_note"))
        hint_note = _normalize_note_object(payload.get("hint_note"))
        if procedure_note is not None and hint_note is not None:
            return procedure_note, hint_note, True, None
    return None, None, False, "parse_failed"
