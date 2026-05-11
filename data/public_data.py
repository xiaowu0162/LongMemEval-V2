from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.exists(), f"Missing JSONL file: {path}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        require(isinstance(payload, dict), f"Line {line_no} in {path} is not a JSON object")
        records.append(payload)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_questions(data_root: Path, domain: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(data_root / "questions.jsonl")
    if domain is not None:
        rows = [row for row in rows if row.get("domain") == domain]
    return rows


def load_trajectories(data_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(data_root / "trajectories.jsonl"):
        trajectory_id = row.get("id")
        require(isinstance(trajectory_id, str) and trajectory_id, "Invalid trajectory id")
        require(trajectory_id not in out, f"Duplicate trajectory id: {trajectory_id}")
        out[trajectory_id] = row
    return out


def load_haystack(data_root: Path, tier: str) -> dict[str, list[str]]:
    path = data_root / "haystacks" / f"lme_v2_{tier}.json"
    require(path.exists(), f"Missing haystack file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Haystack file must be a JSON object: {path}")
    out: dict[str, list[str]] = {}
    for question_id, trajectory_ids in payload.items():
        require(isinstance(question_id, str) and question_id, "Invalid haystack question id")
        require(isinstance(trajectory_ids, list), f"Haystack for {question_id} must be a list")
        require(
            all(isinstance(item, str) and item for item in trajectory_ids),
            f"Haystack for {question_id} contains invalid trajectory ids",
        )
        out[question_id] = list(trajectory_ids)
    return out


def resolve_question_image(data_root: Path, image_value: Any) -> str | None:
    if image_value is None:
        return None
    require(isinstance(image_value, str) and image_value.strip(), "question image must be null or a string")
    path = Path(image_value)
    if not path.is_absolute():
        path = data_root / path
    require(path.exists(), f"Missing question image: {path}")
    return str(path.resolve())


def materialize_runtime_questions(
    *,
    data_root: Path,
    domain: str,
    question_ids: list[str] | None,
    limit: int | None,
    output_path: Path,
) -> list[dict[str, Any]]:
    questions = load_questions(data_root, domain=domain)
    if question_ids:
        requested = set(question_ids)
        questions = [row for row in questions if row.get("id") in requested]
        found = {str(row.get("id")) for row in questions}
        missing = requested - found
        require(not missing, f"Unknown question ids for domain {domain}: {sorted(missing)}")
    if limit is not None:
        require(limit > 0, "--limit must be positive")
        questions = questions[:limit]
    require(questions, "No questions selected")
    runtime_rows: list[dict[str, Any]] = []
    for row in questions:
        item = dict(row)
        image_path = resolve_question_image(data_root, item.get("image"))
        if image_path is not None:
            item["question"] = {"text": item["question"], "image": image_path}
        item.pop("image", None)
        runtime_rows.append(item)
    write_json(output_path, runtime_rows)
    return runtime_rows


def materialize_runtime_haystack(
    *,
    data_root: Path,
    tier: str,
    selected_questions: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, list[str]]:
    haystack = load_haystack(data_root, tier)
    selected_ids = [str(row["id"]) for row in selected_questions]
    out: dict[str, list[str]] = {}
    for question_id in selected_ids:
        require(question_id in haystack, f"Missing haystack entry for question {question_id}")
        out[question_id] = list(haystack[question_id])
    write_json(output_path, out)
    return out


def _safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as archive:
        destination_resolved = destination.resolve()
        for member in archive.getmembers():
            member_target = (destination / member.name).resolve()
            require(
                destination_resolved == member_target or destination_resolved in member_target.parents,
                f"Refusing unsafe archive member path: {member.name}",
            )
        archive.extractall(destination)


def _relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(relative_target, target_is_directory=src.is_dir())


def _link_or_copy_dir(src: Path, dst: Path, *, mode: str, replace: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not replace:
            return "skipped"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if mode == "symlink":
        try:
            _relative_symlink(src.resolve(), dst)
            return "symlinked"
        except OSError:
            shutil.copytree(src, dst)
            return "copied"
    shutil.copytree(src, dst)
    return "copied"


def prepare_screenshots(data_root: Path, *, mode: str = "symlink", extract_archives: bool = True) -> dict[str, Any]:
    require(mode in {"symlink", "copy"}, "--mode must be symlink or copy")
    screenshot_root = data_root / "screenshots"
    sources_root = data_root / "trajectory_screenshots"
    source_specs = [
        ("web_screenshots", False),
        ("enterprise_screenshots_base", False),
        ("enterprise_screenshots_patch", True),
        ("trajectory_screenshots", False),
    ]

    expanded_dirs: list[tuple[Path, bool]] = []
    if sources_root.exists():
        for name, replace in source_specs[:3]:
            source_dir = sources_root / name
            tar_path = sources_root / f"{name}.tar.gz"
            if not source_dir.exists() and tar_path.exists() and extract_archives:
                _safe_extract_tar(tar_path, source_dir)
            if source_dir.exists() and source_dir.is_dir():
                expanded_dirs.append((source_dir, replace))
    direct_sample_dir = data_root / "trajectory_screenshots"
    if direct_sample_dir.exists() and any(
        path.is_dir() and any(path.glob("*.png")) for path in direct_sample_dir.iterdir()
    ):
        expanded_dirs.append((direct_sample_dir, False))

    require(
        expanded_dirs,
        (
            "No expanded screenshot directories found. Expected screenshot directories "
            "under trajectory_screenshots/ or local tarballs that can be extracted."
        ),
    )

    counts = {"symlinked": 0, "copied": 0, "skipped": 0}
    for source_dir, replace in expanded_dirs:
        for trajectory_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            status = _link_or_copy_dir(
                trajectory_dir,
                screenshot_root / trajectory_dir.name,
                mode=mode,
                replace=replace,
            )
            counts[status] += 1
    return {
        "data_root": str(data_root),
        "screenshots_root": str(screenshot_root),
        "source_dirs": [str(path) for path, _ in expanded_dirs],
        "counts": counts,
    }


def validate_public_data(data_root: Path, *, tier: str, check_screenshots: bool = True) -> dict[str, Any]:
    require(tier in {"small", "medium"}, "--tier must be small or medium")
    questions = load_questions(data_root)
    trajectories = load_trajectories(data_root)
    haystack = load_haystack(data_root, tier)
    question_ids = {str(row["id"]) for row in questions}
    require(set(haystack).issubset(question_ids), "Haystack contains unknown question ids")
    missing_haystack = question_ids - set(haystack)
    require(not missing_haystack, f"Questions missing haystack entries: {sorted(missing_haystack)[:10]}")
    for row in questions:
        require(row.get("domain") in {"web", "enterprise"}, f"Invalid question domain: {row.get('id')}")
        require(isinstance(row.get("question"), str) and row["question"].strip(), f"Invalid question text: {row.get('id')}")
        resolve_question_image(data_root, row.get("image"))
    for question_id, trajectory_ids in haystack.items():
        question_domain = next(row["domain"] for row in questions if row["id"] == question_id)
        seen: set[str] = set()
        for trajectory_id in trajectory_ids:
            require(trajectory_id in trajectories, f"Unknown trajectory id {trajectory_id} in {question_id}")
            require(trajectory_id not in seen, f"Duplicate trajectory id {trajectory_id} in {question_id}")
            seen.add(trajectory_id)
            require(
                trajectories[trajectory_id].get("domain") == question_domain,
                f"Cross-domain trajectory {trajectory_id} in haystack for {question_id}",
            )
    missing_screenshots = 0
    if check_screenshots:
        for trajectory in trajectories.values():
            for state in trajectory.get("states", []):
                if not isinstance(state, dict):
                    continue
                screenshot_value = state.get("screenshot")
                if isinstance(screenshot_value, str) and not (data_root / screenshot_value).exists():
                    missing_screenshots += 1
    require(
        missing_screenshots == 0,
        f"Missing {missing_screenshots} trajectory screenshots. Run data/prepare_data.py first.",
    )
    return {
        "questions": len(questions),
        "trajectories": len(trajectories),
        "haystack_questions": len(haystack),
        "tier": tier,
        "check_screenshots": check_screenshots,
    }
