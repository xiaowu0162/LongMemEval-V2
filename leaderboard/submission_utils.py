from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any

try:
    from combine_aggregated_metrics import combine_metrics
except ModuleNotFoundError:
    from leaderboard.combine_aggregated_metrics import combine_metrics


EXPECTED_READER_MODEL_SUBSTRING = "qwen3.5-9b"
EXPECTED_EVALUATOR_MODEL_SUBSTRING = "gpt-5.2"
REQUIRED_RUN_FILES = (
    "aggregated_metrics.json",
    "per_question.jsonl",
    "run_args.json",
)
REQUIRED_OPERATING_POINT_FILES = (
    "metric_overview.json",
    "operating_point_metadata.json",
)
QUESTION_ID_KEYS = ("id", "question_id")
SUPPORTED_TIERS = {"small", "medium"}
SAFE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


class SubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunValidation:
    run_dir: Path
    domain: str
    method: str
    tier: str
    question_count: int
    question_type_counts: dict[str, int]
    run_args: dict[str, Any]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class OperatingPointValidation:
    path: Path
    name: str
    submission_name: str
    method: str
    tier: str
    metric_overview: dict[str, Any]
    metadata: dict[str, Any]
    question_ids_by_domain: dict[str, list[str]]
    haystack_by_domain: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SubmissionError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise SubmissionError(f"Expected object in {path}:{line_number}")
        rows.append(row)
    return rows


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SubmissionError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SubmissionError(f"Missing {label}: {path}")


def require_single_file(path: Path, label: str) -> None:
    if path.is_dir():
        raise SubmissionError(f"{label} must be a single file, not a directory: {path}")
    require_file(path, label)


def validate_safe_name(value: str, label: str) -> None:
    if not SAFE_NAME_RE.fullmatch(value):
        raise SubmissionError(
            f"{label} may contain only letters, numbers, '.', '_', and '-'"
        )


def validate_tier(tier: str) -> None:
    if tier not in SUPPORTED_TIERS:
        supported = ", ".join(sorted(SUPPORTED_TIERS))
        raise SubmissionError(f"Unsupported tier {tier!r}; expected one of: {supported}")


def question_id(row: dict[str, Any], path: Path) -> str:
    for key in QUESTION_ID_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)
    raise SubmissionError(f"Missing question id in {path}")


def check_no_duplicate_ids(ids: list[str], label: str, path: Path) -> None:
    counts = Counter(ids)
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise SubmissionError(f"Duplicate {label} ids in {path}: {preview}")


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def infer_method(run_dir: Path, domain: str, tier: str) -> str | None:
    suffix = f"_{domain}_{tier}"
    if run_dir.name.endswith(suffix) and len(run_dir.name) > len(suffix):
        return run_dir.name[: -len(suffix)]
    return None


def get_method(
    run_dir: Path,
    run_args: dict[str, Any],
    domain: str,
    tier: str,
    method_override: str | None = None,
) -> str:
    inferred_method = infer_method(run_dir, domain, tier)
    if method_override:
        if run_args.get("method") and run_args["method"] != method_override:
            raise SubmissionError(
                f"run_args method for {run_dir} does not match the requested method: "
                f"{run_args['method']} != {method_override}"
            )
        if run_args.get("tier") and run_args["tier"] != tier:
            raise SubmissionError(
                f"run_args tier for {run_dir} does not match the requested tier: "
                f"{run_args['tier']} != {tier}"
            )
        return method_override

    method = run_args.get("method") or inferred_method
    if not method:
        raise SubmissionError(
            f"Could not determine method for {run_dir}; use a "
            f"<method>_{domain}_{tier} directory name or include method in run_args.json."
        )
    if inferred_method and run_args.get("method") and run_args["method"] != inferred_method:
        raise SubmissionError(
            f"run_args method for {run_dir} does not match directory name: "
            f"{run_args['method']} != {inferred_method}"
        )
    if run_args.get("tier") and run_args["tier"] != tier:
        raise SubmissionError(
            f"run_args tier for {run_dir} does not match the requested tier: "
            f"{run_args['tier']} != {tier}"
        )
    return str(method)


def validate_model_fields(run_dir: Path, run_args: dict[str, Any]) -> None:
    model = str(run_args.get("model", "")).lower()
    evaluator_model = str(run_args.get("evaluator_model", "")).lower()
    if EXPECTED_READER_MODEL_SUBSTRING not in model:
        raise SubmissionError(
            f"{run_dir}/run_args.json model must contain "
            f"{EXPECTED_READER_MODEL_SUBSTRING!r}"
        )
    if EXPECTED_EVALUATOR_MODEL_SUBSTRING not in evaluator_model:
        raise SubmissionError(
            f"{run_dir}/run_args.json evaluator_model must contain "
            f"{EXPECTED_EVALUATOR_MODEL_SUBSTRING!r}"
        )


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = read_json(path)
    if not isinstance(questions, list):
        raise SubmissionError(f"Expected list in {path}")
    for index, row in enumerate(questions):
        if not isinstance(row, dict):
            raise SubmissionError(f"Expected object at {path}[{index}]")
    return questions


def sorted_question_ids(path: Path) -> list[str]:
    questions = load_questions(path)
    ids = [question_id(row, path) for row in questions]
    check_no_duplicate_ids(ids, "runtime input question", path)
    return sorted(ids)


def validate_run(
    run_dir: Path,
    expected_domain: str,
    tier: str,
    method_override: str | None = None,
) -> RunValidation:
    validate_tier(tier)
    run_dir = run_dir.resolve()
    require_dir(run_dir, f"{expected_domain} run directory")
    for filename in REQUIRED_RUN_FILES:
        require_file(run_dir / filename, filename)
    runtime_inputs_dir = run_dir / "runtime_inputs"
    require_dir(runtime_inputs_dir, "runtime_inputs directory")
    questions_path = runtime_inputs_dir / "questions.json"
    require_file(questions_path, "runtime_inputs/questions.json")
    require_file(runtime_inputs_dir / "haystack.json", "runtime_inputs/haystack.json")

    run_args = read_json(run_dir / "run_args.json")
    metrics = read_json(run_dir / "aggregated_metrics.json")
    questions = load_questions(questions_path)
    records = read_jsonl(run_dir / "per_question.jsonl")
    if not isinstance(run_args, dict):
        raise SubmissionError(f"Expected object in {run_dir / 'run_args.json'}")
    if not isinstance(metrics, dict):
        raise SubmissionError(f"Expected object in {run_dir / 'aggregated_metrics.json'}")

    if run_args.get("domain") != expected_domain:
        raise SubmissionError(
            f"{run_dir}/run_args.json domain must be {expected_domain!r}, "
            f"got {run_args.get('domain')!r}"
        )
    question_domains = {row.get("domain") for row in questions}
    if question_domains and question_domains != {expected_domain}:
        raise SubmissionError(
            f"{questions_path} contains domains {question_domains}, "
            f"expected only {expected_domain!r}"
        )

    validate_model_fields(run_dir, run_args)
    method = get_method(run_dir, run_args, expected_domain, tier, method_override)

    question_ids = [question_id(row, questions_path) for row in questions]
    record_ids = [question_id(row, run_dir / "per_question.jsonl") for row in records]
    check_no_duplicate_ids(question_ids, "runtime input question", questions_path)
    check_no_duplicate_ids(
        record_ids,
        "per-question output",
        run_dir / "per_question.jsonl",
    )
    missing = sorted(set(question_ids) - set(record_ids))
    extra = sorted(set(record_ids) - set(question_ids))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise SubmissionError(
            f"{run_dir} does not cover the runtime questions: {'; '.join(details)}"
        )

    missing_question_types = [
        question_id(row, questions_path)
        for row in questions
        if not row.get("question_type")
    ]
    missing_record_types = [
        question_id(row, run_dir / "per_question.jsonl")
        for row in records
        if not row.get("question_type")
    ]
    if missing_question_types or missing_record_types:
        raise SubmissionError(
            f"{run_dir} has missing question_type values: "
            f"runtime_inputs={missing_question_types[:5]} "
            f"per_question={missing_record_types[:5]}"
        )

    question_type_counts = Counter(str(row.get("question_type")) for row in questions)
    record_type_counts = Counter(str(row.get("question_type")) for row in records)
    if question_type_counts != record_type_counts:
        raise SubmissionError(
            f"{run_dir} question_type counts do not match runtime inputs: "
            f"expected={counter_to_dict(question_type_counts)} "
            f"got={counter_to_dict(record_type_counts)}"
        )

    overall = metrics.get("overall")
    if not isinstance(overall, dict):
        raise SubmissionError(
            f"{run_dir}/aggregated_metrics.json is missing overall metrics"
        )
    count_all_questions = overall.get("count_all_questions")
    if count_all_questions != len(questions) or count_all_questions != len(records):
        raise SubmissionError(
            f"{run_dir}/aggregated_metrics.json count_all_questions={count_all_questions} "
            f"does not match questions={len(questions)} and per_question={len(records)}"
        )

    return RunValidation(
        run_dir=run_dir,
        domain=expected_domain,
        method=method,
        tier=tier,
        question_count=len(questions),
        question_type_counts=counter_to_dict(question_type_counts),
        run_args=run_args,
        metrics=metrics,
    )


def validate_run_pair(web: RunValidation, enterprise: RunValidation) -> None:
    if web.method != enterprise.method:
        raise SubmissionError(
            f"Run methods do not match: {web.method} != {enterprise.method}"
        )
    if web.tier != enterprise.tier:
        raise SubmissionError(
            f"Run tiers do not match: {web.tier} != {enterprise.tier}"
        )


def copy_run_artifacts(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for filename in REQUIRED_RUN_FILES:
        shutil.copy2(source / filename, target / filename)
    shutil.copytree(source / "runtime_inputs", target / "runtime_inputs")


def copy_root_file(source: Path, target: Path, overwrite: bool = False) -> None:
    if target.exists():
        if not overwrite:
            raise SubmissionError(
                f"Refusing to overwrite package root file: {target.name}"
            )
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copy2(source, target)


def get_required_metric(metrics: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = metrics
    for key in path:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(path)
            raise SubmissionError(f"Combined metrics missing {joined}")
        value = value[key]
    return value


def build_metric_overview(combined_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_full_set": get_required_metric(
            combined_metrics,
            ("overall", "overall_full_set"),
        ),
        "gotchas_accuracy": get_required_metric(
            combined_metrics,
            ("non_abstention_by_category", "gotchas", "pct_correct"),
        ),
        "static_accuracy": get_required_metric(
            combined_metrics,
            ("combined_abstention_by_category", "static", "pct_correct"),
        ),
        "dynamic_accuracy": get_required_metric(
            combined_metrics,
            ("combined_abstention_by_category", "dynamic", "pct_correct"),
        ),
        "procedure_accuracy": get_required_metric(
            combined_metrics,
            ("combined_abstention_by_category", "procedure", "pct_correct"),
        ),
        "memory_query_avg_seconds": get_required_metric(
            combined_metrics,
            ("memory_query", "avg_seconds"),
        ),
    }


def combine_domain_metrics(web: RunValidation, enterprise: RunValidation) -> dict[str, Any]:
    return combine_metrics(
        web.metrics,
        enterprise.metrics,
        web.run_dir / "aggregated_metrics.json",
        enterprise.run_dir / "aggregated_metrics.json",
    )


def run_metadata(run: RunValidation) -> dict[str, Any]:
    return {
        "source_run_dir": str(run.run_dir),
        "domain": run.domain,
        "question_count": run.question_count,
        "question_type_counts": run.question_type_counts,
        "model": run.run_args.get("model"),
        "evaluator_model": run.run_args.get("evaluator_model"),
    }


def ensure_regular_files(package_dir: Path) -> None:
    for path in package_dir.rglob("*"):
        if path.is_symlink():
            raise SubmissionError(f"Package contains symlink: {path}")
        if not path.is_file() and not path.is_dir():
            raise SubmissionError(
                f"Package contains unsupported filesystem entry: {path}"
            )


def create_tarball(package_dir: Path, archive_path: Path) -> None:
    ensure_regular_files(package_dir)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(package_dir, arcname=package_dir.name)


def validate_operating_point_dir(
    path: Path,
    expected_submission_name: str,
) -> OperatingPointValidation:
    path = path.resolve()
    require_dir(path, "operating point directory")
    for filename in REQUIRED_OPERATING_POINT_FILES:
        require_file(path / filename, filename)
    for domain in ("web", "enterprise"):
        require_dir(path / domain, f"{domain} operating point directory")
        for filename in REQUIRED_RUN_FILES:
            require_file(path / domain / filename, f"{domain}/{filename}")
        require_file(
            path / domain / "runtime_inputs" / "questions.json",
            f"{domain}/runtime_inputs/questions.json",
        )
        require_file(
            path / domain / "runtime_inputs" / "haystack.json",
            f"{domain}/runtime_inputs/haystack.json",
        )

    metric_overview = read_json(path / "metric_overview.json")
    metadata = read_json(path / "operating_point_metadata.json")
    if not isinstance(metric_overview, dict):
        raise SubmissionError(f"Expected object in {path / 'metric_overview.json'}")
    if not isinstance(metadata, dict):
        raise SubmissionError(
            f"Expected object in {path / 'operating_point_metadata.json'}"
        )

    name = str(metadata.get("operating_point_name") or path.name)
    if name != path.name:
        raise SubmissionError(
            f"{path} name does not match operating_point_metadata.json: "
            f"{path.name} != {name}"
        )
    if metadata.get("submission_name") != expected_submission_name:
        raise SubmissionError(
            f"{path} belongs to submission {metadata.get('submission_name')!r}, "
            f"expected {expected_submission_name!r}"
        )
    method = metadata.get("method")
    tier = metadata.get("tier")
    if not isinstance(method, str) or not method:
        raise SubmissionError(
            f"{path} is missing method in operating_point_metadata.json"
        )
    if not isinstance(tier, str):
        raise SubmissionError(f"{path} is missing tier in operating_point_metadata.json")
    validate_tier(tier)

    question_ids_by_domain = {
        domain: sorted_question_ids(path / domain / "runtime_inputs" / "questions.json")
        for domain in ("web", "enterprise")
    }
    haystack_by_domain = {
        domain: read_json(path / domain / "runtime_inputs" / "haystack.json")
        for domain in ("web", "enterprise")
    }
    return OperatingPointValidation(
        path=path,
        name=name,
        submission_name=expected_submission_name,
        method=method,
        tier=tier,
        metric_overview=metric_overview,
        metadata=metadata,
        question_ids_by_domain=question_ids_by_domain,
        haystack_by_domain=haystack_by_domain,
    )
