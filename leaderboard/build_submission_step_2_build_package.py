#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
from typing import Any

try:
    from compute_lafs import Point, lafs_summary_for_submission
    from submission_utils import (
        OperatingPointValidation,
        SubmissionError,
        copy_root_file,
        create_tarball,
        require_single_file,
        utc_now_iso,
        validate_operating_point_dir,
        validate_safe_name,
        write_json,
    )
except ModuleNotFoundError:
    from leaderboard.compute_lafs import Point, lafs_summary_for_submission
    from leaderboard.submission_utils import (
        OperatingPointValidation,
        SubmissionError,
        copy_root_file,
        create_tarball,
        require_single_file,
        utc_now_iso,
        validate_operating_point_dir,
        validate_safe_name,
        write_json,
    )


RESERVED_ROOT_NAMES = {
    "SYSTEM_DESCRIPTION.md",
    "submission_overview.json",
    "operating_points",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LongMemEval-V2 leaderboard package from one or more "
            "validated operating point folders."
        )
    )
    parser.add_argument("submission_name")
    parser.add_argument("system_description_path", type=Path)
    parser.add_argument("code_file_path", type=Path)
    parser.add_argument("operating_point_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("leaderboard/submissions"),
        help="Directory containing leaderboard submission workspaces.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace root package files, archive, and stale operating point folders.",
    )
    return parser.parse_args()


def require_numeric(value: Any, metric_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubmissionError(f"{metric_name} must be numeric, got {value!r}")
    return float(value)


def submission_point_from_overview(name: str, overview: dict[str, Any]) -> Point:
    accuracy = require_numeric(overview.get("overall_full_set"), "overall_full_set")
    latency = require_numeric(
        overview.get("memory_query_avg_seconds"),
        "memory_query_avg_seconds",
    )
    if latency <= 0:
        raise SubmissionError(
            f"memory_query_avg_seconds must be positive for operating point {name!r}"
        )
    return Point(name=name, acc=accuracy * 100.0, latency=latency)


def validate_operating_points(
    submission_name: str,
    operating_point_dirs: list[Path],
) -> list[OperatingPointValidation]:
    points = [
        validate_operating_point_dir(path, submission_name)
        for path in operating_point_dirs
    ]
    names = [point.name for point in points]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise SubmissionError(
            f"Duplicate operating point folder names: {', '.join(duplicate_names)}"
        )

    methods = {point.method for point in points}
    tiers = {point.tier for point in points}
    if len(methods) != 1:
        raise SubmissionError(
            f"Operating points use different methods: {sorted(methods)}"
        )
    if len(tiers) != 1:
        raise SubmissionError(
            f"Operating points use different tiers: {sorted(tiers)}"
        )

    reference = points[0]
    for point in points[1:]:
        for domain in ("web", "enterprise"):
            if (
                point.question_ids_by_domain[domain]
                != reference.question_ids_by_domain[domain]
            ):
                raise SubmissionError(
                    f"{point.path} uses different {domain} question ids than "
                    f"{reference.path}"
                )
            if (
                point.haystack_by_domain[domain]
                != reference.haystack_by_domain[domain]
            ):
                raise SubmissionError(
                    f"{point.path} uses a different {domain} haystack than "
                    f"{reference.path}"
                )
    return points


def prepare_package_paths(
    output_root: Path,
    submission_name: str,
    force: bool,
) -> tuple[Path, Path, Path]:
    output_root = output_root.resolve()
    package_dir = output_root / submission_name
    operating_points_dir = package_dir / "operating_points"
    archive_path = output_root / f"{submission_name}.tar.gz"
    if archive_path.exists():
        if not force:
            raise SubmissionError(
                f"Archive already exists: {archive_path}. Use --force."
            )
        archive_path.unlink()
    package_dir.mkdir(parents=True, exist_ok=True)
    operating_points_dir.mkdir(parents=True, exist_ok=True)
    return package_dir, operating_points_dir, archive_path


def copy_operating_points(
    operating_points_dir: Path,
    source_points: list[OperatingPointValidation],
    force: bool,
) -> None:
    requested_names = {point.name for point in source_points}
    existing_names = {
        path.name
        for path in operating_points_dir.iterdir()
        if path.is_dir()
    }
    extra_names = existing_names - requested_names
    if extra_names and not force:
        raise SubmissionError(
            "Submission contains operating point folders not passed to step 2: "
            f"{sorted(extra_names)}. Use --force to remove them."
        )
    if force:
        for name in extra_names:
            shutil.rmtree(operating_points_dir / name)

    for point in source_points:
        destination = operating_points_dir / point.name
        if point.path == destination.resolve():
            continue
        if destination.exists():
            if not force:
                raise SubmissionError(
                    f"Operating point destination already exists: {destination}. "
                    "Use --force to replace it."
                )
            shutil.rmtree(destination)
        shutil.copytree(point.path, destination)


def build_submission_overview(
    submission_name: str,
    system_description_name: str,
    code_file_name: str,
    archive_name: str,
    source_points: list[OperatingPointValidation],
) -> dict[str, Any]:
    method = source_points[0].method
    tier = source_points[0].tier
    lafs_points = [
        submission_point_from_overview(point.name, point.metric_overview)
        for point in source_points
    ]
    lafs = lafs_summary_for_submission(tier, lafs_points)
    return {
        "submission_name": submission_name,
        "method": method,
        "tier": tier,
        "generated_at_utc": utc_now_iso(),
        "archive_name": archive_name,
        "system_description_file": system_description_name,
        "code_file": code_file_name,
        "lafs": lafs,
        "operating_points": [
            {
                "name": point.name,
                "metric_overview_file": (
                    f"operating_points/{point.name}/metric_overview.json"
                ),
                "overall_full_set": point.metric_overview["overall_full_set"],
                "memory_query_avg_seconds": point.metric_overview[
                    "memory_query_avg_seconds"
                ],
                "lafs_accuracy_percentage_points": lafs_point.acc,
                "lafs_latency_seconds": lafs_point.latency,
            }
            for point, lafs_point in zip(source_points, lafs_points)
        ],
    }


def build_package(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_safe_name(args.submission_name, "submission_name")
    require_single_file(args.system_description_path, "SYSTEM_DESCRIPTION.md")
    require_single_file(args.code_file_path, "code file")
    if args.code_file_path.name in RESERVED_ROOT_NAMES:
        raise SubmissionError(
            f"Code file name is reserved: {args.code_file_path.name}"
        )

    source_points = validate_operating_points(
        args.submission_name,
        args.operating_point_dirs,
    )
    package_dir, operating_points_dir, archive_path = prepare_package_paths(
        args.output_root,
        args.submission_name,
        args.force,
    )
    copy_operating_points(operating_points_dir, source_points, args.force)
    copy_root_file(
        args.system_description_path,
        package_dir / "SYSTEM_DESCRIPTION.md",
        overwrite=args.force,
    )
    copy_root_file(
        args.code_file_path,
        package_dir / args.code_file_path.name,
        overwrite=args.force,
    )
    overview_path = package_dir / "submission_overview.json"
    if overview_path.exists() and not args.force:
        raise SubmissionError(f"File already exists: {overview_path}. Use --force.")
    write_json(
        overview_path,
        build_submission_overview(
            args.submission_name,
            "SYSTEM_DESCRIPTION.md",
            args.code_file_path.name,
            archive_path.name,
            source_points,
        ),
    )
    create_tarball(package_dir, archive_path)
    return package_dir, archive_path


def main() -> None:
    args = parse_args()
    try:
        package_dir, archive_path = build_package(args)
    except SubmissionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Created submission folder: {package_dir}")
    print(f"Created archive: {archive_path}")


if __name__ == "__main__":
    main()
