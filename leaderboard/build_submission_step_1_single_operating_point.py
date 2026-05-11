#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

try:
    from submission_utils import (
        SubmissionError,
        build_metric_overview,
        combine_domain_metrics,
        copy_run_artifacts,
        run_metadata,
        utc_now_iso,
        validate_run,
        validate_run_pair,
        validate_safe_name,
        validate_tier,
        write_json,
    )
except ModuleNotFoundError:
    from leaderboard.submission_utils import (
        SubmissionError,
        build_metric_overview,
        combine_domain_metrics,
        copy_run_artifacts,
        run_metadata,
        utc_now_iso,
        validate_run,
        validate_run_pair,
        validate_safe_name,
        validate_tier,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and reshape one web+enterprise operating point for a "
            "LongMemEval-V2 leaderboard submission."
        )
    )
    parser.add_argument("web_output_dir", type=Path)
    parser.add_argument("enterprise_output_dir", type=Path)
    parser.add_argument("submission_name")
    parser.add_argument("operating_point_name")
    parser.add_argument("tier", choices=["small", "medium"])
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "Override the method name recorded for this operating point. "
            "Useful when run folder names include latency labels."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("leaderboard/submissions"),
        help="Directory containing leaderboard submission workspaces.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing operating point folder.",
    )
    return parser.parse_args()


def build_operating_point(args: argparse.Namespace) -> Path:
    validate_safe_name(args.submission_name, "submission_name")
    validate_safe_name(args.operating_point_name, "operating_point_name")
    validate_tier(args.tier)

    output_root = args.output_root.resolve()
    submission_dir = output_root / args.submission_name
    operating_points_dir = submission_dir / "operating_points"
    output_dir = operating_points_dir / args.operating_point_name
    archive_path = output_root / f"{args.submission_name}.tar.gz"

    if archive_path.exists() and not args.force:
        raise SubmissionError(
            f"Existing final archive would become stale: {archive_path}. "
            "Use --force to remove it before rebuilding this operating point."
        )
    if output_dir.exists() and not args.force:
        raise SubmissionError(f"Operating point already exists: {output_dir}")
    if args.force:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if archive_path.exists():
            archive_path.unlink()

    web = validate_run(args.web_output_dir, "web", args.tier, args.method)
    enterprise = validate_run(
        args.enterprise_output_dir,
        "enterprise",
        args.tier,
        args.method,
    )
    validate_run_pair(web, enterprise)

    output_dir.mkdir(parents=True, exist_ok=False)
    copy_run_artifacts(web.run_dir, output_dir / "web")
    copy_run_artifacts(enterprise.run_dir, output_dir / "enterprise")

    combined_metrics = combine_domain_metrics(web, enterprise)
    write_json(
        output_dir / "metric_overview.json",
        build_metric_overview(combined_metrics),
    )
    write_json(
        output_dir / "operating_point_metadata.json",
        {
            "submission_name": args.submission_name,
            "operating_point_name": args.operating_point_name,
            "method": web.method,
            "tier": args.tier,
            "generated_at_utc": utc_now_iso(),
            "runs": {
                "web": run_metadata(web),
                "enterprise": run_metadata(enterprise),
            },
            "included_run_files": [
                "aggregated_metrics.json",
                "per_question.jsonl",
                "run_args.json",
                "runtime_inputs/",
            ],
        },
    )
    return output_dir


def main() -> None:
    args = parse_args()
    try:
        output_dir = build_operating_point(args)
    except SubmissionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Created operating point folder: {output_dir}")


if __name__ == "__main__":
    main()
