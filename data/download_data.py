#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "xiaowu0162/longmemeval-v2"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "longmemeval-v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the LongMemEval-V2 dataset from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository id")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory where the dataset snapshot should be downloaded",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing data-root before downloading",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if args.force and data_root.exists():
        shutil.rmtree(data_root)
    if (data_root / "questions.jsonl").exists() and (data_root / "trajectories.jsonl").exists():
        print(
            json.dumps(
                {
                    "repo_id": args.repo_id,
                    "repo_type": "dataset",
                    "revision": args.revision,
                    "data_root": str(data_root),
                    "status": "already_present",
                },
                indent=2,
            )
        )
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface_hub. Install project requirements before running data/download_data.py."
        ) from exc

    data_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(data_root),
    )
    require((data_root / "questions.jsonl").exists(), f"Download did not create {data_root / 'questions.jsonl'}")
    require((data_root / "trajectories.jsonl").exists(), f"Download did not create {data_root / 'trajectories.jsonl'}")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "repo_type": "dataset",
                "revision": args.revision,
                "data_root": str(data_root),
                "status": "downloaded",
                "next": [
                    f"python data/prepare_data.py --data-root {data_root}",
                    f"python data/validate_data.py --data-root {data_root} --tier small",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
