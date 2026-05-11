#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_data import prepare_screenshots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LongMemEval-V2 screenshot runtime layout.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--no-extract-archives", action="store_true")
    args = parser.parse_args()
    result = prepare_screenshots(
        Path(args.data_root).expanduser().resolve(),
        mode=args.mode,
        extract_archives=not args.no_extract_archives,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
