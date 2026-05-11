#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.public_data import validate_public_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LongMemEval-V2 public data files.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tier", choices=["small", "medium"], default="small")
    parser.add_argument("--check-screenshots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = validate_public_data(
        Path(args.data_root).expanduser().resolve(),
        tier=args.tier,
        check_screenshots=args.check_screenshots,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
