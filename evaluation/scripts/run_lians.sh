#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT_VALUE="${DATA_ROOT:?Set DATA_ROOT to the LongMemEval-V2 dataset directory}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-runs}"
TIER_VALUE="${TIER:-small}"

for domain in web enterprise; do
  python "$REPO_ROOT/evaluation/run_eval.py" \
    --method lians \
    --data-root "$DATA_ROOT_VALUE" \
    --domain "$domain" \
    --tier "$TIER_VALUE" \
    --output-dir "$OUTPUT_ROOT_VALUE/lians_${domain}_${TIER_VALUE}" \
    "$@"
done
