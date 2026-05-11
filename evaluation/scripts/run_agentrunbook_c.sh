#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$REPO_ROOT/evaluation/run_eval.py" \
  --method agentrunbook_c \
  --data-root "${DATA_ROOT:?Set DATA_ROOT to the LongMemEval-V2 dataset directory}" \
  --domain "${DOMAIN:-web}" \
  --tier "${TIER:-small}" \
  --output-dir "${OUTPUT_ROOT:-runs}/agentrunbook_c_${DOMAIN:-web}_${TIER:-small}" \
  "$@"
