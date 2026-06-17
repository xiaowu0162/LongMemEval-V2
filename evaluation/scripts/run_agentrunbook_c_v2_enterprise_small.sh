#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python "$REPO_ROOT/evaluation/run_eval.py" \
  --method agentrunbook_c_v2 \
  --data-root "${DATA_ROOT:?Set DATA_ROOT to the LongMemEval-V2 dataset directory}" \
  --domain enterprise \
  --tier small \
  --output-dir "${OUTPUT_ROOT:-runs}/agentrunbook_c_v2_enterprise_small_gpt54mini_medium" \
  --openai-sdk-model "${OPENAI_SDK_MODEL:-gpt-5.4-mini}" \
  --openai-sdk-reasoning-effort "${OPENAI_SDK_REASONING_EFFORT:-medium}" \
  "$@"
