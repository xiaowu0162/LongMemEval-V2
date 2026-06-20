#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
METHOD="agentrunbook_c_v2"
DATA_ROOT_VALUE="${DATA_ROOT:?Set DATA_ROOT to the LongMemEval-V2 dataset directory}"
OUTPUT_ROOT_VALUE="${OUTPUT_ROOT:-runs}"
TIER_VALUE="${TIER:-small}"
OPENAI_SDK_MODEL_VALUE="${OPENAI_SDK_MODEL:-gpt-5.4-mini}"
OPENAI_SDK_REASONING_EFFORT_VALUE="${OPENAI_SDK_REASONING_EFFORT:-medium}"

for arg in "$@"; do
  case "$arg" in
    --method|--method=*|--data-root|--data-root=*|--domain|--domain=*|--tier|--tier=*|--output-dir|--output-dir=*)
      echo "This wrapper owns --method, --data-root, --domain, --tier, and --output-dir. Set DATA_ROOT, TIER, or OUTPUT_ROOT instead." >&2
      exit 2
      ;;
  esac
done

echo "Note: agentrunbook_c_v2 does not enable a view_image tool; prior runs found low usage and high latency overhead. Screenshot-grounded items may rely on text/AXTree evidence instead." >&2

for domain in web enterprise; do
  python "$REPO_ROOT/evaluation/run_eval.py" \
    --method "$METHOD" \
    --data-root "$DATA_ROOT_VALUE" \
    --domain "$domain" \
    --tier "$TIER_VALUE" \
    --output-dir "$OUTPUT_ROOT_VALUE/${METHOD}_${domain}_${TIER_VALUE}" \
    --openai-sdk-model "$OPENAI_SDK_MODEL_VALUE" \
    --openai-sdk-reasoning-effort "$OPENAI_SDK_REASONING_EFFORT_VALUE" \
    "$@"
done
