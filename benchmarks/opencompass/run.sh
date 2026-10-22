#!/usr/bin/env bash
set -euo pipefail
: "${OPENCOMPASS_ROOT:?Set the reviewed OpenCompass checkout}"
: "${LCCP_MODEL:?Set the local checkpoint folder}"
: "${LCCP_BENCHMARK:?Choose longbench, graphwalks, longreason or babilong}"
PYTHON=${PYTHON:-python}
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT=${OUTPUT:-"$PWD/outputs/$LCCP_BENCHMARK"}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
LCCP_MODEL=$(cd "$LCCP_MODEL" && pwd)
OPENCOMPASS_ROOT=$(cd "$OPENCOMPASS_ROOT" && pwd)
export LCCP_MODEL
export PYTHONPATH="$REPOSITORY/benchmarks/opencompass:$OPENCOMPASS_ROOT:$REPOSITORY:${PYTHONPATH:-}"
export OPENCOMPASS_RUN_DIR="$OUTPUT/runtime"
export LONGBENCH_DISABLE_LEGACY_EXPORT=1
export LCCP_ATTENTION=${LCCP_ATTENTION:-flash_attention_2}
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export LOCAL_RANK=${LOCAL_RANK:-0}
cd "$OUTPUT"
"$PYTHON" "$OPENCOMPASS_ROOT/run.py" "$REPOSITORY/benchmarks/opencompass/eval_config.py" \
  --work-dir "$OUTPUT" --max-num-workers 1 "$@"
