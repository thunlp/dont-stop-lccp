#!/usr/bin/env bash
set -euo pipefail
: "${RULER_ROOT:?Set the reviewed upstream RULER checkout}"
: "${MODEL:?Set the local model folder}"
PYTHON=${PYTHON:-python}
CONTEXT=${CONTEXT:-32768}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-outputs/ruler}
if [[ "$CONTEXT" == 32768 ]]; then
  SAMPLES=${SAMPLES:-500}
elif [[ "$CONTEXT" == 131072 ]]; then
  SAMPLES=${SAMPLES:-100}
else
  echo "Expected CONTEXT=32768 or 131072" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
TASKS=(niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2)
for task in "${TASKS[@]}"; do
  if [[ -n "${RULER_DATA:-}" ]]; then
    test -f "$RULER_DATA/$task/validation.jsonl" || { echo "Missing fixed data for $task" >&2; exit 1; }
  fi
done
if [[ -n "${RULER_DATA:-}" ]]; then
  RULER_DATA=$(cd "$RULER_DATA" && pwd)
  if [[ "$RULER_DATA" == "$(cd "$OUTPUT" && pwd)" ]]; then
    echo "Use a separate output directory; fixed data must remain read-only" >&2
    exit 1
  fi
fi
for task in "${TASKS[@]}"; do
  if [[ -z "${RULER_DATA:-}" ]]; then
  "$PYTHON" "$RULER_ROOT/scripts/data/prepare.py" --save_dir "$OUTPUT" \
    --benchmark synthetic --task "$task" --tokenizer_path "$MODEL" --tokenizer_type hf \
    --max_seq_length "$CONTEXT" --model_template_type base --num_samples "$SAMPLES" --random_seed "$SEED"
  fi
  "$PYTHON" "$RULER_ROOT/scripts/pred/call_api.py" --data_dir "${RULER_DATA:-$OUTPUT}" --save_dir "$OUTPUT" \
    --benchmark synthetic --task "$task" --server_type hf --model_name_or_path "$MODEL" \
    --temperature 0 --top_k 32 --top_p 1 --batch_size 1
done
"$PYTHON" "$RULER_ROOT/scripts/eval/evaluate.py" --data_dir "$OUTPUT" --benchmark synthetic
