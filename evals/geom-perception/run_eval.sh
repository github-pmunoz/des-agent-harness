#!/usr/bin/env bash
#
# run_eval.sh — one perception run: every case asked once through send_direct, then graded.
#
# Usage:
#   ./run_eval.sh <EVAL_SETTINGS.json>
#
# The settings is a flat JSON object read by run.py (see baseline.json):
#   {"model": "...-mmproj", "port": 8012, "think": true, "temperature": 0.7, "max_tokens": 16384,
#    "timeout": 900, "grid": false, "image_px": 800, "cases": []}
#
# The run dir ./runs/run-<timestamp>-<hash6> will contain
# - run_settings.json: a copy of the settings
# - run_manifest.json: exit status, wall time, harness snapshot identity
# - stdout / stderr:   of run.py
# - cases.json, images/, truth/, responses/, completions.jsonl: see run.py
# - result.json, grade.log: see grade.py

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <EVAL_SETTINGS.json>" >&2
  exit 2
fi
SETTINGS="$1"
if [[ ! -f "$SETTINGS" ]]; then
  echo "error: settings file not found: $SETTINGS" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to parse the settings" >&2
  exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
timestamp="$(date +%Y%m%d-%H%M%S)"
hash6="$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n' | cut -c1-6)"
run_id="run-${timestamp}-${hash6}"
run_dir="${here}/runs/${run_id}"
mkdir -p "$run_dir"
cp "$SETTINGS" "${run_dir}/run_settings.json"

# send_direct runs from a frozen copy of src/ (../harness_snapshot.sh), one per sweep via
# DESH_HARNESS or one per run otherwise, so an edit mid-sweep cannot change what later runs send.
evals_root="$(cd "${here}/.." && pwd)"
harness="${DESH_HARNESS:-}"
if [[ -z "$harness" ]]; then
  harness="${run_dir}/harness"
  "${evals_root}/harness_snapshot.sh" "$harness"
fi
python="${DESH_PYTHON:-$(git -C "$evals_root" rev-parse --show-toplevel)/venv/bin/python}"

port="$(jq -r '.port' "$SETTINGS")"
model="$(jq -r '.model' "$SETTINGS")"

echo "run id : $run_id"
echo "settings : $SETTINGS"
echo "harness: ${harness} ($(jq -r '.commit[:7] + (if .dirty then "+dirty" else "" end)' "${harness}/harness.json"))"
echo "model  : $model  think=$(jq -r '.think' "$SETTINGS")  grid=$(jq -r '.grid' "$SETTINGS")  image_px=$(jq -r '.image_px' "$SETTINGS")"
echo "----------------------------------------"

# Start cold: the first case must not reuse a previous run's cache.
printf "erasing kv cache... "
reply="$(curl -s -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$model\"}" \
  "http://127.0.0.1:${port}/slots/0?action=erase")"
printf '%s\n' "$reply"
if [[ "$reply" == *'"error"'* ]]; then
  echo "error: kv cache erase failed" >&2
  exit 2
fi

start_ns="$(date +%s%N)"
set +e
PYTHONPATH="${harness}/src" "$python" "${here}/run.py" "$SETTINGS" "$run_dir" \
  > >(tee "${run_dir}/stdout") 2> "${run_dir}/stderr"
status=$?
set -e
end_ns="$(date +%s%N)"
elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

jq -n \
  --arg run_id "$run_id" \
  --arg settings "$SETTINGS" \
  --argjson status "$status" \
  --argjson elapsed_ms "$elapsed_ms" \
  --argjson harness "$(cat "${harness}/harness.json")" \
  --arg started "$(date -d "@$((start_ns / 1000000000))" +%Y-%m-%dT%H:%M:%S%z)" \
  --arg finished "$(date +%Y-%m-%dT%H:%M:%S%z)" \
  '{run_id: $run_id, settings: $settings, status: $status, elapsed_ms: $elapsed_ms,
    harness: $harness, started: $started, finished: $finished}' > "${run_dir}/run_manifest.json"

if [[ "$status" -eq 0 ]]; then
  echo "grading..."
  "$python" "${here}/grade.py" "$run_dir" 2>&1 | tee "${run_dir}/grade.log"
fi
echo "----------------------------------------"
echo "run.py exited with status $status (wall time: ${elapsed_ms} ms)"
exit "$status"
