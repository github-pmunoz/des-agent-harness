#!/usr/bin/env bash
#
# run_eval.sh — one Track A run: every case of the curriculum worked by its own chat-des agent at
# the same settings, then graded.
#
# Usage:
#   ./run_eval.sh <EVAL_SETTINGS.json>
#
# The settings is a chat-des --config file (keys are the long flag names with underscores, plus a
# "prompts" object) with one key of the eval's own, removed before chat-des sees it:
#   "cases": []          case ids or levels to run ("L3", "L6-polygon-2"); empty = all of cases/
# The arm is ordinary chat-des settings: geo_tools, geo_feedback, geo_snap, geo_ruler_bias, ...
# Per case, the harness passes --geo (the case file) and the run's own paths; the task text is the
# case's prompt, so the settings carry no "task".
#
# Run it with the venv active (python on PATH is the venv's), or set DESH_PYTHON.
#
# The run dir ./runs/run-<timestamp>-<hash6> will contain
# - run_settings.json:  the effective chat-des configuration (chat-des --print-config)
# - run_manifest.json:  settings path, cases, harness identity, wall time
# - cases/<id>/         per case: stdout, stderr, session.json, completions.jsonl, des.jsonl,
#                       case_manifest.json (exit status, wall time) and geo/ (task.json,
#                       submissions.jsonl, render PNGs)
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
run_dir="$(pwd)/runs/${run_id}"
mkdir -p "$run_dir/cases"

# The harness code the run executes, frozen (../harness_snapshot.sh): a sweep passes its one
# snapshot in DESH_HARNESS so every run of the set executes the same copy.
evals_root="$(cd "${here}/.." && pwd)"
harness="${DESH_HARNESS:-}"
if [[ -z "$harness" ]]; then
  harness="${run_dir}/harness"
  "${evals_root}/harness_snapshot.sh" "$harness"
fi
python="${DESH_PYTHON:-$(command -v python)}"
chat_des=(env PYTHONPATH="${harness}/src" "$python" -m desh_chat.cli)

# The chat-des config: the settings without the eval's own keys.
config="${run_dir}/chat_config.json"
jq 'del(.cases)' "$SETTINGS" > "$config"
if ! "${chat_des[@]}" --config "$config" --print-config > "${run_dir}/run_settings.json"; then
  echo "error: chat-des rejected the settings" >&2
  exit 2
fi
port="$(jq -r '.port' "${run_dir}/run_settings.json")"
model="$(jq -r '.model' "${run_dir}/run_settings.json")"

# The cases: every file under cases/, filtered by the "cases" key (an id or a level prefix).
mapfile -t wanted < <(jq -r '(.cases // [])[]' "$SETTINGS")
cases=()
for f in "${here}"/cases/*.json; do
  id="$(basename "$f" .json)"
  if [[ ${#wanted[@]} -eq 0 ]]; then
    cases+=("$f")
    continue
  fi
  for w in "${wanted[@]}"; do
    if [[ "$id" == "$w" || "$id" == "$w"-* ]]; then
      cases+=("$f")
      break
    fi
  done
done
if [[ ${#cases[@]} -eq 0 ]]; then
  echo "error: no case matches ${wanted[*]}" >&2
  exit 2
fi

echo "run id : $run_id"
echo "settings : $SETTINGS"
echo "harness: ${harness} ($(jq -r '.commit[:7] + (if .dirty then "+dirty" else "" end)' "${harness}/harness.json"))"
echo "cases  : ${#cases[@]}"
echo "----------------------------------------"

erase_kv_cache() {
  curl -s -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$model\"}" \
    "http://127.0.0.1:$port/slots/0?action=erase"
}

# Each case works in an empty directory outside the repository; nothing in it is read or graded.
live_root="${EVAL_WORK_ROOT:-${TMPDIR:-/tmp}/desh-eval}/${run_id}"
run_start_ns="$(date +%s%N)"
for f in "${cases[@]}"; do
  id="$(basename "$f" .json)"
  case_dir="${run_dir}/cases/${id}"
  workspace="${live_root}/${id}"
  mkdir -p "$case_dir" "$workspace"
  printf "%-18s " "$id"
  reply="$(erase_kv_cache)"
  if [[ "$reply" == *'"error"'* ]]; then
    echo "error: kv cache erase failed: $reply" >&2
    exit 2
  fi
  start_ns="$(date +%s%N)"
  set +e
  GIT_CEILING_DIRECTORIES="$live_root" "${chat_des[@]}" --config "$config" --geo "$f" \
    --geo-out "${case_dir}/geo" -w "$workspace" -s "${case_dir}/session.json" \
    -cl "${case_dir}/completions.jsonl" -dl "${case_dir}/des.jsonl" \
    > "${case_dir}/stdout" 2> "${case_dir}/stderr"
  status=$?
  set -e
  elapsed_ms=$(( ($(date +%s%N) - start_ns) / 1000000 ))
  jq -n --arg id "$id" --arg file "$f" --argjson status "$status" --argjson elapsed_ms "$elapsed_ms" \
    '{id: $id, file: $file, status: $status, elapsed_ms: $elapsed_ms}' > "${case_dir}/case_manifest.json"
  last="$(tail -n 1 "${case_dir}/geo/submissions.jsonl" 2>/dev/null | jq -r 'if .ok then (if .exact then "exact" else "iou \(.layers.M1.iou)" end) else "rejected" end' 2>/dev/null || echo "no submission")"
  echo "exit ${status}  $((elapsed_ms / 1000)) s  ${last}"
done
rm -rf "$live_root"
run_elapsed_ms=$(( ($(date +%s%N) - run_start_ns) / 1000000 ))

jq -n --arg run_id "$run_id" --arg settings "$SETTINGS" --argjson elapsed_ms "$run_elapsed_ms" \
  --argjson harness "$(cat "${harness}/harness.json")" \
  --args '{run_id: $run_id, settings: $settings, elapsed_ms: $elapsed_ms, harness: $harness,
           cases: $ARGS.positional}' "${cases[@]##*/}" > "${run_dir}/run_manifest.json"

echo "grading..."
PYTHONPATH="${harness}/src" "$python" "${here}/grade.py" "$run_dir" 2>&1 | tee "${run_dir}/grade.log"
echo "----------------------------------------"
echo "wall time: $((run_elapsed_ms / 1000)) s"
