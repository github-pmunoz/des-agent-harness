#!/usr/bin/env bash
#
# run_eval.sh — launch a single chat-des agent run from a flat JSON settings.
#
# Usage:
#   ./run_eval.sh <EVAL_SETTINGS.json>
#
# The settings is a flat JSON object whose keys map 1:1 to chat-des flags.
# Example:
#   {
#     "model": "claude-3-5-sonnet",
#     "temperature": 0.2,
#     "context": 200000,
#     "max_turn_tokens": 8192,
#     "max_tool_rounds": 60,
#     "system_prompt": "You are a coding agent.",
#     "timeout": 600,
#     "task": "Read task.txt and implement the wc-like CLI using TDD.",
#     "auto": true,
#     "cont": true,
#     "debug": false,
#     "think": true,
#     "read": true, "write": true, "edit": true, "bash": true,
#     "memory": "scratchpad", "current_time": true, delegate": false
#   }
#
# Each run gets its own workspace, outside the repository while it runs and moved to
# ./runs/run-<timestamp>-<hash6>/workspace when it is done.
# The run dir ./runs/run-<timestamp>-<hash6> will contain
# - run_settings.json: the effective chat-des configuration (chat-des --print-config)
# - run_manifest.json: a copy of the chat-des exit status and wall time
# - stdout: stdout of the chat-des run
# - stderr: stderr of the chat-des run
# - session.json: the chat-des session file
# - des.jsonl: the chat-des des log
# - completions.jsonl: the chat-des completions log

set -euo pipefail

# --- arguments -------------------------------------------------------------
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <EVAL_SETTINGS.json>" >&2
  exit 2
fi

SETTINGS="$1"
if [[ ! -f "$SETTINGS" ]]; then
  echo "error: settings file not found: $SETTINGS" >&2
  exit 2
fi

# --- run id: run-<timestamp>-<6 digit hash> --------------------------------
timestamp="$(date +%Y%m%d-%H%M%S)"
hash6="$(od -An -N3 -tx1 /dev/urandom | tr -d ' \n' | cut -c1-6)"
run_id="run-${timestamp}-${hash6}"

runs_root="$(pwd)/runs"
run_dir="${runs_root}/${run_id}"
# The agent works in a directory outside this repository and is moved into the run folder when
# it is done. Nested in the repository, its absolute path named the live repo around it, and a
# Bash `cd` there read the real git log (run 8e7447). GIT_CEILING_DIRECTORIES below keeps git
# from discovering any repository above it as well.
live_dir="${EVAL_WORK_ROOT:-${TMPDIR:-/tmp}/desh-eval}/${run_id}"
workspace="${live_dir}/workspace"
mkdir -p "$workspace" "$run_dir"

# The harness code the run executes, frozen (../harness_snapshot.sh): the venv installs it in
# editable mode, so a plain `chat-des` would run whatever src/ holds when each run starts, and a
# sweep that outlives an edit would run its later arms on other code. A sweep passes its one
# snapshot in DESH_HARNESS so every run of the set executes the same copy; a run launched on its
# own makes its own. The manifest records which (harness.json: commit, dirty flag, content hash).
evals_root="$(cd "$(dirname "$0")/.." && pwd)"
harness="${DESH_HARNESS:-}"
if [[ -z "$harness" ]]; then
  harness="${run_dir}/harness"
  "${evals_root}/harness_snapshot.sh" "$harness"
fi
python="${DESH_PYTHON:-$(git -C "$evals_root" rev-parse --show-toplevel)/venv/bin/python}"
chat_des=(env PYTHONPATH="${harness}/src" "$python" -m desh_chat.cli)


# Copy work material into workspace
cp task.txt "${workspace}/task.txt"
cp -rf train/ "${workspace}/train"

# Set up telemetry files
session_file="${run_dir}/session.json"
des_log="${run_dir}/des_log.jsonl"
completions_log="${run_dir}/completions_log.jsonl"

# --- build chat-des arguments ------------------------------------------------
# The settings file IS a chat-des --config file: its keys are the long flag names with
# underscores, plus a "prompts" object of instruction-prompt overrides (see --print-config).
# chat-des checks it: an unknown key, a mistyped value or a bad prompt key stops the run before it
# starts (a sweep once overrode "tool-cap" while the flag was "tool_cap": 45 runs at the default).
# Only what the harness decides per run is passed as flags, which win over the file.
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to parse the settings" >&2
  exit 2
fi

# erase llama-server kv cache for starting the run cold
erase_kv_cache() {
  local port="$1" model="$2"
  curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\"}" \
    "http://127.0.0.1:$port/slots/0?action=erase"
}

args=(--config "$SETTINGS"
      -cl "$completions_log" -s "$session_file" -dl "$des_log" -w "$workspace")

# The effective configuration — every option and every prompt with its text, defaults and @files
# resolved — is the record of what the run was built with.
if ! "${chat_des[@]}" "${args[@]}" --print-config > "${run_dir}/run_settings.json"; then
  echo "error: chat-des rejected the settings" >&2
  exit 2
fi
port="$(jq -r '.port' "${run_dir}/run_settings.json")"
model="$(jq -r '.model' "${run_dir}/run_settings.json")"

# --- launch ----------------------------------------------------------------
echo "run id : $run_id"
echo "settings : $SETTINGS"
echo "workspace: $workspace"
echo "harness: ${harness} ($(jq -r '.commit[:7] + (if .dirty then "+dirty" else "" end)' "${harness}/harness.json"))"
echo "cmd    : chat-des ${args[*]}"
echo "----------------------------------------"

# Erase the llama-server kv cache for starting the run cold.
# use printf to avoid the newline
printf "erasing kv cache... "
reply="$(erase_kv_cache "$port" "$model")"
printf "${reply}\n"
if [[ "$reply" == *'"error"'* ]]; then
  echo "error: kv cache erase failed" >&2
  exit 2
fi

# Measure wall time of the chat-des run.
# Capture stdout/stderr separately and persist them in the run folder.
stdout_file="${run_dir}/stdout"
stderr_file="${run_dir}/stderr"
echo "running..."
start_ns="$(date +%s%N)"
set +e
GIT_CEILING_DIRECTORIES="$live_dir" "${chat_des[@]}" "${args[@]}" >"$stdout_file" 2>"$stderr_file"
status=$?
set -e
end_ns="$(date +%s%N)"
elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

# The workspace joins the rest of the run: the grader reads it at ${run_dir}/workspace.
mv "$workspace" "${run_dir}/workspace"
rmdir "$live_dir" 2>/dev/null || true

# Write a small run manifest into the run folder.
manifest="${run_dir}/run_manifest.json"
jq -n \
  --arg run_id "$run_id" \
  --arg settings "$SETTINGS" \
  --argjson status "$status" \
  --argjson elapsed_ms "$elapsed_ms" \
  --arg workspace "$workspace" \
  --argjson harness "$(cat "${harness}/harness.json")" \
  --arg started "$(date -d "@$((start_ns / 1000000000))" +%Y-%m-%dT%H:%M:%S%z 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)" \
  --arg finished "$(date +%Y-%m-%dT%H:%M:%S%z)" \
  '{
    run_id: $run_id,
    settings: $settings,
    status: $status,
    elapsed_ms: $elapsed_ms,
    workspace: $workspace,
    harness: $harness,
    started: $started,
    finished: $finished
  }' > "$manifest"

# Grade the run: result.json lands next to the manifest, and what the grader printed lands in
# grade.log — a run once finished clean and was excluded from its sweep with no trace of why.
echo "grading..."
"$(dirname "$0")/grade.py" "${run_dir}" 2>&1 | tee "${run_dir}/grade.log"

echo "----------------------------------------"
echo "chat-des exited with status $status (wall time: ${elapsed_ms} ms)"
echo "manifest: $manifest"
exit "$status"
