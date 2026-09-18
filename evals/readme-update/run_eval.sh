#!/usr/bin/env bash
#
# run_eval.sh — launch a single chat-des agent run from a flat JSON settings.
#
# Usage:
#   ./run_eval.sh <EVAL_SETTINGS.json>
#
# The settings is a flat JSON object whose keys map 1:1 to chat-des flags, and it is the run's
# whole configuration: chat-des is launched directly, never through coding.sh. The orchestrator
# system prompt, with the fixture's file tree inlined, is the literal "system_prompt" value, so a
# sweep can override it like any other key and an edit to coding.sh cannot change a result.
#
# The workspace is a clone of the fixture branch `readme_eval` (the commit the README lagged at,
# plus a regenerated INDEX.md): full history up to there and nothing after it, so the README that
# was later committed is not reachable from inside the run.
# Example:
#   {
#     "model": "claude-3-5-sonnet",
#     "temperature": 0.2,
#     "context": 200000,
#     "max_turn_tokens": 8192,
#     "max_tool_rounds": 60,
#     "tool_expiration": 10,
#     "system_prompt": "You are a coding agent.",
#     "timeout": 600,
#     "task": "Read task.txt and implement the wc-like CLI using TDD.",
#     "auto": true,
#     "cont": true,
#     "debug": false,
#     "think": true,
#     "read": true, "write": true, "edit": true, "bash": true,
#     "scratchpad": true, "current_time": true, delegate": false
#   }
#
# Each run gets its own workspace under ./runs/run-<timestamp>-<hash6>/workspace.
# The run dir ./runs/run-<timestamp>-<hash6> will contain
# - run_settings.json: a copy of the chat-des cli args
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
workspace="${run_dir}/workspace"
mkdir -p "$workspace"

# Save a copy of the settings into the run folder.
cp "$SETTINGS" "${run_dir}/run_settings.json"

# The workspace is the fixture branch, cloned over a real pack transfer: a plain local clone
# hardlinks the whole object store, and every later commit would be readable by its hash.
FIXTURE_BRANCH="readme_eval"
ANSWER_COMMIT="a5d7e31"   # the README update that was committed; must not exist in the workspace
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
git clone -q --no-local --single-branch --branch "$FIXTURE_BRANCH" "$repo_root" "$workspace"
git -C "$workspace" remote remove origin
fixture_sha="$(git -C "$workspace" rev-parse HEAD)"
if git -C "$workspace" cat-file -e "$ANSWER_COMMIT" 2>/dev/null; then
  echo "error: $ANSWER_COMMIT is reachable in the workspace; the fixture leaks the answer" >&2
  exit 2
fi
cp task.txt "${workspace}/task.txt"

# Set up telemetry files
session_file="${run_dir}/session.json"
des_log="${run_dir}/des_log.jsonl"
completions_log="${run_dir}/completions_log.jsonl"

# --- build chat-des arguments from the flat settings -------------------------
# jq is required.
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to parse the settings" >&2
  exit 2
fi

# has <key>  -> 0 if the key is present (even if null/false)
has() { jq -e --arg k "$1" 'has($k)' "$SETTINGS" >/dev/null 2>&1; }

# Every key a flag is built from is recorded here (by add_value / add_bool, which run in this
# shell — get() runs in a command substitution, so it cannot record), and the launch refuses a
# settings key nothing read. port and model are read directly below.
consumed=" port model"

# get <key> fails if `has <key>` is false, no default
get() {
  local key="$1" 
  if has "$key"; then
    jq -r --arg k "$key" '.[$k]' "$SETTINGS"
  else
    echo "error: settings key $key is missing" >&2
    exit 2
  fi
}

# is_true <key> -> 0 if the key is present and true, fails if key is missing
is_true() {
  local key="$1"
  if has "$key"; then
    jq -e --arg k "$key" '.[$k] == true' "$SETTINGS" >/dev/null 2>&1
  else
    echo "error: settings key $key is missing" >&2
    exit 2
  fi
}

args=()

# value flags:  flag  settings-key  default
add_value() {
  local flag="$1" key="$2"
  consumed="$consumed $key"
  val="$(get "$key")"
  args+=("$flag" "$val")
}

# boolean flags:  flag  settings-key
add_bool() {
  local flag="$1" key="$2"
  consumed="$consumed $key"
  if is_true "$key"; then
    args+=("$flag")
  fi
}

# force external value
force_value() {
  local flag="$1" value="$2"
  args+=("$flag" "$value")
}

# erase llama-server kv cache for starting the run cold
erase_kv_cache() {
  local port="$1" model="$2"
  curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\"}" \
    "http://127.0.0.1:$port/slots/0?action=erase"
}

# Extract port and model for reuse in erase_kv_cache command
port="$(get port)"
model="$(get model)"

# value flags
add_value "-t"   "temperature"
add_value "-c"   "context"
add_value "-mt"  "max_turn_tokens"
add_value "-mtr" "max_tool_rounds"
add_value "-te"  "tool_expiration"
add_value "-sp"  "system_prompt"
add_value "-to"  "timeout"
add_value "-ta"  "task"
add_value "-tt"  "task_timeout"
add_value "-tc"  "tool_cap"
add_value "-ct"  "checkpoint_target"

# harness forced values
force_value "-cl"  "${completions_log}"
force_value "-s"   "${session_file}"
force_value "-dl"  "${des_log}"
force_value "-w"   "${workspace}"
force_value "-p"   "${port}"
force_value "-m"   "${model}"

# boolean flags
add_bool "-a"   "auto"
add_bool "-co"  "cont"
add_bool "-th"  "think"
add_bool "-d"   "debug"
add_bool "--read"         "read"
add_bool "--write"        "write"
add_bool "--edit"         "edit"
add_bool "--bash"         "bash"
add_bool "--delegate"     "delegate"
add_bool "--scratchpad"   "scratchpad"
add_bool "--current_time" "current_time"

# A key this script never read is a typo that would otherwise run silently on the flag's default
# (a sweep once overrode "tool-cap" while this read "tool_cap": 45 runs at the default).
for key in $(jq -r 'keys[]' "$SETTINGS"); do
  case " $consumed " in
    *" $key "*) ;;
    *) echo "error: settings key '$key' is not one run_eval.sh reads (keys map 1:1 to chat-des flags, underscores)" >&2; exit 2;;
  esac
done

# --- launch ----------------------------------------------------------------
echo "run id : $run_id"
echo "settings : $SETTINGS"
echo "workspace: $workspace"
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
chat-des "${args[@]}" >"$stdout_file" 2>"$stderr_file"
status=$?
set -e
end_ns="$(date +%s%N)"
elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

# Write a small run manifest into the run folder.
manifest="${run_dir}/run_manifest.json"
jq -n \
  --arg run_id "$run_id" \
  --arg settings "$SETTINGS" \
  --arg fixture_sha "$fixture_sha" \
  --argjson status "$status" \
  --argjson elapsed_ms "$elapsed_ms" \
  --arg started "$(date -d "@$((start_ns / 1000000000))" +%Y-%m-%dT%H:%M:%S%z 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)" \
  --arg finished "$(date +%Y-%m-%dT%H:%M:%S%z)" \
  '{
    run_id: $run_id,
    settings: $settings,
    fixture_sha: $fixture_sha,
    status: $status,
    elapsed_ms: $elapsed_ms,
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
