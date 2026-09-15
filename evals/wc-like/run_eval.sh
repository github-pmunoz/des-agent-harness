#!/usr/bin/env bash
#
# run_eval.sh — launch a single chat-des agent run from a flat JSON config.
#
# Usage:
#   ./run_eval.sh <EVAL_CONFIG.json>
#
# The config is a flat JSON object whose keys map 1:1 to chat-des flags.
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
  echo "usage: $0 <EVAL_CONFIG.json>" >&2
  exit 2
fi

CONFIG="$1"
if [[ ! -f "$CONFIG" ]]; then
  echo "error: config file not found: $CONFIG" >&2
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
cp "$CONFIG" "${run_dir}/run_settings.json"

# Copy work material into workspace
cp task.txt "${workspace}/task.txt"
cp -rf train/ "${workspace}/train"

# Set up telemetry files
session_file="${run_dir}/session.json"
des_log="${run_dir}/des_log.jsonl"
completions_log="${run_dir}/completions_log.jsonl"

# --- build chat-des arguments from the flat config -------------------------
# jq is required.
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to parse the config" >&2
  exit 2
fi

# has <key>  -> 0 if the key is present (even if null/false)
has() { jq -e --arg k "$1" 'has($k)' "$CONFIG" >/dev/null 2>&1; }

# get <key> fails if `has <key>` is false, no default
get() {
  local key="$1" 
  if has "$key"; then
    jq -r --arg k "$key" '.[$k]' "$CONFIG"
  else
    echo "error: config key $key is missing" >&2
    exit 2
  fi
}

# is_true <key> -> 0 if the key is present and true, fails if key is missing
is_true() {
  local key="$1"
  if has "$key"; then
    jq -e --arg k "$key" '.[$k] == true' "$CONFIG" >/dev/null 2>&1
  else
    echo "error: config key $key is missing" >&2
    exit 2
  fi
}

args=()

# value flags:  flag  config-key  default
add_value() {
  local flag="$1" key="$2"
  val="$(get "$key")"
  args+=("$flag" "$val")
}

# boolean flags:  flag  config-key
add_bool() {
  local flag="$1" key="$2"
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

# --- launch ----------------------------------------------------------------
echo "run id : $run_id"
echo "config : $CONFIG"
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
  --arg config "$CONFIG" \
  --argjson status "$status" \
  --argjson elapsed_ms "$elapsed_ms" \
  --arg started "$(date -d "@$((start_ns / 1000000000))" +%Y-%m-%dT%H:%M:%S%z 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)" \
  --arg finished "$(date +%Y-%m-%dT%H:%M:%S%z)" \
  '{
    run_id: $run_id,
    config: $config,
    status: $status,
    elapsed_ms: $elapsed_ms,
    started: $started,
    finished: $finished
  }' > "$manifest"

# Grade the run: result.json lands next to the manifest.
echo "grading..."
"$(dirname "$0")/grade.py" "${run_dir}"

echo "----------------------------------------"
echo "chat-des exited with status $status (wall time: ${elapsed_ms} ms)"
echo "manifest: $manifest"
exit "$status"
