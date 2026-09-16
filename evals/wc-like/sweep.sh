#!/usr/bin/env bash
#
# sweep.sh — run one eval settings N times and summarise the graded results.
#
# Usage:
#   ./sweep.sh <CONFIG.json> <REPEATS> [NAME]
#
# Each repeat is a full ./run_eval.sh launch (own run dir under ./runs, own grade).
# The sweep itself lands in ./sweeps/<NAME>-<timestamp>/ with
# - settings.json:   a copy of the settings every repeat used
# - manifest.json: name, settings path, repeats, and the run ids in order
# - summary.json:  aggregates over the graded runs (written by summarise.py)
# A failing run does not stop the sweep: its exit status is in its own run manifest,
# and summarise.py decides whether it enters the aggregates.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <CONFIG.json> <REPEATS> [NAME]" >&2
  exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"   # run_eval.sh resolves ./runs, task.txt and train/ against the cwd

settings="$1"
repeats="$2"
name="${3:-$(basename "$settings" .json)}"

if [[ ! -f "$settings" ]]; then
  echo "error: settings file not found: $settings" >&2
  exit 2
fi
if ! [[ "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: REPEATS must be a positive integer, got '$repeats'" >&2
  exit 2
fi

sweep_id="${name}-$(date +%Y%m%d-%H%M%S)"
sweep_dir="sweeps/${sweep_id}"
mkdir -p "$sweep_dir"
cp "$settings" "${sweep_dir}/settings.json"

run_ids=()
for i in $(seq 1 "$repeats"); do
  echo "=== ${sweep_id}: repeat ${i}/${repeats} ==="
  set +e
  out="$(./run_eval.sh "$settings" | tee /dev/stderr)"
  set -e
  run_id="$(sed -n 's/^run id : //p' <<<"$out")"
  if [[ -z "$run_id" ]]; then
    echo "error: run_eval.sh printed no run id; aborting sweep" >&2
    exit 2
  fi
  run_ids+=("$run_id")
done

jq -n \
  --arg name "$name" \
  --arg settings "$settings" \
  --argjson repeats "$repeats" \
  --args '{name: $name, settings: $settings, repeats: $repeats, runs: $ARGS.positional}' \
  "${run_ids[@]}" > "${sweep_dir}/manifest.json"

echo "=== ${sweep_id}: summarising ==="
./summarise.py "$sweep_dir"
