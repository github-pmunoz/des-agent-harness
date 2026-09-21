#!/usr/bin/env bash
#
# sweep.sh — repeat eval runs and summarise the graded results.
#
# Usage:
#   ./sweep.sh <SWEEP.json>                      a sweep set: several settings, as deltas to a baseline
#   ./sweep.sh <SETTINGS.json> <REPEATS> [NAME]   one settings file, repeated
#
# A sweep set names a baseline settings file and a list of sweeps, each a name, a repeat count
# and an override merged over the baseline, recursively: a set's "prompts" object adds to the
# baseline's instead of replacing it (an @file prompt must be an absolute path: the merged settings
# are written to the sweep folder, and @paths resolve against the file that names them).
# Override keys must exist in the baseline, so a typo
# fails before any inference is spent. Example:
#   {
#     "baseline": "settings.json",
#     "sweeps": [
#       {"name": "8k", "reps": 5, "override": {"context": 8192, "max_turn_tokens": 8192}},
#       {"name": "4k", "reps": 5, "override": {"context": 4096, "max_turn_tokens": 4096}}
#     ]
#   }
#
# Every repeat is a full ./run_eval.sh launch (own run dir under ./runs, own grade). A sweep
# set lands in ./sweeps/<set name>-<timestamp>/ with a copy of the spec, one sub-directory per
# sweep, and compare.json across them; a single sweep lands in ./sweeps/<NAME>-<timestamp>/.
# Each sweep directory holds
# - settings.json: the settings every repeat used (baseline + override for a set)
# - manifest.json: name, settings path, repeats, and the run ids in order
# - summary.json:  aggregates over the graded runs (written by summarise.py)
# A failing run does not stop the sweep: its exit status is in its own run manifest, and
# summarise.py decides whether it enters the aggregates.

set -euo pipefail

usage() {
  echo "usage: $0 <SWEEP.json> | $0 <SETTINGS.json> <REPEATS> [NAME]" >&2
  exit 2
}

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"   # run_eval.sh resolves ./runs against the cwd

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 2
fi

# run_sweep <settings file> <repeats> <name> <sweep dir>
# Runs the settings <repeats> times, writes manifest.json, summarises.
run_sweep() {
  local settings="$1" repeats="$2" name="$3" sweep_dir="$4"
  local run_ids=() i out run_id
  for i in $(seq 1 "$repeats"); do
    echo "=== ${name}: repeat ${i}/${repeats} ==="
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
  echo "=== ${name}: summarising ==="
  ./summarise.py "$sweep_dir"
}

[[ $# -ge 1 ]] || usage
spec="$1"
if [[ ! -f "$spec" ]]; then
  echo "error: file not found: $spec" >&2
  exit 2
fi
stamp="$(date +%Y%m%d-%H%M%S)"

# --- single settings file, repeated ------------------------------------------
if ! jq -e 'has("sweeps")' "$spec" >/dev/null; then
  [[ $# -ge 2 && $# -le 3 ]] || usage
  repeats="$2"
  name="${3:-$(basename "$spec" .json)}"
  if ! [[ "$repeats" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: REPEATS must be a positive integer, got '$repeats'" >&2
    exit 2
  fi
  sweep_dir="sweeps/${name}-${stamp}"
  mkdir -p "$sweep_dir"
  cp "$spec" "${sweep_dir}/settings.json"
  run_sweep "${sweep_dir}/settings.json" "$repeats" "$name" "$sweep_dir"
  exit 0
fi

# --- sweep set: deltas over a baseline ---------------------------------------
[[ $# -eq 1 ]] || usage
baseline="$(jq -r '.baseline' "$spec")"
if [[ ! -f "$baseline" ]]; then
  echo "error: baseline settings not found: $baseline" >&2
  exit 2
fi

# Validate the whole spec before running anything: names present and unique, reps positive,
# override keys all present in the baseline.
problems="$(jq -r --slurpfile base "$baseline" '
  [ .sweeps[] | select((.name // "") == "") | "a sweep has no name" ],
  [ .sweeps | group_by(.name)[] | select(length > 1) | "duplicate sweep name: \(.[0].name)" ],
  [ .sweeps[] | select(((.reps // 0) | type) != "number" or (.reps // 0) < 1) | "\(.name): reps must be a positive integer" ],
  [ .sweeps[] | .name as $n | ((.override // {}) | keys[]) | select(in($base[0]) | not) | "\($n): override key not in baseline: \(.)" ]
  | .[]' "$spec")"
if [[ -n "$problems" ]]; then
  echo "error: invalid sweep spec $spec" >&2
  sed 's/^/  /' <<<"$problems" >&2
  exit 2
fi

set_name="$(basename "$spec" .json)"
set_dir="sweeps/${set_name}-${stamp}"
mkdir -p "$set_dir"
cp "$spec" "${set_dir}/sweep.json"

sweep_dirs=()
count="$(jq '.sweeps | length' "$spec")"
for ((k = 0; k < count; k++)); do
  name="$(jq -r ".sweeps[$k].name" "$spec")"
  repeats="$(jq -r ".sweeps[$k].reps" "$spec")"
  sweep_dir="${set_dir}/${name}"
  mkdir -p "$sweep_dir"
  jq --slurpfile base "$baseline" ".sweeps[$k].override // {} | \$base[0] * ." "$spec" > "${sweep_dir}/settings.json"
  run_sweep "${sweep_dir}/settings.json" "$repeats" "$name" "$sweep_dir"
  sweep_dirs+=("$sweep_dir")
done

echo "=== ${set_name}: comparing ==="
./compare.py "$set_dir"
