#!/usr/bin/env python3
"""summarise.py — aggregate the graded runs of one sweep.

Usage:
    ./summarise.py <SWEEP_DIR>

Reads <SWEEP_DIR>/manifest.json (written by sweep.sh), loads runs/<run id>/result.json and
run_manifest.json for every run, and writes <SWEEP_DIR>/summary.json: per-metric mean / min /
max / stdev over the runs that count, plus the distribution of stop reasons and exit statuses
over all runs. Pure function of the graded files: rerun it any time, on any old sweep.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# Metrics aggregated per run: (name, path into result.json).
METRICS = (
    ("held_out_score", ("held_out", "score")),
    ("wall_ms", ("stats", "wall_ms")),
    ("prompt_tokens", ("stats", "prompt_tokens")),
    ("prompt_tokens_peak", ("stats", "prompt_tokens_peak")),
    ("completion_tokens", ("stats", "completion_tokens")),
    ("tool_calls", ("stats", "tool_calls")),
    ("turns", ("stats", "turns")),
    ("compactions", ("stats", "compactions")),
    ("checkpoints", ("stats", "checkpoints")),
    ("summary_retries", ("stats", "summary_retries")),
    ("fallback_summaries", ("stats", "fallback_summaries")),
    ("salvaged_turns", ("stats", "salvaged_turns")),
    ("cut_results", ("stats", "cut_results")),
    ("rereads_refused", ("stats", "rereads_refused")),
    ("offset_reads", ("stats", "offset_reads")),
    ("spill_reads", ("stats", "spill_reads")),
    ("scratchpad_writes", ("stats", "scratchpad_writes")),
    ("first_green_round", ("stats", "first_green_round")),
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def dig(d: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def counts(result: dict | None, run_manifest: dict | None) -> bool:
    """Whether a run enters the aggregates.

    `result` is the grader's result.json (None when the run never got graded), `run_manifest`
    is run_eval.sh's run_manifest.json (None when the run never launched). Runs that return
    False are still listed in the summary, under `excluded`, so nothing is silently dropped.
    """
    if run_manifest is None or result is None:
        return False
    status = run_manifest.get("status")
    return status == 0 or (status == 1 and result["stats"]["stop"] == "overflow")


def aggregate(values: list[float]) -> dict:
    """mean / min / max / stdev over the values present; stdev needs two."""
    if not values:
        return {"n": 0}
    out = {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    if len(values) >= 2:
        out["stdev"] = statistics.stdev(values)
    return out


def tally(values: list) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def summarise(sweep_dir: Path) -> dict:
    manifest = read_json(sweep_dir / "manifest.json")
    if manifest is None:
        sys.exit(f"error: no manifest.json in {sweep_dir}")

    rows, excluded = [], []
    for run_id in manifest["runs"]:
        run_dir = RUNS / run_id
        result = read_json(run_dir / "result.json")
        run_manifest = read_json(run_dir / "run_manifest.json")
        row = {
            "run_id": run_id,
            "exit_status": run_manifest.get("status") if run_manifest else None,
            "stop": dig(result or {}, ("stats", "stop")),
            "final_answer": dig(result or {}, ("stats", "final_answer")),
        }
        for name, path in METRICS:
            row[name] = dig(result or {}, path)
        (rows if counts(result, run_manifest) else excluded).append(row)

    metrics = {}
    for name, _ in METRICS:
        values = [r[name] for r in rows if isinstance(r[name], (int, float)) and not isinstance(r[name], bool)]
        metrics[name] = aggregate(values)

    return {
        "sweep": sweep_dir.name,
        "name": manifest["name"],
        "settings": manifest["settings"],
        "repeats": manifest["repeats"],
        "counted": len(rows),
        "excluded": excluded,
        "stop": tally([r["stop"] for r in rows + excluded]),
        "exit_status": tally([r["exit_status"] for r in rows + excluded]),
        "final_answer": tally([r["final_answer"] for r in rows + excluded]),
        "metrics": metrics,
        "runs": rows,
    }


def print_table(summary: dict) -> None:
    print(f"{summary['name']}  ({summary['counted']} counted, {len(summary['excluded'])} excluded of {summary['repeats']})")
    print(f"  stop: {summary['stop']}   exit: {summary['exit_status']}   final answer: {summary['final_answer']}")
    print(f"  {'metric':<20}{'n':>3}{'mean':>12}{'min':>10}{'max':>10}{'stdev':>10}")
    for name, agg in summary["metrics"].items():
        if agg["n"] == 0:
            print(f"  {name:<20}{0:>3}")
            continue
        stdev = f"{agg['stdev']:.2f}" if "stdev" in agg else "-"
        print(f"  {name:<20}{agg['n']:>3}{agg['mean']:>12.2f}{agg['min']:>10.2f}{agg['max']:>10.2f}{stdev:>10}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: summarise.py <SWEEP_DIR>")
    sweep_dir = Path(sys.argv[1]).resolve()
    summary = summarise(sweep_dir)
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_table(summary)


if __name__ == "__main__":
    main()
