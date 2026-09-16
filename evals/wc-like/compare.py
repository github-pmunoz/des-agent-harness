#!/usr/bin/env python3
"""compare.py — one table across the sweeps of a sweep set.

Usage:
    ./compare.py <SET_DIR>

Reads <SET_DIR>/sweep.json (the spec sweep.sh copied there) and every <SET_DIR>/<name>/summary.json
(written by summarise.py), and writes <SET_DIR>/compare.json: one row per sweep, in spec order,
with the override that sweep applied. Pure function of the summaries: rerun it any time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def row(summary: dict) -> dict:
    """The columns of one sweep in the comparison table, from its summary.json.

    `summary["metrics"]` maps each metric name (held_out_score, wall_ms, prompt_tokens_peak,
    tool_calls, first_green_round, ...) to {n, mean, min, max, stdev}; `summary["stop"]` tallies
    stop reasons over all runs; `summary["counted"]` / `summary["excluded"]` say how many runs
    entered the aggregates. Column order here is column order in the table.
    """
    metrics = summary.get("metrics", {})
    counted = summary.get("counted", 0)
    total = counted + len(summary.get("excluded", []))

    def spread(name: str, scale: float = 1.0) -> str | None:
        """mean±stdev, or the mean alone under two runs; None when nothing counted, so the
        column stays in the table and renders as a dash."""
        m = metrics.get(name, {})
        if m.get("n", 0) == 0:
            return None
        mean = m["mean"] * scale
        stdev = m.get("stdev")
        if stdev is None:
            return f"{mean:.2f}"
        return f"{mean:.2f}±{stdev * scale:.2f}"

    return {
        "held_out_score": spread("held_out_score"),
        "wall_s": spread("wall_ms", scale=0.001),
        "scored": f"{counted}/{total}",
        # a turn that ends normally has an empty stop reason
        "stop": {k or "done": v for k, v in summary.get("stop", {}).items()},
        "prompt_tokens_peak": spread("prompt_tokens_peak"),
        "first_green_round": spread("first_green_round"),
    }


def compare(set_dir: Path) -> dict:
    spec = read_json(set_dir / "sweep.json")
    if spec is None:
        sys.exit(f"error: no sweep.json in {set_dir}")
    rows = []
    for sweep in spec["sweeps"]:
        summary = read_json(set_dir / sweep["name"] / "summary.json")
        if summary is None:
            rows.append({"name": sweep["name"], "override": sweep.get("override", {}), "missing": True})
            continue
        rows.append({"name": sweep["name"], "override": sweep.get("override", {}), **row(summary)})
    return {"set": set_dir.name, "baseline": spec["baseline"], "rows": rows}


def fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items()) or "-"
    return "-" if value is None else str(value)


def print_table(result: dict) -> None:
    rows = [r for r in result["rows"] if not r.get("missing")]
    missing = [r["name"] for r in result["rows"] if r.get("missing")]
    columns = ["name"] + [c for c in (rows[0] if rows else {}) if c not in ("name", "override")]
    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(line[i]) for line in cells)) if cells else len(c) for i, c in enumerate(columns)]
    print(f"{result['set']}  (baseline {result['baseline']})")
    print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r, line in zip(rows, cells):
        print("  " + "  ".join(v.ljust(w) for v, w in zip(line, widths)) + f"    {fmt(r['override'])}")
    if missing:
        print(f"  no summary: {', '.join(missing)}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: compare.py <SET_DIR>")
    set_dir = Path(sys.argv[1]).resolve()
    result = compare(set_dir)
    (set_dir / "compare.json").write_text(json.dumps(result, indent=2) + "\n")
    print_table(result)


if __name__ == "__main__":
    main()
