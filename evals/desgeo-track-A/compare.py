#!/usr/bin/env python3
"""compare.py — put the sweeps of one sweep set side by side.

Usage:
    ./compare.py <SET_DIR>

Reads <SET_DIR>/<sweep>/summary.json for every sweep named in <SET_DIR>/sweep.json, writes
<SET_DIR>/compare.json, and prints two tables: score, exact, level means, parsimony, rounds,
tokens and wall per sweep; then the exact rate of every case per sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(v, spec: str) -> str:
    return format(v, spec) if v is not None else "-".rjust(len(format(0, spec)))


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    set_dir = Path(argv[0])
    spec = json.loads((set_dir / "sweep.json").read_text())
    rows = {}
    for sweep in spec["sweeps"]:
        path = set_dir / sweep["name"] / "summary.json"
        if path.exists():
            rows[sweep["name"]] = json.loads(path.read_text())
    (set_dir / "compare.json").write_text(json.dumps(rows, indent=1))
    graded = {n: s for n, s in rows.items() if s.get("graded")}
    if not graded:
        print("no graded sweeps")
        return 0
    levels = list(next(iter(graded.values()))["levels"])
    width = max(len(n) for n in graded) + 2
    print(f"{'sweep':{width}} runs  score         exact  " + " ".join(f"{lv:>5}" for lv in levels)
          + "  ops_r  var_r  rounds  subs  compl_tok  wall_s")
    for name, s in graded.items():
        sc, st = s["score"], s["stats"]
        print(f"{name:{width}} {s['graded']:>4}  {sc['mean']:.3f}±{sc['stdev']:.3f}  {s['exact']['mean']:5.1f}  "
              + " ".join(fmt(s["levels"].get(lv), "5.2f") for lv in levels)
              + f"  {fmt(st['ops_ratio'], '5.2f')}  {fmt(st['var_ratio'], '5.2f')}  {fmt(st['rounds'], '6.1f')}"
              f"  {fmt(st['submissions'], '4.1f')}  {st['completion_tokens']:9.0f}  {st['wall_ms'] / 1000:6.0f}")
    print()
    cases = list(next(iter(graded.values()))["exact_rate"])
    print(f"{'exact rate':20} " + " ".join(f"{n[:12]:>12}" for n in graded))
    for cid in cases:
        print(f"{cid:20} " + " ".join(f"{s['exact_rate'].get(cid, 0):12.2f}" for s in graded.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
