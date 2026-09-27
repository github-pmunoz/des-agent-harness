#!/usr/bin/env python3
"""compare.py — put the sweeps of one sweep set side by side.

Usage:
    ./compare.py <SET_DIR>

Reads <SET_DIR>/<sweep>/summary.json for every sweep named in <SET_DIR>/sweep.json, writes
<SET_DIR>/compare.json, and prints two tables: score, exact, tier and level means, tokens and wall per
sweep, then the mean score of every case per sweep (where the arms differ case by case).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


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

    first = next(iter(graded.values()))
    levels, tiers = list(first["levels"]), list(first.get("tiers", {}))
    width = max(len(n) for n in graded) + 2
    print(f"{'sweep':{width}} runs  score         exact  " + " ".join(f"{'tier' + t:>6}" for t in tiers) + " " + " ".join(f"{lv[:6]:>6}" for lv in levels)
          + "  compl_tok  wall_s")
    for name, s in graded.items():
        sc, st = s["score"], s["stats"]
        print(f"{name:{width}} {s['graded']:>4}  {sc['mean']:.3f}±{sc['stdev']:.3f}  {s['exact']['mean']:5.1f}  "
              + " ".join(f"{s.get('tiers', {}).get(t, 0):6.2f}" for t in tiers) + " " + " ".join(f"{s['levels'].get(lv, 0):6.2f}" for lv in levels)
              + f"  {st['completion_tokens']:9.0f}  {st['wall_ms'] / 1000:6.0f}")
    print()
    cases = list(next(iter(graded.values()))["per_case"])
    print(f"{'case':28} " + " ".join(f"{n[:10]:>10}" for n in graded))
    for cid in cases:
        print(f"{cid:28} " + " ".join(f"{s['per_case'].get(cid, 0):10.2f}" for s in graded.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
