#!/usr/bin/env python3
"""summarise.py — aggregate the graded runs of one sweep.

Usage:
    ./summarise.py <SWEEP_DIR>

Reads <SWEEP_DIR>/manifest.json (written by sweep.sh) and runs/<run id>/result.json for each run,
and writes <SWEEP_DIR>/summary.json: score and exact count over runs (mean / stdev / min / max),
the mean score per tier, per level and per case, and mean tokens, wall time and unreadable answers per run.
A run without result.json (run.py failed) is listed as missing and left out of the aggregates.
Pure function of the graded files: rerun it any time.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def spread(values: list[float]) -> dict:
    return {"mean": round(statistics.mean(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    sweep_dir = Path(argv[0])
    manifest = json.loads((sweep_dir / "manifest.json").read_text())
    results, missing = [], []
    for run_id in manifest["runs"]:
        path = RUNS / run_id / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            missing.append(run_id)
    summary = {"name": manifest["name"], "runs": len(manifest["runs"]), "graded": len(results),
               "missing": missing}
    if results:
        cases = [c["id"] for c in results[0]["per_case"]]
        summary.update({
            "score": spread([r["score"] for r in results]),
            "exact": spread([r["exact"] for r in results]),
            "cases": results[0]["cases"],
            "tiers": {t: round(statistics.mean(r.get("tiers", {}).get(t, 0) for r in results), 4)
                      for t in results[0].get("tiers", {})},
            "levels": {lv: round(statistics.mean(r["levels"][lv] for r in results), 4)
                       for lv in results[0]["levels"]},
            "per_case": {cid: round(statistics.mean(
                next(c["score"] for c in r["per_case"] if c["id"] == cid) for r in results), 4)
                for cid in cases},
            "stats": {k: round(statistics.mean(r["stats"][k] for r in results), 1)
                      for k in results[0]["stats"]},
        })
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{summary['name']}: {summary['graded']}/{summary['runs']} graded")
    if results:
        s, e = summary["score"], summary["exact"]
        print(f"score {s['mean']:.3f} ± {s['stdev']:.3f} [{s['min']:.3f}, {s['max']:.3f}]  "
              f"exact {e['mean']:.1f}/{summary['cases']}")
        print("tiers:  " + "  ".join(f"{k} {v:.2f}" for k, v in summary["tiers"].items()))
        print("levels: " + "  ".join(f"{k} {v:.2f}" for k, v in summary["levels"].items()))
        print("stats:  " + "  ".join(f"{k} {v}" for k, v in summary["stats"].items()))
        for cid, v in summary["per_case"].items():
            print(f"  {cid:28} {v:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
