#!/usr/bin/env python3
"""summarise.py — aggregate the graded runs of one sweep.

Usage:
    ./summarise.py <SWEEP_DIR>

Reads <SWEEP_DIR>/manifest.json (written by sweep.sh) and runs/<run id>/result.json for each run,
and writes <SWEEP_DIR>/summary.json: score and exact count over runs (mean / stdev / min / max),
the mean score per level and per case, the exact rate per case, and the per-run stats (tokens,
wall, rounds, submissions, parsimony ratios on exact answers). A run without result.json is listed
as missing and left out. Pure function of the graded files: rerun it any time.
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


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


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
    summary = {"name": manifest["name"], "runs": len(manifest["runs"]), "graded": len(results), "missing": missing}
    if results:
        by_case: dict[str, list[dict]] = {}
        for r in results:
            for c in r["per_case"]:
                by_case.setdefault(c["id"], []).append(c)
        summary.update({
            "score": spread([r["score"] for r in results]),
            "exact": spread([r["exact"] for r in results]),
            "cases": results[0]["cases"],
            "levels": {lv: mean(r["levels"].get(lv) for r in results) for lv in results[0]["levels"]},
            "per_case": {cid: mean(c["score"] for c in cs) for cid, cs in by_case.items()},
            "exact_rate": {cid: round(sum(c["exact"] for c in cs) / len(cs), 3) for cid, cs in by_case.items()},
            "stats": {k: mean(r["stats"][k] for r in results) for k in results[0]["stats"]},
        })
    (sweep_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{summary['name']}: {summary['graded']}/{summary['runs']} graded")
    if results:
        s, e = summary["score"], summary["exact"]
        print(f"score {s['mean']:.3f} ± {s['stdev']:.3f} [{s['min']:.3f}, {s['max']:.3f}]  "
              f"exact {e['mean']:.1f}/{summary['cases']}")
        print("levels: " + "  ".join(f"{k} {v:.2f}" for k, v in summary["levels"].items()))
        print("stats:  " + "  ".join(f"{k} {v}" for k, v in summary["stats"].items()))
        for cid, v in summary["per_case"].items():
            print(f"  {cid:20} {v:.2f}  exact {summary['exact_rate'][cid]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
