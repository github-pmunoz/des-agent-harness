#!/usr/bin/env python3
"""grade.py — grade one Track A run.

Usage:
    ./grade.py <RUN_DIR>

The answer to a case is its last accepted submission (geo_submit's own rule: "the last submission
is your answer"); a rejected one after it does not erase it. Per case:
    score        1.0 when the answer is exact, else its IoU against the target (0 with no answer)
    exact, iou   of the answer; first_exact is the submission number that first matched, if any
    ops, variables and their ratios to the reference program (ops_ratio, var_ratio, on exact
                 answers only: < 1 beats the reference, > 1 is a longer program than needed)
    submissions, rejected, rounds, tools (calls per tool), stop (how the last turn ended)
    completion_tokens, peak_prompt (largest prompt of the case), wall_ms
The run's result.json holds per_case, the mean score and exact count, the mean score per level,
and per-run stats (sums or means over cases). A pure function of the run dir: rerun any time.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def grade_case(case_dir: Path) -> dict:
    manifest = json.loads((case_dir / "case_manifest.json").read_text())
    task = json.loads(Path(manifest["file"]).read_text())
    subs = lines(case_dir / "geo" / "submissions.jsonl")
    ok = [s for s in subs if s.get("ok")]
    answer = ok[-1] if ok else None
    ref = task["reference"]
    out = {"id": task["id"], "level": task["level"], "status": manifest["status"],
           "wall_ms": manifest["elapsed_ms"], "submissions": len(subs), "rejected": len(subs) - len(ok),
           "ref_ops": ref["ops"], "ref_variables": ref["variables"]}
    if answer is None:
        out.update(score=0.0, exact=False, iou=0.0, first_exact=None, ops=None, variables=None,
                   ops_ratio=None, var_ratio=None)
    else:
        ious = [v["iou"] for v in answer["layers"].values()]
        exact = bool(answer["exact"])
        out.update(exact=exact, iou=round(statistics.mean(ious), 4), score=1.0 if exact else round(statistics.mean(ious), 4),
                   first_exact=next((s["n"] for s in ok if s["exact"]), None),
                   ops=answer["ops"], variables=answer["variables"],
                   ops_ratio=round(answer["ops"] / ref["ops"], 3) if exact else None,
                   var_ratio=round(answer["variables"] / ref["variables"], 3) if exact else None,
                   answer=answer["program"])
    session = case_dir / "session.json"
    turns = json.loads(session.read_text()).get("turns", []) if session.exists() else []
    out["rounds"] = sum(len(t.get("rounds", [])) for t in turns)
    out["stop"] = turns[-1]["stop"] if turns else "none"
    out["tools"] = dict(Counter(tc["function"]["name"] for t in turns for r in t.get("rounds", [])
                                for tc in r.get("tool_calls", [])))
    usage = [(c.get("response") or {}).get("usage") or {} for c in lines(case_dir / "completions.jsonl")]
    out["completion_tokens"] = sum(u.get("completion_tokens", 0) for u in usage)
    out["peak_prompt"] = max((u.get("prompt_tokens", 0) for u in usage), default=0)
    return out


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    per_case = [grade_case(d) for d in sorted((run_dir / "cases").iterdir()) if (d / "case_manifest.json").exists()]
    levels = sorted({c["level"] for c in per_case})
    result = {
        "cases": len(per_case),
        "score": mean(c["score"] for c in per_case),
        "exact": sum(c["exact"] for c in per_case),
        "levels": {lv: mean(c["score"] for c in per_case if c["level"] == lv) for lv in levels},
        "stats": {
            "completion_tokens": sum(c["completion_tokens"] for c in per_case),
            "wall_ms": sum(c["wall_ms"] for c in per_case),
            "rounds": mean(c["rounds"] for c in per_case),
            "submissions": mean(c["submissions"] for c in per_case),
            "peak_prompt": max((c["peak_prompt"] for c in per_case), default=0),
            "ops_ratio": mean(c["ops_ratio"] for c in per_case),
            "var_ratio": mean(c["var_ratio"] for c in per_case),
        },
        "per_case": per_case,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(f"score {result['score']}  exact {result['exact']}/{result['cases']}  "
          + "  ".join(f"{lv} {v:.2f}" for lv, v in result["levels"].items()))
    for c in per_case:
        cost = f"ops {c['ops']}/{c['ref_ops']} vars {c['variables']}/{c['ref_variables']}" if c["ops"] is not None else "no answer"
        print(f"  {c['id']:18} {'EXACT' if c['exact'] else f'iou {c['iou']:.3f}':10} {cost:26} "
              f"subs {c['submissions']:2}  rounds {c['rounds']:2}  {c['wall_ms'] / 1000:5.0f} s  {c['stop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
