#!/usr/bin/env python3
"""grade.py — score one perception run from its run dir alone.

Usage:
    ./grade.py <RUN_DIR>

The answer is the JSON after the LAST "ANSWER:" in the reply's content. Per case: score in [0, 1],
exact (bool), the parsed answer, and an error string when the answer could not be read. Scoring by
kind (see cases.py): exact match for int / bool, linear credit inside tol for int_tol / num_rel,
fraction of fields for fields_int, IoU against the truth mask for rects / polygon. A rects answer
with as many rectangles as the truth has regions also reports max_err, the largest coordinate
error after pairing each answer rectangle with the truth rectangle it overlaps most (only for cases
that list their truth rectangles).

Writes <RUN_DIR>/result.json and prints a per-case table.
"""
from __future__ import annotations

import ast
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import iou, load_mask, polygon_mask, rect, union   # noqa: E402

LEVELS = {1: "count", 2: "locate", 3: "topology", 4: "outline", 5: "measure", 6: "boolean",
          7: "layers", 8: "reconstruct"}


class Unreadable(ValueError):
    pass


def parse_answer(content: str):
    at = content.rfind("ANSWER:")
    if at < 0:
        raise Unreadable("no ANSWER line")
    text = content[at + len("ANSWER:"):].strip().split("\n")[0].strip().strip("`").strip()
    if text.lower() in ("true", "yes"):
        return True
    if text.lower() in ("false", "no"):
        return False
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(text)
        except (ValueError, SyntaxError):
            continue
        try:
            json.dumps(value)            # a Python literal JSON cannot hold (a set, a tuple key) is no answer
        except TypeError:
            raise Unreadable(f"not JSON: {text[:80]!r}")
        return value
    raise Unreadable(f"not JSON: {text[:80]!r}")


def as_int(v) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise Unreadable(f"not a number: {v!r}")
    try:
        f = float(v)
    except ValueError:
        raise Unreadable(f"not a number: {v!r}")
    return round(f)


def as_rects(v) -> list[list[float]]:
    if isinstance(v, list) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v):
        v = [v]
    if not isinstance(v, list) or not v or not all(
            isinstance(r, list) and len(r) == 4 and all(isinstance(n, (int, float)) for n in r) for r in v):
        raise Unreadable("not a list of [x, y, width, height]")
    return v


def as_vertices(v) -> list[tuple[float, float]]:
    if not isinstance(v, list) or len(v) < 3 or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(n, (int, float)) for n in p) for p in v):
        raise Unreadable("not a list of at least 3 [x, y] vertices")
    return [tuple(p) for p in v]


def max_rect_error(answer: list[list[float]], truth: list[list[int]]) -> float:
    def overlap(a, b):
        return max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * \
               max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    worst, left = 0.0, list(truth)
    for a in answer:
        t = max(left, key=lambda t: overlap(a, t))
        left.remove(t)
        edges_a = (a[0], a[1], a[0] + a[2], a[1] + a[3])
        edges_t = (t[0], t[1], t[0] + t[2], t[1] + t[3])
        worst = max(worst, max(abs(p - q) for p, q in zip(edges_a, edges_t)))
    return worst


def score_case(case: dict, answer, run_dir: Path) -> dict:
    kind, truth, tol = case["kind"], case["truth"], case["tol"]
    if kind == "int":
        v = as_int(answer)
        return {"score": float(v == truth), "exact": v == truth, "err": v - truth}
    if kind == "int_tol":
        err = as_int(answer) - truth
        return {"score": max(0.0, 1 - abs(err) / tol), "exact": err == 0, "err": err}
    if kind == "num_rel":
        rel = abs(as_int(answer) - truth) / truth
        return {"score": max(0.0, 1 - rel / tol), "exact": rel == 0, "err": round(rel, 4)}
    if kind == "bool":
        if not isinstance(answer, bool):
            raise Unreadable(f"not a boolean: {answer!r}")
        return {"score": float(answer == truth), "exact": answer == truth}
    if kind == "fields_int":
        if not isinstance(answer, dict):
            raise Unreadable("not an object")
        hits = [k in answer and as_int(answer[k]) == v for k, v in truth.items()]
        return {"score": sum(hits) / len(hits), "exact": all(hits)}
    truth_mask = load_mask(run_dir / "truth" / f"{case['id']}.png")
    if kind == "rects":
        rects = as_rects(answer)
        got = union(*[rect(*r) for r in rects])
        out = {"score": iou(got, truth_mask), "n_rects": len(rects)}
        truth_rects = case.get("truth_rects")
        if truth_rects and len(truth_rects) == len(rects):
            out["max_err"] = max_rect_error(rects, truth_rects)
    elif kind == "polygon":
        vertices = as_vertices(answer)
        out = {"score": iou(polygon_mask(vertices), truth_mask), "n_vertices": len(vertices)}
    else:
        raise ValueError(f"unknown kind {kind}")
    out["score"] = round(out["score"], 4)
    out["exact"] = out["score"] == 1.0
    return out


def grade(run_dir: Path) -> dict:
    cases = json.loads((run_dir / "cases.json").read_text())
    per_case = []
    for case in cases:
        resp = json.loads((run_dir / "responses" / f"{case['id']}.json").read_text())
        completion = resp["completion"] or {}
        choice = (completion.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = completion.get("usage") or {}
        row = {"id": case["id"], "level": case["level"], "tier": case.get("tier", 1), "kind": case["kind"], "truth": case["truth"],
               "finish_reason": choice.get("finish_reason"), "wall_ms": resp["wall_ms"],
               "prompt_tokens": usage.get("prompt_tokens", 0),
               "completion_tokens": usage.get("completion_tokens", 0)}
        if resp["returncode"] != 0 or not completion:
            row.update(score=0.0, exact=False, error=f"send_direct exit {resp['returncode']}")
        else:
            try:
                answer = parse_answer(content)
                row["answer"] = answer
                row.update(score_case(case, answer, run_dir))
            except Unreadable as e:
                row.update(score=0.0, exact=False, error=str(e))
        per_case.append(row)

    levels = {}
    for lv, name in LEVELS.items():
        scores = [r["score"] for r in per_case if r["level"] == lv]
        if scores:
            levels[name] = round(statistics.mean(scores), 4)
    tiers = {str(t): round(statistics.mean(r["score"] for r in per_case if r["tier"] == t), 4)
             for t in sorted({r["tier"] for r in per_case})}
    return {
        "score": round(statistics.mean(r["score"] for r in per_case), 4),
        "tiers": tiers,
        "exact": sum(r["exact"] for r in per_case),
        "cases": len(per_case),
        "levels": levels,
        "stats": {
            "wall_ms": sum(r["wall_ms"] for r in per_case),
            "prompt_tokens": sum(r["prompt_tokens"] for r in per_case),
            "completion_tokens": sum(r["completion_tokens"] for r in per_case),
            "unreadable": sum("error" in r for r in per_case),
            "length_stops": sum(r["finish_reason"] == "length" for r in per_case),
        },
        "per_case": per_case,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(argv[0])
    result = grade(run_dir)
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    for r in result["per_case"]:
        detail = r.get("error") or f"answer={json.dumps(r.get('answer'))[:60]}"
        extra = f" max_err={r['max_err']}" if "max_err" in r else ""
        print(f"{r['id']:28} {r['score']:5.2f} {'=' if r['exact'] else ' '} "
              f"{r['completion_tokens']:6} tok {r['wall_ms'] / 1000:6.1f}s  {detail}{extra}")
    s = result["stats"]
    print(f"score {result['score']:.3f}  exact {result['exact']}/{result['cases']}  "
          f"completion {s['completion_tokens']} tok  wall {s['wall_ms'] / 1000:.0f}s  "
          f"unreadable {s['unreadable']}  length {s['length_stops']}")
    print("tiers:  " + "  ".join(f"{k} {v:.2f}" for k, v in result["tiers"].items()))
    print("levels: " + "  ".join(f"{k} {v:.2f}" for k, v in result["levels"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
