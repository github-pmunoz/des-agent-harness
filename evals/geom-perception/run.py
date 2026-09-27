#!/usr/bin/env python3
"""run.py — ask every perception case once, through send_direct, and keep what came back.

Usage:
    run.py <SETTINGS.json> <RUN_DIR>

Settings keys: model, port, think, temperature, max_tokens, timeout (seconds per case), grid
(overlay on or off: a setting of the run, never the model's choice), image_px (rendered size of
the 800-unit layout), cases (ids to run; empty runs all).

Writes into RUN_DIR, so grading needs nothing but the run dir:
- cases.json            id, level, tier, topic, kind, prompt, truth, truth_rects, tol per case
- images/<id>.png       the image the model saw
- truth/<id>.png        the truth mask for rects / polygon cases (1-bit, viewable)
- responses/<id>.json   send_direct's full completion, its exit status and the wall time
- completions.jsonl     send_direct's telemetry log

send_direct runs as `python -m desh.llama.send_direct` with this interpreter and environment, so a
PYTHONPATH pointing at a harness snapshot decides which code sends the requests.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import build                     # noqa: E402
from geom import render, save_mask          # noqa: E402

KEYS = {"model", "port", "think", "temperature", "max_tokens", "timeout", "grid", "image_px", "cases"}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    settings = json.loads(Path(argv[0]).read_text())
    if unknown := set(settings) - KEYS:
        print(f"error: unknown settings keys: {sorted(unknown)}", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    for sub in ("images", "truth", "responses"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    cases = build()
    if wanted := settings.get("cases"):
        if missing := set(wanted) - {c.id for c in cases}:
            print(f"error: no such cases: {sorted(missing)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in wanted]

    grid, image_px = settings.get("grid", False), settings.get("image_px", 800)
    records = []
    for case in cases:
        image = run_dir / "images" / f"{case.id}.png"
        render(case.layers, image, grid=grid, image_px=image_px)
        if case.truth_mask is not None:
            save_mask(case.truth_mask, run_dir / "truth" / f"{case.id}.png")
        prompt = case.prompt(grid)
        records.append({"id": case.id, "level": case.level, "topic": case.topic, "kind": case.kind,
                        "tier": case.tier, "prompt": prompt, "truth": case.truth,
                        "truth_rects": case.truth_rects, "tol": case.tol})

        cmd = [sys.executable, "-m", "desh.llama.send_direct",
               "-p", str(settings["port"]), "-m", settings["model"],
               "-t", str(settings["temperature"]), "-mt", str(settings["max_tokens"]),
               "-to", str(settings["timeout"]), "-up", prompt, "-i", str(image),
               "-fo", "-l", str(run_dir / "completions.jsonl")]
        if settings.get("think"):
            cmd.append("-th")
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall_ms = round((time.monotonic() - start) * 1000)
        try:
            completion = json.loads(proc.stdout) if proc.returncode == 0 else None
        except ValueError:
            completion = None
        (run_dir / "responses" / f"{case.id}.json").write_text(json.dumps(
            {"returncode": proc.returncode, "wall_ms": wall_ms, "stderr": proc.stderr[-2000:],
             "completion": completion}, indent=1, ensure_ascii=False))
        usage = (completion or {}).get("usage") or {}
        print(f"{case.id:28} exit={proc.returncode} {wall_ms / 1000:6.1f}s "
              f"prompt={usage.get('prompt_tokens', '-')} completion={usage.get('completion_tokens', '-')}",
              flush=True)

    (run_dir / "cases.json").write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
