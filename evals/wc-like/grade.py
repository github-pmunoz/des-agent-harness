#!/usr/bin/env python3
"""Grade one wc-like eval run and write result.json into the run folder.

Usage:
    ./grade.py runs/run-<timestamp>-<hash>

Inputs read from the run folder:
    workspace/            the agent's project root: wc.py, test_wc.py, train/
    completions*.json     JSONL, one {"timestamp","payload","response"} per completion
    des_log*.json         JSONL of engine steps (optional; only step and error counts are used)
    run_manifest.json     wall time and exit status written by run_eval.sh (optional)
    run_settings.json     the flat config the run was launched with (optional)

The held-out goldens come from test/goldens.json next to this script; the run
never sees that folder. Everything the grader learns is written to
<run>/result.json and a one-screen summary is printed.

Checks:
    held_out      run `python wc.py FILE` on every held-out file, compare the three counts
    cli_contract  no args and a missing file exit non-zero with something on stderr
    pytest        the agent's own suite, run from the workspace
    stdlib_only   imports of wc.py resolve to the standard library (test_wc.py may add pytest)
    holdout_refs  no deliverable and no file-touching tool call names test/ or a parent dir
    stats         tokens, cache hits, prompt/generation time, tool-call mix, pytest rounds
"""
import ast
import glob
import json
import os
import io
import re
import subprocess
import sys
import tokenize

EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
HOLDOUT_DIR = os.path.join(EVAL_ROOT, "test")
HOLDOUT_GOLDENS = os.path.join(HOLDOUT_DIR, "goldens.json")

# Tools whose arguments name paths or shell commands. Only these can reach the held-out set;
# scratchpad notes and prose mention test/ legitimately because the task text forbids it.
PATH_ARGS = {"Read": ("file_path",), "Write": ("file_path",), "Edit": ("file_path",), "Bash": ("command",)}
HOLDOUT_PATTERN = re.compile(r"(^|[\s'\"=(/])test/|\.\./")


# --- helpers -------------------------------------------------------------------

def first_glob(run_dir: str, pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
    return hits[0] if hits else None


def read_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_cli(workspace: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "wc.py", *args], cwd=workspace,
                          capture_output=True, text=True, timeout=timeout)


# --- checks --------------------------------------------------------------------

def check_held_out(workspace: str) -> dict:
    """Run the CLI on every held-out file and compare lines/words/bytes to the goldens."""
    goldens = read_json(HOLDOUT_GOLDENS) or {}
    passed, failures = 0, []
    for name, expected in sorted(goldens.items()):
        path = os.path.join(HOLDOUT_DIR, name)
        try:
            proc = run_cli(workspace, path)
        except subprocess.TimeoutExpired:
            failures.append({"file": name, "error": "timeout"})
            continue
        parts = proc.stdout.split()
        try:
            got = {"lines": int(parts[0]), "words": int(parts[1]), "bytes": int(parts[2])}
        except (IndexError, ValueError):
            failures.append({"file": name, "error": "unparseable output", "stdout": proc.stdout[:200],
                             "stderr": proc.stderr[:200], "rc": proc.returncode})
            continue
        if proc.returncode == 0 and got == expected:
            passed += 1
        else:
            failures.append({"file": name, "got": got, "expected": expected, "rc": proc.returncode})
    return {"passed": passed, "total": len(goldens), "score": passed / len(goldens) if goldens else 0.0,
            "failures": failures}


def check_cli_contract(workspace: str) -> dict:
    """The two error paths the task spells out, plus one line per file on multiple inputs."""
    no_args = run_cli(workspace)
    missing = run_cli(workspace, "does_not_exist.txt")
    names = sorted(read_json(HOLDOUT_GOLDENS) or {})[:3]
    multi = run_cli(workspace, *[os.path.join(HOLDOUT_DIR, n) for n in names])
    return {
        "no_args_nonzero": no_args.returncode != 0 and no_args.stderr.strip() != "",
        "missing_file_nonzero": missing.returncode != 0 and missing.stderr.strip() != "",
        "one_line_per_file": len(multi.stdout.splitlines()) == len(names),
    }


def check_pytest(workspace: str) -> dict:
    """Run the agent's own suite. The summary line is the only thing parsed."""
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                              cwd=workspace, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ran": False, "error": "timeout"}
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors)", tail)}
    return {"ran": True, "rc": proc.returncode, "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0),
            "summary": tail}


def imports_of(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return sorted(names)


def check_stdlib_only(workspace: str) -> dict:
    """wc.py must import only the standard library; test_wc.py may also import pytest."""
    allowed_extra = {"test_wc.py": {"pytest", "wc"}, "wc.py": set()}
    result = {"ok": True, "files": {}}
    for name, extra in allowed_extra.items():
        path = os.path.join(workspace, name)
        if not os.path.exists(path):
            result["ok"] = False
            result["files"][name] = {"missing": True}
            continue
        try:
            names = imports_of(path)
        except SyntaxError as e:
            result["ok"] = False
            result["files"][name] = {"syntax_error": str(e)}
            continue
        foreign = [n for n in names if n not in sys.stdlib_module_names and n not in extra]
        result["files"][name] = {"imports": names, "foreign": foreign}
        if foreign:
            result["ok"] = False
    return result


def code_lines(path: str) -> list[tuple[int, str]]:
    """(line number, code text) for every line, with docstrings dropped and comments cut off:
    only code can reach the held-out set, and prose legitimately says "test/" because the task
    forbids it. A file that does not parse is scanned whole, so a broken deliverable is never
    given the benefit of the doubt."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=path)
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError):
        return list(enumerate(lines, 1))
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                skipped.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    comment_at = {tok.start[0]: tok.start[1] for tok in tokens if tok.type == tokenize.COMMENT}
    out = []
    for n, line in enumerate(lines, 1):
        if n in skipped:
            continue
        out.append((n, line[:comment_at[n]] if n in comment_at else line))
    return out


def check_holdout_refs(workspace: str, completions: list[dict]) -> dict:
    """Any mention of test/ or a parent directory in a deliverable's code or a path-carrying tool call."""
    in_files = {}
    for name in ("wc.py", "test_wc.py"):
        path = os.path.join(workspace, name)
        if os.path.exists(path):
            lines = [n for n, text in code_lines(path) if HOLDOUT_PATTERN.search(text)]
            if lines:
                in_files[name] = lines
    in_tools = []
    for i, row in enumerate(completions):
        for call in tool_calls_of(row):
            for arg in PATH_ARGS.get(call["name"], ()):
                value = str(call["args"].get(arg, ""))
                if HOLDOUT_PATTERN.search(value):
                    in_tools.append({"completion": i, "tool": call["name"], arg: value[:200]})
    return {"ok": not in_files and not in_tools, "in_files": in_files, "in_tool_calls": in_tools}


# --- stats from the completions log ----------------------------------------------

# What a cut result carries, one marker per way of cutting (desh.tools.ToolRegistry.bound, Workspace.read, Workspace.bash).
CUT_MARKERS = ("characters truncated", "[showing lines", "the whole of it is saved at")
SPILL_DIR = ".desh/out"


def tool_calls_of(row: dict) -> list[dict]:
    choices = (row.get("response") or {}).get("choices") or [{}]
    message = choices[0].get("message") or {}
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": fn.get("arguments")}
        calls.append({"name": fn.get("name", "?"), "args": args if isinstance(args, dict) else {"_raw": args}})
    return calls


def pytest_rounds(completions: list[dict]) -> list[dict]:
    """Each pytest summary the model saw, in order. The tool result of completion i arrives as a
    tool-role message in the payload of completion i+1, so it is attributed to round i."""
    rounds = []
    seen = 0
    for i, row in enumerate(completions):
        messages = (row.get("payload") or {}).get("messages") or []
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        for m in tool_msgs[seen:]:
            text = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
            found = re.search(r"(?:(\d+) failed, )?(\d+) passed in", text or "")
            if found:
                rounds.append({"round": i - 1, "failed": int(found.group(1) or 0), "passed": int(found.group(2))})
        seen = len(tool_msgs)
    return rounds


def session_stats(session: dict | None) -> dict:
    """Turn-level shape of the run from the session file: how many turns, how many ended at the
    round cap and were continued, how many are compaction summaries, and which tool calls the cap
    dropped. Runs under a small cap show their pathology here, not in the token totals."""
    turns = (session or {}).get("turns") or []
    real = [t for t in turns if not t.get("summary")]
    last = real[-1] if real else {}
    results = [res.get("content", "") for t in turns for r in t.get("rounds", []) for res in r.get("results", [])]
    return {
        "turns": len(turns),
        "capped_turns": sum(1 for t in turns if t.get("stop") == "cap"),
        "summary_turns": len(turns) - len(real),
        # results the cap cut: the registry's blind cut, Read's line cut, Bash's spill
        "cut_results": sum(1 for c in results if any(m in c for m in CUT_MARKERS)),
        "stop": last.get("stop", "") if real else None,      # how the run ended; "" is an answer
        "final_answer": bool(last.get("assistant")) and not last.get("stop"),
    }


def calls_not_run(des_log: list[dict]) -> dict:
    """Tool calls a budget refused to run, by tool name. The session does not record them; the
    engine's warnings do, in one shape for both budgets:
    'Tool-call round cap reached (N); Write, Edit not run.' / 'Task deadline reached (Ns); Bash not run.'"""
    dropped = {}
    for e in des_log:
        if e.get("event") != "Warn":
            continue
        found = re.search(r"(?:round cap|deadline) reached \([^)]*\); (.+?) not run", str(e.get("payload", "")))
        if found:
            for name in found.group(1).split(", "):
                dropped[name] = dropped.get(name, 0) + 1
    return dropped


def stats_of(completions: list[dict], des_log: list[dict], manifest: dict | None) -> dict:
    usage = [((r.get("response") or {}).get("usage") or {}) for r in completions]
    timings = [((r.get("response") or {}).get("timings") or {}) for r in completions]
    tools = {}
    offset_reads = spill_reads = 0      # did the model follow a cut's pointer: a Read by range, a Read of a Bash spill
    for r in completions:
        for call in tool_calls_of(r):
            tools[call["name"]] = tools.get(call["name"], 0) + 1
            if call["name"] == "Read":
                args = call["args"]
                offset_reads += int((args.get("offset") or 1) > 1)
                spill_reads += int(str(args.get("file_path", "")).startswith(SPILL_DIR))
    finish = {}
    for r in completions:
        reason = (((r.get("response") or {}).get("choices") or [{}])[0]).get("finish_reason")
        finish[str(reason)] = finish.get(str(reason), 0) + 1
    rounds = pytest_rounds(completions)
    greens = [x["round"] for x in rounds if x["failed"] == 0 and x["passed"] > 0]
    gen_tps = [t.get("predicted_per_second") for t in timings if t.get("predicted_per_second")]
    return {
        "model": (completions[0].get("payload") or {}).get("model") if completions else None,
        "completions": len(completions),
        "tool_calls": sum(tools.values()),
        "tool_mix": tools,
        "offset_reads": offset_reads,
        "spill_reads": spill_reads,
        "finish_reasons": finish,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "prompt_tokens_cached": sum(t.get("cache_n", 0) for t in timings),
        "prompt_tokens_peak": max((u.get("prompt_tokens", 0) for u in usage), default=0),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "prompt_ms": round(sum(t.get("prompt_ms", 0) for t in timings)),
        "generation_ms": round(sum(t.get("predicted_ms", 0) for t in timings)),
        "generation_tps_mean": round(sum(gen_tps) / len(gen_tps), 1) if gen_tps else None,
        "pytest_rounds": rounds,
        "first_green_round": greens[0] if greens else None,
        "engine_steps": len(des_log),
        "engine_errors": sum(1 for e in des_log if e.get("outcome") not in (None, "ok")),
        "compactions": sum(1 for e in des_log if e.get("event") == "CompactHistory"),
        "checkpoints": sum(1 for e in des_log if e.get("event") == "CompactPendingTurn"),
        "calls_not_run": calls_not_run(des_log),
        "wall_ms": (manifest or {}).get("elapsed_ms"),
        "exit_status": (manifest or {}).get("status"),
    }



# --- main --------------------------------------------------------------------------

def grade(run_dir: str) -> dict:
    run_dir = os.path.abspath(run_dir)
    workspace = os.path.join(run_dir, "workspace")
    if not os.path.isdir(workspace):
        workspace = run_dir  # runs made before the workspace subfolder existed
    completions = read_jsonl(first_glob(run_dir, "completions*"))
    des_log = read_jsonl(first_glob(run_dir, "des*"))
    manifest = read_json(os.path.join(run_dir, "run_manifest.json"))
    session = read_json(first_glob(run_dir, "session*") or os.path.join(run_dir, "session.json"))
    settings = read_json(os.path.join(run_dir, "run_settings.json")) or read_json(os.path.join(run_dir, ".eval_config.json"))

    result = {
        "run_id": os.path.basename(run_dir),
        "settings": settings,
        "deliverables": {n: os.path.exists(os.path.join(workspace, n)) for n in ("wc.py", "test_wc.py")},
        "held_out": check_held_out(workspace) if os.path.exists(os.path.join(workspace, "wc.py")) else {"passed": 0, "total": 0, "score": 0.0, "failures": [{"error": "wc.py missing"}]},
        "cli_contract": check_cli_contract(workspace) if os.path.exists(os.path.join(workspace, "wc.py")) else {},
        "pytest": check_pytest(workspace) if os.path.exists(os.path.join(workspace, "test_wc.py")) else {"ran": False, "error": "test_wc.py missing"},
        "stdlib_only": check_stdlib_only(workspace),
        "holdout_refs": check_holdout_refs(workspace, completions),
        "stats": stats_of(completions, des_log, manifest) | session_stats(session),
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def summary(r: dict) -> str:
    s, h, p = r["stats"], r["held_out"], r["pytest"]
    lines = [
        f"{r['run_id']}  model={s['model']}",
        f"  held-out {h['passed']}/{h['total']}   pytest {p.get('passed', 0)} passed {p.get('failed', 0)} failed"
        f"   stdlib_only={r['stdlib_only']['ok']}   holdout_refs_ok={r['holdout_refs']['ok']}",
        f"  cli: {r['cli_contract']}",
        f"  {s['turns']} turns ({s['capped_turns']} capped, {s['summary_turns']} summaries, {s['compactions']} compactions,"
        f" {s['checkpoints']} checkpoints),"
        f" stop={s['stop']!r} final answer: {s['final_answer']}",
        f"  {s['completions']} completions, {s['tool_calls']} tool calls {s['tool_mix']}   not run: {s['calls_not_run']}",
        f"  cut results {s['cut_results']}, reads by offset {s['offset_reads']}, reads of a spill {s['spill_reads']}",
        f"  tokens: prompt {s['prompt_tokens']} (cached {s['prompt_tokens_cached']}, peak {s['prompt_tokens_peak']}),"
        f" completion {s['completion_tokens']}",
        f"  time: wall {s['wall_ms']} ms, prompt {s['prompt_ms']} ms, generation {s['generation_ms']} ms,"
        f" {s['generation_tps_mean']} tok/s",
        f"  pytest rounds {s['pytest_rounds']}  first green: {s['first_green_round']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print(f"usage: {sys.argv[0]} <run folder>", file=sys.stderr)
        sys.exit(2)
    print(summary(grade(sys.argv[1])))
