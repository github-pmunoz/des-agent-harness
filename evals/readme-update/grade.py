#!/usr/bin/env python3
"""Grade one readme-update eval run and write result.json into the run folder.

Usage:
    ./grade.py runs/run-<timestamp>-<hash>

Inputs read from the run folder:
    workspace/                 the agent's project root: a clone of the fixture branch, README.md edited
    completions*.jsonl         one {"timestamp","payload","response"} per completion, main agent and subagents alike
    des_log*.jsonl             engine steps of every run, each bracketed by RUN_START / RUN_END and keyed by "run"
    session.json               the main agent's session; session.delegate-*.json beside it, one per subagent
    run_manifest.json          wall time, exit status and the fixture commit, written by run_eval.sh
    run_settings.json          the flat config the run was launched with

The score is a sanity signal, not the point of the eval. The point is the shape of a long,
delegation-heavy run under a small context: whether it finishes, which agents overflow and are
salvaged, where compactions and checkpoints fall, and what the scratchpad was used for. So the
stats are split per agent, and the summary prints one line per delegation.

Checks:
    score      the held-out fact score, zeroed when the damage check fails: facts added to a README
               that lost sections, shrank or came with stray files are not the task done
    held_out   the fact checklist (facts.py, never copied into the workspace) against README.md
    damage     README.md changed, kept its headings and its length; nothing else touched; not committed
    leak       no path-carrying tool call reaches outside the workspace or names the answer commit
    stats      totals, then main / delegations[] / sub: tokens, stops, compactions, checkpoints,
               salvages, cut results, refused re-reads, scratchpad writes by kind
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

from facts import FACTS

EVAL_ROOT = os.path.dirname(os.path.abspath(__file__))
ANSWER_COMMIT = "a5d7e31"

# Tools whose arguments name paths or shell commands: the only way out of the workspace.
PATH_ARGS = {"Read": ("file_path",), "Write": ("file_path",), "Edit": ("file_path",), "Bash": ("command",)}

# How a subagent's completions are told from the orchestrator's: the opening of
# desh_chat.delegate.DELEGATE_SYSTEM_PROMPT. The completions log records no run id.
DELEGATE_PROMPT_HEAD = "You are a coding agent working inside one project directory."

# What a cut result carries, one marker per way of cutting (desh.tools.ToolRegistry.bound, Workspace.read, Workspace.bash).
CUT_MARKERS = ("characters truncated", "[showing lines", "the whole of it is saved at")
SPILL_DIR = ".desh/out"
# where --delegate-records keeps each delegation's brief and answer (desh_chat.delegate.RECORD_DIR)
RECORD_DIR = ".desh/delegates"
FOLD_EVENTS = ("CompactHistory", "CompactPendingTurn")
# The answer the repeated-round guard leaves (desh_chat.events.AppendRound).
REPEAT_STOP = re.compile(r"\[stopped: .* repeated three times with identical results\]\s*$")

# Fact groups that are reported and not scored: bonus is what the committed update itself missed;
# hygiene (only `absent` patterns) and kept (true before, true still) are groups a README nobody
# touched passes in full.
UNSCORED_GROUPS = ("bonus", "hygiene", "kept")

# What a run may leave in the workspace besides README.md.
ALLOWED_PATHS = ("README.md", ".desh/")


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


def git(workspace: str, *args: str) -> str:
    return subprocess.run(["git", "-C", workspace, *args], capture_output=True, text=True, timeout=30).stdout


# --- checks --------------------------------------------------------------------

def check_facts(readme: str) -> dict:
    """Each fact passes when every `present` pattern is found and no `absent` one is. Every group
    is reported; the score leaves out UNSCORED_GROUPS."""
    groups: dict[str, dict] = {}
    failures = []
    for fact in FACTS:
        missing = [p for p in fact.get("present", ()) if not re.search(p, readme, re.M)]
        stale = [p for p in fact.get("absent", ()) if re.search(p, readme, re.M)]
        ok = not missing and not stale
        g = groups.setdefault(fact["group"], {"passed": 0, "total": 0})
        g["passed"] += int(ok)
        g["total"] += 1
        if not ok:
            failures.append({"id": fact["id"], "group": fact["group"], "missing": missing, "stale": stale})
    scored = [g for name, g in groups.items() if name not in UNSCORED_GROUPS]
    passed, total = sum(g["passed"] for g in scored), sum(g["total"] for g in scored)
    return {"passed": passed, "total": total, "score": passed / total if total else 0.0,
            "groups": groups, "failures": failures}


def headings(text: str) -> list[str]:
    """Markdown heading lines outside code fences (a `# comment` in a shell example is not one)."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} ", line):
            out.append(line.rstrip())
    return out


def check_damage(workspace: str, fixture_sha: str | None) -> dict:
    """What the run did to the workspace besides the intended edit. A small-context Write can
    replace the README with a fragment, so length and headings are checked against the fixture's."""
    base_ref = fixture_sha or "HEAD"
    base = git(workspace, "show", f"{base_ref}:README.md")
    path = os.path.join(workspace, "README.md")
    now = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    touched = set(git(workspace, "diff", "--name-only", base_ref).split())
    touched |= {line[3:] for line in git(workspace, "status", "--porcelain").splitlines() if line.startswith("??")}
    others = sorted(p for p in touched if not any(p == a or p.startswith(a) for a in ALLOWED_PATHS))
    lost = [h for h in headings(base) if h not in headings(now)]
    base_lines, lines = len(base.splitlines()), len(now.splitlines())
    return {
        "changed": now != base,
        "lines": lines,
        "base_lines": base_lines,
        "kept_length": lines >= 0.8 * base_lines,
        "headings_lost": lost,
        "other_files_touched": others,
        "committed": bool(fixture_sha) and git(workspace, "rev-parse", "HEAD").strip() != fixture_sha,
        "ok": now != base and lines >= 0.8 * base_lines and not lost and not others,
    }


def score_of(held_out: dict, damage: dict) -> float:
    """The run's score: the fact score of an undamaged README, 0 otherwise. The fact score alone
    rewards a rewrite that adds the new facts and drops sections a reader relied on."""
    return held_out["score"] if damage["ok"] else 0.0


def check_leak(workspace: str, completions: list[dict], live: str | None = None) -> dict:
    """Any path-carrying tool call that climbs out of the workspace, names the live repository
    around it, or names the commit that holds the answer. `live` is where the workspace was while
    the agent ran (run_manifest.json), when that is not where it is now."""
    repo_root = os.path.dirname(os.path.dirname(EVAL_ROOT))
    hits = []
    for i, row in enumerate(completions):
        for call in tool_calls_of(row):
            for arg in PATH_ARGS.get(call["name"], ()):
                value = str(call["args"].get(arg, ""))
                outside = value.replace(os.path.abspath(workspace), "")
                if live:
                    outside = outside.replace(live, "")
                if "../" in outside or repo_root in outside or ANSWER_COMMIT in outside:
                    hits.append({"completion": i, "tool": call["name"], arg: value[:200]})
    return {"ok": not hits, "in_tool_calls": hits}


# --- splitting the logs per agent --------------------------------------------------

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


def ts_of(row: dict) -> float:
    """A completion's timestamp as epoch seconds. It is written to the second, after the completion."""
    try:
        return datetime.fromisoformat(row.get("timestamp", "")).timestamp()
    except ValueError:
        return 0.0


def is_subagent(row: dict) -> bool:
    messages = (row.get("payload") or {}).get("messages") or [{}]
    content = messages[0].get("content") if messages[0].get("role") == "system" else ""
    return isinstance(content, str) and content.startswith(DELEGATE_PROMPT_HEAD)


def engine_runs(des_log: list[dict]) -> list[dict]:
    """The engine runs in the log, in start order: {run, delegate, session, start, end, rows}.
    A subagent is an engine run of its own, nested in time inside the parent's tool step."""
    runs: dict[str, dict] = {}
    for e in des_log:
        run = runs.setdefault(e.get("run", ""), {"run": e.get("run", ""), "delegate": False, "session": None,
                                                 "start": e.get("ts", 0.0), "end": None, "rows": []})
        if e.get("event") == "RUN_START":
            run.update(delegate=bool(e.get("delegate")), session=e.get("session"), start=e.get("ts", 0.0))
        elif e.get("event") == "RUN_END":
            run["end"] = e.get("ts")
        else:
            run["rows"].append(e)
    return sorted(runs.values(), key=lambda r: r["start"])


def split_completions(completions: list[dict], children: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """The orchestrator's completions, and each subagent's. The system prompt says which kind a
    completion is; among subagents, runs never overlap (a delegation blocks its parent), so the
    first run whose window holds the timestamp owns it."""
    main, per_child = [], [[] for _ in children]
    for row in completions:
        if not is_subagent(row):
            main.append(row)
            continue
        t = ts_of(row)
        for i, child in enumerate(children):
            end = child["end"] if child["end"] is not None else float("inf")
            if child["start"] - 1 <= t <= end + 1:
                per_child[i].append(row)
                break
    return main, per_child


# --- stats -------------------------------------------------------------------------

# A memory tool is named <slot>_<verb> (desh_chat.memory: the slot is the memory's name).
MEMORY_TOOL = re.compile(r"^([a-z]+)_(write|delete|clear)$")


def memory_stats(completions: list[dict], fold_ts: list[float]) -> dict:
    """What each memory was used for, by slot, from the calls the model issued. The questions a
    memory tool has to answer: was it written as the work went or in one burst (write_rounds
    against writes), and was it ever load-bearing — did a compaction or a checkpoint fold results
    away AFTER something was written (folds_after_first_write)? A memory written once at the end,
    or in a run where nothing folded, was never needed, whatever the score."""
    slots: dict[str, dict] = {}
    for r in completions:
        written: set[str] = set()
        for call in tool_calls_of(r):
            found = MEMORY_TOOL.match(call["name"])
            if not found:
                continue
            slot = slots.setdefault(found.group(1), {"calls": {}, "writes": 0, "write_rounds": 0, "first_write_ts": None,
                                                     "writes_before_first_fold": 0})
            slot["calls"][call["name"]] = slot["calls"].get(call["name"], 0) + 1
            if found.group(2) == "write":
                slot["writes"] += 1
                written.add(found.group(1))
                slot["first_write_ts"] = ts_of(r) if slot["first_write_ts"] is None else slot["first_write_ts"]
                slot["writes_before_first_fold"] += int(bool(fold_ts) and ts_of(r) <= min(fold_ts))
        for name in written:
            slots[name]["write_rounds"] += 1
    for slot in slots.values():
        first = slot.pop("first_write_ts")
        slot["folds_after_first_write"] = sum(1 for t in fold_ts if first is not None and t > first)
        if not fold_ts:
            slot["writes_before_first_fold"] = None
    return slots


def usage_stats(completions: list[dict], fold_ts: list[float] | None = None) -> dict:
    first_fold_ts = min(fold_ts) if fold_ts else None
    usage = [((r.get("response") or {}).get("usage") or {}) for r in completions]
    timings = [((r.get("response") or {}).get("timings") or {}) for r in completions]
    tools: dict[str, int] = {}
    kinds: dict[str, int] = {}          # what the model wrote to the scratchpad, by kind
    offset_reads = spill_reads = record_reads = early_writes = 0
    issued: dict[str, int] = {}         # identical calls (name and arguments), over the agent's whole run
    for r in completions:
        for call in tool_calls_of(r):
            tools[call["name"]] = tools.get(call["name"], 0) + 1
            if not call["name"].startswith("scratchpad_"):
                identity = call["name"] + " " + json.dumps(call["args"], sort_keys=True)
                issued[identity] = issued.get(identity, 0) + 1
            if call["name"] == "Read":
                offset_reads += int((call["args"].get("offset") or 1) > 1)
                spill_reads += int(str(call["args"].get("file_path", "")).startswith(SPILL_DIR))
            if call["name"] in ("Read", "Bash"):
                record_reads += int(RECORD_DIR in str(call["args"].get("file_path") or call["args"].get("command") or ""))
            if call["name"] == "scratchpad_write":
                kind = str(call["args"].get("kind"))
                kinds[kind] = kinds.get(kind, 0) + 1
                early_writes += int(first_fold_ts is not None and ts_of(r) <= first_fold_ts)
    finish: dict[str, int] = {}
    for r in completions:
        reason = (((r.get("response") or {}).get("choices") or [{}])[0]).get("finish_reason")
        finish[str(reason)] = finish.get(str(reason), 0) + 1
    return {
        "completions": len(completions),
        "tool_calls": sum(tools.values()),
        "tool_mix": tools,
        # A call issued again with the same arguments. The re-read guard sees one turn's visible
        # rounds; this counts across checkpoints and cap-continues, where a loop actually lives.
        "repeated_calls": sum(n - 1 for n in issued.values()),
        "most_repeated_call": max(issued.values(), default=0),
        "offset_reads": offset_reads,      # did the model follow a cut's pointer: a Read by range,
        "spill_reads": spill_reads,         # a Read of a Bash spill
        "record_reads": record_reads,       # a Read or Bash call on a delegation record (brief or answer)
        "scratchpad_writes": sum(kinds.values()),
        "scratchpad_kinds": kinds,
        # notes taken before the first compaction or checkpoint are the ones that survive it; None when nothing folded
        "scratchpad_writes_before_first_fold": early_writes if first_fold_ts is not None else None,
        "memory": memory_stats(completions, fold_ts or []),
        "finish_reasons": finish,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "prompt_tokens_cached": sum(t.get("cache_n", 0) for t in timings),
        "prompt_tokens_peak": max((u.get("prompt_tokens", 0) for u in usage), default=0),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "prompt_ms": round(sum(t.get("prompt_ms", 0) for t in timings)),
        "generation_ms": round(sum(t.get("predicted_ms", 0) for t in timings)),
    }


def calls_not_run(rows: list[dict]) -> dict:
    """Tool calls a budget refused to run, by tool name, from the engine's warnings:
    'Tool-call round cap reached (N); Write, Edit not run.' / 'Task deadline reached (Ns); Bash not run.'"""
    dropped: dict[str, int] = {}
    for e in rows:
        if e.get("event") != "Warn":
            continue
        found = re.search(r"(?:round cap|deadline) reached \([^)]*\); (.+?) not run", str(e.get("payload", "")))
        if found:
            for name in found.group(1).split(", "):
                dropped[name] = dropped.get(name, 0) + 1
    return dropped


def engine_stats(rows: list[dict]) -> dict:
    def warns(text: str) -> int:
        return sum(1 for e in rows if e.get("event") == "Warn" and text in str(e.get("payload", "")))
    return {
        "engine_steps": len(rows),
        "engine_errors": sum(1 for e in rows if e.get("outcome") not in (None, "ok")),
        "compactions": sum(1 for e in rows if e.get("event") == "CompactHistory"),
        "checkpoints": sum(1 for e in rows if e.get("event") == "CompactPendingTurn"),
        "calls_not_run": calls_not_run(rows),
        # compactions the model first answered with nothing (asked again), and those folded with the harness's digest after two
        "summary_retries": warns("asking again"),
        "fallback_summaries": warns("digest"),
        # turns that ended by overflow, deadline or error and had their record stand as the answer
        "salvaged_turns": warns("salvaged"),
    }


def is_summary(turn: dict) -> bool:
    """A compaction's synthetic turn: stop "summary" in a format-6 session, the summary flag before."""
    return turn.get("stop") == "summary" or bool(turn.get("summary"))


def stop_of(turn: dict) -> str:
    """How the turn ended, "" for an answer: format 6 writes "answer", older sessions no key, and
    the summaries of older runs tally "" — so both read the same here."""
    stop = turn.get("stop", "")
    return "" if stop == "answer" else stop


def session_stats(session: dict | None) -> dict:
    """Turn-level shape of one agent's run from its session file, and what its scratchpad held
    at the end. `stop` is how the run ended; "" is an answer."""
    turns = (session or {}).get("turns") or []
    real = [t for t in turns if not is_summary(t)]
    last = real[-1] if real else {}
    results = [res.get("content", "") for t in turns for r in t.get("rounds", []) for res in r.get("results", [])]
    # format 7 records every memory under "memory", by slot; format 6 had the scratchpad alone
    memory = (turns[-1].get("memory") if turns else None) or {}
    pad = memory.get("scratchpad") or (turns[-1].get("scratchpad") if turns else None) or {}
    # Before format 6 the repeated-round guard ended a turn without a stop reason: the session
    # records an answered turn whose answer is the guard's own note. It is told apart here by that
    # note, so those runs tally "repeat" as the newer ones do.
    stop = stop_of(last) if real else None
    if stop == "" and REPEAT_STOP.search(last.get("assistant") or ""):
        stop = "repeat"
    final_kinds: dict[str, int] = {}
    for entry in pad.values():
        kind = entry.get("kind", "fact") if isinstance(entry, dict) else "fact"
        final_kinds[kind] = final_kinds.get(kind, 0) + 1
    return {
        "turns": len(turns),
        "rounds": sum(len(t.get("rounds", [])) for t in turns),
        "capped_turns": sum(1 for t in turns if t.get("stop") == "cap"),
        # replies cut at the token limit: the turn ended on its record and, with the length prompt, was continued once
        "length_turns": sum(1 for t in turns if t.get("stop") == "length"),
        "summary_turns": len(turns) - len(real),
        # results the cap cut: the registry's blind cut, Read's line cut, Bash's spill
        "cut_results": sum(1 for c in results if any(m in c for m in CUT_MARKERS)),
        # read-only calls answered with the re-read notice instead of running (a loop the harness caught)
        "rereads_refused": sum(1 for c in results if c.startswith("Not run: ") and "already been answered" in c),
        # calls the round budget held back: the round's results would not have fit once folded
        "calls_deferred": sum(1 for c in results if c.startswith("Not run: this round's results")),
        "scratchpad_final": final_kinds,
        # entries held at the end, by slot
        "memory_final": {slot: len(value) for slot, value in (memory or ({"scratchpad": pad} if pad else {})).items()},
        "open_todos": final_kinds.get("todo", 0),
        "stop": stop,
        "final_answer": bool(last.get("assistant")) and not stop,
    }


def agent_stats(completions: list[dict], rows: list[dict], session: dict | None) -> dict:
    folds = [e.get("ts") for e in rows if e.get("event") in FOLD_EVENTS]
    return usage_stats(completions, folds) | engine_stats(rows) | session_stats(session)


def delivered(answer: str, main_session: dict | None) -> str | None:
    """The tool result the orchestrator got for this subagent's answer: the answer, the check
    block after it, and the registry's cut over both. None when it is no longer in the session."""
    head = answer[:200]
    for t in (main_session or {}).get("turns") or []:
        for r in t.get("rounds", []):
            for res in r.get("results", []):
                if res.get("name") == "delegate" and head and res.get("content", "").startswith(head):
                    return res["content"]
    return None


def check_exit_of(result: str | None) -> int | None:
    """The exit code in '[check `cmd`: exit N]'; None when there was no check."""
    found = re.search(r"\[check `.*?`: exit (-?\d+)\]", result or "", re.S)
    return int(found.group(1)) if found else None


def chars_lost_of(result: str | None) -> int:
    """What the registry's cut dropped from the middle of the answer. A delegate answer has no
    pointer to the rest and no spill: what is cut here never reaches the orchestrator."""
    found = re.search(r"\[\.\.\. (\d+) characters truncated \.\.\.\]", result or "")
    return int(found.group(1)) if found else 0


# What a delegation was FOR, read off its brief. A subagent is for work whose details the
# orchestrator does not need — an investigation that reports findings, a rewrite, a test loop. Two
# kinds of brief spend a whole child run on something else: "transport" asks for file text back
# (the orchestrator has Read, and the answer is capped, so the text arrives cut and is paged back in
# from the spill file), and "verify" asks a subagent to look at a result (the check argument, or
# one call of the orchestrator's own, does that).
TRANSPORT_BRIEF = re.compile(r"verbatim|word[- ]for[- ]word|exact(?: current)? (?:text|contents?)|full (?:text|contents?) of", re.I)
VERIFY_BRIEF = re.compile(r"^\W*(?:verify|confirm|validate|double[- ]check|check (?:that|whether|if))\b", re.I)


def brief_kind(brief: str) -> str:
    if VERIFY_BRIEF.search(brief):
        return "verify"
    return "transport" if TRANSPORT_BRIEF.search(brief) else "work"


def orchestration_stats(main_session: dict | None, delegations: list[dict], main: dict) -> dict:
    """How the orchestrator divided the work between itself and its subagents: the delegations by
    kind (brief_kind), briefs issued twice, and what its own rounds went to."""
    seen: dict[str, int] = {}
    for d in delegations:
        seen[d["brief_head"]] = seen.get(d["brief_head"], 0) + 1
    rounds = [r for t in (main_session or {}).get("turns") or [] for r in t.get("rounds", []) if r.get("tool_calls")]
    def names(r: dict) -> list[str]:
        return [(c.get("function") or c).get("name", "") for c in r["tool_calls"]]
    kinds = [d["kind"] for d in delegations]
    return {
        "delegations": len(delegations),
        "work": kinds.count("work"), "transport": kinds.count("transport"), "verify": kinds.count("verify"),
        "repeated_briefs": sum(n - 1 for n in seen.values()),
        "with_check": sum(1 for d in delegations if d["check_exit"] is not None),
        "main_rounds": len(rounds),
        "main_delegate_rounds": sum(1 for r in rounds if "delegate" in names(r)),
        # rounds that only persisted: the memory calls that were meant to ride with other calls
        "main_memory_only_rounds": sum(1 for r in rounds if all(MEMORY_TOOL.match(n) for n in names(r))),
        "main_spill_reads": main["spill_reads"],     # a delegate answer over the cap, paged back in
    }


def stats_of(run_dir: str, completions: list[dict], des_log: list[dict], manifest: dict | None,
             main_session: dict | None) -> dict:
    runs = engine_runs(des_log)
    children = [r for r in runs if r["delegate"]]
    main_rows = [e for r in runs if not r["delegate"] for e in r["rows"]]
    main_completions, child_completions = split_completions(completions, children)

    delegations = []
    for child, rows in zip(children, child_completions):
        session = read_json(os.path.join(run_dir, os.path.basename(child["session"]))) if child["session"] else None
        turns = (session or {}).get("turns") or []
        real = [t for t in turns if not is_summary(t)]
        s = agent_stats(rows, child["rows"], session)
        answer = real[-1].get("assistant", "") if real else ""
        result = delivered(answer, main_session)
        delegations.append({
            "session": os.path.basename(child["session"] or ""),
            "brief": " ".join((turns[0].get("user") or "").split())[:120] if turns else "",
            "brief_head": " ".join((turns[0].get("user") or "").split())[:400] if turns else "",
            "kind": brief_kind(turns[0].get("user") or "") if turns else "work",
            # the brief points at an earlier delegation's record instead of retyping its findings
            "points_to_record": bool(turns) and RECORD_DIR in (turns[0].get("user") or ""),
            "wall_ms": round((child["end"] - child["start"]) * 1000) if child["end"] else None,
            "check_exit": check_exit_of(result),
            "answer_chars": len(answer),
            "answer_chars_lost": chars_lost_of(result),
            **s,
        })

    def total(key: str) -> int:
        return sum(d[key] for d in delegations)
    sub = {
        "count": len(delegations),
        "overflows": sum(1 for d in delegations if d["stop"] == "overflow"),
        "deadlines": sum(1 for d in delegations if d["stop"] == "deadline"),
        "errors": sum(1 for d in delegations if d["stop"] == "error"),
        "capped": sum(1 for d in delegations if d["stop"] == "cap"),
        "repeat_stops": sum(1 for d in delegations if d["stop"] == "repeat"),
        "length_stops": sum(1 for d in delegations if d["stop"] == "length"),
        "salvaged": total("salvaged_turns"),
        "answered": sum(1 for d in delegations if d["final_answer"]),
        "checks_failed": sum(1 for d in delegations if d["check_exit"] not in (None, 0)),
        "answers_cut": sum(1 for d in delegations if d["answer_chars_lost"]),
        "answer_chars_lost": total("answer_chars_lost"),
        "prompt_tokens_peak": max((d["prompt_tokens_peak"] for d in delegations), default=0),
        "rounds_mean": round(total("rounds") / len(delegations), 1) if delegations else None,
        "compactions": total("compactions"),
        "checkpoints": total("checkpoints"),
        "scratchpad_writes": total("scratchpad_writes"),
        "record_reads": total("record_reads"),
        "briefs_to_records": sum(1 for d in delegations if d["points_to_record"]),
    }
    main = agent_stats(main_completions, main_rows, main_session)
    everything = usage_stats(completions) | engine_stats([e for r in runs for e in r["rows"]])
    return everything | {
        "model": (completions[0].get("payload") or {}).get("model") if completions else None,
        "cut_results": main["cut_results"] + total("cut_results"),
        "rereads_refused": main["rereads_refused"] + total("rereads_refused"),
        "calls_deferred": main["calls_deferred"] + total("calls_deferred"),
        "unattributed_completions": len(completions) - len(main_completions) - sum(len(c) for c in child_completions),
        "stop": main["stop"],                   # the orchestrator's: how the run ended
        "final_answer": main["final_answer"],
        "wall_ms": (manifest or {}).get("elapsed_ms"),
        "exit_status": (manifest or {}).get("status"),
        "main": main,
        "sub": sub,
        "orchestration": orchestration_stats(main_session, delegations, main),
        "delegations": delegations,
    }


# --- main --------------------------------------------------------------------------

def grade(run_dir: str) -> dict:
    run_dir = os.path.abspath(run_dir)
    workspace = os.path.join(run_dir, "workspace")
    completions = read_jsonl(first_glob(run_dir, "completions*"))
    des_log = read_jsonl(first_glob(run_dir, "des*"))
    manifest = read_json(os.path.join(run_dir, "run_manifest.json"))
    # The main agent's session is session.json by contract; the subagents' sit beside it.
    session = read_json(os.path.join(run_dir, "session.json"))
    if session is None:
        raise FileNotFoundError(f"no session.json in {run_dir}")
    readme_path = os.path.join(workspace, "README.md")
    readme = open(readme_path, encoding="utf-8").read() if os.path.exists(readme_path) else ""

    held_out = check_facts(readme)
    damage = check_damage(workspace, (manifest or {}).get("fixture_sha"))
    result = {
        "run_id": os.path.basename(run_dir),
        "settings": read_json(os.path.join(run_dir, "run_settings.json")),
        # the frozen code the run executed (harness_snapshot.sh), None for runs launched before it
        "harness": (manifest or {}).get("harness"),
        "score": score_of(held_out, damage),
        "held_out": held_out,
        "damage": damage,
        "leak": check_leak(workspace, completions, (manifest or {}).get("workspace")),
        "stats": stats_of(run_dir, completions, des_log, manifest, session),
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


def summary(r: dict) -> str:
    s, h, d, m, sub = r["stats"], r["held_out"], r["damage"], r["stats"]["main"], r["stats"]["sub"]
    groups = "  ".join(f"{name} {g['passed']}/{g['total']}" for name, g in h["groups"].items())
    lines = [
        f"{r['run_id']}  model={s['model']}  context={(r['settings'] or {}).get('context')}",
        f"  score {r['score']:.2f}{'' if d['ok'] else ' (damaged)' if d['changed'] else ' (unchanged)'}  facts {h['passed']}/{h['total']} ({groups})",
        f"  readme: changed={d['changed']} lines {d['base_lines']}->{d['lines']} headings lost {len(d['headings_lost'])}"
        f" other files {d['other_files_touched']} committed={d['committed']}   leak_ok={r['leak']['ok']}",
        f"  run: stop={s['stop']!r} final answer: {s['final_answer']}  exit {s['exit_status']}  wall {s['wall_ms']} ms"
        f"  tokens prompt {s['prompt_tokens']} (cached {s['prompt_tokens_cached']}) completion {s['completion_tokens']}",
        f"  main: {m['turns']} turns ({m['capped_turns']} capped, {m['length_turns']} length-cut, {m['summary_turns']} summaries) {m['rounds']} rounds,"
        f" peak {m['prompt_tokens_peak']}, {m['compactions']} compactions, {m['checkpoints']} checkpoints,"
        f" {m['salvaged_turns']} salvaged, tools {m['tool_mix']}   not run: {m['calls_not_run']}",
        f"  main scratchpad: writes {m['scratchpad_kinds']} (before first fold: {m['scratchpad_writes_before_first_fold']}),"
        f" at the end {m['scratchpad_final']}",
        *[f"  main <{slot}>: {x['writes']} writes in {x['write_rounds']} rounds, {x['folds_after_first_write']} folds after the first"
          f" (before first fold: {x['writes_before_first_fold']}), {m['memory_final'].get(slot, 0)} entries at the end, calls {x['calls']}"
          for slot, x in m["memory"].items()],
        "  orchestration: {delegations} delegations ({work} work, {transport} transport, {verify} verify, {repeated_briefs} repeated, "
        "{with_check} with a check); main {main_rounds} rounds = {main_delegate_rounds} delegate, {main_spill_reads} spill reads, "
        "{main_memory_only_rounds} memory-only".format(**s["orchestration"]),
        f"  subagents: {sub['count']} runs, {sub['answered']} answered, {sub['overflows']} overflow, {sub['capped']} cap, {sub['repeat_stops']} repeat-stop, {sub['length_stops']} length,"
        f" {sub['deadlines']} deadline, {sub['errors']} error, {sub['salvaged']} salvaged, {sub['checks_failed']} checks failed,"
        f" peak {sub['prompt_tokens_peak']}",
        f"  records: {sub['briefs_to_records']} briefs point at one, read {m['record_reads']} times by main, {sub['record_reads']} by subagents",
        f"  {'#':>3} {'stop':<9}{'rnds':>5}{'peak':>7}{'cmp':>4}{'ckp':>4}{'slv':>4}{'cut':>4}{'rep':>4}{'rrd':>4}{'pad':>4}{'chk':>5}{'ans':>7}{'lost':>6}{'s':>6}  brief",
    ]
    for i, x in enumerate(s["delegations"]):
        check = "-" if x["check_exit"] is None else str(x["check_exit"])
        wall = "-" if x["wall_ms"] is None else f"{x['wall_ms'] / 1000:.0f}"
        lines.append(f"  {i:>3} {(x['stop'] or 'done'):<9}{x['rounds']:>5}{x['prompt_tokens_peak']:>7}{x['compactions']:>4}"
                     f"{x['checkpoints']:>4}{x['salvaged_turns']:>4}{x['cut_results']:>4}{x['repeated_calls']:>4}{x['rereads_refused']:>4}"
                     f"{x['scratchpad_writes']:>4}{check:>5}{x['answer_chars']:>7}{x['answer_chars_lost']:>6}{wall:>6}"
                     f"  {x['brief'][:60]}")
    if s["unattributed_completions"]:
        lines.append(f"  {s['unattributed_completions']} subagent completions matched no engine run")
    for f in h["failures"]:
        lines.append(f"  miss [{f['group']}] {f['id']}" + (f"  stale: {f['stale']}" if f["stale"] else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print(f"usage: {sys.argv[0]} <run folder>", file=sys.stderr)
        sys.exit(2)
    print(summary(grade(sys.argv[1])))
