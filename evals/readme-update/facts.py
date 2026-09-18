"""The held-out checklist of the readme-update eval: what a README brought up to date says, and
what it no longer says. grade.py imports it; nothing here is ever copied into a run's workspace.

A fact is {"id", "group", "present": [regex...], "absent": [regex...]}. It passes when every
`present` pattern is found in README.md and no `absent` one is (re.search with re.M, so ^ and $
are line anchors and `.` stays within a line). A corrected value needs both halves: the new one
written down and the stale one gone.

Every fact is grounded in the code of the fixture commit, not in the README that was committed
afterwards: that README is one acceptable answer, and where it fell short the fact lives in the
`bonus` group, which is reported and never scored.

Patterns match the way a careful writer could phrase the fact, not one wording. A flag is matched
by its name; a default by the flag and the value sharing a line, which holds for a table row and
for a sentence alike. The README's "fully specified session" example passes explicit values, which
are not defaults, so the stale-default patterns are anchored to a table row.
"""


def flag(name: str) -> str:
    """The flag itself: --task must not pass on --task-timeout, nor --cont on --context."""
    return rf"--{name}(?![-\w])"


def row(name: str, value: str) -> str:
    """The flag's row of the CLI options table, giving this value as its default."""
    return rf"^\|[^|]*--{name}(?![-\w])[^|]*\|\s*`?{value}`?\s*\|"


FACTS = [
    # --- flags: added to the CLI since the README was last touched (src/desh_chat/cli.py) ---------
    {"id": "flag-tool-cap", "group": "flags", "present": [flag("tool-cap") + r".*\b10(\.0)?\b"]},
    {"id": "flag-checkpoint-target", "group": "flags", "present": [flag("checkpoint-target") + r".*\b0\.15\b"]},
    {"id": "flag-task", "group": "flags", "present": [flag("task")]},
    {"id": "flag-task-timeout", "group": "flags", "present": [flag("task-timeout")]},
    {"id": "flag-auto", "group": "flags", "present": [flag("auto")]},
    {"id": "flag-cont", "group": "flags", "present": [flag("cont")]},

    # --- defaults: documented before, and wrong now ------------------------------------------------
    {"id": "default-model", "group": "defaults",
     "present": [flag("model") + r".*Qwen3\.8-27B-UD-Q4_K_M-64K"],
     "absent": [row("model", r"Qwen3\.8-27B-UD-Q4_K_M")]},
    {"id": "default-context", "group": "defaults",
     "present": [flag("context") + r".*\b65536\b"], "absent": [row("context", "16384")]},
    {"id": "default-max-turn-tokens", "group": "defaults",
     "present": [flag("max-turn-tokens") + r".*\b65536\b"], "absent": [row("max-turn-tokens", "8192")]},
    {"id": "default-max-tool-rounds", "group": "defaults",
     "present": [flag("max-tool-rounds") + r".*\b30\b"], "absent": [row("max-tool-rounds", "10")]},
    {"id": "default-tool-expiration", "group": "defaults",
     "present": [flag("tool-expiration") + r".*\b10\b"], "absent": [row("tool-expiration", "6")]},
    {"id": "default-system-prompt", "group": "defaults",
     "present": [r"You are a helpful assistant\. Reply concisely\."],
     "absent": [r"^\|[^|]*--system-prompt[^|]*\|\s*toolset-specific"]},

    # --- features: behaviour a user meets and the README did not describe ----------------------------
    # The flags group asks that a flag is named with its default. These ask that the README says what
    # the thing does: the flag and the words that explain it share a line, as a table row or a sentence.

    # The cap is a share of the context window, not a number of characters.
    {"id": "tool-cap-meaning", "group": "features",
     "present": [flag("tool-cap") + r".*(?i:percent|%|share|fraction|proportion)"]},
    # What the target is a target of: the size of a mid-turn checkpoint summary.
    {"id": "checkpoint-target-meaning", "group": "features",
     "present": [flag("checkpoint-target") + r".*(?i:summar|share|fraction|proportion|percent|%)"]},
    # Task mode runs without an operator, and its timeout is a wall-clock budget for the whole run.
    {"id": "task-mode", "group": "features",
     "present": [flag("task") + r".*(?i:non-?interactive|unattended|headless|without (an |the )?operator|and exits?)",
                 flag("task-timeout") + r".*(?i:seconds|wall[- ]clock|budget|deadline)"]},
    # --auto starts the session in auto mode; --cont continues a turn that stopped at the round cap.
    {"id": "auto-and-continue", "group": "features",
     "present": [flag("auto") + r".*(?i:auto mode|without asking|confirm)",
                 flag("cont") + r".*(?i:continu|round cap|\bcap\b)"]},
    # delegate's check: a shell command the harness runs after the subagent, reporting its exit code.
    # The stale README already names a "`check` command" in passing; the exit code is what is new.
    {"id": "delegate-check", "group": "features",
     "present": [r"(?i)\bcheck\b.*\bexit\b|\bexit\b.*\bcheck\b"]},
    # The kinds are an enum the model must pick from, so a README that documents the scratchpad
    # names them; two of the five are enough to tell it from a passing mention.
    {"id": "scratchpad-kinds", "group": "features",
     "present": [r"(?i)\bkind\b", r"\bhypothesis\b", r"\bblock\b"]},
    # The second rung: when history is already folded, the pending turn is checkpointed mid-turn.
    # The stale README says "compaction mid-turn" of the first rung, so only a checkpoint counts;
    # and not in a table row, where the --checkpoint-target flag alone would pass for the mechanism.
    {"id": "compaction-ladder", "group": "features",
     "present": [r"CompactPendingTurn|^(?!\|).*(?i:mid-turn|pending turn|within a turn).*(?i:checkpoint)"
                 r"|^(?!\|).*(?i:checkpoint).*(?i:mid-turn|pending turn|within a turn)"]},
    # A turn ended by overflow, deadline or error keeps its record as the answer.
    {"id": "salvage", "group": "features",
     "present": [r"(?i)salvag.*\b(overflow|deadline|error)|\b(overflow|deadline|error).*salvag"]},
    # A call that ran is echoed with its long argument removed. \b keeps --sessions-folder out.
    {"id": "fold-functions", "group": "features",
     "present": [r"(?i)\bfold(s|ed|ing)?\b"]},

    # --- hygiene: what task.txt asks the README not to become ------------------------------------------
    {"id": "no-changelog-section", "group": "hygiene",
     "absent": [r"(?i)^#{1,6} .*(what'?s new|change ?log|recent changes|release notes)"]},
    {"id": "no-dates", "group": "hygiene", "absent": [r"\b2026-\d\d-\d\d\b"]},
    # an abbreviated or full commit hash: 7-40 hex characters holding both a digit and a letter
    {"id": "no-commit-hashes", "group": "hygiene",
     "absent": [r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b"]},

    # --- kept: true in the stale README and still true; an update must not talk itself out of it. Never
    # scored, since a README nobody touched passes. A subagent's tools are fixed (cli.build_tools gives
    # it Read, Write, Edit, Bash and the scratchpad whatever the parent was launched with); a run
    # once rewrote this as "the parent's tool registry minus delegate", after delegate.py's docstring.
    {"id": "subagent-tools-fixed", "group": "kept",
     "present": [r"(?i)always gets `?Read`?, `?Write`?, `?Edit`? and `?Bash`?"],
     "absent": [r"(?i)parent'?s tool (registry|set|s\b)"]},

    # --- bonus: stale statements the committed update itself left in place; never scored ---------------
    # The prose under the turn diagram says one compaction is all a turn gets; the ladder has two rungs.
    {"id": "diagram-prose-one-compaction", "group": "bonus", "absent": [r"(?i)after one compaction"]},
    {"id": "diagram-second-rung", "group": "bonus", "present": [r"[─►>]\s*CompactPendingTurn"]},
    # Session files carry a format number, and older formats still load.
    {"id": "session-format", "group": "bonus", "present": [r"(?i)session (file )?format|format[- ]5\b"]},
]
