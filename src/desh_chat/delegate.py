"""
The delegate tool: the model hands a subtask to a subagent and gets back one answer.

A subagent is another Engine run over a fresh ChatState — its own history, its own pending turn,
its own round budget — sharing only the transport with the parent. It runs synchronously inside
the parent's ExecuteToolCalls step, the way Bash blocks on its subprocess, so the parent's DES
step simply takes longer. Nothing about nesting is known to the engine: the child's normal exit
becomes tool text, a harness bug inside it becomes a "tool raised" result through the registry's
invoke boundary, and Ctrl+C (a BaseException the registry lets through) reaches the parent's own
interrupt handler. The child engine therefore gets no handlers of its own.

What the tool is for is context management: the parent's window holds the task and the answer,
never the subagent's reads, searches and tool rounds. The model is told this and nothing else —
its schema is task, context, gate and check. The tools are fixed at startup (the operator's
choice, not the model's), minus delegate itself, so a subagent cannot delegate. The settings are
the parent's CURRENT ones: the registry injects state.settings on every call (see Tool.inject),
so /model, /temperature and auto mode reach the next subagent the moment they change.

The child streams to the terminal exactly as the parent does and its confirmed tools ask the
operator exactly as the parent's do; a banner marks the hand-over and the return, and every line
in between is drawn behind a gutter (sys.stdout is wrapped for the duration). When the parent
keeps a session file, each subagent run keeps its own beside it, named from the parent's, so the
work the parent never saw stays inspectable.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import uuid
import json
from dataclasses import dataclass, field, replace
from typing import TextIO

from desh.engine import Engine
from desh.llama.logger import Logger
from desh.render import Gutter, Palette, c_out
from desh.tools import DEFAULT_RESULT_CHARS, ToolRegistry
from desh_chat.coding import spill
from desh_chat.state import ChatHistory, ChatState, Deadline, InferenceEngine, Settings, StopReason
from desh_chat.events import TurnStart
from desh_chat.memory import Memories


DELEGATE_SYSTEM_PROMPT = (
"""You are a coding agent working inside one project directory. Use the tools to look before you act: Read a file before editing it, prefer Edit over Write for changes to existing files, and use Bash for listing, searching, running tests and anything else. Paths are relative to the project root. Every Write, Edit and Bash call is shown to the user for approval before it runs; a declined call comes back as a message explaining why — do not retry it, adapt. Never modify files through Bash; instead use your Edit and Write tools.

Orient yourself before searching: if the project root has an INDEX.md, `grep -n \"#\" INDEX.md` maps its files; if it has a README.md, read it for the design. When you use find or grep, exclude venv, .git and __pycache__ and skip .log, .json and .jsonl files. Tool output is cut to a fixed size (the cut says where the rest is), so keep it short: run tests with -q and pipe long output through tail.

You are handling a subtask delegated by another agent. Complete it using your tools, then reply with your final answer only: what you found or did, concretely, without narrating the steps. Your reply is all the delegating agent will see, and it is read into a context that is smaller than yours: report exactly what the task asked to be reported, name files by path and line rather than pasting them, and never include whole files, whole logs or whole command outputs. If the task cannot be completed as specified, stop and report why."""
)

CAP_CONTINUE_MSG = ("Checkpoint: the tool round cap was reached. Any memory call in your last reply ran; its other tool calls were not run. "
                    "If the task is not finished, continue from here and ask again for any call you still need. "
                    "If it is finished, reply with your final answer.")

# The message a turn cut at the token limit is continued with (ChatState.length_prompt): the note
# the turn ends with says what it was writing.
LENGTH_CONTINUE_MSG = ("Your last reply was cut at the token limit before it finished; its closing note says what it was "
                       "writing. A reply must fit in one completion: a tool call whose argument carries a whole file or a "
                       "long document does not — give instructions and line anchors instead of content, or split the work. "
                       "Continue from here: ask again for what you still need, or reply with your final answer.")

BRIEF_HEAD_CHARS = 400


def fold_brief(args: dict) -> dict:
    """The echoed form of an answered delegate call (Tool.fold): the head of the task and a note
    that the rest was folded. The answer supersedes the brief, and the full brief is the child's
    first user message in its session file. A brief that fits in the head is kept whole, and so
    is one whose task is not a string — there is nothing sensible to keep of it."""
    task = args.get("task")
    if not isinstance(task, str) or len(json.dumps(args, ensure_ascii=False)) <= BRIEF_HEAD_CHARS:
        return args
    return {"task": task[:BRIEF_HEAD_CHARS] + "...", "folded": "context, gate and check omitted; see the result"}

def child_settings(parent: Settings) -> Settings:
    """The subagent's settings: the parent's, as they are. Compaction stays on: the round cap is the
    child's checkpoint, and NextRound compacts the capped turn into a summary when the auto prompt
    that continues it would not otherwise fit."""
    return parent


def child_session_file(parent: str | None) -> str | None:
    """A fresh session file beside the parent's: <parent stem>.delegate-<timestamp>_<hash>.json.
    None when the parent keeps no session."""
    if parent is None:
        return None
    stem, _ = os.path.splitext(parent)
    return f"{stem}.delegate-{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}.json"


@dataclass(frozen=True)
class Delegate:
    """What every subagent is built from. The tool method below is what the model calls; the
    registry derives its schema from the method alone, so none of these fields reach the model."""
    root: str
    inference: InferenceEngine = field(repr=False)
    settings: Settings                  # the fallback when no settings are injected at call time
    tools: ToolRegistry = field(default_factory=ToolRegistry, repr=False)
    session_file: str | None = None     # the PARENT's; each run derives its own from it
    completions_log: Logger | None = field(default=None, repr=False)
    des_log: TextIO | None = field(default=None, repr=False)
    debug: bool = False
    # The cap on one result, the registry's bound made known to the tool (as Workspace.result_chars
    # is): an answer over it is saved whole where the parent can Read it, and says so.
    result_chars: int = DEFAULT_RESULT_CHARS
    # The working memory every subagent starts with: the memories whose tools `tools` registers,
    # empty. A value, so one serves every run.
    memory: Memories = field(default_factory=Memories)
    # The texts a run may be built with others of (desh_chat.prompts): the subagent's system
    # prompt, and the message a capped turn of its is continued with.
    system_prompt: str = DELEGATE_SYSTEM_PROMPT
    cap_continue: str = CAP_CONTINUE_MSG
    length_continue: str = LENGTH_CONTINUE_MSG

    def delegate(self, task: str, context: str = "", gate: str = "", check: str = "", *,
                 settings: Settings | None = None, deadline: Deadline | None = None) -> str:
        """Hand a self-contained task to a subagent and get back only its final answer. Use it
        to keep your own context small: the subagent does the reading, searching and tool calls in
        its own conversation, and none of that comes back to you, only the answer. Prefer it for
        anything that needs many tool calls or large file reads whose details you will not need
        afterwards. The subagent has your tools and settings but none of your conversation, so the
        task must stand on its own.

        Args:
            task: What the subagent must do and what it must report back, complete and specific.
            context: Background it needs that is not in the task: relevant facts, paths, constraints.
            gate: The success criterion in words: when the subagent is done. Appended to the subagent's instructions, so it knows what "done" means.
            check: A shell command the harness runs in the project root after the subagent finishes; its exit code and last output lines are appended to the answer you receive. The subagent never sees it. Put every verification here (a grep, a test run) instead of delegating it: a verification subagent is a whole run spent on one command.
        """
        # `settings` is not in the Args block on purpose: the registry injects it (Tool.inject) and
        # the schema leaves it out, so the model cannot pass it. Register with inject=("settings",).
        child_config = child_settings(settings if settings is not None else self.settings)
        # A subagent starts with its memories empty: it has none of the parent's conversation, so
        # it has none of the parent's memory either.
        memory = self.memory
        system_prompt = memory.system_prompt(self.system_prompt, child_config.max_tool_rounds)
        if context:
            system_prompt += "\n\nContext from the delegating agent:\n" + context
        if gate:
            system_prompt += "\n\nSuccess criterion:\n" + gate
        session_file = child_session_file(self.session_file)
        child = ChatState(
            settings=child_config,
            inference=self.inference,
            history=ChatHistory(),
            running=True,
            system_prompt=system_prompt,
            completions_log=self.completions_log,
            session_file=session_file,
            tools=self.tools,
            operator=False,                 # nobody to prompt: a finished turn returns the run
            auto_prompt=self.cap_continue,  # checkpoint: a capped turn is continued, not returned
            length_prompt=self.length_continue,
            memory=memory,
            deadline=deadline,              # the parent's, injected like settings: no child outlives the run
        )
        print(c_out(Palette.CHROME, "╭─ delegate ─ subagent starts" + (f" ({session_file})" if session_file else "")))
        try:
            # Everything the child prints — its streamed answer, tool echoes, results, prompts —
            # goes through sys.stdout, so one redirect for the duration of the run draws the gutter.
            # The prefix starts with a reset: it is drawn in whatever colour state the child left.
            with contextlib.redirect_stdout(Gutter(sys.stdout, c_out(Palette.RESET + Palette.CHROME, "│") + " ")):
                final = Engine[ChatState](des_log=self.des_log, debug=self.debug).run(
                    child, seed=[TurnStart(task)], log_header={"delegate": True, "session": session_file})
        finally:
            print(c_out(Palette.CHROME, "╰─ delegate ─ back to the main agent"))
        last = final.history.last_non_summary()
        has_answer = last is not None and last.visible and last.assistant != ""
        text = answer(final)
        return self._spilled(self._checked(text, check) if has_answer else text, task)

    def _spilled(self, text: str, task: str) -> str:
        """An answer over the cap goes whole to a file the parent can Read by range, and its last
        line says so. The registry's cut keeps the tail, so the pointer survives whatever it drops:
        the middle of a long answer, or of a salvaged record, is no longer lost to the parent."""
        if len(text) <= self.result_chars:
            return text
        path = spill(self.root, "delegate", task, text)
        return text + f"\n[answer is {len(text)} characters, cut to {self.result_chars}; the whole of it is saved at {path} — Read it with offset and limit]"

    def _checked(self, text: str, check: str) -> str:
        """Append the check block to a real answer. Canned strings are not checked by this 
        method and are assumed to be checked earlier. The check runs in the project root
        and must never raise — a timeout or a crash is reported inside the block."""
        if not check:
            return text
        code, output, note = 0, "", ""
        try:
            proc = subprocess.run(check, shell=True, cwd=self.root, capture_output=True, text=True, timeout=60)
            code, output = proc.returncode, (proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            note = "timed out after 60s"
        except Exception as e:
            note = str(e)
        block = "\n".join(output.splitlines()[-20:])
        if len(block) > 2000:
            block = block[-2000:]
        suffix = f"\n\n[check `{check}`: exit {code}]"
        if block:
            suffix += f"\n{block}"
        if note:
            suffix += f"\n{note}"
        return text + suffix

def answer(state: ChatState) -> str:
    """The subagent's final state as the text the parent's model reads.

    The turn that matters is the last one that is not a summary: a child that checkpointed has
    capped turns and summaries before it. None at all means the request never fit the context
    window even on round one; otherwise that turn's `stop` (StopReason) says how it ended:
      OVERFLOW             the window filled up mid-task
      CANCELLED, INTERRUPT the operator pressed ESC, cancelled at a confirmation prompt, or hit Ctrl+C
      CAP                  the round cap cut the turn short and the auto prompt could not go on;
                           the model's text so far is the answer
      DEADLINE             the run's wall-clock budget ran out; the model's text so far is the answer
      REPEAT               the repeated-round guard ended the turn
      LENGTH               the reply hit the token limit twice running; the text so far and the record are the answer
      ANSWER               the answer, verbatim ("(no answer)" when the model said nothing)
    Every case must come back as text the parent can act on — it cannot see the child's history.
    A turn that ended by overflow, deadline, error or repeat carries what it got down as its answer
    (TurnEnd salvages it from the checkpoint, the memories and the rounds), so the parent
    reads that, followed by a note naming how the turn ended: "[Subagent ran out of context
    window]", "[Subagent hit the task deadline]", "[Subagent hit an error]"; a capped turn's
    answer is followed by "[Subagent hit the tool round cap]". An operator's cancel passes
    nothing on: "The operator cancelled the request."
    """
    assert state.pending is None
    turn = state.history.last_non_summary()
    if turn is None:
        return "The request didn't fit the context window."
    child_msg = turn.assistant or "(no answer)"
    notes = {
        StopReason.OVERFLOW: "[Subagent ran out of context window]",
        StopReason.DEADLINE: "[Subagent hit the task deadline]",
        StopReason.ERROR: "[Subagent hit an error]",
        StopReason.CAP: "[Subagent hit the tool round cap]",
        StopReason.REPEAT: "[Subagent ran into a repeat loop]",
        StopReason.LENGTH: "[Subagent's reply was cut at the token limit]",
    }
    if turn.stop in notes:
        return f"{child_msg}\n{notes[turn.stop]}"
    if turn.stop in (StopReason.CANCELLED, StopReason.INTERRUPT):
        return "The operator cancelled the request."
    return child_msg
