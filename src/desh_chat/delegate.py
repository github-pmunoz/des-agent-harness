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
its schema is `task` and `context`; model, temperature, budgets and tools are inherited from the
parent at startup (the operator's choice, not the model's), minus delegate itself, so a subagent
cannot delegate.

The child streams to the terminal exactly as the parent does and its confirmed tools ask the
operator exactly as the parent's do; a banner marks the hand-over and the return. When the parent
keeps a session file, each subagent run keeps its own beside it, named from the parent's, so the
work the parent never saw stays inspectable.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import TextIO

from desh.engine import Engine
from desh.llama.logger import Logger
from desh.render import Palette, c_out
from desh.tools import ToolRegistry
from desh_chat.state import ChatHistory, ChatState, InferenceEngine, Settings
from desh_chat.events import UserMessage


DELEGATE_SYSTEM_PROMPT = (
"You are a coding agent working inside one project directory. Use the tools to look before you act: Read a file before editing it, prefer Edit over Write for changes to existing files, and use Bash for listing, searching, running tests and anything else. Paths are relative to the project root. Every Write, Edit and Bash call is shown to the user for approval before it runs; a declined call comes back as a message explaining why — do not retry it, adapt."
"You are handling a subtask delegated by another agent. Complete it using your tools, then reply with your final answer only: what you found or did, concretely, without narrating the steps. Your reply is all the delegating agent will see. If the task cannot be completed as specified, stop and report why."
)


def child_settings(parent: Settings) -> Settings:
    """The subagent's settings: the parent's, with compaction off. A child's history is one turn,
    so a summary would replace the very answer the parent is waiting for."""
    return replace(parent, compaction_threshold=float("inf"))


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
    inference: InferenceEngine = field(repr=False)
    settings: Settings
    system_prompt: str
    tools: ToolRegistry = field(default_factory=ToolRegistry, repr=False)
    session_file: str | None = None     # the PARENT's; each run derives its own from it
    completions_log: Logger | None = field(default=None, repr=False)
    des_log: TextIO | None = field(default=None, repr=False)
    debug: bool = False

    def delegate(self, task: str, context: str = "") -> str:
        """Hand a self-contained task to a subagent and get back only its final answer. Use it
        to keep your own context small: the subagent does the reading, searching and tool calls in
        its own conversation, and none of that comes back to you, only the answer. Prefer it for
        anything that needs many tool calls or large file reads whose details you will not need
        afterwards. The subagent has your tools and settings but none of your conversation, so the
        task must stand on its own.

        Args:
            task: What the subagent must do and what it must report back, complete and specific.
            context: Background it needs that is not in the task: relevant facts, paths, constraints.
        """
        system_prompt = DELEGATE_SYSTEM_PROMPT + ("\n\nContext from the delegating agent:\n" + context if context else "")
        session_file = child_session_file(self.session_file)
        child = ChatState(
            settings=self.settings,
            inference=self.inference,
            history=ChatHistory(),
            running=False,          # drain after one turn: MaybeRegenerate prompts nobody
            system_prompt=system_prompt,
            completions_log=self.completions_log,
            session_file=session_file,
            tools=self.tools,
        )
        print(c_out(Palette.CHROME, "╭─ delegate ─ subagent starts" + (f" ({session_file})" if session_file else "")))
        try:
            final = Engine[ChatState](des_log=self.des_log, debug=self.debug).run(
                child, seed=[UserMessage(task)], log_header={"delegate": True, "session": session_file})
        finally:
            print(c_out(Palette.CHROME, "╰─ delegate ─ back to the main agent"))
        return answer(final)


def answer(state: ChatState) -> str:
    """The subagent's final state as the text the parent's model reads.

    A child run leaves at most one turn in history: none if the request never fit the context
    window, a cancelled one if the operator pressed ESC or cancelled at a confirmation prompt, an
    empty answer if the round cap cut the turn short, and otherwise the answer. Every case must
    come back as text the parent can act on — it cannot see the child's history.
    """
    assert len(state.history.turns) <= 1
    assert state.pending is None
    if len(state.history.turns) == 0:
        return "The request didn't fit the context window."
    if state.history.turns[0].cancelled:
        return "The operator cancelled the request."
    child_msg = state.history.turns[0].assistant
    if len(child_msg) == 0:
        child_msg = "(no answer)"
    if len(state.history.turns[0].rounds) == state.settings.max_tool_rounds:
        return f"{child_msg}\n[Subagent used all {state.settings.max_tool_rounds} tool rounds]"
    return child_msg


def with_delegate(tools: ToolRegistry, *, inference: InferenceEngine, settings: Settings, system_prompt: str,
                  session_file: str | None = None, completions_log: Logger | None = None,
                  des_log: TextIO | None = None, debug: bool = False) -> ToolRegistry:
    """`tools` plus the delegate tool, whose subagents get `tools` as given — without delegate."""
    d = Delegate(inference=inference, settings=child_settings(settings), system_prompt=system_prompt, tools=tools,
                 session_file=session_file, completions_log=completions_log, des_log=des_log, debug=debug)
    return tools.add(d.delegate, name="delegate")
