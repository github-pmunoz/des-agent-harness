import json
import time
from dataclasses import dataclass, field, replace
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.llama.wire import ToolCall
from desh.llama.tokens import RESULT_CHARS_PER_TOKEN, estimate_result_tokens, estimate_tokens
from desh.engine import State
from desh.tools import ToolRegistry
from desh_chat.memory import Memories
from typing import Any, Callable
from enum import StrEnum

class StopReason(StrEnum):
    """How a turn ended. Read by whoever must tell the cases apart: the view (HIDDEN_STOPS), the
    salvage (SALVAGE_STOPS), the delegate's answer(), the CLI's exit code. The values are what the
    session file and the eval graders read."""
    ANSWER = "answer"           # the model answered
    CAP = "cap"                 # round cap hit; the model's text so far is the answer, and the auto prompt may continue the turn
    OVERFLOW = "overflow"       # no room left for a completion
    DEADLINE = "deadline"       # the run's wall-clock budget ran out; the model's text so far is the answer, and unlike CAP the turn is never continued
    ERROR = "error"             # a mid-turn exception
    REPEAT = "repeat"           # the same calls asked a third time with identical results
    LENGTH = "length"           # the completion hit its token limit before it finished (a tool call left unclosed, say); the text so far and a note are the answer, and the length prompt may continue the turn once
    CANCELLED = "cancelled"     # the operator pressed ESC or cancelled at a confirmation prompt
    INTERRUPT = "interrupt"     # Ctrl+C in auto mode
    SETTING = "setting"         # not a conversation turn: a settings change made by a /command
    SUMMARY = "summary"         # not a conversation turn: the synthetic turn a compaction leaves

def rounds_since_last_summary(rounds: tuple[Round, ...]) -> tuple[Round, ...]:
    """The rounds since the last summary, including the latest summary round."""
    view = []
    for round in reversed(rounds):
        view.append(round)
        if round.summary:
            break
    return tuple(reversed(view))


@dataclass(frozen=True)
class Section:
    """One piece of a compaction transcript, oldest first in the list: its renderings from the
    fullest to the most reduced (a round with its results whole, then stubbed; a turn with its
    rounds, then without), and whether it may be left out altogether. A summary and the task
    message are fixed: they carry everything before them."""
    renderings: tuple[str, ...]
    fixed: bool = False


def fit_transcript(sections: list[Section], budget_tokens: int | None, unit: str = "rounds") -> str:
    """Join the sections into a transcript that fits `budget_tokens` (None: no bound), giving up
    the least first: the oldest section is reduced to its last rendering, then the next, and so
    on; if that is not enough the oldest non-fixed sections are left out, oldest first, with one
    line saying how many; if the fixed sections alone do not fit, the head of the text is cut.
    A compaction request that does not fit the window would fail outright, and the summary is
    what the model works from next, so a poorer transcript beats none."""
    level = [0] * len(sections)
    omitted = 0

    def render() -> str:
        head = [f"[{omitted} earlier {unit} left out of this transcript]"] if omitted else []
        return "\n".join(head + [s.renderings[l] for s, l in zip(sections, level) if l < len(s.renderings)])

    def over(text: str) -> bool:
        return budget_tokens is not None and estimate_result_tokens(text) > budget_tokens

    text = render()
    for i, s in enumerate(sections):
        while over(text) and level[i] < len(s.renderings) - 1:
            level[i] += 1
            text = render()
    for i, s in enumerate(sections):
        if not over(text):
            break
        if not s.fixed:
            level[i] = len(s.renderings)
            omitted += 1
            text = render()
    if over(text):
        keep = int((budget_tokens or 0) * RESULT_CHARS_PER_TOKEN)
        text = "[transcript cut to fit]\n" + text[-keep:] if keep > 0 else "[transcript cut to fit]"
    return text

# -----------------------
# Chat DES State
# -----------------------

# The system prompt of the compaction request. Written for a working session with tools: what the
# summary must keep is what the model would otherwise re-read or re-derive — paths, names, lines,
# the exact text an edit has to match — and what it may drop is what a later step superseded.
COMPACTION_PROMPT = """You will be sent the transcript of a working session: a task, the assistant's tool calls (file reads, edits, shell commands) with their results, and the assistant's replies. Write the summary the assistant will work from in place of the transcript, so it can carry on without re-reading what it already read.

Keep, exactly as written: the task and its success criterion; every file path touched and what was done to it; the function, class and test names, signatures, line numbers and error messages that were established; short code fragments that will be needed again verbatim, such as the exact text an edit must match or a command that must be re-run; test results as reported; decisions made and why; what remains to do, as concrete next steps.

Drop: narration, the full contents of files that were read, tool output that a later step superseded. Prefer a terse list to prose. Do not mention this instruction and do not repeat the transcript."""

# The system prompt of the checkpoint request (CompactPendingTurn). Unlike a history summary, the
# checkpoint is read right after the task message and next to the memory block, in a window
# that is already tight, and it is rewritten every few rounds: so it must not restate the task,
# must not duplicate what the working memory holds, must say what the next concrete action is, and
# must stay short.
CHECKPOINT_PROMPT = """You will be sent a transcript of agent rounds being folded away. Write the checkpoint the assistant will read in place of them, right after the task message and next to the working memory block.

Do not restate the task — the task message is right before the checkpoint. Do not duplicate what the working memory holds — the memory block follows the checkpoint. The transcript may begin with an EARLIER CHECKPOINT: fold what it says into yours, updated by the rounds after it, so that yours stands alone in its place. Always write a checkpoint; there is always something to keep.

Keep: the one thing the assistant was about to do next, as a single concrete action, with the file paths, names and error messages that action depends on exactly as written; which files were written or edited and whether they pass, with test results as reported; what was verified versus what was only assumed. A result shown as expired was cut to fit this transcript: record what the assistant said or did about it, do not guess its content. Drop narration and superseded detail. Stay short — the checkpoint must fit in a small fraction of the context window. Do not mention this instruction."""

# The user message of a summary request must not END with the transcript: a model reading raw
# tool output up to the last token takes it for the end of a document and stops at once
# (replayed: 0 of 8 such requests answered at temperature 0, 0.1 or 0.3; 8 of 8 with a closing
# line). So the transcript is followed by the instruction to write, and an empty answer is
# asked once more with a firmer one.
SUMMARY_CLOSE = "\n\nWrite the summary now."
CHECKPOINT_CLOSE = "\n\nWrite the checkpoint now."
RETRY_NUDGE = "\n\nAn empty reply is not an answer: write it now."


@dataclass(frozen=True)
class Settings:
    model: str
    temperature: float
    think: bool
    context: int
    max_turn_tokens: int
    max_tool_rounds: int = 10       # tool-call rounds allowed inside one turn before it is forced to end
    compaction_threshold: float = 0.65
    compaction_target: float = 0.25
    turn_token_cap: float = 0.40
    min_compaction_tokens: int = 64
    auto: bool = False           # auto mode: confirmed tools run without asking; Ctrl+C turns it off
    compaction_prompt: str = field(default=COMPACTION_PROMPT, repr=False)    # system prompt of the compaction request
    # A checkpoint (CompactPendingTurn) has its own instruction and share of the context: it sits
    # after the task message and next to the memory block, so it must not repeat either, and it is
    # rewritten every few rounds in a tight window, so it is kept smaller than a history summary.
    checkpoint_target: float = 0.15
    checkpoint_prompt: str = field(default=CHECKPOINT_PROMPT, repr=False)
    # what follows the transcript in a summary and in a checkpoint request, and what an empty
    # answer is asked again with
    summary_close: str = field(default=SUMMARY_CLOSE, repr=False)
    checkpoint_close: str = field(default=CHECKPOINT_CLOSE, repr=False)
    retry_nudge: str = field(default=RETRY_NUDGE, repr=False)
    # The share of the context ONE memory may take (Memories.commit refuses a write past it): the
    # block is re-sent whole with every request, so a memory that grows without bound eats the
    # window its results need.
    memory_target: float = 0.10

    def to_dict(self) -> dict:
        """Plain-JSON form of every field, for the session document (format 4)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "think": self.think,
            "context": self.context,
            "max_turn_tokens": self.max_turn_tokens,
            "max_tool_rounds": self.max_tool_rounds,
            "compaction_threshold": self.compaction_threshold,
            "compaction_target": self.compaction_target,
            "turn_token_cap": self.turn_token_cap,
            "min_compaction_tokens": self.min_compaction_tokens,
            "auto": self.auto,
            "compaction_prompt": self.compaction_prompt,
            "checkpoint_target": self.checkpoint_target,
            "checkpoint_prompt": self.checkpoint_prompt,
            "summary_close": self.summary_close,
            "checkpoint_close": self.checkpoint_close,
            "retry_nudge": self.retry_nudge,
            "memory_target": self.memory_target,
        }

    @classmethod
    def from_dict(cls, d: Any) -> Settings:
        """Inverse of to_dict. Missing keys fall back to the dataclass defaults; the required
        fields (no default) must be present. Raises ValueError on non-dict input."""
        if not isinstance(d, dict):
            raise ValueError(f"settings document must be an object, got {type(d).__name__}")
        return cls(
            model=d["model"],
            temperature=d["temperature"],
            think=d["think"],
            context=d["context"],
            max_turn_tokens=d["max_turn_tokens"],
            max_tool_rounds=d.get("max_tool_rounds", 10),
            compaction_threshold=d.get("compaction_threshold", 0.65),
            compaction_target=d.get("compaction_target", 0.25),
            turn_token_cap=d.get("turn_token_cap", 0.40),
            min_compaction_tokens=d.get("min_compaction_tokens", 64),
            auto=d.get("auto", False),
            compaction_prompt=d.get("compaction_prompt", COMPACTION_PROMPT),
            checkpoint_target=d.get("checkpoint_target", 0.15),
            checkpoint_prompt=d.get("checkpoint_prompt", CHECKPOINT_PROMPT),
            summary_close=d.get("summary_close", SUMMARY_CLOSE),
            checkpoint_close=d.get("checkpoint_close", CHECKPOINT_CLOSE),
            retry_nudge=d.get("retry_nudge", RETRY_NUDGE),
            memory_target=d.get("memory_target", 0.10),
        )

@dataclass(frozen=True)
class InferenceEngine:
    models: list[str]
    max_context: dict[str, int]
    server: LlamaServer = field(repr=False)
    port: int = field(repr=False)

@dataclass(frozen=True)
class Deadline:
    """A wall-clock budget for a run: the instant it ends, on the monotonic clock, and the budget
    it was set from, for the message that reports it. A budget rather than an instant is what a
    reader can make sense of; an instant rather than a budget is what a check can compare."""
    at: float           # time.monotonic() value after which no further tool round starts
    budget: float       # the seconds it was set from, as given on the command line

    @classmethod
    def in_seconds(cls, budget: float) -> "Deadline":
        return cls(at=time.monotonic() + budget, budget=budget)

    def passed(self) -> bool:
        return time.monotonic() > self.at


@dataclass(frozen=True)
class ChatState(State):
    """Immutable chat state."""
    settings: Settings
    history: ChatHistory
    running: bool
    system_prompt: str
    completions_log: Logger | None = field(repr=False)
    inference: InferenceEngine = field(repr=False)
    session_file: str | None = None     # where LoadSession reads / SaveSession writes; None -> no persistence
    pending: PendingTurn | None = None  # the turn between TurnStart and TurnEnd; never persisted
    tools: ToolRegistry = field(default_factory=ToolRegistry, repr=False)  # what the model may call; empty -> no tools offered
    # How a turn begins when the queue runs dry (TurnStart's policy): with an operator, the prompt
    # is shown; without one, the run returns. auto_prompt is the message a capped turn is continued
    # with before either — None means a capped turn is never continued automatically.
    operator: bool = True
    auto_prompt: str | None = None
    # The message a turn cut at the token limit (StopReason.LENGTH) is continued with, once: the
    # model wrote past what a reply may hold — a call carrying a whole file, most often — and is
    # told so. None means such a turn is never continued automatically. Two in a row end the run.
    length_prompt: str | None = None
    # The model's working memory: the memories the run registered and the value each holds; empty
    # when none is offered. Every write goes through ExecuteToolCalls, which commits the new value
    # here; a memory tool only sees a dict built from its slot for the one call. Rendered last in
    # every request, and snapshotted onto each finished Turn so a session restores it.
    memory: Memories = field(default_factory=Memories)
    # What MaybeRegenerate does when the queue runs dry: 'prompt' opens the next turn (the
    # interactive loop); 'exit' ends the run (a one-shot --task run).
    idle_policy: str = "prompt"
    # The run's wall-clock budget, None for no limit. Checked where the round cap is checked, before
    # the next tool round starts, so a run overshoots it by at most one completion. A subagent
    # inherits the parent's (Tool.inject), so no child outlives the run that spawned it.
    deadline: Deadline | None = None

    def change_setting(self, setting: str, value: Any) -> ChatState:
        return replace(self, settings=replace(self.settings, **{setting: value}))

    def record_setting_change(self, setting: str, value: Any) -> ChatState:
        """change_setting plus the settings turn that records it in the history: a "settings"
        turn whose delta is {setting: new_value}. LoadSession replays settings turns in order,
        so the last change wins. tokens=1 keeps the turn out of the token accounting (it is not
        part of the conversation) and stops __post_init__ from re-pricing the empty text."""
        turn = Turn(user="", assistant="", tokens=1, type="settings", delta={setting: value}, stop=StopReason.SETTING)
        return replace(self,
                       settings=replace(self.settings, **{setting: value}),
                       history=self.history.append(turn))

    def pending_tokens(self) -> int:
        """What the pending turn costs in the prompt: priced rounds plus the heuristic for the text no usage frame has priced."""
        p = self.pending
        if p is None:
            return 0
        # the user message is prose, the latest results are tool output: different densities
        unpriced = estimate_tokens(p.unpriced_text()) if not p.rounds else estimate_result_tokens(p.unpriced_text())
        return p.priced_tokens() + unpriced

    def memory_budget(self) -> int:
        """What one memory may take of the window, in tokens (Settings.memory_target)."""
        return int(self.settings.memory_target * self.settings.context)

    def turn_round(self) -> tuple[int, int] | None:
        """The round the next completion is — one past the completed ones — against the turn's cap."""
        if self.pending is None:
            return None
        return (self.pending.non_summary_rounds() + 1, self.settings.max_tool_rounds)

    def fold_near(self) -> bool:
        """Whether the fold alert is due: a round the size of this turn's typical one would leave
        less than the room a request needs, so that the NEXT request folds the pending turn into a
        checkpoint (NextRound's ladder) — and that was not yet so one round ago. Said once, on the
        edge: a reply that only persists adds next to nothing to the prompt, so a line repeated
        while the condition holds is answered with the same write round after round, and the
        rounds it was meant to save are spent on it."""
        p = self.pending
        if p is None or not self._fold_within_a_round():
            return False
        before = replace(self, pending=replace(p, rounds=p.rounds[:-1]))
        return not before._fold_within_a_round()

    def _fold_within_a_round(self) -> bool:
        """The condition fold_near watches the edge of. False while the history still holds
        something to compact — that rung goes first and folds nothing of the turn — and while the
        view holds fewer than two model rounds, when there is nothing to fold. An estimate: the
        next round's results are unknown."""
        p = self.pending
        if p is None or any(not t.summary for t in self.history.since_last_summary()):
            return False
        rounds = sum(1 for r in p.since_last_summary() if not r.summary)
        if rounds < 2:
            return False
        pending_tokens = self.pending_tokens()
        block = self.memory.block(round=self.turn_round(), budget_tokens=self.memory_budget())
        prompt = (estimate_tokens(self.system_prompt) + self.tools_tokens() + self.history.window_tokens()
                  + pending_tokens + (estimate_tokens(block["content"]) if block is not None else 0))
        return self.settings.context - prompt - pending_tokens // rounds < self.min_gen_tokens()

    def memory_block(self) -> dict | None:
        """The working memory as the last message of the next request, framed with the round the
        turn is on and the line that says what is about to expire; None when no memory is
        registered. The one place the block is built, so what is priced is what is sent."""
        if not self.memory:
            return None
        return self.memory.block(round=self.turn_round(), budget_tokens=self.memory_budget(), fold_near=self.fold_near())

    def memory_tokens(self) -> int:
        """What the memory block costs in the prompt: heuristic, it is re-sent whole every request."""
        block = self.memory_block()
        return estimate_tokens(block["content"]) if block is not None else 0

    def tools_tokens(self) -> int:
        """What the tool schemas cost in the prompt: heuristic, they are re-sent whole every request.
        Priced as prior, like the system prompt — left to the usage frames, their cost would sit on
        whichever round's frame first absorbed it, and leave the estimate with that round."""
        schemas = self.tools.schemas()
        return estimate_tokens(json.dumps(schemas)) if schemas else 0

    def prompt_tokens(self, pending_tokens: int) -> int:
        """What the next request costs before generation: system prompt, tool schemas, window since the last summary, pending, memory block."""
        return estimate_tokens(self.system_prompt) + self.tools_tokens() + self.history.window_tokens() + pending_tokens + self.memory_tokens()

    def gen_room(self, pending_tokens: int) -> int:
        """What the window leaves for the next completion, before any cap: context minus the prompt."""
        return self.settings.context - self.prompt_tokens(pending_tokens)

    def gen_budget(self, pending_tokens: int) -> int:
        """Room for the next completion: the turn cap, the fraction cap, and what the window leaves."""
        s = self.settings
        return int(min(s.max_turn_tokens, self.gen_room(pending_tokens), s.turn_token_cap * s.context))

    def min_gen_tokens(self) -> int:
        """The room the window must leave for a completion before a request goes out; less than
        this and the history is compacted first. Derived from the compaction threshold: a window
        past the threshold is one that leaves less than (1 - threshold) of the context."""
        s = self.settings
        return int((1 - s.compaction_threshold) * s.context)

    def session_tokens(self, pending_tokens: int) -> int:
        """Whole priced tokens of the session so far."""
        return estimate_tokens(self.system_prompt) + self.tools_tokens() + self.history.get_total_tokens() + pending_tokens

# -----------------------
# Tool exchange inside a turn
# -----------------------

@dataclass(frozen=True)
class ToolResult:
    """What one tool call came back with. Content is text the model reads; an error or a denial
    is still a result (the model must be able to see it and recover)."""
    tool_call_id: str
    name: str
    content: str

    def message(self) -> dict:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "name": self.name, "content": self.content}

    def to_dict(self) -> dict:
        return {"tool_call_id": self.tool_call_id, "name": self.name, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict) -> ToolResult:
        return cls(tool_call_id=d["tool_call_id"], name=d["name"], content=d["content"])


# What a tool message says once its turn has ended and its result has left the context. Constant on
# purpose: the stubbed prefix of a request must not change from one round to the next, or the server re-prefills it.
EXPIRED_RESULT = "[expired: this result is no longer in context]"
# The stops a turn can end by without an answer, where the record is the answer (PendingTurn.salvage):
# the window overflowed, the run's deadline passed, an error cut the turn, a repeated round, a
# completion cut at its token limit. A capped turn is not one —
# the cap message continues it — and an interrupt is the operator's, who wants no answer.
SALVAGE_STOPS = frozenset((StopReason.OVERFLOW, StopReason.DEADLINE, StopReason.ERROR, StopReason.REPEAT, StopReason.LENGTH))
# The stops that keep a turn out of the conversation (Turn.visible): it is on the record and in the
# session file, but not in the view, the token totals or the compaction transcript.
HIDDEN_STOPS = frozenset((StopReason.CANCELLED, StopReason.INTERRUPT, StopReason.ERROR, StopReason.OVERFLOW, StopReason.REPEAT))

CHECKPOINT_PREFIX = "Checkpoint of this turn so far, in place of the rounds before it: "

# How much of a call's target a digest line shows: enough to recognise a path or a command,
# never a dump — the digest stands in the request for as long as its checkpoint does.
MENTION_CHARS = 60


@dataclass(frozen=True)
class Round:
    """One intermediate model round inside a turn: the assistant asked for tools (with whatever text
    it said alongside), and the tools answered. A turn's final answer is NOT a Round — it is Turn.assistant.

    The record is always complete: results are never dropped from the Round (the session file, the
    repeat detector and the compaction transcript read them). Stubbing is a RENDERING, the one a
    finished turn gets: messages() and text() take `stubbed` and put EXPIRED_RESULT in place of
    every result. The calls stay as they are — the template wants one tool message per call, and
    the model must still see what it asked for."""
    assistant: str
    tool_calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...] = ()
    # Priced from the usage frame of the completion that produced the calls: what that request added
    # over the prior, so the PREVIOUS round's results plus this round's completion. The round kept
    # whole by a checkpoint is re-priced to its own text (PendingTurn.compact), since the results
    # its frame counted are the ones the checkpoint folded.
    tokens: int = 0
    summary: bool = False

    def messages(self, stubbed: bool = False) -> list[dict]:
        """The wire form of this round, as it is echoed back in every later request of the conversation.
        A summary round is the checkpoint of the rounds folded before it: one user message, with no
        ack of its own, because the round kept whole after it supplies the assistant's continuation."""
        if self.summary:
            return [{"role": "user", "content": self.assistant}]
        messages: list[dict] = [{ #stripping away the tc index
            "role": "assistant",
            "content": self.assistant,
            "tool_calls": [{
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
            } for tc in self.tool_calls]
        }]
        for result in self.results:
            messages.append(replace(result, content=EXPIRED_RESULT).message() if stubbed else result.message())
        return messages

    def transcript(self, stubbed: bool = False) -> str:
        """Plain-text rendering for the compaction prompts and the history display. `stubbed`
        renders the results as EXPIRED_RESULT, the way the model last saw them."""
        if self.summary:
            # to the summariser, an earlier checkpoint is input to fold, not a user message to answer
            return f"EARLIER CHECKPOINT: {self.checkpoint_body()}"
        calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in self.tool_calls)
        lines = [f"ASSISTANT (tool calls): {self.assistant + ' ' if self.assistant else ''}{calls}"]
        lines += [f"TOOL {res.name}: {EXPIRED_RESULT if stubbed else res.content}" for res in self.results]
        return "\n".join(lines)

    def checkpoint_body(self) -> str:
        """A checkpoint round's text without the prefix the model reads it under."""
        return self.assistant[len(CHECKPOINT_PREFIX):] if self.assistant.startswith(CHECKPOINT_PREFIX) else self.assistant

    def own_text(self) -> str:
        """The round's own text — the assistant message and its calls, never the results."""
        return self.assistant + "".join(tc.name + tc.arguments for tc in self.tool_calls)

    def text(self, stubbed: bool = False) -> str:
        """All text of the round as rendered, for heuristic pricing when no usage frame priced it."""
        results = "".join(EXPIRED_RESULT if stubbed else r.content for r in self.results)
        return self.own_text() + results

    def mentions(self, describe: Callable[[str, str], str] | None = None) -> str:
        """The calls of the round, one mention each, for the line that stands for a folded round (digest):
        the tool name, followed by what the call was about when `describe` (name, arguments) -> str
        knows it — 'Read tests/conftest.py', 'Bash grep -n "def answer"'. A target is folded onto
        one line and cut at MENTION_CHARS; a call without one is mentioned by name alone."""
        def mention(tc: ToolCall) -> str:
            target = " ".join(describe(tc.name, tc.arguments).split()) if describe is not None else ""
            if len(target) > MENTION_CHARS:
                target = target[:MENTION_CHARS] + "..."
            return f"{tc.name} {target}" if target else tc.name
        return ", ".join(mention(tc) for tc in self.tool_calls)

    def to_dict(self) -> dict:
        return {"assistant": self.assistant, "tokens": self.tokens,
                "tool_calls": [tc.to_dict() for tc in self.tool_calls],
                "results": [r.to_dict() for r in self.results], "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> Round:
        return cls(assistant=d["assistant"], tokens=d.get("tokens", 0),
                   tool_calls=tuple(ToolCall(index=tc["index"], id=tc["id"], type=tc["type"],
                                             name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                                    for tc in d["tool_calls"]),
                   results=tuple(ToolResult.from_dict(r) for r in d.get("results", [])), summary=d.get("summary", False))


@dataclass(frozen=True)
class PendingTurn:
    """A turn between TurnStart and TurnEnd: the user message plus every completed tool round so
    far. Lives on ChatState.pending only; never in ChatHistory, never in the session file.

    The turn exists before its message: TurnStart opens it with user=None and the message source
    (the operator's prompt, an auto prompt, a seed) fills it through with_user(). An empty
    placeholder is not a turn in progress — commands run against it, and Exit drops it."""
    user: str | None = None
    rounds: tuple[Round, ...] = ()

    def with_user(self, message: str) -> PendingTurn:
        """The message arrived: the placeholder becomes a turn. Filling twice is a loop bug."""
        assert self.user is None, "pending turn already has its message"
        return replace(self, user=message)

    # A pending turn renders whole: every round of the view goes out with its results, so each
    # request appends to the previous one and the server keeps its prefix. Results leave the
    # request only when a checkpoint folds their round (compact) or the turn ends (Turn.messages).

    def messages(self) -> list[dict]:
        assert self.user is not None, "pending turn has no message yet"
        return ([{"role": "user", "content": self.user}]
                + [m for r in self.since_last_summary() for m in r.messages()])

    def priced_tokens(self) -> int:
        """What the completed rounds cost in the next request: what their usage frames priced."""
        return sum(r.tokens for r in self.since_last_summary())

    def unpriced_text(self) -> str:
        """The prompt text the NEXT completion's usage frame will price: the user message on round one,
        the latest tool results afterwards."""
        if not self.rounds:
            return self.user or ""
        return "\n".join(r.content for r in self.rounds[-1].results)

    def add_round(self, round: Round) -> PendingTurn:
        return replace(self, rounds=self.rounds + (round,))

    def with_results(self, results: tuple[ToolResult, ...]) -> PendingTurn:
        """Attach results to the latest round (the one whose calls just ran)."""
        last = replace(self.rounds[-1], results=results)
        return replace(self, rounds=self.rounds[:-1] + (last,))

    def fold_call(self, index: int, arguments: str) -> PendingTurn:
        """Replace the arguments of call `index` in the latest round: the form the call is echoed
        back in from now on (Tool.fold), once its result has superseded what it asked for."""
        last = self.rounds[-1]
        calls = last.tool_calls[:index] + (replace(last.tool_calls[index], arguments=arguments),) + last.tool_calls[index + 1:]
        return replace(self, rounds=self.rounds[:-1] + (replace(last, tool_calls=calls),))

    def add_results(self, *results: ToolResult) -> PendingTurn:
        """Append results to the latest round, in call order: the round is answered one call per step."""
        return self.with_results(self.rounds[-1].results + results)

    def finish(self, assistant: str, tokens: int, stop: StopReason, memory: dict[str, dict] | None = None) -> Turn:
        """The final answer arrived (or the turn was cut short): freeze into a history Turn.
        tokens prices only the final completion. A history turn renders its rounds stubbed from
        now on (Turn.messages), so the rounds are priced on that rendering here, once, and not on
        the usage frames that priced them whole. memory is the working memory as it stands when
        the turn ends (Memories.snapshot), recorded on the Turn for the session file."""
        assert self.user is not None, "pending turn has no message yet"
        stubbed_rounds = sum(estimate_tokens(r.text(stubbed=True)) for r in self.since_last_summary())
        # every round goes on the record (the session file shows what a checkpoint folded); the
        # Turn renders and prices its view, as the pending turn did
        return Turn(self.user, assistant, tokens=tokens + stubbed_rounds if tokens else 0,
                    rounds=self.rounds, stop=stop, memory=memory)

    def since_last_summary(self) -> tuple[Round, ...]:
        """The rounds the next request carries: the latest checkpoint, when there is one, and
        every round after it. The rounds a checkpoint folded stay in `rounds` for the record."""
        return rounds_since_last_summary(self.rounds)

    def non_summary_rounds(self) -> int:
        """The model's rounds in the whole turn, checkpoints excluded: what the round cap counts.
        A checkpoint frees context, it does not open a new turn, so the cap does not restart."""
        return sum(1 for r in self.rounds if not r.summary)

    def transcript(self, rounds: tuple[Round, ...], budget_tokens: int | None = None) -> str:
        """Plain-text rendering of the user message and `rounds` (a prefix of the view), for the
        checkpoint prompt. Each round renders whole, as the model saw it, and the whole is fitted
        to `budget_tokens` (fit_transcript):
        the oldest whole results are stubbed first, then the oldest rounds left out. A checkpoint
        among the rounds renders as the user message it is on the wire and is never reduced, so a
        second checkpoint subsumes the first."""
        sections = [Section((f"USER: {self.user}",), fixed=True)]
        for r in rounds:
            if r.summary:
                sections.append(Section((r.transcript(),), fixed=True))
            else:
                sections.append(Section((r.transcript(), r.transcript(stubbed=True))))
        # The budget is the rounds': the checkpoint was written to its own share and the memory
        # is paid for in every request, so both stand whole, and the head cut that a compaction
        # request needs to fit the window never fires here — history compaction and a parent's
        # result cap bound the record later, each in its own way.
        fixed = estimate_result_tokens("\n".join(s.renderings[0] for s in sections if s.fixed))
        return fit_transcript(sections, budget_tokens + fixed if budget_tokens is not None else None, unit="rounds")

    def digest(self, rounds: tuple[Round, ...], describe: Callable[[str, str], str] | None = None, budget_tokens: int | None = None) -> str:
        """What stands in for a checkpoint when the model returns none: the previous checkpoint's
        text, when `rounds` starts with one, and one line per folded round naming its calls
        (Round.mentions) — what was done, never the results. Deterministic and cheap, so a turn
        never folds into nothing; fitted to `budget_tokens` like a transcript."""
        ordinal = {id(r): n for n, r in enumerate((r for r in self.rounds if not r.summary), start=1)}   # as the model counts rounds
        sections = []
        for r in rounds:
            if r.summary:
                sections.append(Section((r.checkpoint_body(),), fixed=True))
            else:
                sections.append(Section((f"round {ordinal.get(id(r), '?')}: {r.mentions(describe)}",)))
        return fit_transcript(sections, budget_tokens, unit="rounds")

    def salvage(self, stop: StopReason, memory: Memories = Memories(), budget_tokens: int | None = None) -> str:
        """The answer of a turn that ended without one (SALVAGE_STOPS): everything the turn got
        down, assembled from the record rather than asked of the model — at an overflow there is
        no room to ask, at the deadline no time, after an error maybe no server. The sources are
        the view's checkpoint (the model's own summary of the folded rounds), the memories as they
        stand, and the rounds after the checkpoint with their results, fitted to `budget_tokens`
        the way a compaction transcript is (fit_transcript): the oldest results are stubbed first,
        then the oldest rounds left out. Read by the operator, by the next turn as history, and by
        a delegating agent as the subagent's result, which cannot see any of these sources."""
        view = self.since_last_summary()
        checkpoint = view[0].checkpoint_body() if view and view[0].summary else None
        rounds = [r for r in view if not r.summary]
        # Oldest first, because fit_transcript reduces from the front: the lead line and the
        # checkpoint are the oldest and most important, and the rounds run in order after them.
        # The parent agent reads this as a tool result, head and tail first: the lead line tells
        # every reader why there is a record instead of an answer, and the newest round — the
        # turn's last work — stands at the tail.
        sections = [Section((f"TURN ENDED: {stop}",), fixed=True)]
        if checkpoint is not None:
            sections.append(Section((f"CHECKPOINT: {checkpoint}",), fixed=True))
        sections += [Section((text,), fixed=True) for text in memory.salvage_sections()]
        for r in rounds:
            sections.append(Section((r.transcript(), r.transcript(stubbed=True))))
        # The budget is the rounds': the checkpoint was written to its own share and the memory
        # is paid for in every request, so both stand whole, and the head cut that a compaction
        # request needs to fit the window never fires here — history compaction and a parent's
        # result cap bound the record later, each in its own way.
        fixed = estimate_result_tokens("\n".join(s.renderings[0] for s in sections if s.fixed))
        return fit_transcript(sections, budget_tokens + fixed if budget_tokens is not None else None, unit="rounds")

    def compact(self, summary: str, tokens: int = 0) -> PendingTurn:
        """Fold the view's rounds but the last into one checkpoint round: the summary as the model
        will read it, priced by `tokens` (0 -> heuristic). The last round is kept whole because it
        is the model's live continuation: its calls and results are what the next completion is
        about. Requires two non-summary rounds in the view, so at least one is folded."""
        assert sum(1 for r in self.since_last_summary() if not r.summary) >= 2, "nothing to fold"
        text = CHECKPOINT_PREFIX + summary
        # a Round has no self-pricing (a Turn does): without a usage frame the checkpoint would cost 0
        summary_round = Round(text, (), (), summary=True, tokens=tokens or estimate_tokens(text))
        # The kept round's frame priced the results of the round before it, which just folded: its
        # price is now its own text; its results are the unpriced text, as they were.
        kept = replace(self.rounds[-1], tokens=estimate_tokens(self.rounds[-1].own_text()))
        return replace(self, rounds=self.rounds[:-1] + (summary_round, kept))


# -----------------------
# Chat History
# -----------------------



@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str
    stop: StopReason
    tokens: int = 0
    rounds: tuple[Round, ...] = ()   # tool exchanges between user and assistant; () for a plain turn

    # the working memory as it stood when the turn ended, {slot: dict form}; None for a turn made
    # without one (a run that registered no memory, a summary turn). Kept in the dict form so the
    # record needs no plugin to load; LoadSession restores each registered slot from the newest
    # turn that recorded it. Not part of the turn's tokens: the block is priced live.
    memory: dict[str, dict] | None = None

    # what kind of turn this is: "chat" (the default, serialized without the key) or "settings"
    # (a settings change made by a /command; delta carries {setting: new_value}). LoadSession
    # replays settings turns in order to restore the settings.
    type: str = "chat"
    delta: dict | None = None

    def __post_init__(self):
        if self.tokens == 0:
            # the rounds of a history turn render stubbed (messages), so that is what they cost
            rounds = sum(estimate_tokens(r.text(stubbed=True)) for r in rounds_since_last_summary(self.rounds))
            object.__setattr__(self, 'tokens', estimate_tokens(self.user) + estimate_tokens(self.assistant) + rounds)

    def messages(self):
        """The wire form of a finished turn. Its rounds are always stubbed: the final answer is what
        the results led to, and a turn's prefix never changes again once it is in history."""
        return ([{"role": "user", "content": self.user}]
                + [m for r in rounds_since_last_summary(self.rounds) for m in r.messages(stubbed=True)]
                + [{"role": "assistant", "content": self.assistant}])

    def transcript(self, rounds: bool = True, stubbed: bool = False) -> str:
        """Plain-text rendering for the history display and the compaction prompt: the view only,
        since a checkpoint already stands for the rounds it folded. The display reads results
        whole; the compaction reads them `stubbed`, as the model last saw them (a history turn's
        answer is what its results led to). `rounds` False leaves the rounds out altogether: the
        reduced form the compaction transcript falls back to when the window does not fit."""
        middle = [r.transcript(stubbed=stubbed) for r in rounds_since_last_summary(self.rounds)] if rounds else []
        return "\n".join([f"USER: {self.user}"] + middle + [f"ASSISTANT: {self.assistant}"])

    @property
    def summary(self) -> bool:
        return self.stop == StopReason.SUMMARY

    @property
    def cancelled(self) -> bool:
        return self.stop == StopReason.CANCELLED

    @property
    def visible(self) -> bool:
        return self.stop not in HIDDEN_STOPS

    def to_dict(self) -> dict:
        d = {"user": self.user, "assistant": self.assistant, "tokens": self.tokens, "stop": self.stop.value}
        if self.rounds:     # the key exists only when there is something to record
            d["rounds"] = [r.to_dict() for r in self.rounds]
        if self.memory:     # likewise: only a turn made with a memory carries one
            d["memory"] = self.memory
        if self.type != "chat":     # likewise: a chat turn serializes exactly as before
            d["type"] = self.type
        if self.delta is not None:
            d["delta"] = self.delta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(user=d["user"], assistant=d["assistant"], tokens=d.get("tokens", 0),
                   stop=StopReason(d["stop"]),
                   rounds=tuple(Round.from_dict(r) for r in d.get("rounds", [])),
                   # format 6 recorded the one memory there was under its own key
                   memory=d["memory"] if "memory" in d else {"scratchpad": d["scratchpad"]} if "scratchpad" in d else None,
                   type=d.get("type", "chat"),
                   delta=d.get("delta"))


@dataclass(frozen=True)
class ChatHistory:
    turns: tuple[Turn, ...] = ()

    SESSION_FORMAT = 7                 # written
    # readable: 6 = every turn carries its stop reason (StopReason), which replaces the cancelled
    # and summary flags; 7 = a turn's working memory is recorded per slot under "memory", where 6
    # had a "scratchpad" key; older formats are not read
    SESSION_FORMATS = (6, 7)

    def last_memory(self, name: str) -> dict | None:
        """The dict form of the memory `name` as it stood at the end of the newest turn that
        recorded it; None when no turn did. Summary turns record none, so compaction never hides it."""
        for turn in reversed(self.turns):
            if turn.memory is not None and name in turn.memory:
                return turn.memory[name]
        return None

    def to_dict(self) -> dict:
        """Serializable form; the session file is this dict plus whatever metadata the saver adds."""
        return {"version": self.SESSION_FORMAT, "turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> ChatHistory:
        """Inverse of to_dict. Raises ValueError on an unknown format, KeyError/TypeError on a malformed one."""
        if not isinstance(d, dict):
            raise ValueError(f"session document must be an object, got {type(d).__name__}")
        version = d.get("version")
        if version not in cls.SESSION_FORMATS:
            raise ValueError(f"unsupported session format {version!r} (expected one of {cls.SESSION_FORMATS})")
        return cls(turns=tuple(Turn.from_dict(t) for t in d["turns"]))

    def append(self, turn: Turn) -> ChatHistory:
        return replace(self, turns=self.turns+(turn,))

    def view_turns(self, budget: int = 0) -> list[Turn]:
        """Return the longest tail of turns whose tokens fit within budget.
        Scans newest first (turns are appended, so the tail is the most recent), skipping cancelled
        turns and stopping at the last summary; the result is re-reversed into chronological order."""
        view, used  = [], 0
        for turn in reversed(self.turns):
            if not turn.visible:
                continue
            if used + turn.tokens > budget:
                break
            used += turn.tokens
            view.append(turn)
            if turn.summary:
                break
        return list(reversed(view))

    def view(self, budget: int = 0) -> list[dict]:
        """The messages of view_turns(budget), flattened for a Request."""
        return [msg for t in self.view_turns(budget) for msg in t.messages()]

    def get_total_tokens(self) -> int:
        """Return total tokens in history. Only includs vible turns."""
        return sum(turn.tokens for turn in self.turns if turn.visible)

    def window_tokens(self) -> int:
        """Return tokens since last summary."""
        return sum(turn.tokens for turn in self.since_last_summary())

    def messages(self):
        """Return all messages in history."""
        return [msg for turn in self.turns for msg in turn.messages()]

    SUMMARY_PREFIX = "Summary of the earlier conversation: "
    SUMMARY_ACK = "Understood."

    def compact(self, summary: str, tokens: int = 0) -> ChatHistory:
        """Replaces the turn history with a synthetic summary turn. tokens=0 -> heuristic pricing."""
        return self.append(Turn(self.SUMMARY_PREFIX + summary, self.SUMMARY_ACK, tokens=tokens, stop=StopReason.SUMMARY))

    def since_last_summary(self):
        """Return all turns since the last summary, except cancelled."""
        view = []
        for turn in reversed(self.turns):
            if not turn.visible:
                continue
            view.append(turn)
            if turn.summary:
                break
        return list(reversed(view))

    def transcript(self, budget_tokens: int | None = None) -> str:
        """The window since the last summary as the compaction prompt reads it: each turn as the
        model last saw it (rounds stubbed), fitted to `budget_tokens` by fit_transcript — the
        oldest turns lose their rounds first, then are left out; a summary turn never is."""
        sections = [Section((t.transcript(),), fixed=True) if t.summary
                    else Section((t.transcript(stubbed=True), t.transcript(rounds=False)))
                    for t in self.since_last_summary()]
        return fit_transcript(sections, budget_tokens, unit="turns")

    def digest(self, budget_tokens: int | None = None) -> str:
        """What stands in for a summary when the model returns none: the previous summary's text,
        when the window starts with one, and each turn's message and answer without its rounds.
        Deterministic, so a window never folds into nothing; fitted to `budget_tokens`."""
        sections = []
        for t in self.since_last_summary():
            if t.summary:
                sections.append(Section((t.user[len(self.SUMMARY_PREFIX):] if t.user.startswith(self.SUMMARY_PREFIX) else t.user,), fixed=True))
            else:
                sections.append(Section((t.transcript(rounds=False),)))
        return fit_transcript(sections, budget_tokens, unit="turns")

    def before(self, turn: Turn) -> tuple[Turn, ...]:
        """The turns that precede `turn` (by identity), for a policy that asks what the turn before
        the last one did. () when the turn is not in the history."""
        for i, t in enumerate(self.turns):
            if t is turn:
                return self.turns[:i]
        return ()

    def last_non_summary(self) -> Turn | None:
        """The last turn that is not a summary."""
        for turn in reversed(self.turns):
            if not turn.summary:
                return turn
        return None

    def __len__(self):
        return len(self.turns)
