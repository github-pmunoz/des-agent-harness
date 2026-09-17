import json
import time
from dataclasses import dataclass, field, replace
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.llama.wire import ToolCall
from desh.llama.tokens import RESULT_CHARS_PER_TOKEN, estimate_result_tokens, estimate_tokens
from desh.engine import State
from desh.tools import ToolRegistry
from desh_chat.scratchpad import Scratchpad
from typing import Any, Callable

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
# checkpoint is read right after the task message and next to the scratchpad block, in a window
# that is already tight, and it is rewritten every few rounds: so it must not restate the task,
# must not duplicate what the scratchpad holds, must say what the next concrete action is, and
# must stay short.
CHECKPOINT_PROMPT = """You will be sent a transcript of agent rounds being folded away. Write the checkpoint the assistant will read in place of them, right after the task message and next to the scratchpad block.

Do not restate the task — the task message is right before the checkpoint. Do not duplicate what the scratchpad holds — the scratchpad block follows the checkpoint. If the transcript already contains an earlier checkpoint, replace it with this one; do not append to it.

Keep: the one thing the assistant was about to do next, as a single concrete action, with the file paths, names and error messages that action depends on exactly as written; which files were written or edited and whether they pass, with test results as reported; what was verified versus what was only assumed. A result shown as expired was no longer available to the assistant: record only what later rounds establish. Drop narration and superseded detail. Stay short — the checkpoint must fit in a small fraction of the context window. Do not mention this instruction."""


@dataclass(frozen=True)
class Settings:
    model: str
    temperature: float
    think: bool
    context: int
    max_turn_tokens: int
    max_tool_rounds: int = 10       # tool-call rounds allowed inside one turn before it is forced to end
    tool_expiration: int = 6        # rounds after which tool results expire from context
    compaction_threshold: float = 0.65
    compaction_target: float = 0.25
    turn_token_cap: float = 0.40
    min_compaction_tokens: int = 64
    auto: bool = False           # auto mode: confirmed tools run without asking; Ctrl+C turns it off
    compaction_prompt: str = field(default=COMPACTION_PROMPT, repr=False)    # system prompt of the compaction request
    # A checkpoint (CompactPendingTurn) has its own instruction and share of the context: it sits
    # after the task message and next to the scratchpad, so it must not repeat either, and it is
    # rewritten every few rounds in a tight window, so it is kept smaller than a history summary.
    checkpoint_target: float = 0.15
    checkpoint_prompt: str = field(default=CHECKPOINT_PROMPT, repr=False)

    def to_dict(self) -> dict:
        """Plain-JSON form of every field, for the session document (format 4)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "think": self.think,
            "context": self.context,
            "max_turn_tokens": self.max_turn_tokens,
            "max_tool_rounds": self.max_tool_rounds,
            "tool_expiration": self.tool_expiration,
            "compaction_threshold": self.compaction_threshold,
            "compaction_target": self.compaction_target,
            "turn_token_cap": self.turn_token_cap,
            "min_compaction_tokens": self.min_compaction_tokens,
            "auto": self.auto,
            "compaction_prompt": self.compaction_prompt,
            "checkpoint_target": self.checkpoint_target,
            "checkpoint_prompt": self.checkpoint_prompt,
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
            tool_expiration=d.get("tool_expiration", 6),
            compaction_threshold=d.get("compaction_threshold", 0.65),
            compaction_target=d.get("compaction_target", 0.25),
            turn_token_cap=d.get("turn_token_cap", 0.40),
            min_compaction_tokens=d.get("min_compaction_tokens", 64),
            auto=d.get("auto", False),
            compaction_prompt=d.get("compaction_prompt", COMPACTION_PROMPT),
            checkpoint_target=d.get("checkpoint_target", 0.15),
            checkpoint_prompt=d.get("checkpoint_prompt", CHECKPOINT_PROMPT),
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
    # The model's working memory, as a value: None when the tool is not offered. Every write goes
    # through ExecuteToolCalls, which commits the new value here; the scratchpad tools themselves
    # only see a dict built from it for the one call. Rendered last in every request, and snapshotted
    # onto each finished Turn so a session restores it.
    scratchpad: Scratchpad | None = None
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
        turn = Turn(user="", assistant="", tokens=1, type="settings", delta={setting: value})
        return replace(self,
                       settings=replace(self.settings, **{setting: value}),
                       history=self.history.append(turn))

    def expire_after(self) -> int | None:
        """The round distance at which tool results leave the context; None when expiration is off."""
        k = self.settings.tool_expiration
        return k if k > 0 else None

    def pending_tokens(self) -> int:
        """What the pending turn costs in the prompt: priced rounds plus the heuristic for the text no usage frame has priced."""
        p = self.pending
        if p is None:
            return 0
        # the user message is prose, the latest results are tool output: different densities
        unpriced = estimate_tokens(p.unpriced_text()) if not p.rounds else estimate_result_tokens(p.unpriced_text())
        return p.priced_tokens(self.expire_after()) + unpriced

    def scratchpad_block(self) -> dict | None:
        """The scratchpad as the last message of the next request, with the line announcing the
        rounds whose results expire after it; None when the tool is not offered. The one place the
        block is built, so what is priced is what is sent."""
        if self.scratchpad is None:
            return None
        # each expiring call is mentioned with what it was about (the registry knows which argument
        # that is), so the model can decide what to persist without recalling what round N read
        expiring = self.pending.expiring(self.expire_after(), describe=self.tools.target) if self.pending is not None else ()
        # the round the next completion is: one past the completed ones, against the turn's cap
        round = (self.pending.non_summary_rounds() + 1, self.settings.max_tool_rounds) if self.pending is not None else None
        return self.scratchpad.to_context(self.settings.tool_expiration, expiring=expiring, round=round)

    def scratchpad_tokens(self) -> int:
        """What the scratchpad block costs in the prompt: heuristic, it is re-sent whole every request."""
        block = self.scratchpad_block()
        return estimate_tokens(block["content"]) if block is not None else 0

    def tools_tokens(self) -> int:
        """What the tool schemas cost in the prompt: heuristic, they are re-sent whole every request.
        Priced as prior, like the system prompt — left to the usage frames, their cost would sit on
        whichever round's frame first absorbed it, and leave the estimate with that round."""
        schemas = self.tools.schemas()
        return estimate_tokens(json.dumps(schemas)) if schemas else 0

    def prompt_tokens(self, pending_tokens: int) -> int:
        """What the next request costs before generation: system prompt, tool schemas, window since the last summary, pending, scratchpad."""
        return estimate_tokens(self.system_prompt) + self.tools_tokens() + self.history.window_tokens() + pending_tokens + self.scratchpad_tokens()

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


# What a tool message says once its result has expired from the context. Constant on purpose: the
# stubbed prefix of a request must not change from one round to the next, or the server re-prefills it.
EXPIRED_RESULT = "[expired: this result is no longer in context]"
CHECKPOINT_PREFIX = "Checkpoint of this turn so far, in place of the rounds before it: "

# How much of a call's target the expiring line shows: enough to recognise a path or a command,
# never a dump — the line is re-sent with every request while the round is expiring.
MENTION_CHARS = 60


@dataclass(frozen=True)
class Round:
    """One intermediate model round inside a turn: the assistant asked for tools (with whatever text
    it said alongside), and the tools answered. A turn's final answer is NOT a Round — it is Turn.assistant.

    The record is always complete: results are never dropped from the Round (the session file, the
    repeat detector and the compaction transcript read them). Expiration is a RENDERING: messages()
    and text() take `stubbed` and put EXPIRED_RESULT in place of every result. The calls stay as
    they are — the template wants one tool message per call, and the model must still see what it
    asked for."""
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
            return f"USER: {self.assistant}"
        calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in self.tool_calls)
        lines = [f"ASSISTANT (tool calls): {self.assistant + ' ' if self.assistant else ''}{calls}"]
        lines += [f"TOOL {res.name}: {EXPIRED_RESULT if stubbed else res.content}" for res in self.results]
        return "\n".join(lines)

    def own_text(self) -> str:
        """The round's own text — the assistant message and its calls, never the results."""
        return self.assistant + "".join(tc.name + tc.arguments for tc in self.tool_calls)

    def text(self, stubbed: bool = False) -> str:
        """All text of the round as rendered, for heuristic pricing when no usage frame priced it."""
        results = "".join(EXPIRED_RESULT if stubbed else r.content for r in self.results)
        return self.own_text() + results

    def mentions(self, describe: Callable[[str, str], str] | None = None) -> str:
        """The calls of the round, one mention each, for the expiring line of the scratchpad block:
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

    # Tool results age by ROUND, counted back from the latest completed round (distance 0, the
    # results the next completion is about to read). With expire_after = k the request shows three
    # bands: distance < k-1 active, distance == k-1 expiring (shown whole, announced in the
    # scratchpad block), distance >= k stubbed (EXPIRED_RESULT in place of the result). None (or
    # k <= 0) turns expiration off. Pure functions of (rounds, k): nothing is recorded, so the
    # session file and the repeat detector always see the full results, and a changed k re-renders.

    # Both bands and the request are over the VIEW (since_last_summary): the rounds a checkpoint
    # folded are not in the request, so they have no distance, and the checkpoint itself never
    # expires — it is the compressed record of what the model would otherwise re-read.

    def stubbed(self, index: int, expire_after: int | None) -> bool:
        """Whether round `index` of the view renders stubbed in the next request."""
        view = self.since_last_summary()
        return (expire_after is not None and expire_after > 0 and not view[index].summary
                and len(view) - 1 - index >= expire_after)

    def expiring(self, expire_after: int | None, describe: Callable[[str, str], str] | None = None) -> tuple[str, ...]:
        """One line per round whose results are shown for the last time in the next request:
        'round N: Read tests/conftest.py, Bash ls' — the calls with what they were about (Round.mentions),
        never their results: the model reads those where they still are and decides what to persist.
        N is the round's number in the turn as the model counts it (checkpoints do not count)."""
        if expire_after is None or expire_after <= 0:
            return ()
        view = self.since_last_summary()
        index = len(view) - expire_after      # distance == expire_after - 1
        if index < 0 or view[index].summary:
            return ()
        number = self.non_summary_rounds() - (len(view) - 1 - index)     # the rounds after it are all the model's
        return (f"round {number}: {view[index].mentions(describe)}",)

    def messages(self, expire_after: int | None = None) -> list[dict]:
        assert self.user is not None, "pending turn has no message yet"
        return ([{"role": "user", "content": self.user}]
                + [m for i, r in enumerate(self.since_last_summary()) for m in r.messages(stubbed=self.stubbed(i, expire_after))])

    def priced_tokens(self, expire_after: int | None = None) -> int:
        """What the completed rounds cost in the next request. A round's `tokens` is what its usage
        frame priced, with the results whole; a stubbed round no longer costs that."""
        if expire_after is not None:
            return sum(r.tokens if i < expire_after else estimate_tokens(r.text(stubbed=True)) for i, r in enumerate(reversed(self.since_last_summary())))
        else:
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

    def finish(self, assistant: str, tokens: int, cancelled: bool, stop: str = "", scratchpad: Scratchpad | None = None) -> Turn:
        """The final answer arrived (or the turn was cut short): freeze into a history Turn.
        tokens prices only the final completion. A history turn renders its rounds stubbed from
        now on (Turn.messages), so the rounds are priced on that rendering here, once, and not on
        the usage frames that priced them whole. scratchpad is the working memory as it stands
        when the turn ends, recorded on the Turn for the session file."""
        assert self.user is not None, "pending turn has no message yet"
        stubbed_rounds = sum(estimate_tokens(r.text(stubbed=True)) for r in self.since_last_summary())
        # every round goes on the record (the session file shows what a checkpoint folded); the
        # Turn renders and prices its view, as the pending turn did
        return Turn(self.user, assistant, tokens=tokens + stubbed_rounds if tokens else 0,
                    cancelled=cancelled, rounds=self.rounds, stop=stop, scratchpad=scratchpad)

    def since_last_summary(self) -> tuple[Round, ...]:
        """The rounds the next request carries: the latest checkpoint, when there is one, and
        every round after it. The rounds a checkpoint folded stay in `rounds` for the record."""
        return rounds_since_last_summary(self.rounds)

    def non_summary_rounds(self) -> int:
        """The model's rounds in the whole turn, checkpoints excluded: what the round cap counts.
        A checkpoint frees context, it does not open a new turn, so the cap does not restart."""
        return sum(1 for r in self.rounds if not r.summary)

    def transcript(self, rounds: tuple[Round, ...], expire_after: int | None = None, budget_tokens: int | None = None) -> str:
        """Plain-text rendering of the user message and `rounds` (a prefix of the view), for the
        checkpoint prompt. Each round renders as the model last saw it — whole or stubbed by the
        same bands as the request — and the whole is fitted to `budget_tokens` (fit_transcript):
        the oldest whole results are stubbed first, then the oldest rounds left out. A checkpoint
        among the rounds renders as the user message it is on the wire and is never reduced, so a
        second checkpoint subsumes the first."""
        sections = [Section((f"USER: {self.user}",), fixed=True)]
        for i, r in enumerate(rounds):
            if r.summary:
                sections.append(Section((r.transcript(),), fixed=True))
            elif self.stubbed(i, expire_after):
                sections.append(Section((r.transcript(stubbed=True),)))
            else:
                sections.append(Section((r.transcript(), r.transcript(stubbed=True))))
        return fit_transcript(sections, budget_tokens, unit="rounds")

    def digest(self, rounds: tuple[Round, ...], describe: Callable[[str, str], str] | None = None, budget_tokens: int | None = None) -> str:
        """What stands in for a checkpoint when the model returns none: the previous checkpoint's
        text, when `rounds` starts with one, and one line per folded round naming its calls
        (Round.mentions) — what was done, never the results. Deterministic and cheap, so a turn
        never folds into nothing; fitted to `budget_tokens` like a transcript."""
        ordinal = {id(r): n for n, r in enumerate((r for r in self.rounds if not r.summary), start=1)}   # as the model counts rounds
        sections = []
        for r in rounds:
            if r.summary:
                sections.append(Section((r.assistant[len(CHECKPOINT_PREFIX):] if r.assistant.startswith(CHECKPOINT_PREFIX) else r.assistant,), fixed=True))
            else:
                sections.append(Section((f"round {ordinal.get(id(r), '?')}: {r.mentions(describe)}",)))
        return fit_transcript(sections, budget_tokens, unit="rounds")

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
    tokens: int = 0
    cancelled: bool = False
    summary: bool = False
    rounds: tuple[Round, ...] = ()   # tool exchanges between user and assistant; () for a plain turn

    # why the turn ended early
    # - "" when the model answered
    # - "cap" (round cap hit; the model's text so far is the answer)
    # - "overflow" (no room left for a completion; cancelled as well, so the turn stays out of the view)
    # - "interrupt" (Ctrl+C in auto mode; cancelled)
    # - "error" (a mid-turn exception; cancelled)
    # - "deadline" (the run's wall-clock budget ran out; the model's text so far is the answer,
    #   and unlike "cap" the turn is never continued)
    # Read by whoever must tell the cases apart: the delegate's answer(), the CLI's exit code.
    stop: str = ""

    # the scratchpad as it stood when the turn ended; None for a turn made without one (a run
    # without the tool, a summary turn, a file older than format 3). LoadSession restores the
    # newest one. Not part of the turn's tokens: the block is priced live, as the current value.

    scratchpad: Scratchpad | None = None

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

    def to_dict(self) -> dict:
        d = {"user": self.user, "assistant": self.assistant, "tokens": self.tokens,
             "cancelled": self.cancelled, "summary": self.summary}
        if self.rounds:     # plain turns serialize exactly as they did in format 1
            d["rounds"] = [r.to_dict() for r in self.rounds]
        if self.stop:       # likewise: the key exists only when there is a reason to record
            d["stop"] = self.stop
        if self.scratchpad is not None:     # likewise: only a turn made with the tool carries one
            d["scratchpad"] = self.scratchpad.to_dict()
        if self.type != "chat":     # likewise: a chat turn serializes exactly as before
            d["type"] = self.type
        if self.delta is not None:
            d["delta"] = self.delta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(user=d["user"], assistant=d["assistant"], tokens=d.get("tokens", 0),
                   cancelled=d.get("cancelled", False), summary=d.get("summary", False),
                   rounds=tuple(Round.from_dict(r) for r in d.get("rounds", [])),
                   stop=d.get("stop", ""),
                   scratchpad=Scratchpad.from_dict(d["scratchpad"]) if "scratchpad" in d else None,
                   type=d.get("type", "chat"),
                   delta=d.get("delta"))


@dataclass(frozen=True)
class ChatHistory:
    turns: tuple[Turn, ...] = ()

    SESSION_FORMAT = 4              # written
    SESSION_FORMATS = (1, 2, 3, 4)  # readable: 1 = plain turns only; 2 = turns may carry tool rounds; 3 = turns may carry a scratchpad; 4 = the document carries the settings, and turns may be settings turns

    def last_scratchpad(self) -> Scratchpad | None:
        """The working memory as it stood at the end of the newest turn that recorded one; None
        when no turn did. Summary turns record none, so compaction never hides it."""
        for turn in reversed(self.turns):
            if turn.scratchpad is not None:
                return turn.scratchpad
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
            if turn.cancelled:
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
        """Return total tokens in history. Does not include cancelled turns."""
        return sum(turn.tokens for turn in self.turns if not turn.cancelled)

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
        return self.append(Turn(self.SUMMARY_PREFIX + summary, self.SUMMARY_ACK, tokens=tokens, summary=True))

    def since_last_summary(self):
        """Return all turns since the last summary, except cancelled."""
        view = []
        for turn in reversed(self.turns):
            if turn.cancelled:
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

    def last_non_summary(self) -> Turn | None:
        """The last turn that is not a summary."""
        for turn in reversed(self.turns):
            if not turn.summary:
                return turn
        return None

    def __len__(self):
        return len(self.turns)
