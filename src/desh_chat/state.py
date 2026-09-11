from dataclasses import dataclass, field, replace
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.llama.wire import ToolCall
from desh.llama.tokens import estimate_tokens
from desh.engine import State
from desh.tools import ToolRegistry
from typing import Any

# -----------------------
# Chat DES State
# -----------------------

# The system prompt of the compaction request. Written for a working session with tools: what the
# summary must keep is what the model would otherwise re-read or re-derive — paths, names, lines,
# the exact text an edit has to match — and what it may drop is what a later step superseded.
COMPACTION_PROMPT = """You will be sent the transcript of a working session: a task, the assistant's tool calls (file reads, edits, shell commands) with their results, and the assistant's replies. Write the summary the assistant will work from in place of the transcript, so it can carry on without re-reading what it already read.

Keep, exactly as written: the task and its success criterion; every file path touched and what was done to it; the function, class and test names, signatures, line numbers and error messages that were established; short code fragments that will be needed again verbatim, such as the exact text an edit must match or a command that must be re-run; test results as reported; decisions made and why; what remains to do, as concrete next steps.

Drop: narration, the full contents of files that were read, tool output that a later step superseded. Prefer a terse list to prose. Do not mention this instruction and do not repeat the transcript."""


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

@dataclass(frozen=True)
class InferenceEngine:
    models: list[str]
    max_context: dict[str, int]
    server: LlamaServer = field(repr=False)
    port: int = field(repr=False)

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

    def change_setting(self, setting: str, value: Any) -> ChatState:
        return replace(self, settings=replace(self.settings, **{setting: value}))

    def pending_tokens(self) -> int:
        """What the pending turn costs in the prompt: priced rounds plus the heuristic for the text no usage frame has priced."""
        p = self.pending
        return 0 if p is None else p.priced_tokens() + estimate_tokens(p.unpriced_text())

    def prompt_tokens(self, pending_tokens: int) -> int:
        """What the next request costs before generation: system prompt, window since the last summary, pending."""
        return estimate_tokens(self.system_prompt) + self.history.window_tokens() + pending_tokens

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
        return estimate_tokens(self.system_prompt) + self.history.get_total_tokens() + pending_tokens

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


@dataclass(frozen=True)
class Round:
    """One intermediate model round inside a turn: the assistant asked for tools (with whatever text
    it said alongside), and the tools answered. A turn's final answer is NOT a Round — it is Turn.assistant."""
    assistant: str
    tool_calls: tuple[ToolCall, ...]
    results: tuple[ToolResult, ...] = ()
    tokens: int = 0     # priced from the usage frame of the completion that produced the calls

    def messages(self) -> list[dict]:
        """The wire form of this round, as it is echoed back in every later request of the conversation."""
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
            messages.append(result.message())
        return messages

    def text(self) -> str:
        """All text of the round, for heuristic pricing when no usage frame priced it."""
        calls = "".join(tc.name + tc.arguments for tc in self.tool_calls)
        return self.assistant + calls + "".join(r.content for r in self.results)

    def to_dict(self) -> dict:
        return {"assistant": self.assistant, "tokens": self.tokens,
                "tool_calls": [tc.to_dict() for tc in self.tool_calls],
                "results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, d: dict) -> Round:
        return cls(assistant=d["assistant"], tokens=d.get("tokens", 0),
                   tool_calls=tuple(ToolCall(index=tc["index"], id=tc["id"], type=tc["type"],
                                             name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                                    for tc in d["tool_calls"]),
                   results=tuple(ToolResult.from_dict(r) for r in d.get("results", [])))


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

    def messages(self) -> list[dict]:
        assert self.user is not None, "pending turn has no message yet"
        return [{"role": "user", "content": self.user}] + [m for r in self.rounds for m in r.messages()]

    def priced_tokens(self) -> int:
        """Tokens already priced by usage frames: every completed round."""
        return sum(r.tokens for r in self.rounds)

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

    def finish(self, assistant: str, tokens: int, cancelled: bool, stop: str = "") -> Turn:
        """The final answer arrived (or the turn was cut short): freeze into a history Turn.
        tokens prices only the final completion; the rounds carry their own."""
        assert self.user is not None, "pending turn has no message yet"
        return Turn(self.user, assistant, tokens=tokens + self.priced_tokens() if tokens else 0,
                    cancelled=cancelled, rounds=self.rounds, stop=stop)


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
    # why the turn ended early, "" when the model answered: "cap" (round cap hit; the model's text
    # so far is the answer) or "overflow" (no room left for a completion; cancelled as well, so the
    # turn stays out of the view). Read by whoever must tell the cases apart, the delegate's answer().
    stop: str = ""

    def __post_init__(self):
        if self.tokens == 0:
            rounds = sum(r.tokens or estimate_tokens(r.text()) for r in self.rounds)
            object.__setattr__(self, 'tokens', estimate_tokens(self.user) + estimate_tokens(self.assistant) + rounds)

    def messages(self):
        return ([{"role": "user", "content": self.user}]
                + [m for r in self.rounds for m in r.messages()]
                + [{"role": "assistant", "content": self.assistant}])

    def transcript(self) -> str:
        """Plain-text rendering for the compaction prompt and the history display."""
        lines = [f"USER: {self.user}"]
        for r in self.rounds:
            calls = ", ".join(f"{tc.name}({tc.arguments})" for tc in r.tool_calls)
            lines.append(f"ASSISTANT (tool calls): {r.assistant + ' ' if r.assistant else ''}{calls}")
            for res in r.results:
                lines.append(f"TOOL {res.name}: {res.content}")
        lines.append(f"ASSISTANT: {self.assistant}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {"user": self.user, "assistant": self.assistant, "tokens": self.tokens,
             "cancelled": self.cancelled, "summary": self.summary}
        if self.rounds:     # plain turns serialize exactly as they did in format 1
            d["rounds"] = [r.to_dict() for r in self.rounds]
        if self.stop:       # likewise: the key exists only when there is a reason to record
            d["stop"] = self.stop
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(user=d["user"], assistant=d["assistant"], tokens=d.get("tokens", 0),
                   cancelled=d.get("cancelled", False), summary=d.get("summary", False),
                   rounds=tuple(Round.from_dict(r) for r in d.get("rounds", [])),
                   stop=d.get("stop", ""))


@dataclass(frozen=True)
class ChatHistory:
    turns: tuple[Turn, ...] = ()

    SESSION_FORMAT = 2              # written
    SESSION_FORMATS = (1, 2)        # readable: 1 = plain turns only; 2 = turns may carry tool rounds

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

    def last_non_summary(self) -> Turn | None:
        """The last turn that is not a summary."""
        for turn in reversed(self.turns):
            if not turn.summary:
                return turn
        return None

    def __len__(self):
        return len(self.turns)
