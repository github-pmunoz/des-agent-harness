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
    pending: PendingTurn | None = None  # the turn in progress between UserMessage and TurnEnd; never persisted
    tools: ToolRegistry = field(default_factory=ToolRegistry, repr=False)  # what the model may call; empty -> no tools offered

    def change_setting(self, setting: str, value: Any) -> ChatState:
        return replace(self, settings=replace(self.settings, **{setting: value}))


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
    """A turn between UserMessage and TurnEnd: the user message plus every completed tool round so
    far. Lives on ChatState.pending only; never in ChatHistory, never in the session file."""
    user: str
    rounds: tuple[Round, ...] = ()

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}] + [m for r in self.rounds for m in r.messages()]

    def priced_tokens(self) -> int:
        """Tokens already priced by usage frames: every completed round."""
        return sum(r.tokens for r in self.rounds)

    def unpriced_text(self) -> str:
        """The prompt text the NEXT completion's usage frame will price: the user message on round one,
        the latest tool results afterwards."""
        if not self.rounds:
            return self.user
        return "\n".join(r.content for r in self.rounds[-1].results)

    def add_round(self, round: Round) -> PendingTurn:
        return replace(self, rounds=self.rounds + (round,))

    def with_results(self, results: tuple[ToolResult, ...]) -> PendingTurn:
        """Attach results to the latest round (the one whose calls just ran)."""
        last = replace(self.rounds[-1], results=results)
        return replace(self, rounds=self.rounds[:-1] + (last,))

    def add_results(self, *results: ToolResult) -> PendingTurn:
        """Append results to the latest round, in call order: the round is answered one call per step."""
        return self.with_results(self.rounds[-1].results + results)

    def finish(self, assistant: str, tokens: int, cancelled: bool, stop: str = "") -> Turn:
        """The final answer arrived (or the turn was cut short): freeze into a history Turn.
        tokens prices only the final completion; the rounds carry their own."""
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

    def __len__(self):
        return len(self.turns)
