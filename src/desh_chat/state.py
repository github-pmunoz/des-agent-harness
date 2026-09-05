from dataclasses import dataclass, field, replace
from desh.llama.client import LlamaServer, Logger
from desh.llama.tokens import estimate_tokens
from desh.engine import State
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
    compaction_threshold: float = 0.65
    compaction_target: float = 0.25
    turn_token_cap: float = 0.40
    min_compaction_tokens: int = 64

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

    def change_setting(self, setting: str, value: Any) -> ChatState:
        return replace(self, settings=replace(self.settings, **{setting: value}))


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

    def __post_init__(self):
        if self.tokens == 0:
            object.__setattr__(self, 'tokens', estimate_tokens(self.user) + estimate_tokens(self.assistant))

    def messages(self):
        return [ {"role": "user", "content": self.user}, {"role": "assistant", "content": self.assistant} ]

    def to_dict(self) -> dict:
        return {"user": self.user, "assistant": self.assistant, "tokens": self.tokens,
                "cancelled": self.cancelled, "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(user=d["user"], assistant=d["assistant"], tokens=d.get("tokens", 0),
                   cancelled=d.get("cancelled", False), summary=d.get("summary", False))


@dataclass(frozen=True)
class ChatHistory:
    turns: tuple[Turn, ...] = ()

    SESSION_FORMAT = 1

    def to_dict(self) -> dict:
        """Serializable form; the session file is this dict plus whatever metadata the saver adds."""
        return {"version": self.SESSION_FORMAT, "turns": [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> ChatHistory:
        """Inverse of to_dict. Raises ValueError on an unknown format, KeyError/TypeError on a malformed one."""
        if not isinstance(d, dict):
            raise ValueError(f"session document must be an object, got {type(d).__name__}")
        version = d.get("version")
        if version != cls.SESSION_FORMAT:
            raise ValueError(f"unsupported session format {version!r} (expected {cls.SESSION_FORMAT})")
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
