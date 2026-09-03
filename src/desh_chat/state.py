from dataclasses import dataclass, field, replace
from desh.llama.client import LlamaServer
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

@dataclass(frozen=True)
class InferenceEngine:
    models: list[str]
    max_context: dict[str, int]
    server: LlamaServer = field(repr=False)

@dataclass(frozen=True)
class ChatState(State):
    """Immutable chat state."""
    settings: Settings
    history: ChatHistory
    running: bool
    system_prompt: str
    completions_log: str
    inference: InferenceEngine = field(repr=False)

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


@dataclass(frozen=True)
class ChatHistory:
    turns: tuple[Turn, ...] = ()

    def append(self, turn: Turn) -> ChatHistory:
        return replace(self, turns=self.turns+(turn,))

    def view(self, budget: int = 0) -> list[dict]:
        """Return the longest tail of turns which in addition to the extra_tokens fits within max_tokens."""
        view, used  = [], 0
        for turn in reversed(self.turns):
            if used + turn.tokens > budget:
                break
            used += turn.tokens
            view.append(turn)
            if turn.summary:
                break
        return [msg for t in reversed(view) for msg in t.messages()]

    def get_total_tokens(self) -> int:
        """Return total tokens in history."""
        return sum(turn.tokens for turn in self.turns)

    def window_tokens(self) -> int:
        """Return tokens since last summary."""
        return sum(turn.tokens for turn in self.since_last_summary())

    def messages(self):
        """Return all messages in history."""
        return [msg for turn in self.turns for msg in turn.messages()]

    def compact(self, summary: str) -> ChatHistory:
        """Replaces the turn history with a synthetic summary turn"""
        return self.append(Turn(f"Summary of the earlier conversation: {summary}", "Understood.", summary=True))

    def since_last_summary(self):
        """Return all turns since the last summary."""
        view = []
        for turn in reversed(self.turns):
            view.append(turn)
            if turn.summary:
                break
        return list(reversed(view))

    def __len__(self):
        return len(self.turns)
