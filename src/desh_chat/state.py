from dataclasses import dataclass, field, replace
from desh.llama.client import LlamaServer
from desh.llama.tokens import estimate_tokens
from desh.engine import State

# -----------------------
# Chat DES State
# -----------------------

@dataclass(frozen=True)
class ChatState(State):
    """Immutable chat state."""
    history: ChatHistory
    model: str
    temperature: float
    think: bool
    running: bool
    system_prompt: str
    context: int
    max_tokens: int
    log_path: str
    server: LlamaServer = field(repr=False)


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
    max_tokens: int = 16384

    def append(self, turn: Turn) -> ChatHistory:
        if turn.tokens > self.max_tokens:
            raise ValueError("Turn exceeds max_tokens")
        return replace(self, turns=self.turns+(turn,))

    def view(self, extra_tokens: int = 0) -> list[dict]:
        """Return the longest tail of turns which in addition to the extra_tokens fits within max_tokens."""
        budget = self.max_tokens - extra_tokens
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
