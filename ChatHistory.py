from dataclasses import dataclass, field
from Tokens import estimate_tokens

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

@dataclass
class ChatHistory:
    turns: list[Turn] = field(default_factory=list)
    max_tokens: int = 16384

    def append(self, turn: Turn):
        if turn.tokens > self.max_tokens:
            raise ValueError("Turn exceeds max_tokens")
        self.turns.append(turn)

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

    def clear(self):
        """Clear all turns."""
        self.turns.clear()

    def get_total_tokens(self) -> int:
        """Return total tokens in history."""
        return sum(turn.tokens for turn in self.turns)

    def window_tokens(self) -> int:
        """Return tokens since last summary."""
        return sum(turn.tokens for turn in self.since_last_summary())

    def messages(self):
        """Return all messages in history."""
        return [msg for turn in self.turns for msg in turn.messages()]

    def compact(self, summary: str):
        """Replaces the turn history with a synthetic summary turn"""
        self.append(Turn(f"Summary of the earlier conversation: {summary}", "Understood.", summary=True))

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