from __future__ import annotations
from dataclasses import dataclass, replace

# Appended to the system prompt of any run that offers the scratchpad tools. Mechanics first: the
# model must know that results vanish and WHEN, or it learns it the expensive way — by hitting the
# round cap and re-gathering everything in the continuation. Formatted with the run's settings.
SCRATCHPAD_SYSTEM_PROMPT = (
    "How your context works. Each reply of yours that calls tools is a round; the results come back "
    "in the next request. A turn allows at most {max_tool_rounds} rounds. At the cap the turn ends, and "
    "if the task is unfinished you are asked to continue in a new turn in which EVERY tool result of "
    "the previous turn has been replaced by an expired stub. Within a turn, a tool result stays visible "
    "for {tool_expiration} rounds and is then replaced by the same stub. The <scratchpad> block at the "
    "end of every request is the only working memory that survives both: it shows which round you are "
    "on and names the results that expire next. Use scratchpad_write for whatever you will still need "
    "(paths, line numbers, ids, constraints, conclusions), condensed, never raw dumps. Write it in the "
    "same reply as your other tool calls: a scratchpad call costs no round of its own. Persist as you "
    "go; do not wait for the expiry notice or the cap."
)

def write(key: str, value: str, scratchpad: dict[str, str]) -> str:
    """Write an entry in the scratchpad, creating a new key or overwriting an existing one.
    
    Args:
        key: The key to store the value under.
        value: The value to store.
    """
    existed = key in scratchpad
    scratchpad[key] = value
    return f"created {key!r}" if not existed else f"overwrote {key!r}"

def delete(key: str, scratchpad: dict[str, str]) -> str:
    """Delete a value from the scratchpad
    
    Args:
        key: The key to delete.
    """
    if key in scratchpad:
        del scratchpad[key]
        return f"{key!r} deleted"
    return f"{key!r} not found"

def clear(scratchpad: dict[str, str]) -> str:
    """Clear the scratchpad."""
    n_items = len(scratchpad)
    if n_items:
        scratchpad.clear()
        return f"cleared {n_items} items"
    return "already empty"
    

@dataclass(frozen=True)
class Scratchpad:
    """The model's working memory, as a value: ordered (key, value) pairs. Lives on ChatState and
    is snapshotted onto every finished Turn. The functions above are the model's side: they work
    on a dict the harness builds from this value for one call and reads back after it."""
    memory: tuple[tuple[str, str],...] = tuple()

    def with_entry(self, key: str, value: str) -> Scratchpad:
        """Write an entry in the scratchpad, creating a new key or overwriting an existing one.
        
        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        return replace(self, memory=tuple(m for m in self.memory if m[0] != key)+ ((key, value),))

    def cleared(self) -> Scratchpad:
        """Clear the scratchpad."""
        return replace(self, memory=tuple())

    def without(self, key: str) -> Scratchpad:
        """Delete a value from the scratchpad
        
        Args:
            key: The key to delete.
        """
        return replace(self, memory=tuple(m for m in self.memory if m[0] != key))

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> Scratchpad:
        return cls(tuple(d.items()))

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.memory}

    def message(self, tool_expiration: int, expiring: tuple[str, ...] = (), round: tuple[int, int] | None = None) -> str:
        """The scratchpad as a context string. `round` is (this round, the turn's cap), shown so the
        model sees its budget on every request. `expiring` names the rounds whose tool results are
        shown for the last time in this request (PendingTurn.expiring): names only, never the
        results — the model reads them where they still are and decides what to persist."""
        header = f"Round {round[0]} of {round[1]} in this turn. " if round is not None else ""
        message = f"<scratchpad>{header}Working memory. Tool results expire from context after {tool_expiration} rounds, persist here.\n"
        if self.memory:
            message += "\n".join(f"{k}: {v}" for k, v in self.memory) + "\n"
        else:
            message += "\n(empty)\n"
        if expiring:
            message += "Expiring next round, persist what you still need from: " + "; ".join(expiring) + "\n"
        message += "</scratchpad>"
        return message

    def to_context(self, tool_expiration: int, expiring: tuple[str, ...] = (), round: tuple[int, int] | None = None) -> dict:
        """Return a message for injecting into the request."""
        return {"role": "user", "content": self.message(tool_expiration, expiring, round)}