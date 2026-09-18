from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

# Every entry has a kind: what the entry IS to the model's task, not what it is about. The tool's
# schema lists them (a Literal becomes a JSON enum), the block renders them grouped, and the
# system prompt says what each one means.
KINDS = ("todo", "done", "fact", "hypothesis", "block")
Kind = Literal["todo", "done", "fact", "hypothesis", "block"]

# Appended to the system prompt of any run that offers the scratchpad tools. Mechanics first: the
# model must know that results vanish and WHEN, or it learns it the expensive way — by hitting the
# round cap and re-gathering everything in the continuation. Formatted with the run's settings.
SCRATCHPAD_SYSTEM_PROMPT = (
    "How your context works. Each reply of yours that calls tools is a round; the results come back "
    "in the next request. A turn allows at most {max_tool_rounds} rounds. At the cap the turn ends: the "
    "scratchpad calls of that reply still run, its other calls do not. If the task is unfinished you are "
    "asked to continue in a new turn in which EVERY tool result of the previous turn has been replaced "
    "by an expired stub. Within a turn, a tool result stays visible "
    "for {tool_expiration} rounds and is then replaced by the same stub. The <scratchpad> block at the "
    "end of every request is the only working memory that survives both: it shows which round you are "
    "on and names the results that expire next. Use scratchpad_write for whatever you will still need "
    "(paths, line numbers, ids, constraints, conclusions), condensed, never raw dumps. Write it in the "
    "same reply as your other tool calls: a scratchpad call costs no round of its own. Persist as you "
    "go; do not wait for the expiry notice or the cap.\n"
    "Every entry has a kind: todo, a step still to do; done, a finished step and its outcome; fact, "
    "something established from a file or a result; hypothesis, something you believe but have not "
    "verified; block, what stops progress and what it needs. An item keeps its key for life: when a "
    "todo is done, write the SAME key again as done with the outcome; when a hypothesis is checked, "
    "write it again as a fact. Never add a second key for an item that already has one."
)

def write(key: str, kind: Kind, value: str, scratchpad: dict[str, dict]) -> str:
    """Write an entry in the scratchpad, creating a new key or overwriting an existing one.

    Args:
        key: The key to store the value under.
        kind: What the entry is: todo, done, fact, hypothesis or block.
        value: The value to store.
    """
    if kind not in KINDS:
        return f"unknown kind {kind!r}; one of {', '.join(KINDS)}"
    before = scratchpad.get(key)
    scratchpad[key] = {"kind": kind, "value": value}
    if before is None:
        return f"created {key!r} ({kind})"
    old = before.get("kind") if isinstance(before, dict) else "fact"
    return f"overwrote {key!r} ({old} -> {kind})" if old != kind else f"overwrote {key!r} ({kind})"

def fold_write(args: dict) -> dict:
    """The echoed form of an answered scratchpad_write (Tool.fold): key and kind, the value
    replaced by its size. Once the call ran, the value is in the block that ends every request;
    echoing it in the round as well carried the same text twice, and at 4k that was an overflow."""
    value = args.get("value")
    if not isinstance(value, str):
        return args
    return {**args, "value": f"[{len(value)} characters, shown in the scratchpad block]"}

def delete(key: str, scratchpad: dict[str, dict]) -> str:
    """Delete a value from the scratchpad

    Args:
        key: The key to delete.
    """
    if key in scratchpad:
        del scratchpad[key]
        return f"{key!r} deleted"
    return f"{key!r} not found"

def clear(scratchpad: dict[str, dict]) -> str:
    """Clear the scratchpad."""
    n_items = len(scratchpad)
    if n_items:
        scratchpad.clear()
        return f"cleared {n_items} items"
    return "already empty"


@dataclass(frozen=True)
class Entry:
    """One scratchpad entry: the key the model addresses it by, what it is (KINDS), and the text."""
    key: str
    kind: str
    value: str


@dataclass(frozen=True)
class Scratchpad:
    """The model's working memory, as a value: ordered entries, one per key. Lives on ChatState and
    is snapshotted onto every finished Turn. The functions above are the model's side: they work
    on a dict the harness builds from this value for one call and reads back after it."""
    memory: tuple[Entry, ...] = tuple()

    def with_entry(self, key: str, kind: str, value: str) -> Scratchpad:
        """Write an entry, creating a new key or overwriting an existing one (kind and value both;
        the key moves to the end, as the newest)."""
        return replace(self, memory=tuple(m for m in self.memory if m.key != key) + (Entry(key, kind, value),))

    def cleared(self) -> Scratchpad:
        """Clear the scratchpad."""
        return replace(self, memory=tuple())

    def without(self, key: str) -> Scratchpad:
        """Delete a value from the scratchpad

        Args:
            key: The key to delete.
        """
        return replace(self, memory=tuple(m for m in self.memory if m.key != key))

    def of_kind(self, kind: str) -> tuple[Entry, ...]:
        """The entries of one kind, in their order."""
        return tuple(m for m in self.memory if m.kind == kind)

    @classmethod
    def from_dict(cls, d: dict[str, dict | str]) -> Scratchpad:
        """The dict form back to a value. A plain string value is an entry written before kinds
        existed (a session file of format 4 or older): it loads as a fact, the neutral kind."""
        return cls(tuple(Entry(k, v["kind"], v["value"]) if isinstance(v, dict) else Entry(k, "fact", v)
                         for k, v in d.items()))

    def to_dict(self) -> dict[str, dict]:
        """The dict the tools work on and the session file stores: {key: {"kind", "value"}}."""
        return {m.key: {"kind": m.kind, "value": m.value} for m in self.memory}

    def message(self, tool_expiration: int, expiring: tuple[str, ...] = (), round: tuple[int, int] | None = None) -> str:
        """The scratchpad as a context string. `round` is (this round, the turn's cap), shown so the
        model sees its budget on every request. `expiring` names the rounds whose tool results are
        shown for the last time in this request (PendingTurn.expiring): names only, never the
        results — the model reads them where they still are and decides what to persist."""
        header = f"Round {round[0]} of {round[1]} in this turn. " if round is not None else ""
        message = f"<scratchpad>{header}Working memory. Tool results expire from context after {tool_expiration} rounds, persist here.\n"
        if self.memory:
            message += self.entries_text() + "\n"
        else:
            message += "\n(empty)\n"
        if expiring:
            message += "Expiring next round, persist what you still need from: " + "; ".join(expiring) + "\n"
        message += "</scratchpad>"
        return message

    def entries_text(self) -> str:
        """The entries grouped by kind, for the block: one header line per kind that has entries,
        kinds without any left out, each entry on its own `key: value` line under its header."""

        rendering_order = ["done", "fact", "hypothesis", "block", "todo"]
        headers: dict[str, str] = {
            "done": "DONE",
            "todo": "TO DO",
            "fact": "FACTS",
            "hypothesis": "HYPOTHESES",
            "block": "BLOCKS",
        }
        lines = []
        for kind in rendering_order:
            entries = self.of_kind(kind)
            if not entries:
                continue
            lines.append(headers[kind])
            for entry in entries:
                lines.append(f"{entry.key}: {entry.value}")
        return "\n".join(lines)

    def to_context(self, tool_expiration: int, expiring: tuple[str, ...] = (), round: tuple[int, int] | None = None) -> dict:
        """Return a message for injecting into the request."""
        return {"role": "user", "content": self.message(tool_expiration, expiring, round)}
