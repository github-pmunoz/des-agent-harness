from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Protocol

from desh.llama.tokens import estimate_tokens
from desh.tools import ToolRegistry

# Memory tools, as plugins. A memory is what the model writes down to outlive its tool results: a
# scratchpad, a work plan, a knowledge graph. The split of ownership:
# - a plugin (Memory) owns its value, its tools and the prompt that says what it is FOR;
# - the harness (this module) owns everything about WHEN memory matters: the system prompt that
#   explains rounds, the cap and checkpoints, the one block that ends every request, the lines that
#   announce what is about to expire, and the budget each memory may take of the window.
# A plugin never explains context mechanics, and no event knows a memory by name: the slot name is
# the tools' injected parameter (Tool.inject), the key on the state, the key in the session file
# and the tag in the block.


class MemoryValue(Protocol):
    """What a memory holds, as an immutable value. The tools never see it: they work on the dict
    form, built for one call and read back after it (Memories.provide / commit)."""
    def to_dict(self) -> dict: ...
    def render(self) -> str:
        """The body of this memory's section of the block: no tags, no mechanics, "" when empty."""
        ...


@dataclass(frozen=True)
class Memory:
    """One pluggable memory: everything the harness needs to offer it, none of how it survives."""
    name: str                                   # the slot: inject key, state key, session key, block tag
    empty: Callable[[], MemoryValue] = field(repr=False)            # the value a run starts with
    from_dict: Callable[[dict], MemoryValue] = field(repr=False)    # the dict form back to a value
    # (function, Tool.define overrides) per tool; registered unconfirmed with inject=(name,)
    tools: tuple[tuple[Callable[..., Any], dict], ...] = ()
    prompt: str = ""                            # what it is for and how to use its tools; nothing else
    # What a subagent gets: a "fresh" empty one of its own, or none ("off"). A subagent's memory
    # dies with it, so a memory only worth having when it is handed back is better left off.
    subagent: Literal["fresh", "off"] = "fresh"

    def tool_names(self) -> tuple[str, ...]:
        return tuple(overrides.get("name", fn.__name__) for fn, overrides in self.tools)

    def register(self, tools: ToolRegistry) -> ToolRegistry:
        """The registry with this memory's tools added. They are unconfirmed — a memory call has no
        effect on the workspace — and declare the slot as injected, which is what makes a call a
        memory call (Memories.owns)."""
        for fn, overrides in self.tools:
            tools = tools.add(fn, inject=(self.name,), confirm=False, **overrides)
        return tools


# Appended to the system prompt of any run that registers a memory, formatted with the run's own
# settings and memories. Mechanics first: the model must know that results vanish and WHEN (at the
# round cap, and when a checkpoint folds its rounds), or it learns it the expensive way — by
# re-gathering everything in the continuation. Each memory's own prompt follows it.
CONTEXT_MECHANICS_PROMPT = (
    "How your context works. Each reply of yours that calls tools is a round; the results come back "
    "in the next request. A turn allows at most {max_tool_rounds} rounds. At the cap the turn ends: the "
    "memory calls of that reply ({tool_names}) still run, its other calls do not. If the task is "
    "unfinished you are asked to continue in a new turn in which EVERY tool result of the previous "
    "turn has been replaced by an expired stub. When the context runs short inside a turn, your "
    "earlier rounds are folded into a checkpoint and their results are gone; only your memory is kept "
    "verbatim. The <memory> block at the end of every request is the only working memory that "
    "survives both: it holds {sections}, shows which round you are on, and tells you when the cap or "
    "a checkpoint is one round away. Record there whatever you will still need, condensed, never raw "
    "dumps. Make memory calls in the same reply as your other tool calls: a memory call costs no "
    "round of its own. Persist as you go; do not wait for the cap. Each memory has a budget, shown "
    "in its tag: a write that would exceed it is refused. Your earlier calls are echoed with their "
    "long arguments removed and a `folded` note in their place; that is the record, not a form to write."
)

# The lines of the block the harness writes (Memories.block).
BLOCK_HEAD = "Working memory. Tool results do not survive the turn, persist here."
LAST_ROUND_LINE = ("Last round before the cap: every tool result of this turn is stubbed in the next one. "
                   "Persist what you still need now.")
CAP_REACHED_LINE = "Round cap reached: only memory calls ({tool_names}) in this reply will run."
FOLD_NEAR_LINE = ("The context is nearly full: your earlier rounds are folded into a checkpoint within a round "
                  "or two and their results are gone. Persist what you still need in this reply, with your other calls; "
                  "this is said once.")
OVER_BUDGET_TEXT = ("Not recorded: <{name}> would take {tokens} tokens, over its budget of {budget}. "
                    "Condense or delete entries first.")


@dataclass(frozen=True)
class MemoryFrame:
    """The texts the harness writes around the memories, as a value: the defaults are the
    constants above, and a run built with other texts (desh_chat.prompts) carries them here. The
    templates keep their placeholders: mechanics {max_tool_rounds} {tool_names} {sections},
    cap_reached {tool_names}, over_budget {name} {tokens} {budget}."""
    mechanics: str = CONTEXT_MECHANICS_PROMPT
    block_head: str = BLOCK_HEAD
    last_round: str = LAST_ROUND_LINE
    cap_reached: str = CAP_REACHED_LINE
    fold_near: str = FOLD_NEAR_LINE
    over_budget: str = OVER_BUDGET_TEXT


@dataclass(frozen=True)
class Memories:
    """The memories a run registered and the value each one holds, in registration order. Lives on
    ChatState; empty means no memory is offered: no block, no mechanics in the system prompt.
    Every write goes through ExecuteToolCalls, which commits the new value on the state; nothing
    mutable survives a step, so a step the engine rolls back leaves the memory untouched."""
    slots: tuple[tuple[Memory, MemoryValue], ...] = ()
    frame: MemoryFrame = MemoryFrame()

    @classmethod
    def of(cls, *memories: Memory, frame: MemoryFrame = MemoryFrame()) -> Memories:
        """Every memory registered and empty: what a run starts with (LoadSession restores on top)."""
        return cls(tuple((m, m.empty()) for m in memories), frame)

    def __bool__(self) -> bool:
        return bool(self.slots)

    def __contains__(self, name: str) -> bool:
        return any(m.name == name for m, _ in self.slots)

    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m, _ in self.slots)

    def get(self, name: str) -> MemoryValue | None:
        return next((v for m, v in self.slots if m.name == name), None)

    def with_value(self, name: str, value: MemoryValue) -> Memories:
        return replace(self, slots=tuple((m, value if m.name == name else v) for m, v in self.slots))

    def tool_names(self) -> tuple[str, ...]:
        return tuple(n for m, _ in self.slots for n in m.tool_names())

    # -- a call ---------------------------------------------------------------------------------

    def owns(self, inject: tuple[str, ...]) -> bool:
        """Whether a tool declaring these injections is a memory tool of this run: it asked for a
        registered slot. A memory tool of an unregistered slot is not — its call is left unrun at
        the cap like any other, rather than answered with an error nobody reads."""
        return any(name in self for name in inject)

    def provide(self, inject: tuple[str, ...]) -> dict[str, dict]:
        """The dict forms of the slots a tool asked for, built for this one call."""
        return {m.name: v.to_dict() for m, v in self.slots if m.name in inject}

    def commit(self, provided: dict[str, dict], budget_tokens: int | None = None) -> tuple[Memories, str | None]:
        """Read the dicts a tool wrote to back into values: the one way a call becomes a state
        transition. A value that grew past `budget_tokens` is not committed — the values are
        immutable, so refusing is keeping the old one — and the refusal is returned as the text
        that answers the call. Shrinking is always allowed, or an over-budget memory (a session
        restored under a smaller window) could never be brought back under."""
        out = self
        for m, old in self.slots:
            if m.name not in provided:
                continue
            new = m.from_dict(provided[m.name])
            tokens = estimate_tokens(new.render())
            if budget_tokens is not None and tokens > budget_tokens and tokens > estimate_tokens(old.render()):
                return self, self.frame.over_budget.format(name=m.name, tokens=tokens, budget=budget_tokens)
            out = out.with_value(m.name, new)
        return out, None

    # -- the record -----------------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict]:
        """The values in their dict form, keyed by slot: what a finished Turn records."""
        return {m.name: v.to_dict() for m, v in self.slots}

    def restored(self, snapshots: Callable[[str], dict | None]) -> Memories:
        """The registered slots, each holding what `snapshots(name)` recorded for it when it
        recorded anything. A slot the file carries but the run did not register is not loaded."""
        out = self
        for m, _ in self.slots:
            saved = snapshots(m.name)
            if saved is not None:
                out = out.with_value(m.name, m.from_dict(saved))
        return out

    def salvage_sections(self) -> list[str]:
        """One `NAME:` section per memory that holds anything, for a salvaged turn's record."""
        return [f"{m.name.upper()}:\n{v.render()}" for m, v in self.slots if v.render()]

    # -- the request ----------------------------------------------------------------------------

    def system_prompt(self, base: str, max_tool_rounds: int) -> str:
        """`base` followed by the context mechanics and each memory's own prompt; `base` alone when
        no memory is registered — mechanics the model can do nothing about are noise."""
        if not self.slots:
            return base
        mechanics = self.frame.mechanics.format(
            max_tool_rounds=max_tool_rounds,
            tool_names=", ".join(self.tool_names()),
            sections=", ".join(f"<{n}>" for n in self.names()))
        return "\n\n".join([base, mechanics] + [m.prompt for m, _ in self.slots if m.prompt])

    def block(self, round: tuple[int, int] | None = None, budget_tokens: int | None = None,
              fold_near: bool = False) -> dict | None:
        """The one message that ends every request: a section per memory inside <memory>, framed by
        what the harness knows and the memories do not. `round` is (this round, the turn's cap),
        shown so the model sees its budget on every request. The round at the cap is the last whose
        calls run, and the reply after it keeps only its memory calls: each says so in a closing
        line, while the results the turn is about to lose can still be read. `fold_near` closes
        with the line that says a checkpoint is about to fold the earlier rounds. None when no
        memory is registered."""
        if not self.slots:
            return None
        header = f"Round {round[0]} of {round[1]} in this turn. " if round is not None else ""
        lines = [f"<memory>{header}{self.frame.block_head}"]
        for m, v in self.slots:
            body = v.render()
            used = f' used="{estimate_tokens(body)}/{budget_tokens} tokens"' if budget_tokens is not None else ""
            lines += [f"<{m.name}{used}>", body or "(empty)", f"</{m.name}>"]
        if round is not None and round[0] == round[1]:
            lines.append(self.frame.last_round)
        elif round is not None and round[0] > round[1]:
            lines.append(self.frame.cap_reached.format(tool_names=", ".join(self.tool_names())))
        elif fold_near:
            lines.append(self.frame.fold_near)
        lines.append("</memory>")
        return {"role": "user", "content": "\n".join(lines)}
