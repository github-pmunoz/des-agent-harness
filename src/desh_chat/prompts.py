from __future__ import annotations
import os
import string
from dataclasses import dataclass, field, replace

from desh.tools import ToolRegistry
from desh_chat import memory as mem
from desh_chat.delegate import CAP_CONTINUE_MSG, DELEGATE_SYSTEM_PROMPT, LENGTH_CONTINUE_MSG, REPEAT_CONTINUE_MSG
from desh_chat.memory import Memory, MemoryFrame
from desh_chat.state import CHECKPOINT_CLOSE, CHECKPOINT_PROMPT, COMPACTION_PROMPT, RETRY_NUDGE, SUMMARY_CLOSE

# Every instruction the model reads, by key, so a run can be built with other texts without a
# change to the harness (--config's "prompts" object, --prompt KEY=TEXT). This module is only the
# catalogue: nothing looks a prompt up at run time. An override is applied once, at construction,
# to the value that already owns the text — Settings, the MemoryFrame, a Memory, the Delegate, a
# Tool's schema — and from there on the run cannot tell it from a default.
#
#   key                                  lands in
#   delegate.system                      Delegate.system_prompt
#   cap_continue                         ChatState.auto_prompt (with --cont) and Delegate.cap_continue
#   length_continue                      ChatState.length_prompt (with --cont) and Delegate.length_continue
#   repeat_continue                      ChatState.repeat_prompt (with --cont) and Delegate.repeat_continue
#   compaction, checkpoint               Settings.compaction_prompt, Settings.checkpoint_prompt
#   compaction.close, checkpoint.close,  Settings.summary_close, Settings.checkpoint_close,
#   summary.retry                        Settings.retry_nudge
#   memory.mechanics, memory.block_head, MemoryFrame
#   memory.last_round, memory.cap_reached,
#   memory.fold_near, memory.over_budget
#   memory.<name>.prompt                 Memory.prompt of a selected memory
#   tool.<Name>.description              the tool's schema, in the main and the subagent registry
#   tool.<Name>.param.<arg>
#
# The run's own system prompt is not here: it is --system-prompt, a setting like any other.
#
# Not overridable, on purpose: the strings the harness or an eval grader parses — the re-read
# notice, the cut and spill markers, the checkpoint prefix, the repeat-stop note, the gate's
# declined/skipped texts, the notes a parent reads for a subagent's stop.

# key -> (default, the placeholders the template may and must carry)
STATIC: dict[str, tuple[str, tuple[str, ...]]] = {
    "delegate.system": (DELEGATE_SYSTEM_PROMPT, ()),
    "cap_continue": (CAP_CONTINUE_MSG, ()),
    "length_continue": (LENGTH_CONTINUE_MSG, ()),
    "repeat_continue": (REPEAT_CONTINUE_MSG, ()),
    "compaction": (COMPACTION_PROMPT, ()),
    "checkpoint": (CHECKPOINT_PROMPT, ()),
    "compaction.close": (SUMMARY_CLOSE, ()),
    "checkpoint.close": (CHECKPOINT_CLOSE, ()),
    "summary.retry": (RETRY_NUDGE, ()),
    "memory.mechanics": (mem.CONTEXT_MECHANICS_PROMPT, ("max_tool_rounds", "tool_names", "sections")),
    "memory.block_head": (mem.BLOCK_HEAD, ()),
    "memory.last_round": (mem.LAST_ROUND_LINE, ()),
    "memory.cap_reached": (mem.CAP_REACHED_LINE, ("tool_names",)),
    "memory.fold_near": (mem.FOLD_NEAR_LINE, ()),
    "memory.over_budget": (mem.OVER_BUDGET_TEXT, ("name", "tokens", "budget")),
}
# the templates: the texts that are formatted, so braces in them mean something
TEMPLATES = tuple(k for k, (_, fields) in STATIC.items() if fields)


class PromptError(ValueError):
    """A prompt override that cannot be applied: an unknown key, a template with the wrong
    placeholders, a tool or a memory the run does not have."""


def from_file(value: str, base_dir: str) -> str:
    """A value spelled `@path` is the text of that file, relative to `base_dir`: a long prompt
    stays out of the JSON and off the command line. Anything else is the value itself."""
    if not value.startswith("@"):
        return value
    path = os.path.join(base_dir, os.path.expanduser(value[1:]))
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().rstrip("\n")
    except OSError as e:
        raise PromptError(f"cannot read {value}: {e}") from e


def placeholders(text: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
    except ValueError as e:
        raise PromptError(f"malformed template: {e}") from e


def tool_key(key: str) -> tuple[str, str | None] | None:
    """(tool name, parameter or None) of a `tool.` key; None when the key is not one. A tool name
    may hold spaces and dots, so the key is read from its ends."""
    if not key.startswith("tool."):
        return None
    rest = key[len("tool."):]
    if rest.endswith(".description"):
        return rest[:-len(".description")], None
    name, sep, param = rest.rpartition(".param.")
    return (name, param) if sep and name and param else None


def memory_key(key: str) -> str | None:
    """The memory name of a `memory.<name>.prompt` key; None when the key is not one."""
    parts = key.split(".")
    return parts[1] if len(parts) == 3 and parts[0] == "memory" and parts[2] == "prompt" else None


@dataclass(frozen=True)
class Prompts:
    """The overrides of one run, checked: every key is one the catalogue knows the shape of, and
    every template carries exactly its placeholders. Whether a `tool.` or `memory.<name>.` key
    names something this run has is known only where it is applied, which is where it is checked."""
    texts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def resolve(cls, *layers: tuple[dict[str, str], str]) -> Prompts:
        """Merge (overrides, base_dir) layers, later ones winning, reading `@file` values against
        each layer's own directory."""
        texts: dict[str, str] = {}
        for overrides, base_dir in layers:
            for key, value in overrides.items():
                if not isinstance(value, str):
                    raise PromptError(f"prompt {key!r} must be a string, got {type(value).__name__}")
                if key not in STATIC and tool_key(key) is None and memory_key(key) is None:
                    raise PromptError(f"unknown prompt key {key!r}; one of {', '.join(STATIC)}, "
                                      f"memory.<name>.prompt, tool.<Name>.description, tool.<Name>.param.<arg>")
                text = from_file(value, base_dir)
                if key in STATIC:
                    expected, found = set(STATIC[key][1]), placeholders(text) if key in TEMPLATES else set()
                    if key in TEMPLATES and found != expected:
                        raise PromptError(f"prompt {key!r} must carry exactly the placeholders "
                                          f"{sorted(expected)}, found {sorted(found)}")
                texts[key] = text
        return cls(texts)

    def get(self, key: str) -> str:
        return self.texts.get(key, STATIC[key][0])

    def settings(self) -> dict[str, str]:
        """The Settings fields the compaction texts land in."""
        return {"compaction_prompt": self.get("compaction"), "checkpoint_prompt": self.get("checkpoint"),
                "summary_close": self.get("compaction.close"), "checkpoint_close": self.get("checkpoint.close"),
                "retry_nudge": self.get("summary.retry")}

    def frame(self) -> MemoryFrame:
        return MemoryFrame(mechanics=self.get("memory.mechanics"), block_head=self.get("memory.block_head"),
                           last_round=self.get("memory.last_round"), cap_reached=self.get("memory.cap_reached"),
                           fold_near=self.get("memory.fold_near"), over_budget=self.get("memory.over_budget"))

    def memories(self, selected: tuple[Memory, ...]) -> tuple[Memory, ...]:
        """The selected memories, each with its own prompt overridden when the run says so. A key
        for a memory the run did not select is an error: the text would reach nobody."""
        names = {m.name for m in selected}
        for key in self.texts:
            name = memory_key(key)
            if name is not None and name not in names:
                raise PromptError(f"prompt {key!r}: memory {name!r} is not selected in this run")
        return tuple(replace(m, prompt=self.texts.get(f"memory.{m.name}.prompt", m.prompt)) for m in selected)

    def describe(self, tools: ToolRegistry) -> tuple[ToolRegistry, set[str]]:
        """The registry with the `tool.` overrides applied to the tools it holds, and the keys that
        were. A run has two registries (the main agent's and the subagents'), and a key is good
        when either holds its tool: check_tools says so once both are described."""
        applied: set[str] = set()
        for key, text in self.texts.items():
            parsed = tool_key(key)
            if parsed is None or parsed[0] not in tools:
                continue
            name, param = parsed
            try:
                tools = tools.described(name, params={param: text}) if param else tools.described(name, description=text)
            except KeyError as e:
                raise PromptError(f"prompt {key!r}: {e.args[0]}") from e
            applied.add(key)
        return tools, applied

    def check_tools(self, applied: set[str]) -> None:
        """Every `tool.` key must have reached a tool: a text for a tool the run does not offer
        would reach nobody."""
        for key in self.texts:
            parsed = tool_key(key)
            if parsed is not None and key not in applied:
                raise PromptError(f"prompt {key!r}: tool {parsed[0]!r} is not offered in this run")

    def effective(self, selected: tuple[Memory, ...], *registries: ToolRegistry) -> dict[str, str]:
        """Every text of the run by key, defaults included: the template a new arm starts from and
        the record of what a run was built with. `selected` and `registries` are the run's, with
        the overrides already applied."""
        out = {key: self.get(key) for key in STATIC}
        out.update({f"memory.{m.name}.prompt": m.prompt for m in selected})
        for tools in registries:
            for tool in tools.tools:
                out.setdefault(f"tool.{tool.name}.description", tool.description)
                for param, schema in tool.parameters.get("properties", {}).items():
                    if "description" in schema:
                        out.setdefault(f"tool.{tool.name}.param.{param}", schema["description"])
        return out
