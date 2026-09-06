"""
The /command surface: one registry (COMMANDS) that completion, dispatch and any help listing all
read from, and the Command event that runs a row of it.

Commands sit inside the chat loop — every handler resolves to MaybeRegenerate, Exit or a turn
event — so this module imports events.py at the top. events.py does NOT import this module at the
top: PromptUser and the completer import Command / COMMANDS at call time. That one late edge is
what keeps the loop's cycle out of the import graph.
"""
from dataclasses import dataclass
from typing import Callable

from desh.engine import Event
from desh_chat.state import ChatState
from desh_chat.display import DisplayHistory, Info, Warn
from desh_chat.events import CompactHistory, Exit, MaybeRegenerate


class CommandError(Exception):
    """User facing command problem; never a bug"""


CommandHandler = Callable[["Command", ChatState], tuple[ChatState, list[Event]]]


@dataclass(frozen=True)
class CommandSpec:
    """One registry row: what a command does, and the handler that does it.

    `aliases` resolve on dispatch only; completion and any help listing show
    the canonical name alone.
    """
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Command(Event):
    """Event to process a command."""
    command: str
    args: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            return self._dispatch(state)
        except CommandError as e:
            return state, [Warn(str(e)), MaybeRegenerate()]

    def _dispatch(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        # command names are case-sensitive by design (see PromptUser)
        spec = _COMMAND_INDEX.get(self.command)
        if spec is None:
            raise CommandError(f"Unknown command: /{self.command}")
        return spec.handler(self, state)

    # --- handlers: one per registry row, in COMMANDS order ---

    def _cmd_compact(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Info(f"compacting conversation history..."), CompactHistory()]

    def _cmd_context(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"context: {state.settings.context}"), MaybeRegenerate()]
        value = self._int(0, state.inference.max_context[state.settings.model])
        return state.change_setting("context", value), [Info(f"↪ context set to: {value}"), MaybeRegenerate()]

    def _cmd_exit(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Exit()]

    def _cmd_history(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [DisplayHistory(), MaybeRegenerate()]

    def _cmd_max_turn_tokens(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"max_turn_tokens: {state.settings.max_turn_tokens}"), MaybeRegenerate()]
        value = self._int(0, state.settings.context)
        return state.change_setting("max_turn_tokens", value), [Info(f"↪ max_turn_tokens set to: {value}"), MaybeRegenerate()]

    def _cmd_max_tool_rounds(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"max_tool_rounds: {state.settings.max_tool_rounds}"), MaybeRegenerate()]
        value = self._int(lo=1)
        return state.change_setting("max_tool_rounds", value), [Info(f"↪ max_tool_rounds set to: {value}"), MaybeRegenerate()]

    def _cmd_models(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Info("\n".join(state.inference.models)), MaybeRegenerate()]

    def _cmd_model(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"model: {state.settings.model}"), MaybeRegenerate()]
        name = self._single_arg()
        if name not in state.inference.models:
            raise CommandError(f"Model {name} not found.")
        return state.change_setting("model", name), [Info(f"↪ model set to: {name}"), MaybeRegenerate()]

    def _cmd_temperature(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"temperature: {state.settings.temperature}"), MaybeRegenerate()]
        value = self._float(lo=0.0, hi=2.0)
        return state.change_setting("temperature", value), [Info(f"↪ temperature set to: {value}"), MaybeRegenerate()]

    def _cmd_think(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state.change_setting("think", True), [Info(f"↪ thinking mode enabled"), MaybeRegenerate()]

    def _cmd_nothink(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state.change_setting("think", False), [Info(f"↪ thinking mode disabled"), MaybeRegenerate()]

    # --- argument parsing ---

    def _no_args(self):
        if self.args:
            raise CommandError(f"/{self.command} takes no argument: {self.args}")

    def _single_arg(self) -> str:
        if not self.args:
            raise CommandError(f"/{self.command} needs an argument")
        if " " in self.args:
            raise CommandError(f"/{self.command} needs a single argument, got {len(self.args.split(" "))}")
        return self.args

    def _float(self, lo, hi) -> float:
        try:
            v = float(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not a float: {self.args}")
        if not lo <= v <= hi:
            raise CommandError(f"/{self.command}: must be in [{lo}, {hi}]")
        return v

    def _int(self, lo=None, hi=None) -> int:
        try:
            v = int(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not an int: {self.args}")
        if (lo is not None and not lo <= v):
            raise CommandError(f"/{self.command}: must be at least {lo}")
        if (hi is not None and not v <= hi):
            raise CommandError(f"/{self.command}: must be at most {hi}")
        return v


# Single source of truth for the command surface: completion, dispatch and
# any help listing all read from here. Keys are canonical names without the
# leading slash. Must follow the Command class so the handlers resolve.
COMMANDS: dict[str, CommandSpec] = {
    "compact":         CommandSpec("compact the conversation history",      Command._cmd_compact),
    "context":         CommandSpec("set the context window size",           Command._cmd_context),
    "exit":            CommandSpec("exit the chat",                         Command._cmd_exit, aliases=("quit",)),
    "history":         CommandSpec("show the conversation history",         Command._cmd_history),
    "max_turn_tokens": CommandSpec("set the max number of tokens per turn", Command._cmd_max_turn_tokens),
    "max_tool_rounds": CommandSpec("set the max number of tool rounds",     Command._cmd_max_tool_rounds),
    "models":          CommandSpec("list available models",                 Command._cmd_models),
    "model":           CommandSpec("set the model to use",                  Command._cmd_model),
    "temperature":     CommandSpec("set the temperature",                   Command._cmd_temperature),
    "think":           CommandSpec("enable thinking",                       Command._cmd_think),
    "nothink":         CommandSpec("disable thinking",                      Command._cmd_nothink),
}

# Dispatch index: canonical names plus aliases, all pointing at the same spec.
_COMMAND_INDEX: dict[str, CommandSpec] = {
    name: spec
    for canonical, spec in COMMANDS.items()
    for name in (canonical, *spec.aliases)
}
