from dataclasses import dataclass, replace
from desh.engine import Event, Priority
from desh.render import Palette, c_out
from desh_chat.state import ChatState
from typing import Any


@dataclass(frozen=True)
class MaybeRegenerate(Event):
    """Schedules PromptUser if running."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not state.running:
            return state, []
        return state, [PromptUser()]


@dataclass(frozen=True)
class Exit(Event):
    """Exit the simulation."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, "Goodbye!"))
        return replace(state, running=False), []


@dataclass(frozen=True)
class PromptUser(Event):
    """Event to prompt user for input."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            user_input = input(c_out(Palette.CHROME_USER, "You: "))
        except EOFError:
            return state, [Exit()]
        if not user_input:
            return state, [MaybeRegenerate()]
        if user_input[0] == "/":
            # command names are case-sensitive by design
            c = s = ""
            if " " in user_input:
                c, s = user_input[1:].split(" ", 1)
            else:
                c, s = user_input[1:], ""
            return state, [Command(command=c.strip(), args=s.strip())]
        return state, [MaybeRegenerate()]


@dataclass(frozen=True)
class Info(Event):
    """Display information in chrome"""
    text: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, self.text))
        return state, [MaybeRegenerate()]


@dataclass(frozen=True)
class Warn(Event):
    """Display information in chrome"""
    text: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.WARNING, self.text))
        return state, [MaybeRegenerate()]

# ---------------------
# Commands
# ---------------------

class CommandError(Exception):
    """User facing command problem; never a bug"""

@dataclass(frozen=True)
class Command(Event):
    """Event to process a command."""
    command: str
    args: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            return self._dispatch(state)
        except CommandError as e:
            return state, [Warn(str(e))]
        
    def _dispatch(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        match self.command:
            case "context":
                if not self.args:
                    return state, [Info(f"context: {state.settings.context}")]
                value = self._int(0, state.inference.max_context[state.settings.model])
                return state.change_setting("context", value), [Info(f"\u21aa context set to: {value}")]
            
            case "exit" | "quit":
                self._no_args()
                return state, [Exit()]

            case "max_turn_tokens":
                if not self.args:
                    return state, [Info(f"max_turn_tokens: {state.settings.max_turn_tokens}")]
                value = self._int(0, state.settings.max_turn_tokens)
                return state.change_setting("max_turn_tokens", value), [Info(f"\u21aa max_turn_tokens set to: {value}")]
            
            case "models":
                self._no_args()
                return state, [Info("\n".join(state.inference.models))]
            
            case "model":
                if not self.args:
                    return state, [Info(f"model: {state.settings.model}")]
                model = self._single_arg()
                if model not in state.inference.models:
                    raise CommandError(f"Model {model} not found.")
                return state.change_setting("model", model), [Info(f"\u21aa model set to: {model}")]
            
            case "temperature":
                if not self.args:
                    return state, [Info(f"temperature: {state.settings.temperature}")]
                value = self._float(lo=0.0, hi=2.0)
                return state.change_setting("temperature", value), [Info(f"\u21aa temperature set to: {value}")]
            
            case "think":
                self._no_args()
                return state.change_setting("think", True), [Info(f"\u21aa thinking mode enabled")]
                
            case "nothink":
                self._no_args()
                return state.change_setting("think", False), [Info(f"\u21aa thinking mode disabled")]
            
            case _:
                
                raise CommandError(f"Unknown command: /{self.command}")

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

    def _int(self, lo, hi) -> int:
        try:
            v = int(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not an int: {self.args}")
        if not lo <= v <= hi:
            raise CommandError(f"/{self.command}: must be in [{lo}, {hi}]")
        return v
