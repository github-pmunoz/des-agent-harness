from dataclasses import dataclass, replace
from desh.engine import Event
from desh.render import Palette, c_out
from desh.llama.client import Request, Seam, CodeFence, Terminal
from desh.llama.esc_watcher import ESCWatcher
from desh.llama.tokens import estimate_tokens
from desh_chat.state import ChatState, Turn
import sys

_TTY = sys.stdout.isatty()



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
        return state, [UserMessage(user_input)]


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


# ---------------------
# Turn logic
# ---------------------

@dataclass(frozen=True)
class StreamCompletion(Event):
    request: Request
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        watcher = ESCWatcher()
        term = Terminal(out=sys.stdout, colour=_TTY)
        print(c_out(Palette.CHROME_ASSISTANT, "Assistant: "), end="", flush=True)
        try:
            watcher.start()
            completion = state.inference.server.stream(self.request, Seam(CodeFence(term)), cancelled=lambda: watcher.interrupted)
            cancelled = completion.finish_reason == "cancelled"
            if cancelled:
                print(c_out(Palette.CHROME, "Response cancelled by user."))
        finally:
            watcher.stop()
        return state, [AppendTurn(user_msg=self.request.messages[-1]["content"], assistant_msg=completion.content, cancelled=cancelled)]

@dataclass(frozen=True)
class AppendTurn(Event):
    user_msg: str
    assistant_msg: str
    cancelled: bool
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        turn = Turn(self.user_msg, self.assistant_msg, cancelled=self.cancelled)
        if not turn.cancelled and state.history.window_tokens() + turn.tokens >= state.settings.compaction_threshold * state.settings.context:
            pass #TODO trigger compaction; cancelled turns cannot trigger a compaction
        return replace(state, history=state.history.append(turn)), [MaybeRegenerate()]

@dataclass(frozen=True)
class UserMessage(Event):
    message: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        user_message = {"role": "user", "content": self.message}
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        msg_tokens = estimate_tokens(self.message)
        used_tokens = msg_tokens + sys_prompt_tokens
        gen_budget = int(min(
            state.settings.max_turn_tokens,
            state.settings.context - used_tokens - state.history.window_tokens(),
            state.settings.turn_token_cap * state.settings.context))
        reserved = sys_prompt_tokens + msg_tokens + gen_budget
        if gen_budget <= 0:
            print(c_out(Palette.ERROR, "Request exceeds context window."))
            return state, [MaybeRegenerate()]
        return state, [StreamCompletion(request=Request(
                messages=[{"role": "system", "content": state.system_prompt}] + state.history.view(state.settings.context - reserved) + [user_message],
                model=state.settings.model,
                temperature=state.settings.temperature,
                max_tokens=gen_budget,
                think=state.settings.think,
                stream=True
                ))]
