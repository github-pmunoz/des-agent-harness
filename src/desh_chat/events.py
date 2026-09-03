from dataclasses import dataclass, replace
from desh.engine import Event, Priority
from desh.render import Palette, c_out
from desh_chat.state import ChatState


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
        print(c_out(Palette.CHROME, "\nGoodbye!"))
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
        if user_input[0] == '/':
            return state, [Command(text=user_input[1:])]
        return state, [Echo(text=user_input)]


@dataclass(frozen=True)
class Echo(Event):
    """Debug echo event."""
    text: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, f"Echo: {self.text}"))
        return state, [MaybeRegenerate()]


@dataclass(frozen=True)
class Command(Event):
    """Event to process a command."""
    text: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        match self.text.lower():
            case 'exit' | 'quit':
                return state, [Exit()]
            case _:
                print(f"Unknown command: {self.text}")
                return state, [MaybeRegenerate()]
