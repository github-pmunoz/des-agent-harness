import traceback
from dataclasses import replace
from desh.engine import Event
from desh_chat.state import ChatState
from desh_chat.display import Info
from desh_chat.events import MaybeRegenerate, Exit, TurnEnd


class AutoOff(Event):
    """Turns auto mode off. The engine's on_interrupt only returns events and keeps the current
    state, so the flip happens here, when the event runs."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        return replace(state, settings=replace(state.settings, auto=False)), []


def on_error(ev: Event, ex: Exception, s: ChatState) -> list[Event]:
    traceback.print_exc()
    # A mid-turn error closes the turn as cancelled first, so the loop head starts from a clean
    # state (TurnEnd already emits the MaybeRegenerate that returns to the prompt).
    if s.pending is not None and s.pending.user is not None:
        return [TurnEnd(assistant="", cancelled=True, stop="error")]
    return [MaybeRegenerate()]

def on_interrupt(s: ChatState) -> list[Event]:
    if s.settings.auto:
        # Ctrl+C in auto mode does not exit: it turns auto mode off and goes back to the prompt.
        # A turn in progress is closed as cancelled first, so the loop head starts from a clean
        # state (TurnEnd already emits the MaybeRegenerate that returns to the prompt).
        if s.pending is not None and s.pending.user is not None:
            return [AutoOff(), Info("auto mode off"), TurnEnd(assistant="", cancelled=True, stop="interrupt")]
        return [AutoOff(), Info("auto mode off"), MaybeRegenerate()]
    return [Info("~ Interrupted"), Exit()]
