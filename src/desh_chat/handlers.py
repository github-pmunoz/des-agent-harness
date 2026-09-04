import traceback
from desh.engine import Event
from desh_chat.state import ChatState
from desh_chat.events import MaybeRegenerate, Exit

def on_error(ev: Event, ex: Exception, s: ChatState) -> list[Event]:
    traceback.print_exc()
    return [MaybeRegenerate()]

def on_interrupt(s: ChatState) -> list[Event]:
    return [Exit()]