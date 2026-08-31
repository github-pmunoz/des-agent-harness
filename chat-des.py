#!/usr/bin/env python3
# filename: chat-des.py
"""
Discrete-Event-Simulation (DES) engine for chatbot
"""
import argparse
import traceback
import itertools
import heapq
import sys
import reprlib

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

_r = reprlib.Repr()

# -----------------------
# Chat State
# -----------------------

@dataclass(frozen=True)
class ChatState:
    """Immutable chat state."""
    history: None # stub as None for now
    model: str
    temperature: float
    think: bool
    running: bool

# -----------------------
# Events
# -----------------------

# Event priority
PRIORITY = {
    'high':    0,
    'normal':  10,
    'low':     20,
}

class Event(ABC):
    """Abstract base class for events."""
    priority: int = PRIORITY['normal']

    @abstractmethod
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        raise NotImplementedError


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
        print("Goodbye!")
        return replace(state, running=False), []


@dataclass(frozen=True)
class PromptUser(Event):
    """Event to prompt user for input."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            user_input = input("User: ")
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
        print(f"Echo: {self.text}")
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


# -----------------------
# Simulation
# -----------------------

COLOR = {
    "debug": "\033[2m", # gray
    "reset": "\033[0m", # reset
    "chrome": "\033[33m", # yellow
    "debug-execute" : "\033[2m\033[36m", # dim cyan
    "debug-trigger" : "\033[2m\033[35m", # dim magenta
}

def c(channel, text):
    return f"{COLOR[channel]}{text}{COLOR["reset"]}" if sys.stdout.isatty() else text

def run(state: ChatState, seed: list[Event], debug: bool) -> ChatState:
    queue, seq = [], itertools.count()
    for e in seed:
        heapq.heappush(queue, (e.priority, next(seq), e))
    while queue:
        _, _, event = heapq.heappop(queue)
        try:
            state, new_events = event.execute(state)
        except KeyboardInterrupt: 
            print(c("chrome", "\nInterrupted by user --- triggering graceful exit..."))
            new_events = [Exit()]
        except Exception:
            traceback.print_exc()
            new_events = [MaybeRegenerate()]
        for ne in new_events:
            heapq.heappush(queue, (ne.priority, next(seq), ne))
        if debug:
            print(c("debug", f"\u21aa {_r.repr(event)} → [{', '.join(_r.repr(e) for e in new_events)}] depth={len(queue)}"), file=sys.stderr)
    return state

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Chat DES Engine")
    parser.add_argument("--model", default="llama3", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    state = ChatState(
        history=None,
        model=args.model,
        temperature=args.temperature,
        think=args.think,
        running=True,
    )
    seed = [
        PromptUser(),
    ]
    run(state, seed, args.debug)


if __name__ == "__main__":
    main()




