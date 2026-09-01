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
import time
import os
import json
import uuid


from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum
from collections.abc import Sequence
from typing import TextIO

from ChatHistory import ImmutableHistory, Turn
from LlamaClient import (Request, LlamaServer, Logger, Seam, CodeFence, Terminal)
from esc_watcher import ESCWatcher
from Tokens import estimate_tokens

_r = reprlib.Repr()
_r.maxother = 80
_r.maxstring = 40

# -----------------------
# Chat State
# -----------------------

@dataclass(frozen=True)
class ChatState:
    """Immutable chat state."""
    history: ImmutableHistory
    model: str
    temperature: float
    think: bool
    running: bool

    # Immutable handles --- not fields
    server: LlamaServer
    system_prompt: str
    context: int
    max_tokens: int
    log_path: str

# -----------------------
# Events
# -----------------------

class Priority(IntEnum):
    HIGH, NORMAL, LOW = 0, 10, 20

class Event(ABC):
    """Abstract base class for events."""
    priority: int = Priority.NORMAL

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


# -----------------------
# Simulation
# -----------------------

class Palette:
    DEBUG   = "\033[2m"     # dim   
    RESET   = "\033[0m"     # reset white
    CHROME  = "\033[33m"    # yellow
    ERROR   = "\033[31m"    # red
    CHROME_USER = "\033[1m\033[32m"  # bright green

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, channel: str, text: str) -> str:
        return f"{channel}{text}{Palette.RESET}" if self.enabled else text
c_out = Palette(enabled=sys.stdout.isatty())
c_err = Palette(enabled=sys.stderr.isatty())

def pretty_log(record: dict) -> str:
    return f"\u21aa {record['payload']} → [{', '.join(record['emitted'])}] " f"depth={record['depth']} {record['dur_ms']}ms" + ("" if record['outcome'] == "ok" else f" ⚠ {record['outcome']}")

def run(state: ChatState, seed: Sequence[Event], debug: bool, run_id : str, des_log: TextIO | None) -> ChatState:
    queue, seq = [], itertools.count()
    n = 0
    t0 = t1 = 0
    for e in seed:
        heapq.heappush(queue, (e.priority, next(seq), e))
    while queue:
        _, _, event = heapq.heappop(queue)
        try:
            t0 = time.time()
            state, new_events = event.execute(state)
            outcome = "ok"
        except KeyboardInterrupt: 
            print(c_out(Palette.CHROME, "\n\nInterrupted by user. Triggering graceful exit..."))
            new_events = [Exit()]
            outcome = "interrupted"
        except Exception:
            traceback.print_exc()
            new_events = [MaybeRegenerate()]
            outcome = "recovered"
        finally:
            t1 = time.time()

        for ne in new_events:
            heapq.heappush(queue, (ne.priority, next(seq), ne))
        log_record = {
            "run": run_id,
            "ts": time.time(),
            "n" : n,
            "depth" : len(queue),
            "event" : type(event).__name__,
            "payload" : _r.repr(event),
            "emitted" : [type(e).__name__ for e in new_events],
            "dur_ms"  : round((t1 - t0) * 1e3, 1),
            "outcome" : outcome
        }
        if debug:
            print(c_err(Palette.DEBUG, pretty_log(log_record)), file=sys.stderr)
        if des_log:
            des_log.write(json.dumps(log_record) + "\n")
            des_log.flush()
        n += 1


    return state

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Simple chatbot using LlamaClient")
    ap.add_argument("-p",   "--port", type=int, default=8012)
    ap.add_argument("-m",   "--model", default="Qwen3.8-27B-UD-Q4_K_M", help="model id (router mode)")
    ap.add_argument("-t",   "--temperature", type=float, default=0.3)
    ap.add_argument("-c",   "--context", type=int, default=16384, help="context window size")
    ap.add_argument("-mt",  "--max_tokens", type=int, default=8192, help="max tokens per turn")
    ap.add_argument("-sp",  "--system-prompt", default="You are a helpful assistant. Reply concisely.")
    ap.add_argument("-th",  "--think", action="store_true", help="enable thinking")
    ap.add_argument("-cl",  "--completions-log", default="~/pablo/llama-server.jsonl", help="JSONL telemetry file ('' to disable)")
    ap.add_argument("-dl",  "--des-log", default="~/pablo/chat-des.jsonl", help="DES engine log")
    ap.add_argument("-to",  "--timeout", type=float, default=30)
    ap.add_argument("-d",   "--debug", action="store_true", help="Enable debug output")
    args = ap.parse_args()

    print(c_out(Palette.CHROME, f"\n{"═"*50}"))
    print(c_out(Palette.CHROME, f""" DES Chat v0.1
    Server:       http://127.0.0.1:{args.port}
    Model:        {args.model}
    Temperature:  {args.temperature}
    Think mode:   {"enabled" if args.think else "disabled"}
    Context:      {args.context}
    Max tokens:   {args.max_tokens}
    Compl log:    {args.completions_log}
    DES log:      {args.des_log}
    Debug:        {"enabled" if args.debug else "disabled"}
    Timeout:      {args.timeout}s"""))
    print(c_out(Palette.CHROME, f"\n{"═"*50}"))


    # Setup logging
    if args.des_log:
        os.makedirs(os.path.dirname(os.path.expanduser(args.des_log)), exist_ok=True)
        des_log = open(os.path.expanduser(args.des_log), "w")
    else:
        des_log = None


    state = ChatState(
        history=ImmutableHistory(max_tokens=args.context - estimate_tokens(args.system_prompt)),
        model=args.model,
        temperature=args.temperature,
        think=args.think,
        running=True,
        server=LlamaServer(f"http://127.0.0.1:{args.port}",timeout=args.timeout),
        system_prompt=args.system_prompt,
        context=args.context,
        max_tokens=args.max_tokens,
        log_path=args.completions_log
    )

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    des_log.write(json.dumps({
        "run" : run_id,
        "ts"  : time.time(),
        "event" : "RUN_START",
        "model" : state.model,
        "context" : state.context,
        "temperature" : state.temperature,
        "think" : state.think,
        "argv" : sys.argv[1:]
        }) + "\n")
    des_log.flush()

    seed = [
        PromptUser(),
    ]
    run(state, seed, args.debug, run_id, des_log)


if __name__ == "__main__":
    main()




