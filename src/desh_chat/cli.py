#!/usr/bin/env python3
# filename: chat-des.py
"""
Discrete-Event-Simulation (DES) engine for chatbot
"""
import argparse
import sys
import time
import os
import uuid
import traceback

from desh_chat.state import ChatState
from desh.llama.client import LlamaServer
from desh.llama.tokens import estimate_tokens
from desh.render import Palette, c_out
from desh.engine import Engine, Event
from desh_chat.events import Exit, MaybeRegenerate, PromptUser
from desh_chat.state import ChatHistory, Settings, InferenceEngine


def main():
    ap = argparse.ArgumentParser(description="Simple chatbot using LlamaClient")
    ap.add_argument("-p",   "--port",           type=int, default=8012)
    ap.add_argument("-m",   "--model",          default="Qwen3.8-27B-UD-Q4_K_M", help="model id (router mode)")
    ap.add_argument("-t",   "--temperature",    type=float, default=0.3)
    ap.add_argument("-c",   "--context",        type=int, default=16384, help="context window size")
    ap.add_argument("-mt",  "--max-turn-tokens",type=int, default=8192, help="max tokens per turn")
    ap.add_argument("-sp",  "--system-prompt",  default="You are a helpful assistant. Reply concisely.")
    ap.add_argument("-th",  "--think",          action="store_true", help="enable thinking")
    ap.add_argument("-cl",  "--completions-log", default="", help="JSONL telemetry file")
    ap.add_argument("-dl",  "--des-log",        default="", help="DES engine log")
    ap.add_argument("-to",  "--timeout",        type=float, default=30)
    ap.add_argument("-d",   "--debug",          action="store_true", help="Enable debug output")
    args = ap.parse_args()

    print(c_out(Palette.CHROME, f"\n{"═"*50}"))
    print(c_out(Palette.CHROME, f""" DES Chat v0.1
    Server:       http://127.0.0.1:{args.port}
    Model:        {args.model}
    Temperature:  {args.temperature}
    Think mode:   {"enabled" if args.think else "disabled"}
    Context:      {args.context}
    Turn tokens:  {args.max_turn_tokens}
    Compl log:    {args.completions_log}
    DES log:      {args.des_log}
    Debug:        {"enabled" if args.debug else "disabled"}
    Timeout:      {args.timeout}s"""))
    print(c_out(Palette.CHROME, f"\n{"═"*50}"))

    # Setup logging
    if args.des_log:
        if(d := os.path.dirname(args.des_log)):
            os.makedirs(os.path.expanduser(d), exist_ok=True)
        des_log = open(os.path.expanduser(args.des_log), "a", encoding="utf-8")
    else:
        des_log = None

    client = LlamaServer(f"http://127.0.0.1:{args.port}",timeout=args.timeout)
    state = ChatState(
        settings=Settings(
            model=args.model,
            temperature=args.temperature,
            think=args.think,
            context=args.context,
            max_turn_tokens=args.max_turn_tokens
        ),
        inference=InferenceEngine(
            server=client,
            models=client.models(),
            max_context=client.max_context()
        ),
        history=ChatHistory(),
        running=True,
        system_prompt=args.system_prompt,
        completions_log=args.completions_log
    )
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    log_header = {
        "model": args.model,
        "context": args.context,
        "temperature": args.temperature,
        "think": args.think,
        "max_turn_tokens": args.max_turn_tokens,
        "argv" : sys.argv[1:]
    }

    def on_error(ev, ex, s) -> list[Event]:
        traceback.print_exc()
        return [MaybeRegenerate()]
    
    Engine[ChatState](
        des_log=des_log,
        debug=args.debug,
        on_error=on_error,
        on_interrupt=lambda s: [Exit()]
    ).run(state, seed=[PromptUser()], run_id=run_id, log_header=log_header)


if __name__ == "__main__":
    main()
