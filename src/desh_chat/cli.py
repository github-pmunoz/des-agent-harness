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

from desh_chat.state import ChatState
from desh.llama.client import LlamaServer, Logger
from desh.render import Palette, c_out
from desh.engine import Engine
from desh_chat.events import LoadSession, PromptUser
from desh_chat.state import ChatHistory, Settings, InferenceEngine
from desh_chat.handlers import on_error, on_interrupt


def resolve_session_file(session: str, sessions_folder: str, run_id: str) -> str | None:
    """
    Turn the two session flags into one path (or None for no persistence).

      --session PATH            an explicit file to load-or-create.
      --sessions-folder DIR     a new file in DIR, named from run_id, so every run leaves a session.
      both                      ignores --sessions-folder, uses --session as-is.

    The folder is created if missing. Paths are ~-expanded. Return None when neither flag is set.
    """
    if sessions_folder and not session:
        # Create a new session file in the folder, named from the run_id.
        session = os.path.join(os.path.expanduser(sessions_folder), f"{run_id}.json")
        os.makedirs(os.path.expanduser(sessions_folder), exist_ok=True)
        return session
    if session:
        # Ignore the folder, use the session file as-is. Includes branches with and without a directory part.
        # The folder is created if missing.
        parent = os.path.dirname(os.path.expanduser(session))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        return os.path.expanduser(session)
    return None  # no session persistence


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
    ap.add_argument("-to",  "--timeout",        default=0, type=float)
    ap.add_argument("-s",   "--session",        default="", help="session file to load or create")
    ap.add_argument("-sf",  "--sessions-folder", default="", help="folder where a new session file is created per run")
    ap.add_argument("-d",   "--debug",          action="store_true", help="Enable debug output")
    args = ap.parse_args()

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    session_file = resolve_session_file(args.session, args.sessions_folder, run_id)

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
    Timeout:      {args.timeout}s
    Session:      {session_file or "-"}"""))
    print(c_out(Palette.CHROME, f"\n{"═"*50}"))

    # Setup logging
    if args.des_log:
        if(d := os.path.dirname(args.des_log)):
            os.makedirs(os.path.expanduser(d), exist_ok=True)
        des_log = open(os.path.expanduser(args.des_log), "a", encoding="utf-8")
    else:
        des_log = None

    if args.timeout == 0:
        args.timeout = args.context // 10 # assuming worst case one shot at 10tokens/sec

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
            port=args.port,
            models=client.models(),
            max_context=client.max_context()
        ),
        history=ChatHistory(),
        running=True,
        system_prompt=args.system_prompt,
        completions_log=Logger(args.completions_log) if args.completions_log else None,
        session_file=session_file,
    )
    log_header = {
        "model": args.model,
        "context": args.context,
        "temperature": args.temperature,
        "think": args.think,
        "max_turn_tokens": args.max_turn_tokens,
        "argv" : sys.argv[1:]
    }
    
    Engine[ChatState](
        des_log=des_log,
        debug=args.debug,
        on_error=on_error,
        on_interrupt=on_interrupt,
    ).run(state, seed=[LoadSession(), PromptUser()], run_id=run_id, log_header=log_header)


if __name__ == "__main__":
    main()
