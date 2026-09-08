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
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.render import Palette, c_out
from desh.engine import Engine
from desh.tools import ToolRegistry
from desh_chat.events import PromptUser
from desh_chat.session import LoadSession
from desh_chat.state import ChatHistory, Settings, InferenceEngine
from desh_chat.handlers import on_error, on_interrupt
from desh_chat.toolset import default_registry
from desh_chat.coding import CODING_SYSTEM_PROMPT, coding_registry
from desh_chat.delegate import with_delegate


def build_tools(basic: bool, coding: bool, workspace: str) -> ToolRegistry:
    """The toolsets are additive: each flag contributes its tools, none of them means no tools.
    delegate is added afterwards by the caller, once the inputs its subagents inherit exist."""
    tools = ToolRegistry()
    if basic:
        tools = ToolRegistry(tools.tools + default_registry().tools)
    if coding:
        tools = ToolRegistry(tools.tools + coding_registry(workspace).tools)
    return tools


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
    ap.add_argument("-mtr",  "--max-tool-rounds",type=int, default=10, help="max tool rounds per turn")
    ap.add_argument("-sp",  "--system-prompt",  default=None, help="default: a plain assistant prompt, or the coding-agent prompt with --coding")
    ap.add_argument("-th",  "--think",          action="store_true", help="enable thinking")
    ap.add_argument("-cl",  "--completions-log", default="", help="JSONL telemetry file")
    ap.add_argument("-dl",  "--des-log",        default="", help="DES engine log")
    ap.add_argument("-to",  "--timeout",        default=0, type=float)
    ap.add_argument("-s",   "--session",        default="", help="session file to load or create")
    ap.add_argument("-sf",  "--sessions-folder", default="", help="folder where a new session file is created per run")
    ap.add_argument("-d",   "--debug",          action="store_true", help="Enable debug output")
    # toolsets are additive flags: any combination, none means the model is offered no tools
    ap.add_argument("--basic",    action="store_true", help="offer the basic tools (current time)")
    ap.add_argument("--coding",   action="store_true", help="offer the coding tools: Read, Write, Edit, Bash")
    ap.add_argument("--delegate", action="store_true", help="offer delegate: subagents with the same tools and settings")
    ap.add_argument("-w",   "--workspace",      default=".", help="project root for the coding toolset")
    args = ap.parse_args()

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    session_file = resolve_session_file(args.session, args.sessions_folder, run_id)
    tools = build_tools(args.basic, args.coding, args.workspace)
    system_prompt = args.system_prompt if args.system_prompt is not None else (
        CODING_SYSTEM_PROMPT if args.coding else "You are a helpful assistant. Reply concisely.")

    print(c_out(Palette.CHROME, f"\n{"═"*50}"))
    print(c_out(Palette.CHROME, f""" DES Chat v0.1
    Server:       http://127.0.0.1:{args.port}
    Model:        {args.model}
    Temperature:  {args.temperature}
    Think mode:   {"enabled" if args.think else "disabled"}
    Context:      {args.context}
    Turn tokens:  {args.max_turn_tokens}
    Tool rounds:  {args.max_tool_rounds}
    Compl log:    {args.completions_log}
    DES log:      {args.des_log}
    Debug:        {"enabled" if args.debug else "disabled"}
    Timeout:      {args.timeout}s
    Session:      {session_file or "-"}
    Tools:        {", ".join(t.name + (" (asks)" if t.confirm else "") for t in tools.tools) or "none"}{" + delegate (asks)" if args.delegate else ""}
    Workspace:    {os.path.realpath(args.workspace) if args.coding else "-"}
    System:       {system_prompt[:40]}{f"...[{len(system_prompt) - 40} more chars]" if len(system_prompt) > 40 else ""}"""))
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
    settings = Settings(
        model=args.model,
        temperature=args.temperature,
        think=args.think,
        context=args.context,
        max_turn_tokens=args.max_turn_tokens,
        max_tool_rounds=args.max_tool_rounds
    )
    inference = InferenceEngine(
        server=client,
        port=args.port,
        models=client.models(),
        max_context=client.max_context()
    )
    completions_log = Logger(args.completions_log) if args.completions_log else None
    if args.delegate:
        # subagents inherit what the parent has at this point: settings, prompt, tools, logs, session
        tools = with_delegate(tools, inference=inference, settings=settings, system_prompt=system_prompt,
                              session_file=session_file, completions_log=completions_log, des_log=des_log, debug=args.debug)
    state = ChatState(
        settings=settings,
        inference=inference,
        history=ChatHistory(),
        running=True,
        system_prompt=system_prompt,
        completions_log=completions_log,
        session_file=session_file,
        tools=tools,
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
