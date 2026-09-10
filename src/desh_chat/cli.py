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
from typing import TextIO

from desh_chat.state import ChatState
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.render import Palette, c_out
from desh.engine import Engine
from desh.tools import ToolRegistry
from desh_chat.events import PromptUser, Continue
from desh_chat.session import LoadSession
from desh_chat.state import ChatHistory, Settings, InferenceEngine
from desh_chat.handlers import on_error, on_interrupt
from desh_chat.toolset import current_time, ToolRegistry
from desh_chat.coding import Workspace, edit_preview
from desh_chat.delegate import Delegate, CAP_CONTINUE_MSG


def build_tools(args: argparse.Namespace, inference: InferenceEngine, settings: Settings, *,
                session_file: str | None = None, completions_log: Logger | None = None,
                des_log: TextIO | None = None) -> ToolRegistry:
    """The toolsets are additive: each flag contributes its tools, none of them means no tools.
    The delegate tool is built from the RESOLVED session path, completions Logger and DES log file,
    never from the raw flags: a subagent writes to the same log objects the parent's engine does."""
    ws = Workspace(args.workspace)
    tools = ToolRegistry(debug=args.debug)
    if args.read:
        tools = tools.add(ws.read, name="Read", confirm=False)
    if args.write:
        tools = tools.add(ws.write, name="Write")
    if args.edit:
        tools = tools.add(ws.edit, name="Edit", preview=edit_preview)
    if args.bash:
        tools = tools.add(ws.bash, name="Bash", identity=("command",))
    if args.current_time:
        tools = tools.add(current_time, name="Current time")
    if args.delegate:
        delegate_tools = ToolRegistry(debug=args.debug)
        delegate_tools = (delegate_tools.add(ws.read, name="Read", confirm=False)
                          .add(ws.write, name="Write")
                          .add(ws.edit, name="Edit", preview=edit_preview)
                          .add(ws.bash, name="Bash", identity=("command",)))
        delegate = Delegate(root=ws.root, inference=inference, settings=settings, tools=delegate_tools,
                            session_file=session_file, completions_log=completions_log, des_log=des_log, debug=args.debug)
        # the parent's CURRENT settings travel with every call; the child derives its own from them
        tools = tools.add(delegate.delegate, name="delegate", inject=("settings",))
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
    ap.add_argument("-a",   "--auto",    action="store_true", help="enable auto mode")
    ap.add_argument("-co",   "--cont",    action="store_true", help="enable auto-continue prompt on tool round cap of orchestrator")
    ap.add_argument("-p",   "--port",           type=int, default=8012)
    ap.add_argument("-m",   "--model",          default="Qwen3.8-27B-UD-Q4_K_M", help="model id (router mode)")
    ap.add_argument("-t",   "--temperature",    type=float, default=0.3)
    ap.add_argument("-c",   "--context",        type=int, default=16384, help="context window size")
    ap.add_argument("-mt",  "--max-turn-tokens",type=int, default=8192, help="max tokens per turn")
    ap.add_argument("-mtr",  "--max-tool-rounds",type=int, default=10, help="max tool rounds per turn")
    ap.add_argument("-sp",  "--system-prompt",  default="You are a helpful assistant. Reply concisely.", help="default: a plain assistant prompt, or the coding-agent prompt with --coding")
    ap.add_argument("-th",  "--think",          action="store_true", help="enable thinking")
    ap.add_argument("-cl",  "--completions-log", default="", help="JSONL telemetry file")
    ap.add_argument("-dl",  "--des-log",        default="", help="DES engine log")
    ap.add_argument("-to",  "--timeout",        default=0, type=float)
    ap.add_argument("-s",   "--session",        default="", help="session file to load or create")
    ap.add_argument("-sf",  "--sessions-folder", default="", help="folder where a new session file is created per run")
    ap.add_argument("-d",   "--debug",          action="store_true", help="Enable debug output")
    ap.add_argument("-w",   "--workspace",      default=".", help="project root for the coding toolset")
    # toolsets are additive flags: any combination, none means the model is offered no tools
    ap.add_argument("--read",      action="store_true", help="offer the Read tool")
    ap.add_argument("--write",     action="store_true", help="offer the Write tool")
    ap.add_argument("--edit",      action="store_true", help="offer the Edit tool")
    ap.add_argument("--bash",      action="store_true", help="offer the Bash tool")
    ap.add_argument("--delegate", action="store_true", help="offer delegate: subagents with the same tools and settings")
    ap.add_argument("--current_time", action="store_true", help="offer the current time")
    args = ap.parse_args()

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    session_file = resolve_session_file(args.session, args.sessions_folder, run_id)
    system_prompt = args.system_prompt if args.system_prompt else "You are a helpful assistant. Reply concisely."



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
        max_tool_rounds=args.max_tool_rounds,
        auto=args.auto,
    )
    inference = InferenceEngine(
        server=client,
        port=args.port,
        models=client.models(),
        max_context=client.max_context()
    )
    completions_log = Logger(args.completions_log) if args.completions_log else None
    tools = build_tools(args, inference, settings, session_file=session_file, completions_log=completions_log, des_log=des_log)
    state = ChatState(
        settings=settings,
        inference=inference,
        history=ChatHistory(),
        running=True,
        system_prompt=system_prompt,
        completions_log=completions_log,
        session_file=session_file,
        tools=tools,
        on_idle=Continue(CAP_CONTINUE_MSG) if args.cont else None,

    )
    log_header = {
        "model": args.model,
        "context": args.context,
        "temperature": args.temperature,
        "think": args.think,
        "max_turn_tokens": args.max_turn_tokens,
        "argv" : sys.argv[1:]
    }

    print(c_out(Palette.CHROME, f"\n{"═"*50}"))
    print(c_out(Palette.CHROME, f""" DES Chat v0.1
    Server:       http://127.0.0.1:{args.port}
    Model:        {args.model}
    Temperature:  {args.temperature}
    Think mode:   {"enabled" if args.think else "disabled"}
    Auto mode:    {"on" if settings.auto else "off"}
    Context:      {args.context}
    Turn tokens:  {args.max_turn_tokens}
    Tool rounds:  {args.max_tool_rounds}
    Compl log:    {args.completions_log}
    DES log:      {args.des_log}
    Debug:        {"enabled" if args.debug else "disabled"}
    Timeout:      {args.timeout}s
    Session:      {session_file or "-"}
    Tools:        {", ".join(t.name + (" (asks)" if t.confirm else "") for t in tools.tools) or "none"}
    Workspace:    {os.path.realpath(args.workspace)}
    System prompt:{args.system_prompt[0:40] if args.system_prompt else "none"}{"..." if len(args.system_prompt) > 40 else ""}
"""))
    print(c_out(Palette.CHROME, f"\n{"═"*50}"))
    
    Engine[ChatState](
        des_log=des_log,
        debug=args.debug,
        on_error=on_error,
        on_interrupt=on_interrupt,
    ).run(state, seed=[LoadSession(), PromptUser()], run_id=run_id, log_header=log_header)


if __name__ == "__main__":
    main()
