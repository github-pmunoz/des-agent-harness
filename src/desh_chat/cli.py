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
from dataclasses import replace

from desh_chat.state import ChatState
from desh.llama.logger import Logger
from desh.llama.server import LlamaServer
from desh.engine import Engine, Event
from desh.tools import ToolRegistry
from desh_chat.events import TurnStart
from desh_chat.session import LoadSession
from desh_chat.display import DisplayBanner
from desh_chat.state import ChatHistory, Deadline, Settings, InferenceEngine
from desh_chat.handlers import on_error, on_interrupt
from desh_chat.toolset import current_time, ToolRegistry
from desh_chat.coding import Workspace, edit_preview, fold_edited, fold_written
from desh_chat.delegate import Delegate, CAP_CONTINUE_MSG, fold_brief
from desh_chat import scratchpad
from desh_chat.scratchpad import Scratchpad, SCRATCHPAD_SYSTEM_PROMPT


def build_tools(args: argparse.Namespace, inference: InferenceEngine, settings: Settings, *,
                session_file: str | None = None, completions_log: Logger | None = None,
                des_log: TextIO | None = None) -> ToolRegistry:
    """The toolsets are additive: each flag contributes its tools, none of them means no tools.
    The delegate tool is built from the RESOLVED session path, completions Logger and DES log file,
    never from the raw flags: a subagent writes to the same log objects the parent's engine does."""
    # The cap is a share of the context, in chars (4 per token). The registry bounds every result to
    # it as a backstop; Read and Bash cut to it themselves, with a pointer to the rest.
    max_result_chars = int(settings.context * 4 * args.tool_cap / 100.0)
    ws = Workspace(args.workspace, result_chars=max_result_chars)
    tools = ToolRegistry(debug=args.debug, max_result_chars=max_result_chars)
    # `target` is the argument a one-line mention of the call shows (the expiring line of the
    # scratchpad block): a file tool is about its path, Bash about its command, a delegate about
    # its task, a scratchpad tool about its key.
    if args.read:
        tools = tools.add(ws.read, name="Read", confirm=False, target="file_path")
    if args.write:
        tools = tools.add(ws.write, name="Write", fold=fold_written, target="file_path")
    if args.edit:
        tools = tools.add(ws.edit, name="Edit", preview=edit_preview, fold=fold_edited, target="file_path")
    if args.bash:
        tools = tools.add(ws.bash, name="Bash", identity=("command",), target="command")
    if args.current_time:
        tools = tools.add(current_time, name="Current time")
    if args.delegate:
        delegate_tools = ToolRegistry(debug=args.debug, max_result_chars=max_result_chars)
        delegate_tools = (delegate_tools.add(ws.read, name="Read", confirm=False, target="file_path")
                          .add(ws.write, name="Write", fold=fold_written, target="file_path")
                          .add(ws.edit, name="Edit", preview=edit_preview, fold=fold_edited, target="file_path")
                          .add(ws.bash, name="Bash", identity=("command",), target="command")
                          .add(scratchpad.write, name="scratchpad_write", inject=("scratchpad",), confirm=False, target="key")
                          .add(scratchpad.delete, name="scratchpad_delete", inject=("scratchpad",), confirm=False, target="key")
                          .add(scratchpad.clear, name="scratchpad_clear", inject=("scratchpad",), confirm=False))
        delegate = Delegate(root=ws.root, inference=inference, settings=settings, tools=delegate_tools,
                            session_file=session_file, completions_log=completions_log, des_log=des_log, debug=args.debug)
        # the parent's CURRENT settings travel with every call; the child derives its own from them
        tools = tools.add(delegate.delegate, name="delegate", inject=("settings", "deadline"), fold=fold_brief, target="task")
    if args.scratchpad:
        # the working memory itself lives on ChatState; the tools only get a dict for the call
        tools = (tools.add(scratchpad.write, name="scratchpad_write", inject=("scratchpad",), confirm=False, target="key")
                      .add(scratchpad.delete, name="scratchpad_delete", inject=("scratchpad",), confirm=False, target="key")
                      .add(scratchpad.clear, name="scratchpad_clear", inject=("scratchpad",), confirm=False))
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
    ap.add_argument("-mtr", "--max-tool-rounds",type=int, default=10, help="max tool rounds per turn")
    ap.add_argument("-te",  "--tool-expiration",type=int, default=6, help="rounds after which tool results expire from context")
    ap.add_argument("-sp",  "--system-prompt",  default="You are a helpful assistant. Reply concisely.", help="default: a plain assistant prompt, or the coding-agent prompt with --coding")
    ap.add_argument("-th",  "--think",          action="store_true", help="enable thinking")
    ap.add_argument("-cl",  "--completions-log", default="", help="JSONL telemetry file")
    ap.add_argument("-dl",  "--des-log",        default="", help="DES engine log")
    ap.add_argument("-to",  "--timeout",        default=0, type=float)
    ap.add_argument("-tt",  "--task-timeout",   default=0, type=float, help="wall-clock budget in seconds for a --task run, 0 = none; checked before each tool round, so a run overshoots by at most one completion")
    ap.add_argument("-s",   "--session",        default="", help="session file to load or create")
    ap.add_argument("-sf",  "--sessions-folder", default="", help="folder where a new session file is created per run")
    ap.add_argument("-d",   "--debug",          action="store_true", help="Enable debug output")
    ap.add_argument("-w",   "--workspace",      default=".", help="project root for the coding toolset")
    ap.add_argument("-ta",  "--task",           default="", help="task to run")
    # toolsets are additive flags: any combination, none means the model is offered no tools
    ap.add_argument("--read",      action="store_true", help="offer the Read tool")
    ap.add_argument("--write",     action="store_true", help="offer the Write tool")
    ap.add_argument("--edit",      action="store_true", help="offer the Edit tool")
    ap.add_argument("--bash",      action="store_true", help="offer the Bash tool")
    ap.add_argument("--delegate",  action="store_true", help="offer delegate: subagents with the same tools and settings")
    ap.add_argument("--scratchpad", action="store_true", help="offer the scratchpad tool")
    ap.add_argument("--current_time", action="store_true", help="offer the current time")
    ap.add_argument("-tc",  "--tool-cap",       type=float, default=10.0, help="cap on one tool result, as a percentage of the context window (in chars, 4 per token); the rest is reachable by Read")
    ap.add_argument("-ct",  "--checkpoint-target", type=float, default=0.15, help="share of the context a mid-turn checkpoint summary may take")
    args = ap.parse_args()

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"  # Unique run ID
    session_file = resolve_session_file(args.session, args.sessions_folder, run_id)
    system_prompt = args.system_prompt if args.system_prompt else "You are a helpful assistant. Reply concisely."
    if args.scratchpad:
        system_prompt += "\n\n" + SCRATCHPAD_SYSTEM_PROMPT.format(tool_expiration=args.tool_expiration, max_tool_rounds=args.max_tool_rounds)
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
        tool_expiration=args.tool_expiration,
        auto=args.auto,
        checkpoint_target=args.checkpoint_target,
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
        operator=True,
        auto_prompt=CAP_CONTINUE_MSG if args.cont else None,
        scratchpad=Scratchpad() if args.scratchpad else None,   # empty here; LoadSession restores a saved one
        # the clock starts here, before the session loads and the router loads the model: both are run time
        deadline=Deadline.in_seconds(args.task_timeout) if args.task and args.task_timeout > 0 else None,
    )
    log_header = {
        "model": args.model,
        "context": args.context,
        "temperature": args.temperature,
        "think": args.think,
        "max_turn_tokens": args.max_turn_tokens,
        "argv" : sys.argv[1:]
    }

    runtime_seed: list[Event] = [LoadSession()]
    runtime_seed += [DisplayBanner(args.des_log, args.debug, args.timeout, args.workspace, args.completions_log)] if not args.task else []
    runtime_seed += [TurnStart(args.task)] if args.task else [TurnStart()]

    if args.task:
        state = replace(state, idle_policy="prompt", operator=False)

    final_state: ChatState = Engine[ChatState](
        des_log=des_log,
        debug=args.debug,
        on_error=on_error,
        on_interrupt=on_interrupt
    ).run(state, seed=runtime_seed, run_id=run_id, log_header=log_header)

    if args.task:
        sys.exit(task_exit_code(final_state))


def task_exit_code(final: ChatState) -> int:
    """How a one-shot run reports its end to the shell, from the stop reason of the last turn that
    is not a summary. Every way a turn can end commits a Turn with a stop reason, an exception on
    the very first request included, so the only turn-less run is one that never got to a turn
    (a session file that failed to load, say). Interactive runs never come here."""
    turn = final.history.last_non_summary()
    if turn is None:
        return 1
    stop_to_exit_code: dict[str, int] = {
        "" : 0,
        "cap" : 0,
        "deadline" : 0,
        "overflow" : 1,
        "interrupt" : 1,
        "error" : 1,
    }
    return stop_to_exit_code.get(turn.stop, 1)


if __name__ == "__main__":
    main()
