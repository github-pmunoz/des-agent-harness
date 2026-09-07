"""
The operator's side of a tool call: how a call is shown before it runs, how the operator answers,
and what the model is told when they refuse. No events here — ExecuteToolCalls (events.py) calls
into this module; nothing here calls back. Tests replace `ask` on this module to script the prompt.
"""
import json
import readline
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Literal, Optional

from desh.llama.wire import ToolCall
from desh.render import Palette, c_out, rl_prompt
from desh.tools import Tool


DENIED_TEXT = "The user declined this tool call. Do not retry it; ask the user or take another approach."
SKIPPED_TEXT = "Not run: the user declined an earlier tool call in this round. Reconsider the plan before retrying."


@dataclass(frozen=True)
class Answer:
    """One operator decision at the confirmation prompt. `message` is guidance the operator typed
    with a "no": it replaces DENIED_TEXT as the tool message, so the model learns why."""
    kind: Literal["yes", "no", "cancel"]
    message: str = ""


def ask(tc: ToolCall) -> Answer:
    """The operator's decision for one call whose tool wants confirmation. Never raises.
    ESC is cancel, Enter is yes, silence (EOF, a closed or broken stdin) is cancel, [y/n/m/c]
    don't need Enter to be pressed, and any other key asks again."""
    def prompt(text: str) -> str | None:
        """Read one control line without adding it to the conversation history."""
        try:
            before = readline.get_current_history_length()
            try:
                value = input(rl_prompt(Palette.CHROME, text))
            finally:
                after = readline.get_current_history_length()
                for index in range(after - 1, before - 1, -1):
                    readline.remove_history_item(index)
            return value
        except (EOFError):
            return None
        except Exception:
            return None

    def key_prompt(text: str) -> str | None:
        """Read one decision key immediately, without putting the terminal in line mode."""
        old_settings = None
        try:
            sys.stdout.write(c_out(Palette.TOOL_CONFIRM, text))    # plain write: readline's \001/\002 markers do not apply here
            sys.stdout.flush()
            if sys.stdin.isatty():
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            key = sys.stdin.read(1)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return key or None
        except Exception:
            return None
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass
    choice_prompt = "[y]es / [n]o / [m]essage / [c]ancel: (ESC to cancel, ENTER for yes)"
    print(c_out(Palette.CHROME, f"{"──"*(len(choice_prompt)//2)}"))
    while True:
        choice = key_prompt(choice_prompt)
        if choice is None:
            return Answer(kind="cancel")
        choice = choice.lower()
        if choice in {"\r", "\n"}:
            return Answer(kind="yes")
        if choice == "\x1b":
            return Answer(kind="cancel")
        if choice == "y":
            return Answer(kind="yes")
        if choice == "c":
            return Answer(kind="cancel")
        if choice == "n":
            return Answer(kind="no")
        if choice == "m":
            message = prompt("Message for the model: ") or ""
            return Answer(kind="no", message=message)


def shorten(text: str, limit: int = 200, indent: int = 4) -> str:
    """At most `limit` characters, for terminal echoes of calls and results."""
    short = text if len(text) <= limit else text[:limit - 1] + f" ... {len(text)} chars"
    return "\n".join(f"{' '*indent}{line}" for line in short.splitlines())


def describe_call(tc: ToolCall, tool: Optional[Tool] = None) -> str:
    """The call as the operator must see it to approve it. A tool with a `preview` renders its own
    (an Edit as a diff); otherwise one line per argument, multi-line values (file contents) as
    indented blocks. Falls back to the raw wire string when the arguments are not a JSON object,
    and to the generic rendering when a preview raises — the gate must always show something."""
    try:
        args = json.loads(tc.arguments)
    except ValueError:
        args = None
    color_arrow = c_out(Palette.CHROME, f"→")
    color_name = c_out(Palette.TOOL_NAME, f"{tc.name}")
    if not isinstance(args, dict) or not args:
        color_args = c_out(Palette.TOOL_ARG_VALUE, f"{tc.arguments}")
        return f"{color_arrow} {color_name}({color_args})"
    if tool is not None and tool.preview is not None:
        try:
            color_preview = c_out(Palette.TOOL_ARG_VALUE, f"({tool.preview(args)})")
            return f"{color_arrow} {color_name}\n{color_preview}"
        except Exception:
            pass
    color_key = c_out(Palette.TOOL_ARG_KEY, "{}")
    color_value = c_out(Palette.TOOL_ARG_VALUE, "{}")
    lines = [f"{color_arrow} {color_name}"]
    for key, value in args.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"  {color_key.format(key)}:")
            lines.extend(f"    {color_value.format(line)}" for line in value.splitlines())
        else:
            lines.append(f"  {color_key.format(key)}: {color_value.format(json.dumps(value, ensure_ascii=False))}")
    return "\n".join(lines)
