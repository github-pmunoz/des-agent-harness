import re
import reprlib
import sys

bounded_repr = reprlib.Repr()
bounded_repr.maxother = 80
bounded_repr.maxstring = 40

class Palette:
    DEBUG   = "\033[2m"     # dim   
    RESET   = "\033[0m"     # reset white
    CHROME  = "\033[33m"    # yellow
    DIM_CHROME = "\033[2m\033[33m"  # dim yellow
    ERROR   = "\033[31m"    # red
    WARNING = "\033[38;5;208m"    # orange
    CHROME_USER = "\033[1m\033[32m"  # bright green
    CHROME_ASSISTANT = "\033[1m\033[33m"  # bright yellow
    HISTORY_USER = "\033[2m\033[32m"  # dim green
    HISTORY_ASSISTANT = "\033[2m\033[33m"  # dim yellow
    HISTORY_SUMMARY = "\033[2m\033[95m"  # dim magenta
    STATS_LINE = "\033[1m\033[36m"  # bright cyan
    DIFF_DEL = "\033[31m"     # red: a removed line in an edit preview
    DIFF_ADD = "\033[32m"     # green: an added line
    DIFF_CTX = "\033[2m"      # dim: unchanged context
    TOOL_NAME = "\033[1m\033[36m"  # bright cyan: a tool name
    TOOL_ARG_KEY = "\033[36m"  # cyan: a tool argument key
    TOOL_ARG_VALUE = "\033[0m"  # default: a tool argument value
    TOOL_REASON = "\033[38;5;208m"  # orange: the value of a "reason" argument
    TOOL_CONFIRM = "\033[32m"  # green: a tool confirm message
    TOOL_RESULT = "\033[2m"   # dim: a tool result
    TOOL_STATS = "\033[0m\033[36m"  # dim cyan: context stats during tool execution
    PY_KEYWORD = "\033[1m\033[35m"  # bright magenta: a python keyword
    PY_BUILTIN = "\033[36m"   # cyan: a builtin name or type
    PY_CALL = "\033[94m"      # bright blue: a name the code introduced, being called
    PY_STRING = "\033[32m"    # green: a string literal
    PY_NUMBER = "\033[38;5;208m"  # orange: a number literal
    PY_COMMENT = "\033[2m"    # dim: a comment

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, channel: str, text: str) -> str:
        return f"{channel}{text}{Palette.RESET}" if self.enabled else text
    
c_out = Palette(enabled=sys.stdout.isatty())
c_err = Palette(enabled=sys.stderr.isatty())


class Gutter:
    """A text stream that draws a prefix at the start of every line written through it, so a
    nested run (a subagent) shows as an indented block under its banner. Wraps the real stream;
    everything but write() is delegated to it, so isatty(), flush() and fileno() still answer for
    the terminal.

    Writes arrive as arbitrary chunks (a streamed answer splits mid-word), so "at a line start"
    is state carried across writes, not a property of one chunk. Both "\\n" and "\\r" start a line:
    the tool-progress stage redraws its line in place with "\\r\\033[K…", and the prefix has to
    land before the clear-to-end or it is wiped with the rest.

    Colour: the prefix is drawn in whatever SGR state the content left at the line break — a dim
    result's second line, or a progress line whose dim was set BEFORE its "\\r" — so a coloured
    prefix must begin with a reset (see delegate.py), and the state the content had is re-emitted
    after the prefix: a dim multi-line tool result stays dim on every line. A segment that is
    only escape codes (the reset after a trailing newline) does not open a line: it is written
    through as-is and the prefix waits for real content."""
    _BREAKS = re.compile(r"(\r|\n)")
    _SGR = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, out, prefix: str):
        self.out = out
        self.prefix = prefix
        self.at_line_start = True
        self.colour = ""         # SGR sequences the content set since its last reset

    def write(self, text: str) -> int:
        for part in self._BREAKS.split(text):
            if part in ("\n", "\r"):
                self.out.write(part)
                self.at_line_start = True
            elif part:
                if self.at_line_start and self._SGR.sub("", part):    # real content opens the line
                    self.out.write(self.prefix + self.colour)
                    self.at_line_start = False
                self.out.write(part)
                for sgr in self._SGR.findall(part):
                    self.colour = "" if sgr in ("\x1b[0m", "\x1b[m") else self.colour + sgr
        return len(text)

    def __getattr__(self, name):
        return getattr(self.out, name)

def rl_prompt(channel: str, text: str) -> str:
    return f"\001{channel}\002{text}\001{Palette.RESET}\002"

def pretty_log(record: dict) -> str:
    return f"\u21aa {record['payload']} → [{', '.join(record['emitted'])}] " f"depth={record['depth']} {record['dur_ms']}ms" + ("" if record['outcome'] == "ok" else f" ⚠ {record['outcome']}")
