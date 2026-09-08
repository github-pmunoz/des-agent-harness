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

def rl_prompt(channel: str, text: str) -> str:
    return f"\001{channel}\002{text}\001{Palette.RESET}\002"

def pretty_log(record: dict) -> str:
    return f"\u21aa {record['payload']} → [{', '.join(record['emitted'])}] " f"depth={record['depth']} {record['dur_ms']}ms" + ("" if record['outcome'] == "ok" else f" ⚠ {record['outcome']}")
