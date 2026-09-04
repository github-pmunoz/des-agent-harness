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
