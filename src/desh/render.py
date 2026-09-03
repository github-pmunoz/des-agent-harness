import reprlib
import sys

bounded_repr = reprlib.Repr()
bounded_repr.maxother = 80
bounded_repr.maxstring = 40

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
