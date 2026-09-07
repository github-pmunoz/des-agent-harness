"""
Renderer stages: the chain a streaming completion is drawn through. Contract per link is
feed(channel, text) / flush(); a stage transforms events and passes them on, the last one is a
sink. Chain with Seam(CodeFence(PyHighlight(ToolProgress(Terminal())))) — outermost receives first.

Channels are llama-server's frame vocabulary as wire.events() emits it — reasoning, content,
tool_name, tool_args — plus the ones this chain creates: code and code_py (CodeFence), tool
(ToolProgress), and the py_* token channels (PyHighlight). Only Terminal knows a colour.
"""
from __future__ import annotations

import sys
from typing import Optional

from desh.llama.pyscan import ScanState, finish, scan
from desh.render import Palette


class Stage:
    """
    One link of the renderer chain. Contract: feed(channel, text) / flush().
    A stage transforms events and passes them to `self.next`; the last stage is a sink (next=None).
    Chain with `Seam(CodeFence(Terminal()))` — outermost receives first.
    """
    def __init__(self, next: Optional["Stage"] = None):
        self.next = next

    def feed(self, channel: str, text: str) -> None:
        self.emit(channel, text)

    def flush(self) -> None:
        if self.next:
            self.next.flush()

    def emit(self, channel: str, text: str) -> None:
        if self.next:
            self.next.feed(channel, text)


class ToolProgress(Stage):
    """
    Make a streaming tool call visible. Turns tool_name / tool_args events into one dim "tool" line
    per call that is rewritten in place as the arguments grow:

        ⚙ Write … 1234 chars

    The line is closed with a newline when another call opens, when other text arrives, or at
    flush(). Updates are throttled to every `step` characters so a non-TTY sink (a log, a test)
    is not flooded with carriage returns. Nothing here touches the fold: the arguments themselves
    are never rendered, only their size.
    """
    def __init__(self, next: Stage, step: int = 256):
        super().__init__(next)
        self.step = step
        self.name: Optional[str] = None
        self.size = 0
        self.shown = 0

    def feed(self, channel: str, text: str) -> None:
        if channel == "tool_name":
            self._close()
            self.name, self.size, self.shown = text, 0, 0
            self.emit("tool", f"\r⚙ {text}")
        elif channel == "tool_args":
            self.size += len(text)
            if self.name is not None and self.size - self.shown >= self.step:
                self.shown = self.size
                self.emit("tool", f"\r⚙ {self.name} … {self.size} chars")
        else:
            self._close()
            self.emit(channel, text)

    def _close(self) -> None:
        if self.name is not None:
            self.emit("tool", f"\r⚙ {self.name} … {self.size} chars\n" if self.size else "\n")
            self.name = None

    def flush(self) -> None:
        self._close()
        super().flush()


class PyHighlight(Stage):
    """
    Colour the body of a python fence by token class. Consumes "code_py" and re-emits it as the
    py_* channels (plus plain "code_py" for default text) via the streaming scanner in pyscan;
    every other channel passes through. A pass-through event first finishes the scan — the held
    tail must not overtake it, and the closing fence (a "code" event) ends the program, so the
    next fence starts with fresh scanner state.
    """
    def __init__(self, next: Stage):
        super().__init__(next)
        self.state = ScanState()

    def feed(self, channel: str, text: str) -> None:
        if channel == "code_py":
            spans, self.state = scan(self.state, text)
        else:
            spans, self.state = finish(self.state)
            spans.append((channel, text))
        for span_channel, span_text in spans:
            self.emit(span_channel, span_text)

    def flush(self) -> None:
        spans, self.state = finish(self.state)
        for span_channel, span_text in spans:
            self.emit(span_channel, span_text)
        super().flush()


class Terminal(Stage):
    """Sink: writes to a stream with per-channel colour. reasoning dim, content plain, code yellow,
    tool (a streaming call's progress line) dim; inside a python fence the token channels take
    their own colours and default text (code_py) renders in the terminal's foreground."""
    COLOURS = {
        "reasoning": Palette.DEBUG, "content": "", "code": Palette.CHROME, "tool": Palette.DEBUG,
        "code_py": "", "py_kw": Palette.PY_KEYWORD, "py_builtin": Palette.PY_BUILTIN,
        "py_call": Palette.PY_CALL, "py_str": Palette.PY_STRING, "py_num": Palette.PY_NUMBER,
        "py_comment": Palette.PY_COMMENT,
    }
    RESET = Palette.RESET

    def __init__(self, out=None, colour: bool = True):
        super().__init__(None)
        self.out = out or sys.stdout
        self.tty = self.out.isatty()
        self.colour = colour and self.tty

    def feed(self, channel: str, text: str) -> None:
        if self.tty and text.startswith("\r"):
            # A carriage return redraws over whatever the line already holds (the "Assistant: " prompt
            # on the first tool call); erase to end of line so a shorter redraw leaves no tail.
            text = "\r\033[K" + text[1:]
        if self.colour and self.COLOURS.get(channel):
            self.out.write(self.COLOURS[channel] + text + self.RESET)
        else:
            self.out.write(text)
        self.out.flush()

    def flush(self) -> None:
        self.out.write("\n")
        self.out.flush()


class Seam(Stage):
    """Emit a separator on the first content event after any reasoning (port of the bash foreach)."""
    def __init__(self, next: Stage, separator: str = "\n---\n"):
        super().__init__(next)
        self.separator = separator
        self.pending = False

    def feed(self, channel: str, text: str) -> None:
        if channel == "reasoning":
            self.pending = True
        elif channel == "content" and self.pending:
            self.pending = False
            self.emit("content", self.separator)
        self.emit(channel, text)


class CodeFence(Stage):
    """
    Re-channel text inside ``` fences from "content" to "code" (port of chat-bot.py's fence machine).
    Only the content channel is inspected; reasoning passes through untouched.

    Three regions, each with its own hold-back rule, because any of them can be split across deltas:
      - outside a fence: hold up to two trailing backticks ("``" now, "`python" later);
      - the info string right after an opening fence: hold until its newline ("py" then "thon\\n");
      - the fence body: hold up to two trailing backticks, same as outside.
    The info string names the body's channel: a python fence body goes out as "code_py" so a
    highlighter downstream can colour it; every other body, and every fence line, is plain "code".
    """
    PYTHON_LANGS = frozenset({"python", "py", "python3"})

    def __init__(self, next: Stage):
        super().__init__(next)
        self.in_code = False
        self.in_info = False
        self.lang = ""
        self.buffer = ""

    @property
    def channel(self) -> str:
        """The channel the NEXT span of content text belongs to, given where we are in the stream."""
        if not self.in_code:
            return "content"
        return "code_py" if self.lang in self.PYTHON_LANGS else "code"

    def feed(self, channel: str, text: str) -> None:
        # Pass-through: anything not "content" was classified upstream (reasoning today).
        # "code" never arrives here — this stage is the one that *creates* it, on emit.
        if channel != "content":
            self._flush_buffer()
            self.emit(channel, text)
            return

        # Work on the held-back tail from last time plus the new text, as one string.
        work = self.buffer + text
        self.buffer = ""

        while work:
            if self.in_info:
                # The info line is held whole until its newline: the language is its first word.
                nl = work.find("\n")
                if nl < 0:
                    self.buffer = work
                    return
                info, work = work[:nl + 1], work[nl + 1:]
                words = info.split()
                self.lang = words[0].lower() if words else ""
                self.in_info = False
                self.emit("code", info)
                continue

            # A complete fence: emit the span before it on the CURRENT channel, emit the fence
            # itself as code, flip state. Same body serves opening and closing fences — an opening
            # one additionally enters the info region.
            if "```" in work:
                before, work = work.split("```", 1)
                if before:
                    self.emit(self.channel, before)
                self.emit("code", "```")
                self.in_code = not self.in_code
                self.in_info = self.in_code
                self.lang = ""
                continue

            # No complete fence left. But `work` may END in 1-2 backticks that are the start
            # of a fence whose rest is in the next delta.
            tail_start = -2 if work.endswith("``") else -1 if work.endswith("`") else 0
            if tail_start != 0:
                self.buffer = work[tail_start:]
                work = work[:tail_start]
            if work:
                self.emit(self.channel, work)
            return

    def _flush_buffer(self) -> None:
        if self.buffer:
            self.emit("code" if self.in_info else self.channel, self.buffer)
            self.buffer = ""

    def flush(self) -> None:
        self._flush_buffer()
        super().flush()
