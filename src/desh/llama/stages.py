"""
Renderer stages: the chain a streaming completion is drawn through. Contract per link is
feed(channel, text) / flush(); a stage transforms events and passes them on, the last one is a
sink. Chain with Seam(CodeFence(ToolProgress(Terminal()))) — outermost receives first.

Channels are llama-server's frame vocabulary as wire.events() emits it — reasoning, content,
tool_name, tool_args — plus the two this chain creates: code (CodeFence) and tool (ToolProgress).
"""
from __future__ import annotations

import sys
from typing import Optional

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


class Terminal(Stage):
    """Sink: writes to a stream with per-channel colour. reasoning dim, content plain, code yellow,
    tool (a streaming call's progress line) dim."""
    COLOURS = {"reasoning": Palette.DEBUG, "content": "", "code": Palette.CHROME, "tool": Palette.DEBUG}
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
    Must handle a fence split across deltas ("``" then "`python\n"): hold back up to two trailing
    backticks until the next event or flush() decides.
    """
    def __init__(self, next: Stage):
        super().__init__(next)
        self.in_code = False
        self.buffer = ""

    @property
    def channel(self) -> str:
        """The channel the NEXT span of content text belongs to, given where we are in the stream."""
        return "code" if self.in_code else "content"

    def feed(self, channel: str, text: str) -> None:
        # Pass-through: anything not "content" was classified upstream (reasoning today).
        # "code" never arrives here — this stage is the one that *creates* it, on emit.
        if channel != "content":
            if self.buffer:
                self.emit(self.channel, self.buffer)
                self.buffer = ""
            self.emit(channel, text)
            return

        # Work on the held-back tail from last time plus the new text, as one string.
        work = self.buffer + text
        self.buffer = ""

        # Every complete fence in `work`: emit the span before it on the CURRENT channel,
        # emit the fence itself as code, flip state. Same loop body serves opening and closing
        # fences — the only difference is the state we start in.
        while "```" in work:
            before, work = work.split("```", 1)
            if before:
                self.emit(self.channel, before)
            self.emit("code", "```")
            self.in_code = not self.in_code

        # `work` now has no complete fence. But it may END in 1-2 backticks that are the start
        # of a fence whose rest is in the next delta ("``" now, "`python" later).
        tail_start = -2 if work.endswith("``") else -1 if work.endswith("`") else 0
        if tail_start != 0:
            self.buffer = work[tail_start:]
            work = work[:tail_start]
        if work:
            self.emit(self.channel, work)

    def flush(self) -> None:
        if self.buffer:
            self.emit("code" if self.in_code else "content", self.buffer)
            self.buffer = ""
        super().flush()
