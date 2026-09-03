#!/usr/bin/env python3
"""
Thin typed client for a local llama-server (single-model or router mode).

Port of an easrlier bash prototype (send_direct.sh)

    Request      builds the payload; the ONLY place optional keys are decided
    Completion   the settled chat.completion; built from a JSON response OR folded from SSE frames
    LlamaServer  transport + error taxonomy (LlamaUnreachable vs LlamaServerError)
    Renderer     stateful sink over (channel, text) events — seam, code fences, colour
    Logger       one JSONL record per completion; never sees frames

No Delta type (yet): frames flow as raw dicts; the renderer adapter turns them into events.
No queue awareness: that is another project (QueueManager.py).

TODO(transport):
  - Extract Transport Protocol (complete, stream(req, renderer, cancelled), models, props) into
    desh/transport.py; LlamaServer becomes its first impl. Engine depends on the protocol only.
  - Request.seed optional field + payload() emission 
  - Completion.from_frames per-index tool_calls fold 
  - Fixture-based self-check became orphaned when fixtures went — replace with
    tests/test_client.py using inline minimal SSE frames (3–4 lines each), so the fold logic
    keeps a regression test without binary fixtures in the repo.

"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Optional



# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class Request:
    messages: list[dict] = field(default_factory=list)  # [ {role, content}, ... ]
    model: Optional[str] = None          # router mode: section id from the preset INI
    temperature: float = 0.7
    max_tokens: int = 256
    think: bool = False
    stream: bool = False

    @classmethod
    def single(cls, user: str, system: str = "", **params) -> "Request":
        """Build a request with a single user message and optional system prompt."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        return cls(messages=messages, **params)

    def payload(self) -> dict:
        """Build the /v1/chat/completions body. Mirrors send_direct.sh Stage 2."""
        body = {
            "messages": self.messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "chat_template_kwargs": {"enable_thinking": self.think},
        }
        # Optional keys: present exactly when they mean something (a single-model
        # server ignores "model"; the router 400s without it).
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        if self.model:
            body["model"] = self.model
        return body


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

@dataclass
class Completion:
    id: str
    model: str
    created: int
    system_fingerprint: str
    content: str
    reasoning: str
    finish_reason: str
    usage: Optional[dict]
    timings: Optional[dict]
    streamed: bool = False

    @classmethod
    def from_response(cls, d: dict) -> "Completion":
        """Non-streaming: the server's chat.completion JSON as-is."""
        msg = d["choices"][0]["message"]
        return cls(
            id=d["id"],
            model=d["model"],
            created=d["created"],
            system_fingerprint=d.get("system_fingerprint", ""),
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or "",
            finish_reason=d["choices"][0].get("finish_reason") or "unknown",
            usage=d.get("usage"),
            timings=d.get("timings"),
            streamed=False,
        )

    @classmethod
    def from_frames(cls, frames: list[dict]) -> "Completion":
        """
        Streaming: fold the SSE frames (already parsed, [DONE] excluded) into one Completion.

        Frame anatomy (llama-server b10643)
          - every frame carries the envelope: id, model, created, system_fingerprint
          - content frames:   choices[0].delta.content            (never with reasoning in the same frame)
          - reasoning frames: choices[0].delta.reasoning_content
          - finish frame:     choices[0].finish_reason != null, delta == {}
          - trailing frame:   choices == []  with usage + timings  (only with stream_options.include_usage)
        """
        content = ""
        reasoning = ""
        usage = None
        timings = None
        finish_reason = "unknown"
        for frame in frames:
            if "choices" not in frame:
                continue
            if not frame["choices"]:
                # trailing frame with usage and timings
                usage = frame.get("usage")
                timings = frame.get("timings")
                continue
            B
            finish_reason = frame["choices"][0].get("finish_reason") or finish_reason
            delta = frame["choices"][0]["delta"]
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""

        return cls(
            id=frames[0]["id"],
            model=frames[0]["model"],
            created=frames[0]["created"],
            system_fingerprint=frames[0]["system_fingerprint"],
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            streamed=True
        )

    def to_dict(self) -> dict:
        """chat.completion-shaped dict, same layout the bash script logs (so old JSONL queries keep working)."""
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [{
                "index": 0,
                "finish_reason": self.finish_reason,
                "message": {"role": "assistant", "content": self.content, "reasoning_content": self.reasoning},
            }],
            "usage": self.usage,
            "timings": self.timings,
            "streamed": self.streamed,
        }


# ---------------------------------------------------------------------------
# SSE helpers (pure; used by LlamaServer.stream_raw 
# ---------------------------------------------------------------------------

def parse_sse(lines: Iterator[str]) -> Iterator[dict]:
    """Yield parsed frames from 'data: {...}' lines; skip blanks and the [DONE] sentinel."""
    for line in lines:
        line = line.rstrip("\n")
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):]
        if body == "[DONE]":
            return
        yield json.loads(body)


def events(frame: dict) -> Iterator[tuple[str, str]]:
    """
    Frame -> (channel, text) events, channel in {"reasoning", "content"}.
    Envelope-only frames (finish frame, trailing usage frame) yield nothing.
    This is the only place the renderer stack touches the OpenAI frame shape.
    """
    if not frame.get("choices"):
        return
    delta = frame["choices"][0].get("delta") or {}
    if delta.get("reasoning_content"):
        yield "reasoning", delta["reasoning_content"]
    if delta.get("content"):
        yield "content", delta["content"]


# ---------------------------------------------------------------------------
# LlamaServer
# ---------------------------------------------------------------------------

class LlamaUnreachable(Exception):
    """Transport failure: connection refused, timeout (curl exit 7 / 28 in the bash version)."""


class LlamaServerError(Exception):
    """The server answered with {error: {code, message, type}}."""
    def __init__(self, code: int, type_: str, message: str):
        super().__init__(f"{code} {type_}: {message}")
        self.code, self.type, self.message = code, type_, message


class LlamaServer:
    """
    Transport to one llama-server (single-model or router). Two error classes:
      LlamaUnreachable  — could not connect / timed out (nothing came back)
      LlamaServerError  — the server answered with {error:{code,message,type}},
                          either as an HTTP 4xx/5xx body or as a frame mid-stream
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8012", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- plumbing -----------------------------------------------------------

    def _open(self, path: str, body: Optional[dict] = None):
        """Return an open HTTP response (caller closes). Maps transport failures and error bodies."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            # llama-server pairs every 4xx/5xx with a JSON error body (send_direct.sh dropped curl -f for this)
            raw = e.read().decode(errors="replace")
            try:
                err = json.loads(raw)["error"]
            except (ValueError, KeyError, TypeError):
                raise LlamaServerError(e.code, "http_error", raw[:200]) from None
            raise LlamaServerError(err.get("code", e.code), err.get("type", "unknown"), err.get("message", "")) from None
        except urllib.error.URLError as e:
            raise LlamaUnreachable(f"{self.base_url}: {e.reason}") from None
        except TimeoutError as e:
            raise LlamaUnreachable(f"{self.base_url}: timeout after {self.timeout}s") from None

    def _json(self, path: str, body: Optional[dict] = None) -> dict:
        with self._open(path, body) as r:
            return json.load(r)

    # --- API ----------------------------------------------------------------

    def complete(self, req: Request) -> Completion:
        """Non-streaming. Sends req.payload() with stream forced off."""
        payload = req.payload()
        payload["stream"] = False
        payload.pop("stream_options", None)
        return Completion.from_response(self._json("/v1/chat/completions", payload))

    def stream_raw(self, req: Request) -> Iterator[dict]:
        """
        Streaming. Yields parsed SSE frames (dicts) as they arrive; [DONE] ends the iteration.
        A frame carrying {"error": {...}} mid-stream raises LlamaServerError from inside the
        generator — consumers (renderer / fold / logger) never see error frames.
        """
        payload = req.payload()
        payload["stream"] = True
        payload.setdefault("stream_options", {"include_usage": True})
        with self._open("/v1/chat/completions", payload) as raw:
            for frame in parse_sse(line.decode("utf-8") for line in raw):
                if "error" in frame:
                    err = frame["error"]
                    raise LlamaServerError(err.get("code"), err.get("type"), err.get("message"))
                yield frame

    def stream(self, req: Request, renderer: "Stage", cancelled=lambda: False) -> Completion:
        """Streaming T-junction: every frame is kept for the fold AND turned into events for the renderer."""
        frames = []
        for frame in self.stream_raw(req):
            if cancelled():
                break
            frames.append(frame)
            for channel, text in events(frame):
                renderer.feed(channel, text)
        renderer.flush()
        completion = Completion.from_frames(frames) if frames else Completion(
            id="", model=req.model or "", created=0, system_fingerprint="",
            content="", reasoning="", finish_reason="cancelled",
            usage=None, timings=None, streamed=True,
        )
        if cancelled():
            completion.finish_reason = "cancelled"
        return completion

    def models(self) -> list[dict]:
        """Router: GET /models -> [{id, status:{value, args, preset}, ...}]. Single-model: one entry."""
        return self._json("/models")["data"]

    def props(self, model: Optional[str] = None) -> dict:
        """GET /props, or /props?model=<id> in router mode (the child's real launch config)."""
        path = "/props" + (f"?model={urllib.parse.quote(model, safe='')}" if model else "")
        return self._json(path)


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


class Terminal(Stage):
    """Sink: writes to a stream with per-channel colour. reasoning dim, content plain, code yellow."""
    COLOURS = {"reasoning": "\033[2m", "content": "", "code": "\033[33m"}
    RESET = "\033[0m"

    def __init__(self, out=None, colour: bool = True):
        super().__init__(None)
        self.out = out or sys.stdout
        self.colour = colour and self.out.isatty()

    def feed(self, channel: str, text: str) -> None:
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


class Logger:
    """
    One JSONL record per completion, same layout send_direct.sh writes
    ({timestamp, port, payload, response}) so existing jq queries over the file keep working.
    Errors are not recorded (design choice from the bash version: one record = one completion).
    """
    def __init__(self, path: str):
        self.path = pathlib.Path(path).expanduser()

    def record(self, req: Request, completion: Completion, port: int) -> None:
        rec = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "port": port,
            "payload": req.payload(),
            "response": completion.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
