"""
Transport to one llama-server (single-model or router mode): complete(), stream(), models(),
props(), max_context(), and the error taxonomy. stream() is a T-junction — every frame is kept
for the fold AND turned into (channel, text) events for a Renderer, which is any object with
feed()/flush() (see stages.py for the ones the chat uses; this module never imports them).

Port of an earlier bash prototype (send_direct.sh).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Optional, Protocol

from desh.llama.wire import Completion, Request, events, parse_sse


class Renderer(Protocol):
    """What stream() needs from a renderer: (channel, text) events in, a final flush."""
    def feed(self, channel: str, text: str) -> None: ...
    def flush(self) -> None: ...


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

    def stream(self, req: Request, renderer: Renderer, cancelled=lambda: False) -> Completion:
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

    def _models(self) -> list[dict]:
        """Router: GET /models -> [{id, status:{value, args, preset}, ...}]. Single-model: one entry."""
        return self._json("/models")["data"]

    def models(self) -> list[str]:
        return [m["id"] for m in self._models()]

    def props(self, model: Optional[str] = None) -> dict:
        """GET /props, or /props?model=<id> in router mode (the child's real launch config)."""
        path = "/props" + (f"?model={urllib.parse.quote(model, safe='')}" if model else "")
        return self._json(path)

    def max_context(self) -> dict[str, int]:
        result = {}
        for m in self._models():
            args = m.get("status", {}).get("args", [])
            if "--ctx-size" in args:
                result[m["id"]] = int(args[args.index("--ctx-size") + 1])
        return result
