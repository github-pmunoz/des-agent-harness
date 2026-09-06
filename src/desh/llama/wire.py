"""
The wire shape of a llama-server chat completion, and nothing else: the request payload, the
settled completion (from a JSON body or folded from SSE frames), the tool-call record inside it,
and the two pure frame helpers (parse_sse, events). No I/O, no imports from the rest of desh.

    Request      builds the payload; the ONLY place optional keys are decided
    ToolCall     one entry of a completion's tool_calls array
    Completion   the settled chat.completion; built from a JSON response OR folded from SSE frames
    parse_sse    'data: {...}' lines -> frames
    events       frame -> (channel, text) events for a renderer; the one place the frame shape is read
"""
from __future__ import annotations

import json
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
    tools: list[dict] = field(default_factory=list)   # OpenAI tool schemas: [{type:"function", function:{name, description, parameters}}]
    tool_choice: Optional[str | dict] = None          # "auto" | "none" | "required" | {type:"function", function:{name}}
    seed: Optional[int] = None                        # None -> server picks; 0 is a valid seed

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
        if self.tools:
            body["tools"] = self.tools
            if self.tool_choice is not None:
                body["tool_choice"] = self.tool_choice
        if self.seed is not None:
            body["seed"] = self.seed
        return body

# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A tool call, as it appears in a completion's tool_calls array."""
    index: int
    id: str
    type: str
    name: str
    arguments: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "id": self.id,
            "type": self.type,
            "function": {"name": self.name, "arguments": self.arguments},
        }

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
    tool_calls: list[ToolCall] = field(default_factory=list)

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
            tool_calls=[
                ToolCall(
                    index=i,
                    id=t["id"],
                    type=t["type"],
                    name=t["function"]["name"],
                    arguments=t["function"]["arguments"]
                )for i, t in enumerate(msg.get("tool_calls") or [])
            ]
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
          - tool-call frames: choices[0].delta.tool_calls = [ {index, id?, type?, function:{name?, arguments?}} ... ]
                              one element per call touched by this chunk; the OPENER of an index carries
                              id/type/name once, every later element for that index carries only an
                              arguments fragment. Elements for different indices may share a frame.
        """
        content = ""
        reasoning = ""
        usage = None
        timings = None
        finish_reason = "unknown"
        calls: dict[int, dict] = {}   # index -> {index, id, type, name, arguments} accumulated across frames
        for frame in frames:
            if "choices" not in frame:
                continue
            if not frame["choices"]:
                # trailing frame with usage and timings
                usage = frame.get("usage")
                timings = frame.get("timings")
                continue
            finish_reason = frame["choices"][0].get("finish_reason") or finish_reason
            delta = frame["choices"][0]["delta"]
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for tc in delta.get("tool_calls") or []:
                index = tc["index"]
                call = calls.setdefault(index, {
                    "index": index,
                    "id": "",
                    "type": "",
                    "name": "",
                    "arguments": "",
                })
                function = tc.get("function") or {}
                if "id" in tc:
                    call["id"] = tc["id"]
                if "type" in tc:
                    call["type"] = tc["type"]
                if "name" in function:
                    call["name"] = function["name"]
                call["arguments"] += function.get("arguments") or ""

        return cls(
            tool_calls=[ToolCall(**calls[i]) for i in sorted(calls)],
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
        msg : dict = {"role": "assistant", "content": self.content, "reasoning_content": self.reasoning}
        if self.tool_calls:
            msg["tool_calls"] = [t.to_dict() for t in self.tool_calls]
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "system_fingerprint": self.system_fingerprint,
            "choices": [{
                "index": 0,
                "finish_reason": self.finish_reason,
                "message": msg,
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
    Frame -> (channel, text) events, channel in {"reasoning", "content", "tool_name", "tool_args"}.
    Envelope-only frames (finish frame, trailing usage frame) yield nothing.
    A tool-call opener yields ("tool_name", name) once; every arguments fragment yields
    ("tool_args", fragment), so a renderer can show that the model is writing a call — otherwise a
    long argument (a file's contents) streams in total silence.
    This is the only place the renderer stack touches the OpenAI frame shape.
    """
    if not frame.get("choices"):
        return
    delta = frame["choices"][0].get("delta") or {}
    if delta.get("reasoning_content"):
        yield "reasoning", delta["reasoning_content"]
    if delta.get("content"):
        yield "content", delta["content"]
    for tc in delta.get("tool_calls") or []:
        function = tc.get("function") or {}
        if function.get("name"):
            yield "tool_name", function["name"]
        if function.get("arguments"):
            yield "tool_args", function["arguments"]
