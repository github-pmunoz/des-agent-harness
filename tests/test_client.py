"""
llama-server client baseline: pins the frame-fold and transport contract of
desh/llama/client.py with inline SSE frames, so later changes to the fold (tool calls)
and to the transport seam (a Transport protocol with LlamaServer as one implementation)
have a regression net. No network: LlamaServer._open / urlopen are monkeypatched with
fake responses built from the SSE text below.

Frame anatomy under test (llama-server b10643, see Completion.from_frames):
  - every frame carries the envelope: id, model, created, system_fingerprint
  - content frames:   choices[0].delta.content
  - reasoning frames: choices[0].delta.reasoning_content
  - finish frame:     choices[0].finish_reason != null, delta == {}
  - trailing frame:   choices == [] with usage + timings (stream_options.include_usage)
  - error frame:      {"error": {code, type, message}} mid-stream
"""
import io
import json
import urllib.error

import pytest

from desh.llama.client import (
    Completion, LlamaServer, LlamaServerError, LlamaUnreachable, Request, Stage, events, parse_sse,
)


# ---------------------
# Inline frame builders
# ---------------------

ENVELOPE = {
    "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1700000000,
    "model": "model-a", "system_fingerprint": "b10643-abc",
}
USAGE = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
TIMINGS = {"prompt_ms": 10.0, "predicted_ms": 40.0}


def frame(delta=None, finish=None) -> dict:
    return {**ENVELOPE, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def trailing(usage=USAGE, timings=TIMINGS) -> dict:
    return {**ENVELOPE, "choices": [], "usage": usage, "timings": timings}


def sse(*frames, done=True) -> str:
    """Render frames as the wire text llama-server emits: 'data: {...}' lines, blank separators, [DONE]."""
    text = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
    return text + ("data: [DONE]\n\n" if done else "")


# A complete think-then-answer stream: two reasoning deltas, two content deltas, finish, trailing usage.
FULL_STREAM = [
    frame({"role": "assistant", "reasoning_content": "Let me "}),
    frame({"reasoning_content": "think."}),
    frame({"content": "Hel"}),
    frame({"content": "lo"}),
    frame(finish="stop"),
    trailing(),
]


class FakeResponse:
    """Stand-in for the urllib response `_open` returns: a context manager that iterates raw
    byte lines (streaming path) or is read whole (json.load path)."""
    def __init__(self, text: str):
        self.raw = text.encode("utf-8")
        self.lines = self.raw.splitlines(keepends=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self) -> bytes:
        return self.raw


class Recorder(Stage):
    """Renderer sink that records (channel, text) events and counts flushes."""
    def __init__(self):
        super().__init__(None)
        self.events: list[tuple[str, str]] = []
        self.flushed = 0

    def feed(self, channel: str, text: str) -> None:
        self.events.append((channel, text))

    def flush(self) -> None:
        self.flushed += 1


def fake_open(monkeypatch, server: LlamaServer, text: str) -> list[tuple[str, dict | None]]:
    """Route server._open to a FakeResponse over `text`; return the list of (path, body) it was called with."""
    calls: list[tuple[str, dict | None]] = []

    def _open(path, body=None):
        calls.append((path, body))
        return FakeResponse(text)

    monkeypatch.setattr(server, "_open", _open)
    return calls


# ---------------------
# parse_sse
# ---------------------

class TestParseSSE:
    def test_yields_only_data_lines_and_stops_at_done(self):
        text = ": keepalive comment\n" + sse(frame({"content": "a"}), frame(finish="stop"))
        parsed = list(parse_sse(iter(text.splitlines(keepends=True))))
        assert [f["choices"][0]["delta"] for f in parsed] == [{"content": "a"}, {}]

    def test_frames_after_done_are_not_read(self):
        text = sse(frame({"content": "a"})) + sse(frame({"content": "never"}), done=False)
        parsed = list(parse_sse(iter(text.splitlines(keepends=True))))
        assert len(parsed) == 1

    def test_stream_without_done_ends_at_eof(self):
        text = sse(frame({"content": "a"}), frame({"content": "b"}), done=False)
        assert len(list(parse_sse(iter(text.splitlines(keepends=True))))) == 2


# ---------------------
# events(): frame -> renderer events
# ---------------------

class TestEvents:
    def test_content_frame(self):
        assert list(events(frame({"content": "hi"}))) == [("content", "hi")]

    def test_reasoning_frame(self):
        assert list(events(frame({"reasoning_content": "hmm"}))) == [("reasoning", "hmm")]

    def test_reasoning_precedes_content_when_both_present(self):
        assert list(events(frame({"reasoning_content": "r", "content": "c"}))) == [("reasoning", "r"), ("content", "c")]

    def test_envelope_only_frames_yield_nothing(self):
        assert list(events(frame(finish="stop"))) == []
        assert list(events(trailing())) == []
        assert list(events({"error": {"code": 500}})) == []

    def test_empty_strings_are_not_events(self):
        assert list(events(frame({"content": "", "reasoning_content": ""}))) == []


# ---------------------
# Completion.from_frames
# ---------------------

class TestFromFrames:
    def test_full_stream_folds_channels_finish_and_trailing_usage(self):
        c = Completion.from_frames(FULL_STREAM)
        assert c.reasoning == "Let me think."
        assert c.content == "Hello"
        assert c.finish_reason == "stop"
        assert c.usage == USAGE
        assert c.timings == TIMINGS
        assert c.streamed is True

    def test_envelope_comes_from_first_frame(self):
        c = Completion.from_frames(FULL_STREAM)
        assert (c.id, c.model, c.created, c.system_fingerprint) == (
            ENVELOPE["id"], ENVELOPE["model"], ENVELOPE["created"], ENVELOPE["system_fingerprint"])

    def test_no_finish_frame_leaves_finish_reason_unknown(self):
        c = Completion.from_frames([frame({"content": "partial"})])
        assert c.finish_reason == "unknown"
        assert c.content == "partial"

    def test_no_trailing_frame_leaves_usage_and_timings_none(self):
        c = Completion.from_frames([frame({"content": "a"}), frame(finish="stop")])
        assert c.usage is None and c.timings is None

    def test_finish_reason_is_last_non_null(self):
        c = Completion.from_frames([frame({"content": "a"}, finish=None), frame(finish="length"), trailing()])
        assert c.finish_reason == "length"

    def test_frames_without_choices_key_are_skipped(self):
        c = Completion.from_frames([frame({"content": "a"}), {**ENVELOPE, "unexpected": True}, frame({"content": "b"})])
        assert c.content == "ab"

    def test_null_delta_fields_fold_as_empty(self):
        c = Completion.from_frames([frame({"content": None, "reasoning_content": None}), frame({"content": "x"})])
        assert c.content == "x" and c.reasoning == ""

    @pytest.mark.xfail(raises=AttributeError, reason="tool_calls not yet implemented", strict=True)
    def test_tool_call_deltas_baseline_single_call(self):
        """What from_frames does TODAY with streamed tool_calls deltas.

        A tool call streams as per-index fragments: the first frame carries id/name, later
        frames carry argument-string fragments; parallel calls interleave by index.
        """
        tool_stream = [
            frame({"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": "call_a1", "type": "function",
                 "function": {"name": "get_weather", "arguments": ""}}]}),
            frame({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"San"}}]}),
            frame({"tool_calls": [{"index": 0, "function": {"arguments": "tiago\"}"}}]}),
            frame(finish="tool_calls"),
            trailing(),
        ]
        c = Completion.from_frames(tool_stream)
        assert (c.content, c.reasoning) == ("", "")
        assert c.finish_reason == "tool_calls"
        assert c.usage == USAGE and c.timings == TIMINGS
        assert c.streamed is True

        # Desired behavior
        assert isinstance(c.tool_calls, list)
        assert len(c.tool_calls) == 1
        assert isinstance(c.tool_calls[0], ToolCall)
        assert c.tool_calls[0].id == "call_a1"
        assert c.tool_calls[0].index == 0
        assert c.tool_calls[0].type == "function"
        assert c.tool_calls[0].name == "get_weather"
        assert c.tool_calls[0].arguments == '{"city": "Santiago"}'

    @pytest.mark.xfail(raises=AttributeError, reason="tool_calls not yet implemented", strict=True)
    def test_tool_call_deltas_baseline_multi_call(self):
        """Tests indexing and flattening of multiple tool_calls.
        """
        # Each call is OPENED exactly once (the delta that carries id/type/name); every later delta
        # for that index carries only an arguments fragment. Two openers can share one frame.
        tool_stream = [
            frame({"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": "call_a1", "type": "function", "function": {"name": "get_weather", "arguments": ""}},
                {"index": 1, "id": "call_b2", "type": "function", "function": {"name": "get_time", "arguments": ""}},
            ]}),
            frame({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": "}}]}),
            frame({"tool_calls": [{"index": 1, "function": {"arguments": "{\"tz\": \"CLT\"}"}}]}),
            frame({"tool_calls": [{"index": 0, "function": {"arguments": "\"Santiago\"}"}}]}),
            frame(finish="tool_calls"),
            trailing(),
        ]

        # Desired behavior
        c = Completion.from_frames(tool_stream)
        assert isinstance(c.tool_calls, list)
        assert len(c.tool_calls) == 2
        assert [(t.index, t.name, t.arguments, t.id) for t in c.tool_calls] == [
            (0, "get_weather", '{"city": "Santiago"}', "call_a1"),
            (1, "get_time", '{"tz": "CLT"}', "call_b2"),
        ]

    @pytest.mark.xfail(raises=AttributeError, reason="tool_calls not yet implemented", strict=True)
    def test_tool_call_deltas_baseline_frame_spans_two_indices(self):
        """llama-server shape: the model emits calls sequentially, but the server re-parses its partial
        output per chunk and sends the diff, so ONE frame can carry the closing fragment of index 0
        together with the opener of index 1. The fold must route each element of the tool_calls
        array by its own index, not treat the frame as belonging to one call.
        """
        tool_stream = [
            frame({"role": "assistant", "content": None, "tool_calls": [
                {"index": 0, "id": "call_a1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}),
            frame({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\": \"San"}}]}),
            frame({"tool_calls": [
                {"index": 0, "function": {"arguments": "tiago\"}"}},
                {"index": 1, "id": "call_b2", "type": "function", "function": {"name": "get_time", "arguments": "{\"tz\":"}},
            ]}),
            frame({"tool_calls": [{"index": 1, "function": {"arguments": " \"CLT\"}"}}]}),
            frame(finish="tool_calls"),
            trailing(),
        ]

        # Desired behavior
        c = Completion.from_frames(tool_stream)
        assert [(t.index, t.name, t.arguments, t.id) for t in c.tool_calls] == [
            (0, "get_weather", '{"city": "Santiago"}', "call_a1"),
            (1, "get_time", '{"tz": "CLT"}', "call_b2"),
        ]



# ---------------------
# Completion.from_response / to_dict
# ---------------------

RESPONSE = {
    **{k: v for k, v in ENVELOPE.items() if k != "object"}, "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Hello", "reasoning_content": "Let me think."}}],
    "usage": USAGE, "timings": TIMINGS,
}


class TestFromResponse:
    def test_fields_map_from_chat_completion_json(self):
        c = Completion.from_response(RESPONSE)
        assert (c.content, c.reasoning, c.finish_reason) == ("Hello", "Let me think.", "stop")
        assert c.usage == USAGE and c.timings == TIMINGS
        assert c.streamed is False

    def test_missing_optional_fields_default(self):
        d = {"id": "x", "model": "m", "created": 1, "choices": [{"message": {"content": None}}]}
        c = Completion.from_response(d)
        assert (c.content, c.reasoning, c.finish_reason, c.system_fingerprint) == ("", "", "unknown", "")
        assert c.usage is None and c.timings is None

    def test_streamed_and_non_streamed_fold_to_the_same_content(self):
        assert Completion.from_frames(FULL_STREAM).to_dict()["choices"] == Completion.from_response(RESPONSE).to_dict()["choices"]

    def test_to_dict_keeps_chat_completion_layout(self):
        d = Completion.from_response(RESPONSE).to_dict()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"] == {"role": "assistant", "content": "Hello", "reasoning_content": "Let me think."}
        assert d["streamed"] is False
        assert d["usage"] == USAGE


# ---------------------
# Request.payload: optional keys present exactly when they mean something
# ---------------------

class TestRequestPayload:
    def test_model_key_absent_for_single_model_server(self):
        assert "model" not in Request.single("hi").payload()

    def test_model_key_present_in_router_mode(self):
        assert Request.single("hi", model="model-a").payload()["model"] == "model-a"

    def test_stream_options_follow_stream_flag(self):
        assert "stream_options" not in Request.single("hi").payload()
        assert Request.single("hi", stream=True).payload()["stream_options"] == {"include_usage": True}

    def test_think_flag_goes_through_chat_template_kwargs(self):
        assert Request.single("hi", think=True).payload()["chat_template_kwargs"] == {"enable_thinking": True}
        assert Request.single("hi").payload()["chat_template_kwargs"] == {"enable_thinking": False}

    def test_single_adds_system_message_only_when_given(self):
        assert Request.single("q").messages == [{"role": "user", "content": "q"}]
        assert Request.single("q", system="s").messages == [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]


# ---------------------
# LlamaServer.stream_raw / stream: the T-junction over a fake response
# ---------------------

class TestStream:
    def test_stream_raw_yields_parsed_frames_and_forces_streaming_payload(self, monkeypatch):
        server = LlamaServer("http://fake:1/")
        calls = fake_open(monkeypatch, server, sse(*FULL_STREAM))
        frames = list(server.stream_raw(Request.single("hi", stream=False)))
        assert len(frames) == len(FULL_STREAM)
        (path, body), = calls
        assert path == "/v1/chat/completions"
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}

    def test_stream_feeds_renderer_in_order_and_folds_completion(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        fake_open(monkeypatch, server, sse(*FULL_STREAM))
        sink = Recorder()
        c = server.stream(Request.single("hi"), sink)
        assert sink.events == [("reasoning", "Let me "), ("reasoning", "think."), ("content", "Hel"), ("content", "lo")]
        assert sink.flushed == 1
        assert c.content == "Hello" and c.finish_reason == "stop" and c.usage == USAGE

    def test_error_frame_mid_stream_raises_before_consumers_see_it(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        err = {"error": {"code": 500, "type": "server_error", "message": "context shift disabled"}}
        fake_open(monkeypatch, server, sse(frame({"content": "ok so far"}), err))
        sink = Recorder()
        with pytest.raises(LlamaServerError) as e:
            server.stream(Request.single("hi"), sink)
        assert (e.value.code, e.value.type, e.value.message) == (500, "server_error", "context shift disabled")
        assert sink.events == [("content", "ok so far")]
        assert sink.flushed == 0  # raised out of the loop: flush never reached

    def test_cancel_after_first_frame_stops_reading_and_marks_cancelled(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        fake_open(monkeypatch, server, sse(*FULL_STREAM))
        sink = Recorder()
        seen = []
        c = server.stream(Request.single("hi"), sink, cancelled=lambda: len(seen) >= 1 or seen.append(1))
        # cancelled() is checked before each frame is folded: the first frame lands, the second trips it
        assert len(sink.events) == 1
        assert sink.flushed == 1
        assert c.finish_reason == "cancelled"
        assert c.streamed is True

    def test_cancel_before_any_frame_yields_empty_cancelled_completion(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        fake_open(monkeypatch, server, sse(*FULL_STREAM))
        sink = Recorder()
        c = server.stream(Request.single("hi", model="model-a"), sink, cancelled=lambda: True)
        assert sink.events == [] and sink.flushed == 1
        assert c.content == "" and c.finish_reason == "cancelled" and c.model == "model-a"
        assert c.usage is None


# ---------------------
# LlamaServer.complete / models / props / max_context
# ---------------------

class TestJsonEndpoints:
    def test_complete_forces_stream_off_and_drops_stream_options(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        calls = fake_open(monkeypatch, server, json.dumps(RESPONSE))
        c = server.complete(Request.single("hi", stream=True))
        (_, body), = calls
        assert body["stream"] is False and "stream_options" not in body
        assert c.content == "Hello" and c.streamed is False

    def test_models_and_max_context_read_router_listing(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        listing = {"data": [
            {"id": "model-a", "status": {"value": "loaded", "args": ["llama-server", "--ctx-size", "32768"]}},
            {"id": "model-b", "status": {"value": "unloaded", "args": []}},
        ]}
        calls = fake_open(monkeypatch, server, json.dumps(listing))
        assert server.models() == ["model-a", "model-b"]
        assert server.max_context() == {"model-a": 32768}
        assert all(path == "/models" and body is None for path, body in calls)

    def test_props_quotes_model_id_in_router_mode(self, monkeypatch):
        server = LlamaServer("http://fake:1")
        calls = fake_open(monkeypatch, server, json.dumps({"n_ctx": 4096}))
        assert server.props() == {"n_ctx": 4096}
        assert server.props("qwen/3 b") == {"n_ctx": 4096}
        assert [p for p, _ in calls] == ["/props", "/props?model=qwen%2F3%20b"]


# ---------------------
# Error taxonomy in _open: LlamaUnreachable vs LlamaServerError
# ---------------------

def http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://fake:1/x", code, "msg", {}, io.BytesIO(body.encode()))


def raising(exc):
    """urlopen stand-in that raises `exc` when called."""
    def _urlopen(req, timeout):
        raise exc
    return _urlopen


class TestOpenErrors:
    def test_json_error_body_maps_to_server_error(self, monkeypatch):
        body = json.dumps({"error": {"code": 400, "type": "invalid_request_error", "message": "model missing"}})
        monkeypatch.setattr("urllib.request.urlopen", raising(http_error(400, body)))
        with pytest.raises(LlamaServerError) as e:
            LlamaServer("http://fake:1")._open("/v1/chat/completions", {})
        assert (e.value.code, e.value.type, e.value.message) == (400, "invalid_request_error", "model missing")

    def test_non_json_error_body_maps_to_http_error_type(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", raising(http_error(502, "<html>bad gateway</html>")))
        with pytest.raises(LlamaServerError) as e:
            LlamaServer("http://fake:1")._open("/models")
        assert (e.value.code, e.value.type) == (502, "http_error")
        assert "bad gateway" in e.value.message

    def test_connection_failure_maps_to_unreachable(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", raising(urllib.error.URLError("Connection refused")))
        with pytest.raises(LlamaUnreachable) as e:
            LlamaServer("http://fake:1")._open("/models")
        assert "Connection refused" in str(e.value)

    def test_timeout_maps_to_unreachable_with_timeout_value(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", raising(TimeoutError()))
        with pytest.raises(LlamaUnreachable) as e:
            LlamaServer("http://fake:1", timeout=2.5)._open("/models")
        assert "timeout after 2.5s" in str(e.value)

    def test_base_url_trailing_slash_is_normalised(self):
        assert LlamaServer("http://fake:1/").base_url == "http://fake:1"
