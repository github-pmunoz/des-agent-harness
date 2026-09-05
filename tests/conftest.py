import pytest

from desh.llama.client import Completion, Request
from desh_chat.state import ChatHistory, ChatState, InferenceEngine, Settings

MODELS = ["model-a", "model-b"]
MAX_CONTEXT = {"model-a": 32768, "model-b": 16384}
PORT = 8012


class FakeServer:
    """Stand-in for LlamaServer: no network, no real streaming.

    Each call to stream()/complete() pops one script entry (a dict) and
    returns a Completion built from it, so a test can queue up exactly the
    responses a scenario needs. With no script queued, returns a plain
    "ok"/"stop" completion so tests that don't care about the response
    content still get something well-formed.

    stream() feeds the whole content to the renderer as a single chunk
    (feed + flush), same contract LlamaServer.stream() honors, and records
    every request it was called with so tests can assert on what was sent.
    """
    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls: list[tuple[str, Request]] = []

    def _next(self, default_content: str, default_finish: str):
        """Script entry keys: content, finish_reason, reasoning, usage (all optional).
        usage defaults to None — the "no trailing usage frame" case — so token pricing
        falls back to the heuristic unless a test opts in."""
        if self.script:
            spec = self.script.pop(0)
        else:
            spec = {}
        return (
            spec.get("content", default_content),
            spec.get("finish_reason", default_finish),
            spec.get("reasoning", ""),
            spec.get("usage"),
        )

    def stream(self, req: Request, renderer, cancelled=lambda: False) -> Completion:
        self.calls.append(("stream", req))
        content, finish_reason, reasoning, usage = self._next("ok", "stop")
        if content and finish_reason != "cancelled":
            renderer.feed("content", content)
        renderer.flush()
        return Completion(
            id="fake-stream", model=req.model or "", created=0, system_fingerprint="",
            content=content, reasoning=reasoning, finish_reason=finish_reason,
            usage=usage, timings=None, streamed=True,
        )

    def complete(self, req: Request) -> Completion:
        self.calls.append(("complete", req))
        content, finish_reason, reasoning, usage = self._next("summary", "stop")
        return Completion(
            id="fake-complete", model=req.model or "", created=0, system_fingerprint="",
            content=content, reasoning=reasoning, finish_reason=finish_reason,
            usage=usage, timings=None, streamed=False,
        )


class FakeESCWatcher:
    """No-op stand-in for ESCWatcher: no termios, no thread, no real stdin access.

    StreamCompletion decides "cancelled" from Completion.finish_reason, not
    from watcher.interrupted directly — the real cancel path is exercised via
    FakeServer's script (finish_reason="cancelled"), not via this fake.
    """
    def __init__(self):
        self.interrupted = False

    def start(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def make_state():
    """Factory for an isolated ChatState — no live server, no network.

    InferenceEngine.server defaults to a fresh FakeServer() so any test can
    drive the completion path without overriding inference; pass
    inference=... (or server=... below via FakeServer(script=...)) to script
    specific responses. Keeping models/max_context fixed here means these
    tests don't depend on the router being up.
    """
    def _make(**overrides):
        settings = overrides.pop("settings", None) or Settings(
            model=MODELS[0],
            temperature=0.3,
            think=False,
            context=16384,
            max_turn_tokens=8192,
        )
        inference = overrides.pop("inference", None) or InferenceEngine(
            models=MODELS, max_context=MAX_CONTEXT, server=FakeServer(), port=PORT
        )
        base = dict(
            settings=settings,
            inference=inference,
            history=ChatHistory(),
            running=True,
            system_prompt="You are a helpful assistant.",
            completions_log=None,
        )
        base.update(overrides)
        return ChatState(**base)
    return _make


@pytest.fixture
def no_esc_watcher(monkeypatch):
    """Replace ESCWatcher with a no-op fake for any test that reaches StreamCompletion.

    Real ESCWatcher.start() spawns a thread that calls sys.stdin.fileno() —
    under pytest's captured stdin this raises inside the thread on every
    single call, and stop() joins with a timeout. Opt in explicitly rather
    than autouse, so tests that don't touch StreamCompletion aren't affected.
    """
    monkeypatch.setattr("desh_chat.events.ESCWatcher", FakeESCWatcher)
