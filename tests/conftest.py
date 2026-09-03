import pytest

from desh_chat.state import ChatHistory, ChatState, InferenceEngine, Settings

MODELS = ["model-a", "model-b"]
MAX_CONTEXT = {"model-a": 32768, "model-b": 16384}


@pytest.fixture
def make_state():
    """Factory for a isolated ChatState — no live server, no network.

    InferenceEngine.server is None: the command wiring never touches it
    (only Completion, once ported, will). Keeping models/max_context fixed
    here means these tests don't depend on the router being up.
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
            models=MODELS, max_context=MAX_CONTEXT, server=None
        )
        base = dict(
            settings=settings,
            inference=inference,
            history=ChatHistory(),
            running=True,
            system_prompt="You are a helpful assistant.",
            completions_log="",
        )
        base.update(overrides)
        return ChatState(**base)
    return _make
