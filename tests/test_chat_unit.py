"""Unit tests for desh_chat's single-event types, driven in isolation
(state, events = Event(...).execute(state)), no chain-walking.

Scope mirrors events.py's own section boundaries: everything that is not
turn logic (UserMessage, StreamCompletion, AppendTurn, MaybeCompact,
CompactHistory — test_chat_chain.py) and not Command's own dispatch
(test_command_surface.py). That leaves the # Display section (Info, Warn,
DisplayHistory, DisplayStats), Exit, MaybeRegenerate, LogCompletion, and
PromptUser's own parsing.

Info/Warn/DisplayHistory/DisplayStats/LogCompletion are pure sinks: execute()
returns [] and never mutates state. MaybeRegenerate is the only thing that
reschedules PromptUser; Exit hands off to its own Info("Goodbye!") and never
touches MaybeRegenerate.
"""
from conftest import MODELS, PORT

from desh.llama.client import Completion, Logger, Request
from desh_chat.events import (
    Command, DisplayHistory, DisplayStats, Exit, Info, LogCompletion,
    MaybeRegenerate, PromptUser, UserMessage, Warn,
)
from desh_chat.state import ChatHistory, Turn


class TestInfo:
    def test_prints_and_settles(self, make_state, capsys):
        new_state, events = Info("some notice").execute(make_state())
        assert "some notice" in capsys.readouterr().out
        assert events == []


class TestWarn:
    def test_prints_and_settles(self, make_state, capsys):
        new_state, events = Warn("something's off").execute(make_state())
        assert "something's off" in capsys.readouterr().out
        assert events == []


class TestDisplayHistory:
    def test_lists_turns_in_order(self, make_state, capsys):
        history = ChatHistory().append(Turn("first question", "first answer")) \
                                .append(Turn("second question", "second answer"))
        state = make_state(history=history)
        _, events = DisplayHistory().execute(state)
        out = capsys.readouterr().out
        assert out.index("first question") < out.index("second question")
        assert "USER: first question" in out
        assert "ASSISTANT: first answer" in out
        assert "USER: second question" in out
        assert "ASSISTANT: second answer" in out
        assert events == []

    def test_summary_turn_shown_without_user_assistant_prefix(self, make_state, capsys):
        history = ChatHistory().append(Turn("Summary of the earlier conversation: talked about X", "Understood.", summary=True))
        state = make_state(history=history)
        DisplayHistory().execute(state)
        out = capsys.readouterr().out
        assert "Summary of the earlier conversation: talked about X" in out
        assert "USER:" not in out
        assert "Understood." not in out  # the synthetic assistant ack is not displayed for summary turns

    def test_cancelled_turns_are_still_shown(self, make_state, capsys):
        """DisplayHistory is a raw dump of state.history.turns — unlike the
        model-context path (view()/since_last_summary()), it does not filter
        cancelled turns out; this is the one place they're meant to be visible.
        """
        history = ChatHistory().append(Turn("cancelled question", "partial answer", cancelled=True))
        state = make_state(history=history)
        DisplayHistory().execute(state)
        out = capsys.readouterr().out
        assert "cancelled question" in out
        assert "partial answer" in out


class TestDisplayStats:
    def test_prints_context_and_session_figures(self, make_state, capsys):
        from desh.llama.tokens import estimate_tokens
        history = ChatHistory().append(Turn("q", "a"))
        state = make_state(history=history)
        sys_tokens = estimate_tokens(state.system_prompt)
        expected_window = sys_tokens + state.history.window_tokens()
        expected_total = sys_tokens + state.history.get_total_tokens()

        _, events = DisplayStats().execute(state)

        out = capsys.readouterr().out
        assert f"Context: {expected_window} / {state.settings.context} tokens" in out
        assert f"Session: {expected_total}" in out
        assert events == []  # pure sink — MaybeRegenerate (not DisplayStats) schedules PromptUser


class TestExit:
    def test_sets_running_false_and_hands_off_to_info(self, make_state):
        new_state, events = Exit().execute(make_state())
        assert new_state.running is False
        assert len(events) == 1 and isinstance(events[0], Info)

    def test_info_settles_and_prints_goodbye(self, make_state, capsys):
        new_state, events = Exit().execute(make_state())
        _, further = events[0].execute(new_state)
        assert "Goodbye!" in capsys.readouterr().out
        assert further == []


class TestMaybeRegenerate:
    def test_running_fans_to_stats_then_prompt_in_order(self, make_state):
        state = make_state()
        assert state.running is True
        _, events = MaybeRegenerate().execute(state)
        # order matters: DisplayStats must execute before the next PromptUser blocks
        assert [type(e) for e in events] == [DisplayStats, PromptUser]

    def test_not_running_settles(self, make_state):
        state = make_state(running=False)
        new_state, events = MaybeRegenerate().execute(state)
        assert events == []
        assert new_state is state


class TestLogCompletion:
    def test_records_when_a_logger_is_configured(self, make_state, tmp_path):
        import json
        log_path = tmp_path / "completions.jsonl"
        state = make_state(completions_log=Logger(str(log_path)))
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0])
        completion = Completion(id="x", model=MODELS[0], created=0, system_fingerprint="",
                                 content="hello", reasoning="", finish_reason="stop",
                                 usage=None, timings=None, streamed=False)
        new_state, events = LogCompletion(request=req, completion=completion, port=PORT).execute(state)
        assert events == []
        assert new_state is state  # no state mutation — pure side effect
        record = json.loads(log_path.read_text().strip())
        assert record["port"] == PORT
        assert record["response"]["choices"][0]["message"]["content"] == "hello"

    def test_no_op_when_no_logger_configured(self, make_state):
        state = make_state(completions_log=None)
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0])
        completion = Completion(id="x", model=MODELS[0], created=0, system_fingerprint="",
                                 content="hello", reasoning="", finish_reason="stop",
                                 usage=None, timings=None, streamed=False)
        new_state, events = LogCompletion(request=req, completion=completion, port=PORT).execute(state)
        assert events == []
        assert new_state is state


class TestPromptUserParsing:
    """Slash-vs-plain-text detection and command/args splitting."""

    def test_slash_input_becomes_command(self, make_state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "/model model-b")
        state, events = PromptUser().execute(make_state())
        assert len(events) == 1
        assert isinstance(events[0], Command)
        assert events[0].command == "model"
        assert events[0].args == "model-b"

    def test_slash_input_no_args(self, make_state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "/models")
        _, events = PromptUser().execute(make_state())
        assert events[0].command == "models"
        assert events[0].args == ""

    def test_command_name_not_lowered(self, make_state, monkeypatch):
        # PromptUser does not lowercase — Command's own case-sensitivity
        # (test_command_surface.py's TestUnknownAndEdgeCases) is what
        # actually rejects /Model.
        monkeypatch.setattr("builtins.input", lambda prompt="": "/Model foo")
        _, events = PromptUser().execute(make_state())
        assert events[0].command == "Model"

    def test_plain_text_becomes_user_message(self, make_state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "hello there")
        _, events = PromptUser().execute(make_state())
        assert len(events) == 1
        assert isinstance(events[0], UserMessage)
        assert events[0].message == "hello there"

    def test_empty_input_regenerates(self, make_state, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        _, events = PromptUser().execute(make_state())
        assert isinstance(events[0], MaybeRegenerate)

    def test_eof_exits(self, make_state, monkeypatch):
        def raise_eof(prompt=""):
            raise EOFError
        monkeypatch.setattr("builtins.input", raise_eof)
        _, events = PromptUser().execute(make_state())
        assert isinstance(events[0], Exit)
