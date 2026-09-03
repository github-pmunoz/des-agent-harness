"""Unit tests for the desh_chat command surface : 
Settings, Command dispatch, and PromptUser's slash-command parsing.
"""
import pytest

from desh_chat.events import Command, DisplayHistory, Exit, Info, MaybeRegenerate, PromptUser, UserMessage, Warn


def run_command(make_state, command, args, **state_overrides):
    """Unit-level: resolves Command.execute's two-hop chain by hand
    (Command -> Info|Warn|Exit -> MaybeRegenerate) instead of going
    through the Engine, which would hit real input() at the next PromptUser
    step. Returns (final_state, first_event, terminal_event):
      first_event    Info on success / current-value display, Warn on a
                      rejected command, or Exit for /exit /quit.
      terminal_event Always MaybeRegenerate or Exit
    """
    state = make_state(**state_overrides)
    state, events = Command(command=command, args=args).execute(state)
    assert len(events) == 1, f"/{command} {args!r} emitted {len(events)} events: {events}"
    first = events[0]

    if isinstance(first, Exit):
        state, events = first.execute(state)
        assert events == [], f"Exit emitted further events: {events}"
        return state, first, first

    assert isinstance(first, (Info, Warn)), f"/{command} {args!r} emitted unexpected {first}"
    state, events = first.execute(state)
    assert len(events) == 1 and isinstance(events[0], MaybeRegenerate), (
        f"/{command} {args!r} did not settle at MaybeRegenerate: {events}"
    )
    return state, first, events[0]


class TestModel:
    def test_no_arg_shows_current(self, make_state, capsys):
        final, first, _ = run_command(make_state, "model", "")
        assert isinstance(first, Info)
        assert final.settings.model in capsys.readouterr().out

    def test_valid_switches(self, make_state):
        final, first, _ = run_command(make_state, "model", "model-b")
        assert final.settings.model == "model-b"
        assert isinstance(first, Info)

    def test_invalid_rejected(self, make_state, capsys):
        final, first, _ = run_command(make_state, "model", "nonexistent")
        assert isinstance(first, Warn)
        assert "not found" in capsys.readouterr().out

    def test_multi_arg_rejected(self, make_state):
        final, first, _ = run_command(make_state, "model", "model-a extra")
        assert isinstance(first, Warn)


class TestModels:
    def test_lists_all(self, make_state, capsys):
        run_command(make_state, "models", "")
        out = capsys.readouterr().out
        assert "model-a" in out and "model-b" in out

    def test_takes_no_arg(self, make_state):
        _, first, _ = run_command(make_state, "models", "foo")
        assert isinstance(first, Warn)


class TestTemperature:
    def test_no_arg_shows_current(self, make_state, capsys):
        final, first, _ = run_command(make_state, "temperature", "")
        assert isinstance(first, Info)
        assert str(final.settings.temperature) in capsys.readouterr().out

    @pytest.mark.parametrize("args,expected", [("0.0", 0.0), ("0.7", 0.7), ("2.0", 2.0)])
    def test_valid_sets(self, make_state, args, expected):
        final, first, _ = run_command(make_state, "temperature", args)
        assert final.settings.temperature == expected
        assert isinstance(first, Info)

    @pytest.mark.parametrize("args", ["5.0", "-1", "2.01"])
    def test_out_of_range_rejected(self, make_state, args):
        _, first, _ = run_command(make_state, "temperature", args)
        assert isinstance(first, Warn)

    def test_non_numeric_rejected(self, make_state):
        _, first, _ = run_command(make_state, "temperature", "abc")
        assert isinstance(first, Warn)


class TestContext:
    """Bounded by the (real, per-model) InferenceEngine.max_context"""

    def test_no_arg_shows_current(self, make_state):
        _, first, _ = run_command(make_state, "context", "")
        assert isinstance(first, Info)

    def test_valid_sets(self, make_state):
        final, first, _ = run_command(make_state, "context", "8192")
        assert final.settings.context == 8192
        assert isinstance(first, Info)

    def test_bounded_by_current_models_max_context(self, make_state):
        # model-a's max_context is 32768 (see conftest.MAX_CONTEXT)
        _, first, _ = run_command(make_state, "context", "999999999")
        assert isinstance(first, Warn)

    def test_bound_is_per_model(self, make_state):
        # model-b's max_context is 16384 — a value valid for model-a must be
        # rejected once the active model is model-b.
        switched, _, _ = run_command(make_state, "model", "model-b")
        _, first, _ = run_command(make_state, "context", "20000", settings=switched.settings)
        assert isinstance(first, Warn)


class TestMaxTurnTokens:
    def test_no_arg_shows_current(self, make_state):
        _, first, _ = run_command(make_state, "max_turn_tokens", "")
        assert isinstance(first, Info)

    def test_valid_sets(self, make_state):
        final, first, _ = run_command(make_state, "max_turn_tokens", "4096")
        assert final.settings.max_turn_tokens == 4096
        assert isinstance(first, Info)


class TestThink:
    def test_think_enables(self, make_state):
        final, first, _ = run_command(make_state, "think", "")
        assert final.settings.think is True
        assert isinstance(first, Info)

    def test_think_takes_no_arg(self, make_state):
        _, first, _ = run_command(make_state, "think", "yes")
        assert isinstance(first, Warn)

    def test_nothink_disables(self, make_state):
        on, _, _ = run_command(make_state, "think", "")
        off, first, _ = run_command(make_state, "nothink", "", settings=on.settings)
        assert off.settings.think is False
        assert isinstance(first, Info)


class TestHistory:
    """/history dispatches to DisplayHistory, not Info/Warn — a third
    terminal-ish shape run_command() doesn't cover, so this drives Command
    and DisplayHistory by hand instead of reusing that helper.
    """

    def test_takes_no_arg(self, make_state):
        state = make_state()
        _, events = Command(command="history", args="all").execute(state)
        assert len(events) == 1
        assert isinstance(events[0], Warn)

    def test_empty_history_prints_header_only(self, make_state, capsys):
        state = make_state()
        _, events = Command(command="history", args="").execute(state)
        assert len(events) == 1
        assert isinstance(events[0], DisplayHistory)

        state, events = events[0].execute(state)
        out = capsys.readouterr().out
        assert "History:" in out
        assert "USER:" not in out and "ASSISTANT:" not in out
        assert len(events) == 1 and isinstance(events[0], MaybeRegenerate)

    def test_lists_turns_in_order(self, make_state, capsys):
        from desh_chat.state import ChatHistory, Turn
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
        assert isinstance(events[0], MaybeRegenerate)

    def test_summary_turn_shown_without_user_assistant_prefix(self, make_state, capsys):
        from desh_chat.state import ChatHistory, Turn
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
        from desh_chat.state import ChatHistory, Turn
        history = ChatHistory().append(Turn("cancelled question", "partial answer", cancelled=True))
        state = make_state(history=history)
        DisplayHistory().execute(state)
        out = capsys.readouterr().out
        assert "cancelled question" in out
        assert "partial answer" in out


class TestExitQuit:
    @pytest.mark.parametrize("command", ["exit", "quit"])
    def test_stops_the_run(self, make_state, command):
        final, first, terminal = run_command(make_state, command, "")
        assert final.running is False
        assert isinstance(first, Exit) and isinstance(terminal, Exit)

    @pytest.mark.parametrize("command", ["exit", "quit"])
    def test_takes_no_arg(self, make_state, command):
        final, first, terminal = run_command(make_state, command, "now")
        assert final.running is True
        assert isinstance(first, Warn)
        assert isinstance(terminal, MaybeRegenerate)


class TestUnknownAndEdgeCases:
    def test_unknown_command_rejected(self, make_state, capsys):
        _, first, _ = run_command(make_state, "bogus", "")
        assert isinstance(first, Warn)
        assert "Unknown command" in capsys.readouterr().out

    def test_empty_command_rejected(self, make_state):
        _, first, _ = run_command(make_state, "", "")
        assert isinstance(first, Warn)

    def test_command_names_are_case_sensitive_by_design(self, make_state, capsys):
        # deliberate choice (2026-09-02): /MoDeL must not be accepted
        run_command(make_state, "Model", "")
        assert "Unknown command" in capsys.readouterr().out


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
        # (TestUnknownAndEdgeCases) is what actually rejects /Model.
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
