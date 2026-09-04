"""Unit tests for the desh_chat command surface: Settings and Command
dispatch. PromptUser's own slash-command parsing is a single-event
isolation test — see test_chat_unit.py.
"""
import pytest

from desh_chat.events import Command, DisplayHistory, Exit, Info, MaybeRegenerate, Warn


def run_command(make_state, command, args, **state_overrides):
    """Unit-level: drives Command.execute() and resolves its immediate shape
    by hand instead of going through the Engine, which would hit real
    input() at the next PromptUser step.

    Two shapes, not one chain:
      /exit, /quit  — a real chain: Command -> [Exit] -> [Info] -> [].
        Exit sets running=False and never touches MaybeRegenerate.
      everything else — a 2-fan in one Command.execute() step:
        [display, MaybeRegenerate()], display in (Info, Warn, DisplayHistory).
        display is executed here (a leaf, settles at []); MaybeRegenerate is
        NOT executed — running it would hit real input() at the next
        PromptUser when state.running is True — only its type is checked.
        (See TestMaybeRegenerate for direct coverage of its own execute().)

    Returns (final_state, first_event, terminal_event):
      first_event    Info/Warn/DisplayHistory on the normal path, or Exit
                      for /exit /quit.
      terminal_event MaybeRegenerate (unexecuted) on the normal path, or
                      the same Exit instance as first_event.
    """
    state = make_state(**state_overrides)
    state, events = Command(command=command, args=args).execute(state)

    if len(events) == 1 and isinstance(events[0], Exit):
        exit_event = events[0]
        state, events = exit_event.execute(state)
        assert len(events) == 1 and isinstance(events[0], Info), (
            f"/{command} {args!r} Exit emitted unexpected {events}"
        )
        state, events = events[0].execute(state)
        assert events == [], f"/{command} {args!r} Exit's Info emitted further events: {events}"
        return state, exit_event, exit_event

    assert len(events) == 2, f"/{command} {args!r} emitted {len(events)} events: {events}"
    display, regen = events
    assert isinstance(display, (Info, Warn, DisplayHistory)), (
        f"/{command} {args!r} emitted unexpected {display}"
    )
    assert isinstance(regen, MaybeRegenerate), (
        f"/{command} {args!r} did not pair with MaybeRegenerate: {events}"
    )
    state, further = display.execute(state)
    assert further == [], f"/{command} {args!r} {display} emitted further events: {further}"
    return state, display, regen


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
    """/history dispatches to DisplayHistory, which fans exactly like
    Info/Warn now (DisplayHistory settles at [], MaybeRegenerate is the
    sibling) — run_command() covers it like every other command.
    DisplayHistory's own rendering behavior (turn order, summary turns,
    cancelled turns) is tested directly in test_chat_unit.py.
    """

    def test_takes_no_arg(self, make_state):
        _, first, _ = run_command(make_state, "history", "all")
        assert isinstance(first, Warn)

    def test_empty_history_prints_header_only(self, make_state, capsys):
        _, first, _ = run_command(make_state, "history", "")
        assert isinstance(first, DisplayHistory)
        out = capsys.readouterr().out
        assert "History:" in out
        assert "USER:" not in out and "ASSISTANT:" not in out


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
