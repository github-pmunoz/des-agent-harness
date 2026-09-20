"""
Auto mode: Settings.auto defaults off; Ctrl+C toggles it off instead of exiting when it is on;
delegate children inherit it.
"""
from dataclasses import replace

from desh.llama.wire import ToolCall
from desh_chat.display import Info
from desh_chat.events import Exit, MaybeRegenerate, TurnEnd, TurnStart
from desh_chat.handlers import on_error, on_interrupt
from desh_chat.state import PendingTurn, Round, Settings, ToolResult, StopReason
from desh_chat.delegate import child_settings


def make_settings(**overrides) -> Settings:
    base = dict(model="model-a", temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
    base.update(overrides)
    return Settings(**base)


def pending_turn() -> PendingTurn:
    """A mid-turn pending turn: user message plus one completed tool round (same shape as
    test_turn_loop.py's ROUND)."""
    call = ToolCall(index=0, id="call_0", type="function", name="get_weather", arguments='{"city": "Santiago"}')
    return PendingTurn("q").add_round(Round("", (call,), (ToolResult(tool_call_id=call.id, name=call.name, content="sunny"),), tokens=30))


class TestSettingsDefault:
    def test_auto_defaults_to_false(self):
        assert make_settings().auto is False


class TestOnInterrupt:
    def test_auto_on_disables_auto_mode_instead_of_exiting(self, make_state):
        state = make_state(settings=make_settings(auto=True))
        events = on_interrupt(state)
        assert not any(isinstance(e, Exit) for e in events)
        infos = [e for e in events if isinstance(e, Info)]
        assert infos and any("auto mode off" in e.text for e in infos)
        assert any(isinstance(e, MaybeRegenerate) for e in events)
        # the returned state (after the events run) has auto off
        final = state
        for e in events:
            final, _ = e.execute(final)
        assert final.settings.auto is False

    def test_auto_off_keeps_the_exit_path(self, make_state):
        state = make_state(settings=make_settings(auto=False))
        events = on_interrupt(state)
        assert [type(e) for e in events] == [Info, Exit]
        assert events[0].text == "~ Interrupted"

    def test_auto_off_with_pending_turn_keeps_the_exit_path(self, make_state):
        state = make_state(settings=make_settings(auto=False), pending=pending_turn())
        events = on_interrupt(state)
        assert [type(e) for e in events] == [Info, Exit]
        assert events[0].text == "~ Interrupted"

    def test_auto_on_with_pending_turn_closes_it_as_cancelled(self, make_state):
        state = make_state(settings=make_settings(auto=True), pending=pending_turn())
        events = on_interrupt(state)
        final = state
        for e in events:
            final, _ = e.execute(final)
        assert final.settings.auto is False
        assert final.pending is None
        turn = final.history.turns[-1]
        assert turn.stop == StopReason.INTERRUPT and turn.visible is False

    def test_auto_on_with_pending_turn_leaves_loop_head_safe(self, make_state):
        # The bug symptom: with the pending turn left open, the MaybeRegenerate that
        # on_interrupt returns leads straight into TurnStart, whose assertion trips.
        state = make_state(settings=make_settings(auto=True), pending=pending_turn())
        events = on_interrupt(state)
        final = state
        for e in events:
            final, _ = e.execute(final)
        # the loop head must be reachable: MaybeRegenerate -> TurnStart without raising
        final, _ = MaybeRegenerate().execute(final)
        final, _ = TurnStart().execute(final)
        assert final.pending is not None


class TestOnError:
    def test_on_error_with_filled_pending_turn_closes_it_as_cancelled(self, make_state):
        state = make_state(pending=pending_turn())
        events = on_error(Info("x"), ValueError("boom"), state)
        ends = [e for e in events if isinstance(e, TurnEnd)]
        assert ends, "on_error must close the pending turn with a TurnEnd"
        assert ends[0].stop == StopReason.ERROR
        final = state
        for e in events:
            final, _ = e.execute(final)
        assert final.pending is None
        turn = final.history.turns[-1]
        assert turn.stop == StopReason.ERROR and turn.visible is False

    def test_on_error_with_filled_pending_turn_leaves_loop_head_safe(self, make_state):
        # The bug symptom: with the pending turn left open, the MaybeRegenerate that
        # on_error returns leads straight into TurnStart, whose assertion trips.
        state = make_state(pending=pending_turn())
        events = on_error(Info("x"), ValueError("boom"), state)
        final = state
        for e in events:
            final, _ = e.execute(final)
        # the loop head must be reachable: MaybeRegenerate -> TurnStart without raising
        final, _ = MaybeRegenerate().execute(final)
        final, _ = TurnStart().execute(final)
        assert final.pending is not None

    def test_on_error_with_empty_placeholder_goes_to_loop_head(self, make_state):
        state = make_state(pending=PendingTurn())
        events = on_error(Info("x"), ValueError("boom"), state)
        assert [type(e) for e in events] == [MaybeRegenerate]
        assert not any(isinstance(e, TurnEnd) for e in events)

    def test_on_error_with_no_pending_goes_to_loop_head(self, make_state):
        state = make_state(pending=None)
        events = on_error(Info("x"), ValueError("boom"), state)
        assert [type(e) for e in events] == [MaybeRegenerate]
        assert not any(isinstance(e, TurnEnd) for e in events)


class TestDelegateInheritance:
    def test_child_settings_preserves_auto_true(self):
        child = child_settings(make_settings(auto=True))
        assert child.auto is True

    def test_child_settings_preserves_auto_false(self):
        child = child_settings(make_settings(auto=False))
        assert child.auto is False
