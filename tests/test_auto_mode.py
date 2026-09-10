"""
Auto mode: Settings.auto defaults off; Ctrl+C toggles it off instead of exiting when it is on;
delegate children inherit it.
"""
from dataclasses import replace

from desh_chat.display import Info
from desh_chat.events import Exit, MaybeRegenerate
from desh_chat.handlers import on_interrupt
from desh_chat.state import Settings
from desh_chat.delegate import child_settings


def make_settings(**overrides) -> Settings:
    base = dict(model="model-a", temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
    base.update(overrides)
    return Settings(**base)


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


class TestDelegateInheritance:
    def test_child_settings_preserves_auto_true(self):
        child = child_settings(make_settings(auto=True))
        assert child.auto is True

    def test_child_settings_preserves_auto_false(self):
        child = child_settings(make_settings(auto=False))
        assert child.auto is False
