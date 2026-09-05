"""
Session persistence: pure to_dict/from_dict on Turn/ChatHistory, the LoadSession /
SaveSession events, and the chain hooks that emit SaveSession on every history change.
"""
import json
import os

import pytest

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh_chat.events import (
    AppendTurn, CompactHistory, Error, Info, LoadSession, SaveSession, Warn,
)
from desh_chat.state import ChatHistory, InferenceEngine, Turn


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


def sample_history() -> ChatHistory:
    return (ChatHistory()
            .append(Turn("q1", "a1", tokens=40))
            .append(Turn("partial", "…", tokens=5, cancelled=True))
            .append(Turn(ChatHistory.SUMMARY_PREFIX + "s", ChatHistory.SUMMARY_ACK, tokens=12, summary=True))
            .append(Turn("q2", "a2", tokens=60)))


# ---------------------
# Serialization (pure)
# ---------------------

class TestSerialization:
    def test_turn_round_trips_all_fields(self):
        t = Turn("u", "a", tokens=7, cancelled=True, summary=True)
        assert Turn.from_dict(t.to_dict()) == t

    def test_turn_from_dict_with_zero_tokens_reprices_by_heuristic(self):
        # tokens=0 is "unpriced" everywhere else too; a file that carries 0 gets the same treatment
        t = Turn.from_dict({"user": "hello there", "assistant": "general kenobi", "tokens": 0})
        assert t.tokens == Turn("hello there", "general kenobi").tokens

    def test_history_round_trips_and_stays_a_tuple(self):
        h = sample_history()
        back = ChatHistory.from_dict(h.to_dict())
        assert back == h
        assert isinstance(back.turns, tuple)

    def test_round_trip_survives_json(self):
        h = sample_history()
        assert ChatHistory.from_dict(json.loads(json.dumps(h.to_dict()))) == h

    def test_to_dict_carries_the_format_version(self):
        assert ChatHistory().to_dict()["version"] == ChatHistory.SESSION_FORMAT

    def test_from_dict_rejects_unknown_version(self):
        with pytest.raises(ValueError):
            ChatHistory.from_dict({"version": 99, "turns": []})

    def test_from_dict_rejects_missing_turns(self):
        with pytest.raises(KeyError):
            ChatHistory.from_dict({"version": ChatHistory.SESSION_FORMAT})

    def test_from_dict_ignores_extra_keys_such_as_meta(self):
        doc = {**sample_history().to_dict(), "meta": {"model": "x"}}
        assert ChatHistory.from_dict(doc) == sample_history()


# ---------------------
# LoadSession
# ---------------------

class TestLoadSession:
    def test_no_session_file_is_a_silent_noop(self, make_state):
        state = make_state(session_file=None)
        new_state, events = LoadSession().execute(state)
        assert new_state is state
        assert events == []

    def test_missing_file_starts_a_new_session(self, make_state, tmp_path):
        path = str(tmp_path / "s.json")
        state = make_state(session_file=path)
        new_state, events = LoadSession().execute(state)
        assert new_state.history == ChatHistory()
        assert len(events) == 1 and isinstance(events[0], Info)
        assert "New session" in events[0].text
        assert not os.path.exists(path)     # nothing written until a turn happens

    def test_existing_file_restores_history(self, make_state, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**sample_history().to_dict(), "meta": {"model": "other"}}))
        state = make_state(session_file=str(path))
        new_state, events = LoadSession().execute(state)
        assert new_state.history == sample_history()
        assert isinstance(events[0], Info) and "4 turns" in events[0].text

    def test_restored_history_keeps_window_semantics(self, make_state, tmp_path):
        # summary + cancelled flags survive, so the window is exactly what it was before exit
        path = tmp_path / "s.json"
        path.write_text(json.dumps(sample_history().to_dict()))
        new_state, _ = LoadSession().execute(make_state(session_file=str(path)))
        assert new_state.history.window_tokens() == 12 + 60
        assert new_state.history.get_total_tokens() == 40 + 12 + 60

    @pytest.mark.parametrize("content", [
        "{not json",
        json.dumps({"version": 99, "turns": []}),
        json.dumps({"version": ChatHistory.SESSION_FORMAT}),
        json.dumps({"version": ChatHistory.SESSION_FORMAT, "turns": [{"user": "only"}]}),
        json.dumps([1, 2, 3]),
    ])
    def test_corrupt_file_is_moved_aside_and_session_starts_fresh(self, make_state, tmp_path, content):
        path = tmp_path / "s.json"
        path.write_text(content)
        state = make_state(session_file=str(path))
        new_state, events = LoadSession().execute(state)
        assert new_state.history == ChatHistory()
        assert new_state.session_file == str(path)          # still saving to the original path
        assert not path.exists()
        assert (tmp_path / "s.json.bad").read_text() == content
        assert len(events) == 1 and isinstance(events[0], Warn)

    def test_load_does_not_touch_settings(self, make_state, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**sample_history().to_dict(), "meta": {"model": "other", "context": 1}}))
        state = make_state(session_file=str(path))
        new_state, _ = LoadSession().execute(state)
        assert new_state.settings == state.settings
        assert new_state.system_prompt == state.system_prompt


# ---------------------
# SaveSession
# ---------------------

class TestSaveSession:
    def test_no_session_file_is_a_silent_noop(self, make_state):
        state = make_state(session_file=None, history=sample_history())
        new_state, events = SaveSession().execute(state)
        assert new_state is state and events == []

    def test_writes_history_and_meta(self, make_state, tmp_path):
        path = tmp_path / "s.json"
        state = make_state(session_file=str(path), history=sample_history())
        _, events = SaveSession().execute(state)
        assert events == []
        doc = json.loads(path.read_text())
        assert ChatHistory.from_dict(doc) == sample_history()
        assert doc["meta"]["model"] == state.settings.model
        assert doc["meta"]["context"] == state.settings.context
        assert doc["meta"]["system_prompt"] == state.system_prompt
        assert "saved_at" in doc["meta"]

    def test_write_is_atomic_and_leaves_no_temp_file(self, make_state, tmp_path):
        path = tmp_path / "s.json"
        state = make_state(session_file=str(path), history=sample_history())
        SaveSession().execute(state)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json"]

    def test_save_then_load_round_trips(self, make_state, tmp_path):
        path = str(tmp_path / "s.json")
        SaveSession().execute(make_state(session_file=path, history=sample_history()))
        new_state, _ = LoadSession().execute(make_state(session_file=path))
        assert new_state.history == sample_history()

    def test_unwritable_path_surfaces_an_error_event_not_an_exception(self, make_state, tmp_path):
        path = str(tmp_path / "no-such-dir" / "s.json")
        state = make_state(session_file=path, history=sample_history())
        new_state, events = SaveSession().execute(state)
        assert new_state is state
        assert len(events) == 1 and isinstance(events[0], Error)


# ---------------------
# Chain hooks: every history change persists
# ---------------------

class TestPersistHooks:
    def test_append_turn_emits_save_when_session_file_set(self, make_state, tmp_path):
        state = make_state(session_file=str(tmp_path / "s.json"))
        _, events = AppendTurn(user_msg="q", assistant_msg="a", cancelled=False).execute(state)
        assert any(isinstance(e, SaveSession) for e in events)

    def test_append_turn_does_not_emit_save_without_session_file(self, make_state):
        _, events = AppendTurn(user_msg="q", assistant_msg="a", cancelled=False).execute(make_state())
        assert not any(isinstance(e, SaveSession) for e in events)

    def test_compact_history_emits_save_when_session_file_set(self, make_state, tmp_path):
        server = FakeServer(script=[{"content": "summary"}])
        state = with_server(make_state, server, session_file=str(tmp_path / "s.json"),
                            history=ChatHistory().append(Turn("q", "a", tokens=100)))
        _, events = CompactHistory().execute(state)
        assert any(isinstance(e, SaveSession) for e in events)

    def test_save_emitted_by_append_persists_the_appended_turn(self, make_state, tmp_path):
        # AppendTurn returns the new state; SaveSession must see it — i.e. run against new_state
        path = tmp_path / "s.json"
        state = make_state(session_file=str(path))
        new_state, events = AppendTurn(user_msg="q", assistant_msg="a", cancelled=False).execute(state)
        save = next(e for e in events if isinstance(e, SaveSession))
        save.execute(new_state)
        assert ChatHistory.from_dict(json.loads(path.read_text())).turns[-1].user == "q"


# ---------------------
# CLI: resolve_session_file
# ---------------------

class TestResolveSessionFile:
    def test_neither_flag_means_no_persistence(self):
        from desh_chat.cli import resolve_session_file
        assert resolve_session_file("", "", "run1") is None

    def test_explicit_session_is_used_as_is(self, tmp_path):
        from desh_chat.cli import resolve_session_file
        path = str(tmp_path / "chat.json")
        assert resolve_session_file(path, "", "run1") == path

    def test_explicit_session_expands_tilde(self, monkeypatch, tmp_path):
        from desh_chat.cli import resolve_session_file
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_session_file("~/chat.json", "", "run1") == str(tmp_path / "chat.json")

    def test_folder_creates_a_run_named_file_inside_it(self, tmp_path):
        from desh_chat.cli import resolve_session_file
        folder = tmp_path / "sessions"
        got = resolve_session_file("", str(folder), "run1")
        assert got == str(folder / "run1.json")
        assert folder.is_dir()                     # created
        assert not os.path.exists(got)             # file itself is left to SaveSession

    def test_folder_expands_tilde(self, monkeypatch, tmp_path):
        from desh_chat.cli import resolve_session_file
        monkeypatch.setenv("HOME", str(tmp_path))
        got = resolve_session_file("", "~/sessions", "run1")
        assert got == str(tmp_path / "sessions" / "run1.json")
        assert (tmp_path / "sessions").is_dir()
        assert not os.path.exists("~")             # no literal "~" directory in cwd

    def test_explicit_session_wins_over_folder(self, tmp_path):
        from desh_chat.cli import resolve_session_file
        path = str(tmp_path / "chat.json")
        got = resolve_session_file(path, str(tmp_path / "sessions"), "run1")
        assert got == path
        assert not (tmp_path / "sessions").exists()   # folder untouched when ignored
