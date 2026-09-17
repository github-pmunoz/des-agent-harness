"""
Session persistence: pure to_dict/from_dict on Turn/ChatHistory, the LoadSession /
SaveSession events, and the chain hooks that emit SaveSession on every history change.
"""
import json
import os

import pytest

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh_chat.display import Error, Info, Warn
from desh_chat.events import CompactHistory, TurnEnd
from desh_chat.session import LoadSession, SaveSession
from desh_chat.state import COMPACTION_PROMPT, ChatHistory, InferenceEngine, PendingTurn, Settings, Turn


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

    def test_stop_round_trips_and_is_absent_from_a_plain_turn(self):
        """A plain turn serialises exactly as it did before `stop` existed — no key — so format 2
        files stay byte-identical; a turn that ended early carries the reason."""
        assert "stop" not in Turn("u", "a", tokens=7).to_dict()
        capped = Turn("u", "so far", tokens=7, stop="cap")
        overflow = Turn("u", "", tokens=7, cancelled=True, stop="overflow")
        assert capped.to_dict()["stop"] == "cap"
        assert Turn.from_dict(json.loads(json.dumps(capped.to_dict()))) == capped
        assert Turn.from_dict(json.loads(json.dumps(overflow.to_dict()))) == overflow
        assert Turn.from_dict({"user": "u", "assistant": "a", "tokens": 7}).stop == ""

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
        assert ChatHistory.SESSION_FORMAT >= 4
        assert ChatHistory().to_dict()["version"] >= 4

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
# Settings serialization (pure)
# ---------------------

class TestSettingsSerialization:
    def test_to_dict_is_plain_json_and_round_trips_all_fields(self):
        s = Settings(model="model-b", temperature=0.9, think=True, context=8192, max_turn_tokens=4096,
                     max_tool_rounds=7, tool_expiration=3, compaction_threshold=0.5,
                     compaction_target=0.2, turn_token_cap=0.3, min_compaction_tokens=128,
                     auto=True, compaction_prompt="summarise it", checkpoint_target=0.1, checkpoint_prompt="checkpoint it")
        assert Settings.from_dict(s.to_dict()) == s
        json.dumps(s.to_dict())     # every value is plain JSON

    def test_round_trip_keeps_the_default_compaction_prompt(self):
        s = Settings(model="model-a", temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
        assert s.compaction_prompt == COMPACTION_PROMPT
        assert Settings.from_dict(s.to_dict()) == s

    def test_from_dict_fills_defaults_for_a_partial_dict(self):
        d = {"model": "model-a", "temperature": 0.3, "think": False, "context": 16384, "max_turn_tokens": 8192}
        assert Settings.from_dict(d) == Settings(**d)

    @pytest.mark.parametrize("bad", [None, ["settings"], 42, "bad"])
    def test_from_dict_rejects_non_dict_input(self, bad):
        with pytest.raises(ValueError):
            Settings.from_dict(bad)


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
        assert len(events) == 2 and isinstance(events[0], Info) and isinstance(events[1], SaveSession)
        assert "New session" in events[0].text
        events[1].execute(new_state)    # the session file is written seeding it with the current settings
        assert os.path.exists(path)     

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

    def test_v3_file_still_loads(self, make_state, tmp_path):
        # a format-3 document: turns only, no "settings" key — loads unchanged
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": 3, "turns": sample_history().to_dict()["turns"]}))
        state = make_state(session_file=str(path))
        new_state, _ = LoadSession().execute(state)
        assert new_state.history == sample_history()
        assert new_state.settings == state.settings

    def test_load_restores_settings_from_a_v4_document(self, make_state, tmp_path):
        doc_settings = Settings(model="model-b", temperature=0.9, think=True, context=8192,
                                max_turn_tokens=4096, max_tool_rounds=7, tool_expiration=3,
                                compaction_threshold=0.5, compaction_target=0.2,
                                turn_token_cap=0.3, min_compaction_tokens=128, auto=True,
                                compaction_prompt="summarise it")
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**sample_history().to_dict(), "settings": doc_settings.to_dict()}))
        state = make_state(session_file=str(path))
        new_state, _ = LoadSession().execute(state)
        assert new_state.settings == doc_settings

    def test_load_replays_settings_turns(self, make_state, tmp_path):
        doc_settings = Settings(model="model-a", temperature=0.3, think=False, context=16384,
                                max_turn_tokens=8192, max_tool_rounds=7, tool_expiration=3)
        turns = [
            {"user": "", "assistant": "", "tokens": 1, "type": "settings", "delta": {"model": "model-a", "max_tool_rounds": 5}},
            {"user": "", "assistant": "", "tokens": 1, "type": "settings", "delta": {"model": "model-b"}},
        ]
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": 4, "turns": turns, "settings": doc_settings.to_dict()}))
        new_state, _ = LoadSession().execute(make_state(session_file=str(path)))
        assert new_state.settings.model == "model-b"       # the last settings turn wins
        assert new_state.settings.max_tool_rounds == 5     # an earlier delta still applies
        assert new_state.settings.tool_expiration == 3     # untouched by any turn: the doc-seed value
        assert new_state.settings.temperature == 0.3       # likewise the doc-seed value

    def test_load_replays_settings_turns_on_top_of_the_seed(self, make_state, tmp_path):
        # end-to-end: the seed holds the settings in force when the file was created; the settings
        # turns replay in order on top of it (last change wins), and every field no turn touched
        # comes from the seed — not from the CLI settings
        seed = Settings(model="model-a", temperature=0.5, think=True, context=8192, max_turn_tokens=4096,
                        max_tool_rounds=7, tool_expiration=3)
        cli = Settings(model="model-cli", temperature=0.1, think=False, context=32768, max_turn_tokens=16384)
        turns = [
            {"user": "", "assistant": "", "tokens": 1, "type": "settings", "delta": {"model": "model-b"}},
            {"user": "", "assistant": "", "tokens": 1, "type": "settings", "delta": {"temperature": 0.9}},
        ]
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": 4, "turns": turns, "settings": seed.to_dict()}))
        new_state, _ = LoadSession().execute(make_state(session_file=str(path), settings=cli))
        assert new_state.settings.model == "model-b"       # the first settings turn
        assert new_state.settings.temperature == 0.9       # the last settings turn wins
        assert new_state.settings.think is True            # untouched by any turn: the seed value
        assert new_state.settings.context == 8192          # likewise the seed value
        assert new_state.settings.max_turn_tokens == 4096  # likewise the seed value
        assert new_state.settings.max_tool_rounds == 7     # likewise the seed value
        assert new_state.settings.tool_expiration == 3     # likewise the seed value

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

    def test_first_save_writes_the_current_settings_as_the_seed(self, make_state, tmp_path):
        # no pre-existing file: the first save writes the current settings as the initial seed
        path = tmp_path / "s.json"
        assert not path.exists()
        state = make_state(session_file=str(path), history=sample_history())
        SaveSession().execute(state)
        doc = json.loads(path.read_text())
        assert doc["settings"] == state.settings.to_dict()

    def test_second_save_preserves_the_original_seed(self, make_state, tmp_path):
        # the top-level "settings" key holds the settings in force when the file was first
        # created; a later save must not drift it to the latest settings
        seed = Settings(model="model-a", temperature=0.5, think=False, context=16384, max_turn_tokens=8192)
        path = tmp_path / "s.json"
        path.write_text(json.dumps({**sample_history().to_dict(), "settings": seed.to_dict()}))
        latest = Settings(model="model-b", temperature=0.5, think=False, context=16384, max_turn_tokens=8192)
        history = (sample_history()
                   .append(Turn("", "", tokens=1, type="settings", delta={"model": "model-b"})))
        state = make_state(session_file=str(path), history=history, settings=latest)
        SaveSession().execute(state)
        doc = json.loads(path.read_text())
        assert doc["settings"] == seed.to_dict()          # the seed is untouched
        assert any(t.get("type", "chat") == "settings" and t["delta"] == {"model": "model-b"} for t in doc["turns"])

    def test_save_on_an_existing_file_without_a_settings_key_writes_the_current_settings_as_the_seed(self, make_state, tmp_path):
        # a v3 file has no "settings" key: the first v4-format save of it writes the current settings as the seed
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"version": 3, "turns": sample_history().to_dict()["turns"]}))
        state = make_state(session_file=str(path), history=sample_history())
        SaveSession().execute(state)
        doc = json.loads(path.read_text())
        assert doc["settings"] == state.settings.to_dict()

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
    def test_turn_end_emits_save_when_session_file_set(self, make_state, tmp_path):
        state = make_state(session_file=str(tmp_path / "s.json"), pending=PendingTurn("q"))
        _, events = TurnEnd(assistant="a", cancelled=False).execute(state)
        assert any(isinstance(e, SaveSession) for e in events)

    def test_turn_end_does_not_emit_save_without_session_file(self, make_state):
        _, events = TurnEnd(assistant="a", cancelled=False).execute(make_state(pending=PendingTurn("q")))
        assert not any(isinstance(e, SaveSession) for e in events)

    def test_compact_history_emits_save_when_session_file_set(self, make_state, tmp_path):
        server = FakeServer(script=[{"content": "summary"}])
        state = with_server(make_state, server, session_file=str(tmp_path / "s.json"),
                            history=ChatHistory().append(Turn("q", "a", tokens=100)))
        _, events = CompactHistory().execute(state)
        assert any(isinstance(e, SaveSession) for e in events)

    def test_save_emitted_by_append_persists_the_appended_turn(self, make_state, tmp_path):
        # TurnEnd returns the new state; SaveSession must see it — i.e. run against new_state
        path = tmp_path / "s.json"
        state = make_state(session_file=str(path), pending=PendingTurn("q"))
        new_state, events = TurnEnd(assistant="a", cancelled=False).execute(state)
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
        assert got is not None
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
