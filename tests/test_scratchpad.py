"""
The scratchpad: the model's working memory as a VALUE on ChatState.

  Scratchpad            an immutable record of (key, value) pairs; with_entry / without / cleared
                        return new values.
  write/delete/clear    the model's side: plain functions over a dict the harness injects for one
                        call (Tool.inject), returning an acknowledgement the model reads.
  ExecuteToolCalls      builds that dict from the state value, runs the call, reads the dict back
                        into a new value — the one place a scratchpad call becomes a state transition.
  NextRound             renders the block LAST in the request and prices it as prior tokens.
  TurnEnd               snapshots the value onto the Turn; the session file carries it per turn.
  LoadSession           restores the newest snapshot when the tool is offered.
"""
import json

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.llama.tokens import estimate_tokens
from desh.llama.wire import Request, ToolCall
from desh.tools import ToolRegistry
from desh_chat import scratchpad as pad_tools
from desh_chat.events import ExecuteToolCalls, NextRound, StreamCompletion, TurnEnd
from desh_chat.scratchpad import Scratchpad
from desh_chat.session import LoadSession, SaveSession
from desh_chat.state import ChatHistory, InferenceEngine, PendingTurn, Round, Settings, Turn


TOOLS = (ToolRegistry()
         .add(pad_tools.write, name="scratchpad_write", inject=("scratchpad",), confirm=False)
         .add(pad_tools.delete, name="scratchpad_delete", inject=("scratchpad",), confirm=False)
         .add(pad_tools.clear, name="scratchpad_clear", inject=("scratchpad",), confirm=False))


def call(name: str, index: int = 0, **arguments) -> ToolCall:
    return ToolCall(index=index, id=f"call_{index}", type="function", name=name, arguments=json.dumps(arguments))


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


# ---------------------
# The value
# ---------------------

class TestValue:
    def test_empty_by_default_and_every_op_returns_a_new_value(self):
        p0 = Scratchpad()
        p1 = p0.with_entry("a", "1")
        assert p0.memory == () and p1.memory == (("a", "1"),)
        assert p1.without("a").memory == () and p1.memory == (("a", "1"),)
        assert p1.cleared().memory == () and p1.memory == (("a", "1"),)

    def test_overwrite_replaces_the_value_and_moves_the_key_to_the_end(self):
        p = Scratchpad().with_entry("a", "1").with_entry("b", "2").with_entry("a", "3")
        assert p.memory == (("b", "2"), ("a", "3"))

    def test_without_a_missing_key_is_the_same_value(self):
        p = Scratchpad().with_entry("a", "1")
        assert p.without("zz") == p

    def test_dict_round_trip_keeps_order(self):
        p = Scratchpad().with_entry("b", "2").with_entry("a", "1")
        assert p.to_dict() == {"b": "2", "a": "1"}
        assert Scratchpad.from_dict(p.to_dict()) == p
        assert Scratchpad.from_dict({}) == Scratchpad()

    def test_message_names_the_expiration_and_lists_the_entries_or_says_empty(self):
        empty = Scratchpad().message(tool_expiration=4)
        assert empty.startswith("<scratchpad>") and empty.endswith("</scratchpad>")
        assert "4 rounds" in empty and "(empty)" in empty
        full = Scratchpad().with_entry("path", "src/x.py").with_entry("id", "42").message(tool_expiration=4)
        assert "path: src/x.py\nid: 42\n" in full and "(empty)" not in full

    def test_to_context_is_a_user_message(self):
        pad = Scratchpad().with_entry("k", "v")
        assert pad.to_context(tool_expiration=6) == {"role": "user", "content": pad.message(6)}


# ---------------------
# The tools: schema and acknowledgements
# ---------------------

class TestTools:
    def test_schemas_hide_the_injected_dict_and_describe_the_rest(self):
        by_name = {s["function"]["name"]: s["function"]["parameters"] for s in TOOLS.schemas()}
        assert set(by_name) == {"scratchpad_write", "scratchpad_delete", "scratchpad_clear"}
        assert set(by_name["scratchpad_write"]["properties"]) == {"key", "value"}
        assert by_name["scratchpad_write"]["required"] == ["key", "value"]
        assert set(by_name["scratchpad_delete"]["properties"]) == {"key"}
        assert by_name["scratchpad_clear"]["properties"] == {}

    def test_write_creates_then_overwrites(self):
        d: dict[str, str] = {}
        assert pad_tools.write("a", "1", d) == "created 'a'" and d == {"a": "1"}
        assert pad_tools.write("a", "2", d) == "overwrote 'a'" and d == {"a": "2"}

    def test_delete_acks_the_key_or_reports_it_missing(self):
        d = {"a": "1"}
        missing = pad_tools.delete("zz", d)
        assert "'zz'" in missing and "not found" in missing and d == {"a": "1"}
        assert "'a'" in pad_tools.delete("a", d) and d == {}

    def test_clear_counts_what_it_removed(self):
        d = {"a": "1", "b": "2"}
        assert pad_tools.clear(d) == "cleared 2 items" and d == {}
        assert pad_tools.clear(d) == "already empty"

    def test_invoke_through_the_registry_mutates_the_provided_dict(self):
        d = {"a": "1"}
        assert TOOLS.invoke("scratchpad_write", '{"key": "b", "value": "2"}', scratchpad=d) == "created 'b'"
        assert d == {"a": "1", "b": "2"}

    def test_the_model_cannot_pass_the_dict(self):
        content = TOOLS.invoke("scratchpad_write", '{"key": "b", "value": "2", "scratchpad": {}}', scratchpad={})
        assert "scratchpad" in content and "created" not in content


# ---------------------
# ExecuteToolCalls: the call becomes a state transition
# ---------------------

class TestCommit:
    def run_call(self, make_state, tc: ToolCall, scratchpad=Scratchpad(), tools=TOOLS):
        state = make_state(pending=PendingTurn("q").add_round(Round("", (tc,))), tools=tools, scratchpad=scratchpad)
        new_state, events = ExecuteToolCalls(index=0).execute(state)
        return state, new_state, events

    def test_a_write_is_committed_as_a_new_value_and_the_old_state_is_untouched(self, make_state):
        state, new_state, _ = self.run_call(make_state, call("scratchpad_write", key="path", value="src/x.py"))
        assert new_state.scratchpad == Scratchpad().with_entry("path", "src/x.py")
        assert state.scratchpad == Scratchpad()
        assert new_state.pending.rounds[-1].results[0].content == "created 'path'"

    def test_delete_and_clear_commit_too(self, make_state):
        start = Scratchpad().with_entry("a", "1").with_entry("b", "2")
        _, after_delete, _ = self.run_call(make_state, call("scratchpad_delete", key="a"), scratchpad=start)
        assert after_delete.scratchpad == Scratchpad().with_entry("b", "2")
        _, after_clear, _ = self.run_call(make_state, call("scratchpad_clear"), scratchpad=start)
        assert after_clear.scratchpad == Scratchpad()

    def test_a_tool_that_did_not_ask_for_the_scratchpad_leaves_it_alone(self, make_state):
        def now() -> str:
            """The time."""
            return "10:00"
        start = Scratchpad().with_entry("a", "1")
        _, new_state, _ = self.run_call(make_state, call("now"), scratchpad=start, tools=TOOLS.add(now, name="now", confirm=False))
        assert new_state.scratchpad is start
        assert new_state.pending.rounds[-1].results[0].content == "10:00"

    def test_a_rejected_call_commits_the_old_value(self, make_state):
        """The registry answers a call with a missing argument in text, before the tool runs; the
        dict it never reached is read back unchanged."""
        start = Scratchpad().with_entry("a", "1")
        _, new_state, _ = self.run_call(make_state, call("scratchpad_write", key="b"), scratchpad=start)
        assert new_state.scratchpad == start
        assert "value" in new_state.pending.rounds[-1].results[0].content

    def test_without_a_scratchpad_on_the_state_the_tool_is_answered_with_an_error(self, make_state):
        """The registry offers the tool but the run has no working memory: the injection is
        missing, which invoke reports as text, and the state stays without one."""
        _, new_state, _ = self.run_call(make_state, call("scratchpad_write", key="a", value="1"), scratchpad=None)
        assert new_state.scratchpad is None
        assert "scratchpad" in new_state.pending.rounds[-1].results[0].content


# ---------------------
# NextRound: the block is last in the request and priced as prior
# ---------------------

class TestRequest:
    def stream_event(self, make_state, **overrides) -> StreamCompletion:
        state = make_state(pending=PendingTurn("hello"), tools=TOOLS, **overrides)
        _, events = NextRound().execute(state)
        (ev,) = events
        assert isinstance(ev, StreamCompletion)
        return ev

    def test_the_block_is_the_last_message_after_the_pending_turn(self, make_state):
        pad = Scratchpad().with_entry("k", "v")
        ev = self.stream_event(make_state, scratchpad=pad)
        assert ev.request.messages[-1] == make_state(pending=PendingTurn("hello"), scratchpad=pad).scratchpad_block()
        assert ev.request.messages[-1]["content"].startswith("<scratchpad>") and "k: v" in ev.request.messages[-1]["content"]
        assert ev.request.messages[-2] == {"role": "user", "content": "hello"}

    def test_the_block_shows_the_round_against_the_cap(self, make_state):
        """The next completion is one past the completed rounds; the cap is the turn's."""
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=8)
        first = make_state(pending=PendingTurn("hello"), settings=settings, scratchpad=Scratchpad()).scratchpad_block()
        assert "Round 1 of 8 in this turn." in first["content"]
        tc = ToolCall(index=0, id="c0", type="function", name="Read", arguments="{}")
        later = make_state(pending=PendingTurn("hello").add_round(Round("", (tc,))).add_round(Round("", (tc,))),
                           settings=settings, scratchpad=Scratchpad()).scratchpad_block()
        assert "Round 3 of 8 in this turn." in later["content"]
        assert "Round" not in Scratchpad().message(6)     # the pure form carries no counter

    def test_no_scratchpad_means_no_block(self, make_state):
        ev = self.stream_event(make_state)
        assert ev.request.messages[-1] == {"role": "user", "content": "hello"}
        assert not any("<scratchpad>" in m["content"] for m in ev.request.messages)

    def test_the_block_counts_as_prior_and_the_user_message_stays_the_unpriced_text(self, make_state):
        pad = Scratchpad().with_entry("k", "v")
        plain = self.stream_event(make_state)
        with_pad = self.stream_event(make_state, scratchpad=pad)
        block = make_state(pending=PendingTurn("hello"), scratchpad=pad).scratchpad_block()
        assert with_pad.prior_tokens - plain.prior_tokens == estimate_tokens(block["content"])
        assert with_pad.unpriced == "hello" and plain.unpriced == "hello"

    def test_the_block_uses_the_configured_expiration(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, tool_expiration=3)
        ev = self.stream_event(make_state, settings=settings, scratchpad=Scratchpad())
        assert "after 3 rounds" in ev.request.messages[-1]["content"]

    def test_the_block_takes_room_from_the_prompt_not_the_budget(self, make_state):
        pad = Scratchpad().with_entry("k", "v" * 4000)
        state = make_state(pending=PendingTurn("hello"), scratchpad=pad)
        bare = make_state(pending=PendingTurn("hello"))
        cost = estimate_tokens(state.scratchpad_block()["content"])
        assert state.prompt_tokens(state.pending_tokens()) - bare.prompt_tokens(bare.pending_tokens()) == cost
        assert state.gen_room(state.pending_tokens()) == bare.gen_room(bare.pending_tokens()) - cost

    def test_the_heuristic_fallback_prices_the_unpriced_text_not_the_block(self, make_state, no_esc_watcher):
        """A usage frame without a prompt count: turn_tokens falls back to estimating the new
        prompt text, which must be the user message, not the block that came last."""
        pad = Scratchpad().with_entry("k", "v" * 2000)
        req = Request(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hello"}, pad.to_context(6)],
                      model=MODELS[0], temperature=0.3, max_tokens=100, think=False, stream=True)
        server = FakeServer(script=[{"content": "hi", "usage": {"completion_tokens": 5}}])
        state = with_server(make_state, server, pending=PendingTurn("hello"), scratchpad=pad)
        _, events = StreamCompletion(request=req, prior_tokens=10_000, unpriced="hello").execute(state)
        end = next(e for e in events if isinstance(e, TurnEnd))
        assert end.tokens == estimate_tokens("hello") + 5


# ---------------------
# TurnEnd and the session file
# ---------------------

class TestSnapshot:
    def test_turn_end_records_the_value_on_the_turn(self, make_state):
        pad = Scratchpad().with_entry("k", "v")
        state = make_state(pending=PendingTurn("q"), scratchpad=pad)
        new_state, _ = TurnEnd(assistant="a").execute(state)
        assert new_state.history.turns[-1].scratchpad == pad
        assert new_state.scratchpad == pad     # the value outlives the turn

    def test_a_run_without_the_tool_records_none(self, make_state):
        new_state, _ = TurnEnd(assistant="a").execute(make_state(pending=PendingTurn("q")))
        assert new_state.history.turns[-1].scratchpad is None

    def test_a_plain_turn_serialises_without_the_key(self):
        assert "scratchpad" not in Turn("u", "a", tokens=7).to_dict()
        assert Turn.from_dict({"user": "u", "assistant": "a", "tokens": 7}).scratchpad is None

    def test_a_turn_with_a_scratchpad_round_trips_through_json(self):
        pad = Scratchpad().with_entry("b", "2").with_entry("a", "1")
        turn = Turn("u", "a", tokens=7, scratchpad=pad)
        back = Turn.from_dict(json.loads(json.dumps(turn.to_dict())))
        assert back == turn and back.scratchpad.memory == (("b", "2"), ("a", "1"))
        empty = Turn("u", "a", tokens=7, scratchpad=Scratchpad())
        assert Turn.from_dict(json.loads(json.dumps(empty.to_dict()))).scratchpad == Scratchpad()

    def test_last_scratchpad_skips_turns_without_one(self):
        old = Scratchpad().with_entry("k", "old")
        new = Scratchpad().with_entry("k", "new")
        h = (ChatHistory()
             .append(Turn("q1", "a1", scratchpad=old))
             .append(Turn("q2", "a2", scratchpad=new))
             .compact("summary"))
        assert h.last_scratchpad() == new
        assert ChatHistory().last_scratchpad() is None
        assert ChatHistory().append(Turn("q", "a")).last_scratchpad() is None

    def test_save_writes_the_snapshot_per_turn(self, make_state, tmp_path):
        path = str(tmp_path / "s.json")
        pad = Scratchpad().with_entry("k", "v")
        state = make_state(session_file=path, pending=PendingTurn("q"), scratchpad=pad)
        state, events = TurnEnd(assistant="a").execute(state)
        assert any(isinstance(e, SaveSession) for e in events)
        SaveSession().execute(state)
        with open(path) as f:
            doc = json.load(f)
        assert doc["version"] == 3 and doc["turns"][-1]["scratchpad"] == {"k": "v"}


# ---------------------
# LoadSession: the flag decides whether there is a scratchpad, the file only fills it
# ---------------------

class TestLoad:
    OLD = Scratchpad().with_entry("k", "old")
    NEW = Scratchpad().with_entry("k", "new")

    def saved(self, tmp_path, history: ChatHistory) -> str:
        path = str(tmp_path / "s.json")
        with open(path, "w") as f:
            json.dump(history.to_dict(), f)
        return path

    def test_offered_and_the_file_has_one_restores_the_newest_past_a_summary(self, make_state, tmp_path):
        history = (ChatHistory()
                   .append(Turn("q1", "a1", scratchpad=self.OLD))
                   .append(Turn("q2", "a2", scratchpad=self.NEW))
                   .compact("summary"))
        state = make_state(session_file=self.saved(tmp_path, history), scratchpad=Scratchpad())
        new_state, _ = LoadSession().execute(state)
        assert new_state.scratchpad == self.NEW
        assert len(new_state.history) == 3

    def test_offered_and_the_file_has_none_starts_empty(self, make_state, tmp_path):
        """A format 2 file, or a run that never had the tool: the value stays the empty one."""
        state = make_state(session_file=self.saved(tmp_path, ChatHistory().append(Turn("q", "a"))), scratchpad=Scratchpad())
        new_state, _ = LoadSession().execute(state)
        assert new_state.scratchpad == Scratchpad()

    def test_not_offered_ignores_what_the_file_carries(self, make_state, tmp_path):
        """No tool this run means no working memory, whatever the file says: the model would be
        shown memory it cannot change. The file keeps it for a run that offers the tool again."""
        history = ChatHistory().append(Turn("q", "a", scratchpad=self.NEW))
        state = make_state(session_file=self.saved(tmp_path, history))
        new_state, _ = LoadSession().execute(state)
        assert new_state.scratchpad is None
        assert new_state.history.turns[-1].scratchpad == self.NEW    # still in the history, just not live

    def test_a_missing_file_leaves_the_empty_value(self, make_state, tmp_path):
        state = make_state(session_file=str(tmp_path / "none.json"), scratchpad=Scratchpad())
        new_state, _ = LoadSession().execute(state)
        assert new_state.scratchpad == Scratchpad()

    def test_round_trip_through_save_and_load(self, make_state, tmp_path):
        path = str(tmp_path / "s.json")
        state = make_state(session_file=path, pending=PendingTurn("q"), scratchpad=self.NEW)
        state, _ = TurnEnd(assistant="a").execute(state)
        SaveSession().execute(state)
        fresh = make_state(session_file=path, scratchpad=Scratchpad())
        restored, _ = LoadSession().execute(fresh)
        assert restored.scratchpad == self.NEW
