"""
The turn loop: a turn is (user message, tool rounds*, final answer), driven by the chain

  UserMessage -> NextRound -> StreamCompletion -> AppendRound -> ExecuteToolCalls(i)* -> NextRound -> ...
                                               -> TurnEnd -> MaybeCompact

State between rounds lives on ChatState.pending (a PendingTurn); only TurnEnd writes history.
FakeServer's script drives the model: an entry with tool_calls makes a round, one without ends the turn.
These tests run with an EMPTY registry, so ExecuteToolCalls answers every call "not available" — a
legitimate tool message the model can recover from; the loop shape is independent of any real tool.
Real-tool dispatch is covered in test_tools.py.
"""
import json

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.engine import Engine
from desh.llama.wire import Request, ToolCall
from desh.llama.tokens import estimate_tokens
from desh_chat.display import Error, Info, Warn
from desh_chat.events import (
    AppendRound, ExecuteToolCalls, MaybeCompact, NextRound, PromptUser,
    StreamCompletion, TurnEnd, UserMessage,
)
from desh_chat.session import LoadSession, SaveSession
from desh_chat.state import ChatHistory, InferenceEngine, PendingTurn, Round, Settings, ToolResult, Turn


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


def call(index=0, name="get_weather", arguments='{"city": "Santiago"}', id=None) -> ToolCall:
    return ToolCall(index=index, id=id or f"call_{index}", type="function", name=name, arguments=arguments)


def result(tc: ToolCall, content="sunny") -> ToolResult:
    return ToolResult(tool_call_id=tc.id, name=tc.name, content=content)


WEATHER = call(0)
TIME = call(1, name="get_time", arguments='{"tz": "CLT"}')
ROUND = Round(assistant="Let me check.", tool_calls=(WEATHER, TIME), results=(result(WEATHER), result(TIME, "10:00")), tokens=30)


# ---------------------
# Pure state: Round / PendingTurn / Turn with rounds
# ---------------------

class TestPendingTurn:
    def test_add_round_and_with_results_are_immutable(self):
        p0 = PendingTurn("q")
        p1 = p0.add_round(Round("", (WEATHER,)))
        p2 = p1.with_results((result(WEATHER),))
        assert p0.rounds == ()
        assert p1.rounds[0].results == ()
        assert p2.rounds[0].results == (result(WEATHER),)

    def test_priced_tokens_sums_the_rounds(self):
        p = PendingTurn("q").add_round(Round("", (WEATHER,), tokens=30)).add_round(Round("", (TIME,), tokens=20))
        assert p.priced_tokens() == 50

    def test_unpriced_text_is_the_user_message_then_the_latest_results(self):
        p = PendingTurn("the question")
        assert p.unpriced_text() == "the question"
        p = p.add_round(Round("", (WEATHER, TIME))).with_results((result(WEATHER, "sunny"), result(TIME, "10:00")))
        assert p.unpriced_text() == "sunny\n10:00"

    def test_finish_prices_final_completion_plus_rounds(self):
        p = PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=30))
        turn = p.finish("the answer", tokens=12, cancelled=False)
        assert turn == Turn("q", "the answer", tokens=42, rounds=p.rounds)
        assert turn.tokens == 42

    def test_finish_without_usage_keeps_priced_rounds_and_estimates_the_rest(self):
        p = PendingTurn("hello there").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=30))
        turn = p.finish("general kenobi", tokens=0, cancelled=False)
        assert turn.tokens == estimate_tokens("hello there") + estimate_tokens("general kenobi") + 30

    def test_unpriced_round_is_estimated_from_its_text(self):
        unpriced = Round("Let me check.", (WEATHER,), (result(WEATHER),), tokens=0)
        turn = Turn("q", "a", rounds=(unpriced,))
        assert turn.tokens == estimate_tokens("q") + estimate_tokens("a") + estimate_tokens(unpriced.text())


class TestRoundMessages:
    def test_round_is_assistant_with_tool_calls_then_one_tool_message_per_result(self):
        msgs = ROUND.messages()
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "Let me check."
        assert [tc["id"] for tc in msgs[0]["tool_calls"]] == ["call_0", "call_1"]
        assert msgs[0]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Santiago"}'}
        assert msgs[0]["tool_calls"][0]["type"] == "function"
        assert msgs[1] == {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "sunny"}
        assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "name": "get_time", "content": "10:00"}
        assert len(msgs) == 3

    def test_turn_messages_splice_rounds_between_user_and_final_answer(self):
        turn = Turn("q", "the answer", rounds=(ROUND,))
        roles = [m["role"] for m in turn.messages()]
        assert roles == ["user", "assistant", "tool", "tool", "assistant"]
        assert turn.messages()[-1] == {"role": "assistant", "content": "the answer"}

    def test_plain_turn_messages_are_unchanged(self):
        assert Turn("q", "a").messages() == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


class TestTurnSerialization:
    def test_plain_turn_serializes_exactly_as_format_1_did(self):
        assert Turn("q", "a", tokens=5).to_dict() == {"user": "q", "assistant": "a", "tokens": 5, "cancelled": False, "summary": False}

    def test_turn_with_rounds_round_trips_through_json(self):
        turn = Turn("q", "the answer", tokens=42, rounds=(ROUND,))
        back = Turn.from_dict(json.loads(json.dumps(turn.to_dict())))
        assert back == turn
        assert isinstance(back.rounds, tuple) and isinstance(back.rounds[0].tool_calls, tuple)

    def test_history_writes_format_2_and_still_reads_format_1(self):
        assert ChatHistory().to_dict()["version"] == 2
        v1 = {"version": 1, "turns": [{"user": "q", "assistant": "a", "tokens": 5}]}
        assert ChatHistory.from_dict(v1).turns == (Turn("q", "a", tokens=5),)

    def test_transcript_renders_calls_and_results_between_user_and_answer(self):
        text = Turn("q", "the answer", rounds=(ROUND,)).transcript()
        assert text.splitlines() == [
            "USER: q",
            'ASSISTANT (tool calls): Let me check. get_weather({"city": "Santiago"}), get_time({"tz": "CLT"})',
            "TOOL get_weather: sunny",
            "TOOL get_time: 10:00",
            "ASSISTANT: the answer",
        ]


# ---------------------
# StreamCompletion routes on finish_reason
# ---------------------

class TestStreamCompletionRouting:
    REQ = Request(messages=[{"role": "user", "content": "q"}], model=MODELS[0], stream=True)

    def test_tool_calls_finish_emits_append_round_with_the_calls(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "Let me check.", "tool_calls": [{"name": "get_weather", "arguments": "{}"}]}])
        _, events = StreamCompletion(request=self.REQ).execute(with_server(make_state, server))
        assert len(events) == 1 and isinstance(events[0], AppendRound)
        assert events[0].assistant == "Let me check."
        assert [tc.name for tc in events[0].tool_calls] == ["get_weather"]

    def test_stop_finish_emits_turn_end(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "done"}])
        _, events = StreamCompletion(request=self.REQ).execute(with_server(make_state, server))
        assert isinstance(events[0], TurnEnd) and events[0].cancelled is False

    def test_tool_calls_finish_without_calls_ends_the_turn(self, make_state, no_esc_watcher):
        # defensive: a finish_reason claiming tool calls with nothing folded must not open a round
        server = FakeServer(script=[{"content": "", "finish_reason": "tool_calls"}])
        _, events = StreamCompletion(request=self.REQ).execute(with_server(make_state, server))
        assert isinstance(events[0], TurnEnd)

    def test_cancelled_finish_ends_the_turn_cancelled_even_with_calls(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"finish_reason": "cancelled", "tool_calls": [{"name": "get_weather"}]}])
        _, events = StreamCompletion(request=self.REQ).execute(with_server(make_state, server))
        assert isinstance(events[0], Info)
        assert isinstance(events[1], TurnEnd) and events[1].cancelled is True


# ---------------------
# AppendRound / ExecuteToolCalls
# ---------------------

class TestAppendRound:
    def test_records_the_round_on_pending_and_goes_to_the_first_call(self, make_state):
        state = make_state(pending=PendingTurn("q"))
        new_state, events = AppendRound(assistant="checking", tool_calls=(WEATHER,), tokens=30).execute(state)
        assert new_state.pending.rounds == (Round("checking", (WEATHER,), (), tokens=30),)
        assert events == [ExecuteToolCalls(index=0)]
        assert state.pending.rounds == ()

    def test_round_cap_ends_the_turn_with_a_warning_and_no_new_round(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=1)
        pending = PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=30))
        state = make_state(settings=settings, pending=pending)
        new_state, events = AppendRound(assistant="one more?", tool_calls=(TIME,), tokens=10).execute(state)
        assert new_state.pending == pending                    # the capped round is not recorded
        assert isinstance(events[0], Warn) and "get_time" in events[0].text
        assert isinstance(events[1], TurnEnd)
        assert events[1].assistant == "one more?" and events[1].tokens == 10 and events[1].cancelled is False


class TestExecuteToolCalls:
    def test_one_call_per_step_each_unavailable_then_the_next_round(self, make_state):
        state = make_state(pending=PendingTurn("q").add_round(Round("", (WEATHER, TIME))))
        mid, events = ExecuteToolCalls().execute(state)
        assert [(r.tool_call_id, r.name) for r in mid.pending.rounds[-1].results] == [("call_0", "get_weather")]
        assert [type(e) for e in events] == [Info, ExecuteToolCalls] and events[1].index == 1
        assert "get_weather" in events[0].text
        final, events = events[1].execute(mid)
        results = final.pending.rounds[-1].results
        assert [(r.tool_call_id, r.name) for r in results] == [("call_0", "get_weather"), ("call_1", "get_time")]
        assert all("not available" in r.content for r in results)
        assert [type(e) for e in events] == [Info, NextRound]


# ---------------------
# NextRound with rounds pending
# ---------------------

class TestNextRoundWithRounds:
    def test_request_carries_system_view_user_and_every_round(self, make_state):
        history = ChatHistory().append(Turn("q0", "a0", tokens=20))
        pending = PendingTurn("q1").add_round(ROUND)
        state = make_state(history=history, pending=pending)
        _, events = NextRound().execute(state)
        msgs = events[0].request.messages
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "tool", "tool"]
        assert msgs[3]["content"] == "q1"
        assert msgs[4]["tool_calls"][0]["id"] == "call_0"

    def test_prior_tokens_include_the_priced_rounds(self, make_state):
        history = ChatHistory().append(Turn("q0", "a0", tokens=20))
        state = make_state(history=history, pending=PendingTurn("q1").add_round(ROUND))
        _, events = NextRound().execute(state)
        assert events[0].prior_tokens == estimate_tokens(state.system_prompt) + 20 + ROUND.tokens

    def test_over_budget_mid_loop_ends_the_turn_cancelled_instead_of_dropping_it(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=40, max_turn_tokens=100, turn_token_cap=1.0)
        state = make_state(settings=settings, pending=PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=200)))
        new_state, events = NextRound().execute(state)
        assert new_state.pending is state.pending       # TurnEnd, not NextRound, clears it
        assert isinstance(events[0], Error)
        assert isinstance(events[1], TurnEnd) and events[1].cancelled is True


# ---------------------
# Full loop through the real Engine
# ---------------------

def run_chat(make_state, script, inputs, **overrides):
    """Drive PromptUser through a scripted FakeServer with the given user inputs, then EOF."""
    it = iter(inputs)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    import builtins
    original = builtins.input
    builtins.input = fake_input
    try:
        server = FakeServer(script=script)
        state = with_server(make_state, server, **overrides)
        return Engine[type(state)]().run(state, seed=[PromptUser()]), server
    finally:
        builtins.input = original


class TestFullLoop:
    def test_two_tool_rounds_then_answer_make_one_turn(self, make_state, no_esc_watcher):
        script = [
            {"content": "Checking.", "tool_calls": [{"name": "get_weather", "arguments": '{"city": "Santiago"}'}]},
            {"tool_calls": [{"name": "get_time"}]},
            {"content": "It is sunny at 10:00."},
        ]
        final, server = run_chat(make_state, script, ["what is it like there?"])
        assert final.pending is None
        assert len(final.history.turns) == 1
        turn = final.history.turns[0]
        assert turn.user == "what is it like there?"
        assert turn.assistant == "It is sunny at 10:00."
        assert [tc.name for r in turn.rounds for tc in r.tool_calls] == ["get_weather", "get_time"]
        assert all(r.results for r in turn.rounds)
        assert turn.cancelled is False
        # each round's request carried the whole exchange so far
        assert [len(req.messages) for _, req in server.calls] == [2, 4, 6]

    def test_cancel_during_a_later_round_keeps_the_rounds_on_a_cancelled_turn(self, make_state, no_esc_watcher):
        script = [
            {"tool_calls": [{"name": "get_weather"}]},
            {"content": "partial", "finish_reason": "cancelled"},
        ]
        final, _ = run_chat(make_state, script, ["q"])
        turn = final.history.turns[0]
        assert turn.cancelled is True
        assert len(turn.rounds) == 1 and turn.rounds[0].results
        assert final.pending is None

    def test_round_cap_bounds_the_loop(self, make_state, no_esc_watcher):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=2)
        script = [{"tool_calls": [{"name": f"tool_{i}"}]} for i in range(5)] + [{"content": "never reached"}]
        final, server = run_chat(make_state, script, ["q"], settings=settings)
        turn = final.history.turns[0]
        assert len(turn.rounds) == 2
        assert turn.assistant == ""                 # the capped round's text (none here) is the final answer
        assert len(server.calls) == 3               # 2 rounds + the one that hit the cap; no more
        assert len(server.script) == 3              # the rest of the script was never consumed

    def test_turn_with_rounds_survives_save_and_load(self, make_state, no_esc_watcher, tmp_path):
        path = str(tmp_path / "s.json")
        script = [{"tool_calls": [{"name": "get_weather"}]}, {"content": "sunny"}]
        final, _ = run_chat(make_state, script, ["q"], session_file=path)
        restored, _ = LoadSession().execute(make_state(session_file=path))
        assert restored.history == final.history
        assert restored.history.turns[0].rounds[0].tool_calls[0].name == "get_weather"
