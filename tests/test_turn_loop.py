"""
The turn loop: a turn is (user message, tool rounds*, final answer), driven by the chain

  TurnStart -> UserMessage -> NextRound -> StreamCompletion -> AppendRound -> ExecuteToolCalls(i)* -> NextRound -> ...
                                                            -> TurnEnd -> MaybeRegenerate

State between rounds lives on ChatState.pending (a PendingTurn, opened empty by TurnStart and
filled by UserMessage); only TurnEnd writes history.
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
from desh_chat.display import DisplayStats, Error, Info, Warn
import pytest

import time

from desh_chat.events import (
    AppendRound, ExecuteToolCalls, MaybeRegenerate, NextRound,
    StreamCompletion, TurnEnd, TurnStart,
)
from desh_chat.session import LoadSession, SaveSession
from desh_chat import scratchpad as pad_tools
from desh_chat.scratchpad import Scratchpad
from desh.tools import ToolRegistry
from desh_chat.state import EXPIRED_RESULT, ChatHistory, Deadline, InferenceEngine, PendingTurn, Round, Settings, ToolResult, Turn


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
    def test_opens_empty_and_is_filled_once(self):
        """The turn exists before its message: TurnStart opens it, the message source fills it."""
        placeholder = PendingTurn()
        assert placeholder.user is None and placeholder.rounds == ()
        filled = placeholder.with_user("q")
        assert filled == PendingTurn("q") and placeholder.user is None
        with pytest.raises(AssertionError):
            filled.with_user("again")

    def test_a_placeholder_cannot_be_sent_or_finished(self):
        with pytest.raises(AssertionError):
            PendingTurn().messages()
        with pytest.raises(AssertionError):
            PendingTurn().finish("a", tokens=0, cancelled=False)

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

    def test_finish_prices_final_completion_plus_rounds_as_they_will_render(self):
        """A history turn renders its rounds stubbed (Turn.messages), so finish() prices them on the
        stubbed text, not on the usage frame that priced the results whole — the frame's count
        would reserve room the request no longer spends."""
        round = Round("", (WEATHER,), (result(WEATHER, "a long result " * 40),), tokens=300)
        p = PendingTurn("q").add_round(round)
        turn = p.finish("the answer", tokens=12, cancelled=False)
        assert turn == Turn("q", "the answer", tokens=12 + estimate_tokens(round.text(stubbed=True)), rounds=p.rounds)
        assert turn.tokens < 12 + 300

    def test_finish_without_usage_estimates_everything_on_the_stubbed_rendering(self):
        round = Round("", (WEATHER,), (result(WEATHER),), tokens=30)
        p = PendingTurn("hello there").add_round(round)
        turn = p.finish("general kenobi", tokens=0, cancelled=False)
        assert turn.tokens == estimate_tokens("hello there") + estimate_tokens("general kenobi") + estimate_tokens(round.text(stubbed=True))

    def test_unpriced_round_is_estimated_from_its_stubbed_text(self):
        unpriced = Round("Let me check.", (WEATHER,), (result(WEATHER),), tokens=0)
        turn = Turn("q", "a", rounds=(unpriced,))
        assert turn.tokens == estimate_tokens("q") + estimate_tokens("a") + estimate_tokens(unpriced.text(stubbed=True))


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

    def test_a_history_turn_keeps_its_calls_and_stubs_every_result(self):
        """The final answer is what the results led to; once the turn is in history the results
        are gone from the request and the calls stay, one tool message each, for the template."""
        msgs = Turn("q", "the answer", rounds=(ROUND,)).messages()
        assert msgs[1]["tool_calls"] == ROUND.messages()[0]["tool_calls"]
        assert [m["content"] for m in msgs[2:4]] == [EXPIRED_RESULT, EXPIRED_RESULT]
        assert [m["tool_call_id"] for m in msgs[2:4]] == ["call_0", "call_1"]
        assert ROUND.results[0].content == "sunny"     # the record is intact

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

    def test_history_writes_the_current_format_and_still_reads_formats_1_and_2(self):
        assert ChatHistory().to_dict()["version"] == ChatHistory.SESSION_FORMAT == 5
        v1 = {"version": 1, "turns": [{"user": "q", "assistant": "a", "tokens": 5}]}
        assert ChatHistory.from_dict(v1).turns == (Turn("q", "a", tokens=5),)
        v2 = {"version": 2, "turns": [{"user": "q", "assistant": "a", "tokens": 5, "rounds": [ROUND.to_dict()]}]}
        assert ChatHistory.from_dict(v2).turns == (Turn("q", "a", tokens=5, rounds=(ROUND,)),)

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
        assert events[1].stop == "cap"                          # recorded on the Turn, so a reader need not count rounds


class TestAppendRoundCapScratchpad:
    """At the cap the round is refused, except for its scratchpad calls: they are the persistence
    the cap message asks for and touch nothing but the working memory, so they run, in order,
    before the turn ends. The round is still not recorded; TurnEnd snapshots the value."""
    PAD_TOOLS = (ToolRegistry()
                 .add(pad_tools.write, name="scratchpad_write", inject=("scratchpad",), confirm=False, target="key")
                 .add(pad_tools.delete, name="scratchpad_delete", inject=("scratchpad",), confirm=False, target="key"))
    CAPPED = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=1)

    def capped_state(self, make_state, scratchpad=Scratchpad()):
        pending = PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=30))
        return make_state(settings=self.CAPPED, pending=pending, tools=self.PAD_TOOLS, scratchpad=scratchpad)

    def test_the_scratchpad_calls_run_and_the_others_are_named_as_not_run(self, make_state):
        state = self.capped_state(make_state)
        write = call(0, name="scratchpad_write", arguments='{"key": "next", "kind": "todo", "value": "run the tests"}')
        new_state, events = AppendRound(assistant="so far", tool_calls=(write, TIME), tokens=10).execute(state)
        assert new_state.scratchpad == Scratchpad().with_entry("next", "todo", "run the tests")
        assert new_state.pending == state.pending                 # the capped round is not recorded
        assert isinstance(events[0], Warn) and "get_time not run" in events[0].text and "scratchpad_write" not in events[0].text
        assert isinstance(events[1], Info) and "scratchpad_write next" in events[1].text
        assert isinstance(events[2], TurnEnd) and events[2].stop == "cap"
        ended, _ = events[2].execute(new_state)
        assert ended.history.turns[-1].scratchpad == new_state.scratchpad     # the snapshot carries it

    def test_a_round_of_scratchpad_calls_only_names_nothing_as_not_run(self, make_state):
        state = self.capped_state(make_state)
        write = call(0, name="scratchpad_write", arguments='{"key": "k", "kind": "fact", "value": "v"}')
        new_state, events = AppendRound(assistant="", tool_calls=(write,), tokens=10).execute(state)
        assert new_state.scratchpad == Scratchpad().with_entry("k", "fact", "v")
        assert isinstance(events[0], Warn) and "not run" not in events[0].text and "scratchpad_write k" in events[0].text
        assert isinstance(events[1], TurnEnd) and events[1].stop == "cap"

    def test_the_calls_run_in_order(self, make_state):
        state = self.capped_state(make_state, Scratchpad().with_entry("old", "fact", "1"))
        write = call(0, name="scratchpad_write", arguments='{"key": "k", "kind": "fact", "value": "v"}')
        delete = call(1, name="scratchpad_delete", arguments='{"key": "k"}')
        new_state, _ = AppendRound(assistant="", tool_calls=(write, delete), tokens=10).execute(state)
        assert new_state.scratchpad == Scratchpad().with_entry("old", "fact", "1")

    def test_without_a_working_memory_the_call_is_left_unrun(self, make_state):
        state = self.capped_state(make_state, scratchpad=None)
        write = call(0, name="scratchpad_write", arguments='{"key": "k", "kind": "fact", "value": "v"}')
        new_state, events = AppendRound(assistant="", tool_calls=(write,), tokens=10).execute(state)
        assert new_state.scratchpad is None
        assert isinstance(events[0], Warn) and "scratchpad_write not run" in events[0].text
        assert isinstance(events[1], TurnEnd)


class TestAppendRoundDeadline:
    """The run's wall-clock budget is the second budget checked before a round runs. It ends the turn
    the way the cap does — the model's text so far is the answer, the calls are named as not run —
    under its own stop reason, and it is checked first: a run out of time is never continued."""

    def test_a_passed_deadline_ends_the_turn_and_names_the_calls(self, make_state):
        state = make_state(pending=PendingTurn("q"), deadline=Deadline(at=time.monotonic() - 1, budget=300))
        new_state, events = AppendRound(assistant="so far", tool_calls=(TIME,), tokens=10).execute(state)
        assert new_state.pending == state.pending             # the refused round is not recorded
        assert isinstance(events[0], Warn) and "get_time" in events[0].text and "300s" in events[0].text
        assert isinstance(events[1], TurnEnd)
        assert events[1].assistant == "so far" and events[1].cancelled is False and events[1].stop == "deadline"

    def test_a_deadline_still_ahead_lets_the_round_run(self, make_state):
        state = make_state(pending=PendingTurn("q"), deadline=Deadline.in_seconds(60))
        _, events = AppendRound(assistant="", tool_calls=(WEATHER,), tokens=10).execute(state)
        assert events == [ExecuteToolCalls(index=0)]

    def test_the_deadline_wins_over_the_round_cap(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=1)
        pending = PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=30))
        state = make_state(settings=settings, pending=pending, deadline=Deadline(at=time.monotonic() - 1, budget=5))
        _, events = AppendRound(assistant="", tool_calls=(TIME,), tokens=10).execute(state)
        assert isinstance(events[1], TurnEnd) and events[1].stop == "deadline"

    def test_a_timed_out_turn_is_not_continued_by_the_auto_prompt(self, make_state):
        history = ChatHistory().append(Turn(user="q", assistant="so far", stop="deadline"))
        state = make_state(history=history, operator=False, auto_prompt="go on")
        new_state, events = TurnStart().execute(state)
        assert events == [] and new_state.pending is None     # the drain branch: the run returns


def round_of(assistant, calls, contents, round_no) -> Round:
    """A completed round whose call ids are fresh for round_no — as every real round's are — so a
    guard that compares rounds must look at (name, arguments) and (name, content), never wire ids."""
    fresh = tuple(call(tc.index, tc.name, tc.arguments, id=f"r{round_no}_{tc.index}") for tc in calls)
    return Round(assistant, fresh, tuple(result(tc, c) for tc, c in zip(fresh, contents)), tokens=10)


class TestAppendRoundLoopGuard:
    """A model that asks for the same calls a third time, having twice seen the same results, is
    looping: the third round is not recorded and the turn ends the way the round cap ends it."""

    def two_identical_rounds(self, calls=(WEATHER,), contents=("sunny",)) -> PendingTurn:
        return PendingTurn("q").add_round(round_of("trying", calls, contents, 1)).add_round(round_of("again", calls, contents, 2))

    def test_a_third_identical_call_after_two_identical_results_ends_the_turn(self, make_state):
        pending = self.two_identical_rounds()
        third = call(0, id="r3_0")
        new_state, events = AppendRound(assistant="one more?", tool_calls=(third,), tokens=10).execute(make_state(pending=pending))
        assert new_state.pending == pending                    # the looping round is not recorded
        assert isinstance(events[0], Warn) and "get_weather" in events[0].text
        assert isinstance(events[1], TurnEnd) and events[1].cancelled is False and events[1].tokens == 10
        assert "one more?" in events[1].assistant             # what the model said is kept...
        assert "repeat" in events[1].assistant.lower()        # ...and the reader learns why it stopped

    def test_two_identical_calls_are_allowed(self, make_state):
        pending = PendingTurn("q").add_round(round_of("trying", (WEATHER,), ("sunny",), 1))
        new_state, events = AppendRound(assistant="", tool_calls=(call(0, id="r2_0"),), tokens=10).execute(make_state(pending=pending))
        assert len(new_state.pending.rounds) == 2 and events == [ExecuteToolCalls()]

    def test_identical_calls_with_changing_results_are_polling_not_a_loop(self, make_state):
        pending = PendingTurn("q").add_round(round_of("", (WEATHER,), ("pending",), 1)).add_round(round_of("", (WEATHER,), ("still pending",), 2))
        new_state, events = AppendRound(assistant="", tool_calls=(call(0, id="r3_0"),), tokens=10).execute(make_state(pending=pending))
        assert len(new_state.pending.rounds) == 3 and events == [ExecuteToolCalls()]

    def test_a_changed_argument_breaks_the_streak(self, make_state):
        pending = self.two_identical_rounds()
        other = call(0, arguments='{"city": "Lima"}', id="r3_0")
        new_state, events = AppendRound(assistant="", tool_calls=(other,), tokens=10).execute(make_state(pending=pending))
        assert len(new_state.pending.rounds) == 3 and events == [ExecuteToolCalls()]

    def test_a_reworded_reason_does_not_hide_a_repeated_command(self, make_state):
        """The loophole a model found in a real run: the same Bash command with a counter in the
        reason ("... forty-first try") passed the guard forever, because the guard compared the
        whole arguments string. With Bash declaring identity=("command",), the reason is wording."""
        from desh.tools import ToolRegistry
        def bash(reason: str, command: str) -> str:
            return "same output"
        tools = ToolRegistry().add(bash, name="Bash", identity=("command",))
        def attempt(n: int) -> ToolCall:
            return call(0, name="Bash", arguments=f'{{"reason": "attempt {n}", "command": "pytest -q"}}', id=f"r{n}_0")
        pending = (PendingTurn("q")
                   .add_round(Round("", (attempt(1),), (result(attempt(1), "1 failed"),), tokens=10))
                   .add_round(Round("", (attempt(2),), (result(attempt(2), "1 failed"),), tokens=10)))
        state = make_state(pending=pending, tools=tools)
        new_state, events = AppendRound(assistant="", tool_calls=(attempt(3),), tokens=10).execute(state)
        assert new_state.pending == pending
        assert [type(e) for e in events] == [Warn, TurnEnd]
        # ...while the same three calls through a registry that declares no identity are three different calls
        plain = make_state(pending=pending, tools=ToolRegistry().add(bash, name="Bash"))
        new_state, events = AppendRound(assistant="", tool_calls=(attempt(3),), tokens=10).execute(plain)
        assert len(new_state.pending.rounds) == 3 and events == [ExecuteToolCalls()]

    def test_the_whole_round_is_compared(self, make_state):
        pending = self.two_identical_rounds(calls=(WEATHER, TIME), contents=("sunny", "10:00"))
        same = (call(0, id="r3_0"), call(1, name="get_time", arguments='{"tz": "CLT"}', id="r3_1"))
        _, events = AppendRound(assistant="", tool_calls=same, tokens=10).execute(make_state(pending=pending))
        assert [type(e) for e in events] == [Warn, TurnEnd]
        partly = (call(0, id="r3_0"), call(1, name="get_time", arguments='{"tz": "UTC"}', id="r3_1"))
        new_state, events = AppendRound(assistant="", tool_calls=partly, tokens=10).execute(make_state(pending=pending))
        assert len(new_state.pending.rounds) == 3 and events == [ExecuteToolCalls()]


class TestExecuteToolCalls:
    def test_one_call_per_step_each_unavailable_then_the_next_round(self, make_state):
        state = make_state(pending=PendingTurn("q").add_round(Round("", (WEATHER, TIME))))
        mid, events = ExecuteToolCalls().execute(state)
        assert [(r.tool_call_id, r.name) for r in mid.pending.rounds[-1].results] == [("call_0", "get_weather")]
        assert [type(e) for e in events] == [Info, DisplayStats, ExecuteToolCalls] and events[2].index == 1
        assert "get_weather" in events[0].text
        final, events = events[2].execute(mid)
        results = final.pending.rounds[-1].results
        assert [(r.tool_call_id, r.name) for r in results] == [("call_0", "get_weather"), ("call_1", "get_time")]
        assert all("not available" in r.content for r in results)
        assert [type(e) for e in events] == [Info, DisplayStats, NextRound]


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
        assert events[1].stop == "overflow"             # cancelled keeps it out of the view; stop says why

    def test_turn_end_writes_stop_on_the_turn(self, make_state):
        state = make_state(pending=PendingTurn("q").add_round(Round("", (WEATHER,), (result(WEATHER),), tokens=200)))
        capped, _ = TurnEnd(assistant="so far", tokens=5, stop="cap").execute(state)
        overflow, _ = TurnEnd(assistant="", tokens=0, cancelled=True, stop="overflow").execute(state)
        plain, _ = TurnEnd(assistant="done", tokens=5).execute(state)
        assert capped.history.turns[0].stop == "cap" and capped.history.turns[0].cancelled is False
        assert overflow.history.turns[0].stop == "overflow" and overflow.history.turns[0].cancelled is True
        assert plain.history.turns[0].stop == ""


# ---------------------
# Full loop through the real Engine
# ---------------------

def run_chat(make_state, script, inputs, **overrides):
    """Drive the loop head through a scripted FakeServer with the given user inputs, then EOF."""
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
        return Engine[type(state)]().run(state, seed=[MaybeRegenerate()]), server
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
        assert turn.stop == "cap"
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


# ---------------------
# The re-read guard: a loop of any period
# ---------------------

class TestRereadGuard:
    """A read-only call answered twice this turn, with nothing written, edited or run since, is
    answered a third time with a notice instead of running. The two-round repeat guard sees a
    loop of period one; this one sees a model cycling through the same reads whatever the period
    (at 4k: two 30-round turns of the same three reads, re-read after every checkpoint)."""

    def registry(self):
        from desh.tools import ToolRegistry
        seen = []
        def read(file_path: str, offset: int = 1) -> str:
            seen.append((file_path, offset)); return f"contents of {file_path}@{offset}"
        def edit(file_path: str, old_string: str, new_string: str) -> str:
            return "edited"
        def bash(reason: str, command: str) -> str:
            return "1 failed"
        reg = (ToolRegistry().add(read, name="Read", confirm=False, target="file_path")
                             .add(edit, name="Edit", target="file_path")
                             .add(bash, name="Bash", identity=("command",), target="command"))
        return reg, seen

    @staticmethod
    def rd(path: str, n: int, offset: int = 1) -> ToolCall:
        return call(0, name="Read", arguments=f'{{"file_path": "{path}", "offset": {offset}}}', id=f"r{n}_0")

    def answered(self, *calls: ToolCall) -> PendingTurn:
        p = PendingTurn("q")
        for i, tc in enumerate(calls):
            p = p.add_round(Round("", (tc,), (result(tc, f"contents of {tc.arguments}"),), tokens=10 * (i + 1)))
        return p

    def test_the_third_read_of_the_same_thing_is_answered_with_the_notice(self, make_state):
        reg, seen = self.registry()
        pending = self.answered(self.rd("a.py", 1), self.rd("b.py", 2), self.rd("a.py", 3), self.rd("b.py", 4)).add_round(Round("", (self.rd("a.py", 5),), tokens=50))
        state = make_state(pending=pending, tools=reg, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        new_state, events = ExecuteToolCalls(0).execute(state)
        answer = new_state.pending.rounds[-1].results[0].content
        assert answer.startswith("Not run: Read a.py has already been answered 2 times") and "scratchpad" in answer
        assert seen == []                                            # the tool did not run
        assert [type(e) for e in events] == [Warn, DisplayStats, NextRound]   # the turn goes on

    def test_two_reads_are_allowed_and_a_different_range_is_a_different_read(self, make_state):
        reg, seen = self.registry()
        pending = self.answered(self.rd("a.py", 1), self.rd("a.py", 2)).add_round(Round("", (self.rd("a.py", 3, offset=60),), tokens=30))
        state = make_state(pending=pending, tools=reg, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        new_state, _ = ExecuteToolCalls(0).execute(state)
        assert seen == [("a.py", 60)] and new_state.pending.rounds[-1].results[0].content == "contents of a.py@60"

    def test_acting_in_between_resets_the_count(self, make_state):
        reg, seen = self.registry()
        edit = call(0, name="Edit", arguments='{"file_path": "a.py", "old_string": "x", "new_string": "y"}', id="e_0")
        pending = (self.answered(self.rd("a.py", 1), self.rd("a.py", 2))
                   .add_round(Round("", (edit,), (result(edit, "edited"),), tokens=30))
                   .add_round(Round("", (self.rd("a.py", 4),), tokens=40)))
        state = make_state(pending=pending, tools=reg, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        ExecuteToolCalls(0).execute(state)
        assert seen == [("a.py", 1)]                                  # the third read after an edit is a fresh look

    def test_a_confirming_tool_is_never_counted(self, make_state):
        """Re-running the tests is polling, and a Bash round is itself where the count stops."""
        reg, _ = self.registry()
        def pytest_call(n: int) -> ToolCall:
            return call(0, name="Bash", arguments=f'{{"reason": "try {n}", "command": "pytest -q"}}', id=f"b{n}_0")
        pending = self.answered(pytest_call(1), pytest_call(2)).add_round(Round("", (pytest_call(3),), tokens=30))
        state = make_state(pending=pending, tools=reg, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        new_state, _ = ExecuteToolCalls(0).execute(state)
        assert new_state.pending.rounds[-1].results[0].content == "1 failed"

    def test_folded_rounds_still_count(self, make_state):
        """The reads a checkpoint folded are on the record: the loop the guard is for is the one
        where every re-read follows a checkpoint."""
        reg, seen = self.registry()
        pending = self.answered(self.rd("a.py", 1), self.rd("b.py", 2), self.rd("a.py", 3)).compact("c").add_round(Round("", (self.rd("a.py", 4),), tokens=40))
        state = make_state(pending=pending, tools=reg, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, auto=True))
        new_state, _ = ExecuteToolCalls(0).execute(state)
        assert seen == [] and new_state.pending.rounds[-1].results[0].content.startswith("Not run: Read a.py")
