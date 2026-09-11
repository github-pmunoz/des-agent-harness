"""Unit tests for the ported desh_chat turn chain (the one-round case; multi-round
tool loops are covered in test_turn_loop.py):

  MaybeRegenerate -> TurnStart -> [DisplayStats, PromptUser] -> UserMessage -> NextRound
    -> (CompactHistory -> [Info, LogCompletion], NextRound(compacted=True) ->)
    -> StreamCompletion -> [TurnEnd, LogCompletion] -> TurnEnd -> MaybeRegenerate

TurnStart opens state.pending empty and UserMessage fills it; NextRound is where the
budget, the compaction decision and the request are built, so tests that inspect the
request drive all three via open_turn().

MaybeRegenerate owns the running check only. TurnStart is the 2-fan: with an operator
it schedules DisplayStats and PromptUser as siblings. DisplayStats/DisplayHistory/Info/
Warn are pure sinks (execute() returns [] — no further events).

Each event is driven directly (state, events = Event(...).execute(state)),
same style as test_command_surface.py's Command tests, plus one full
Engine.run() test at the end for the end-to-end termination guarantee.

FakeServer/FakeESCWatcher (conftest.py) stand in for the network and the
real terminal-raw-mode watcher — no live router, no real stdin required.
"""
from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.engine import Engine
from desh.llama.logger import Logger
from desh.llama.wire import Request
from desh.llama.tokens import estimate_tokens
from desh_chat.display import DisplayStats, Error, Info
from desh_chat.events import (
    CompactHistory, Exit, LogCompletion, MaybeRegenerate,
    NextRound, PromptUser, StreamCompletion, TurnEnd, TurnStart, UserMessage,
)
from desh_chat.handlers import on_interrupt
from desh_chat.state import ChatHistory, InferenceEngine, PendingTurn, Settings, Turn


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


def open_turn(state, message: str):
    """TurnStart(message) -> UserMessage -> NextRound, returning NextRound's (state, events): the
    request-building step."""
    state, events = TurnStart(message).execute(state)
    assert [type(e) for e in events] == [UserMessage]
    state, events = events[0].execute(state)
    assert [type(e) for e in events] == [NextRound]
    return events[0].execute(state)


# ---------------------
# TurnStart
# ---------------------

class TestTurnStart:
    """The only creator of state.pending, and the policy over where a turn's message comes from."""

    def test_with_an_operator_opens_an_empty_turn_and_prompts(self, make_state):
        new_state, events = TurnStart().execute(make_state(operator=True))
        assert new_state.pending == PendingTurn() and new_state.pending.user is None
        assert [type(e) for e in events] == [DisplayStats, PromptUser]

    def test_a_seed_message_opens_the_turn_and_delivers_it_in_one_step(self, make_state):
        new_state, events = TurnStart("the task").execute(make_state(operator=False))
        assert new_state.pending == PendingTurn()
        assert events == [UserMessage("the task")]

    def test_a_command_leaves_the_placeholder_and_the_loop_head_reuses_it(self, make_state):
        """PromptUser -> Command -> MaybeRegenerate -> TurnStart: the placeholder opened before the
        command is the one the next message fills, not a second one."""
        opened, _ = TurnStart().execute(make_state(operator=True))
        again, events = TurnStart().execute(opened)
        assert again.pending is opened.pending
        assert [type(e) for e in events] == [DisplayStats, PromptUser]

    def test_without_an_operator_a_finished_turn_drains_and_opens_nothing(self, make_state):
        for history in (ChatHistory(),
                        ChatHistory().append(Turn("q", "done")),
                        ChatHistory().append(Turn("q", "", cancelled=True))):
            new_state, events = TurnStart().execute(make_state(operator=False, history=history))
            assert events == [] and new_state.pending is None, history

    def test_a_capped_turn_is_continued_with_the_auto_prompt(self, make_state):
        capped = Turn("q", "so far", stop="cap")
        for history in (ChatHistory().append(capped), ChatHistory().append(capped).compact("s")):
            new_state, events = TurnStart().execute(make_state(operator=False, auto_prompt="go", history=history))
            assert new_state.pending == PendingTurn()
            assert [type(e) for e in events] == [Info, UserMessage] and events[-1] == UserMessage("go")

    def test_the_auto_prompt_beats_the_operator_and_needs_a_capped_turn(self, make_state):
        capped = ChatHistory().append(Turn("q", "so far", stop="cap"))
        _, events = TurnStart().execute(make_state(operator=True, auto_prompt="go", history=capped))
        assert events[-1] == UserMessage("go")
        done = ChatHistory().append(Turn("q", "done"))
        _, events = TurnStart().execute(make_state(operator=True, auto_prompt="go", history=done))
        assert [type(e) for e in events] == [DisplayStats, PromptUser]
        _, events = TurnStart().execute(make_state(operator=True, auto_prompt=None, history=capped))
        assert [type(e) for e in events] == [DisplayStats, PromptUser]

    def test_a_cancelled_capped_turn_is_not_continued(self, make_state):
        """An overflow recorded by NextRound is a cancelled turn: the loop head must see it and stop,
        or the same auto prompt would go out again forever."""
        history = ChatHistory().append(Turn("go", "", cancelled=True, stop="overflow"))
        new_state, events = TurnStart().execute(make_state(operator=False, auto_prompt="go", history=history))
        assert events == [] and new_state.pending is None

    def test_reaching_the_loop_head_mid_turn_is_a_harness_bug(self, make_state):
        import pytest
        with pytest.raises(AssertionError):
            TurnStart().execute(make_state(pending=PendingTurn("q")))

    def test_maybe_regenerate_only_checks_running(self, make_state):
        _, events = MaybeRegenerate().execute(make_state(running=True))
        assert events == [TurnStart()]
        _, events = MaybeRegenerate().execute(make_state(running=False))
        assert events == []


# ---------------------
# UserMessage
# ---------------------

class TestUserMessageBudget:
    """Hand-verified gen_budget under each of the three ceilings in the min()."""

    def test_max_turn_tokens_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, turn_token_cap=0.5)
        state = make_state(settings=settings)
        _, events = open_turn(state, "hi")
        assert isinstance(events[0], StreamCompletion)
        assert events[0].request.max_tokens == 100

    def test_turn_token_cap_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=10_000, turn_token_cap=0.5)
        state = make_state(settings=settings)
        _, events = open_turn(state, "hi")
        assert events[0].request.max_tokens == 500  # 0.5 * 1000

    def test_remaining_context_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=50, max_turn_tokens=10_000, turn_token_cap=0.9)
        state = make_state(settings=settings)
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        msg_tokens = estimate_tokens("hi")
        expected = 50 - sys_prompt_tokens - msg_tokens  # window_tokens() is 0, no prior history
        _, events = open_turn(state, "hi")
        assert events[0].request.max_tokens == expected
        assert expected < 45  # confirms this is genuinely the tightest of the three ceilings

    def test_a_message_that_cannot_fit_ends_as_an_overflow_turn_without_a_request(self, make_state, capsys):
        """No window to compact (history is empty), so there is nothing to try: the turn is recorded
        cancelled with stop="overflow" — a record, not a dropped message, so an auto prompt that
        cannot fit is not issued again."""
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1, max_turn_tokens=100, turn_token_cap=1.0)
        state = make_state(settings=settings)
        new_state, events = open_turn(state, "a message long enough to blow a context of 1 token")
        assert [type(e) for e in events] == [Error, TurnEnd]
        assert "exceeds context window" in events[0].text
        assert events[1].cancelled is True and events[1].stop == "overflow"
        assert new_state.pending is not None    # TurnEnd, not NextRound, clears it


class TestUserMessageHistoryView:
    """Regression coverage for the view()-budget inversion bug: the value
    passed to history.view() must be room LEFT for history (context minus
    what's reserved for sys prompt + this message + the reply), not the
    reserved figure itself. Getting this backwards makes prior turns vanish
    from the request on the very next turn.
    """

    def test_prior_turn_is_included_when_there_is_room(self, make_state):
        history = ChatHistory().append(Turn("previous question", "previous answer"))
        state = make_state(history=history, settings=Settings(
            model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192))
        _, events = open_turn(state, "a new question")
        contents = [m["content"] for m in events[0].request.messages]
        assert "previous question" in contents
        assert "previous answer" in contents
        assert "a new question" in contents

    def test_prior_turn_still_fits_at_the_tightest_legitimate_boundary(self, make_state):
        """Push gen_budget down to being bound by the context-headroom ceiling
        itself (context - used_tokens - window_tokens()) rather than
        max_turn_tokens/turn_token_cap — the tightest case the formula
        produces without outright rejecting the turn. At this exact boundary,
        view()'s budget comes out equal to window_tokens(), and the prior
        turn must still be included (view()'s break condition is a strict
        `>`, not `>=`).

        Contrast with the pre-fix formula: it passed view(used_tokens) where
        used_tokens = msg_tokens + sys_prompt_tokens only (no gen_budget, no
        window headroom) — here that's 10, far smaller than the prior turn's
        50 tokens, so the old code would have dropped it. The fixed formula
        passes context - reserved = 50, which exactly covers it.
        """
        system_prompt = "You are a helpful assistant."
        message = "a new question"
        prior_turn = Turn("previous question", "previous answer", tokens=50)
        history = ChatHistory().append(prior_turn)
        sys_prompt_tokens = estimate_tokens(system_prompt)
        msg_tokens = estimate_tokens(message)
        old_buggy_budget = msg_tokens + sys_prompt_tokens
        assert old_buggy_budget < prior_turn.tokens, "test setup: old formula must have under-shot the prior turn"

        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=100, max_turn_tokens=1000, turn_token_cap=1.0)
        state = make_state(history=history, settings=settings, system_prompt=system_prompt)
        _, events = open_turn(state, message)

        req = events[0].request
        assert req.max_tokens == 100 - sys_prompt_tokens - msg_tokens - prior_turn.tokens  # context-headroom ceiling bound
        contents = [m["content"] for m in req.messages]
        assert "previous question" in contents
        assert "previous answer" in contents
        assert message in contents

    def test_request_message_order_is_system_then_history_then_new_user_message(self, make_state):
        history = ChatHistory().append(Turn("q1", "a1"))
        state = make_state(history=history)
        _, events = open_turn(state, "q2")
        roles_and_last = [(m["role"], m["content"]) for m in events[0].request.messages]
        assert roles_and_last[0] == ("system", state.system_prompt)
        assert roles_and_last[-1] == ("user", "q2")


class TestUserMessageRequestShape:
    def test_request_carries_current_settings(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.9, think=True, context=16384, max_turn_tokens=8192)
        state = make_state(settings=settings)
        _, events = open_turn(state, "hello")
        req = events[0].request
        assert req.model == MODELS[0]
        assert req.temperature == 0.9
        assert req.think is True
        assert req.stream is True


# ---------------------
# StreamCompletion
# ---------------------

class TestStreamCompletion:
    def test_happy_path_emits_turn_end_with_the_streamed_content(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "the answer", "finish_reason": "stop"}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "the question"}],
                      model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req).execute(state)
        assert len(events) == 1
        assert isinstance(events[0], TurnEnd)
        assert events[0].assistant == "the answer"
        assert events[0].cancelled is False

    def test_prints_assistant_prefix_and_content(self, make_state, no_esc_watcher, capsys):
        server = FakeServer(script=[{"content": "hello world"}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0], stream=True)
        StreamCompletion(request=req).execute(state)
        out = capsys.readouterr().out
        assert "Assistant:" in out
        assert "hello world" in out

    def test_cancelled_completion_produces_a_cancelled_append_turn(self, make_state, no_esc_watcher, capsys):
        server = FakeServer(script=[{"content": "partial", "finish_reason": "cancelled"}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req).execute(state)
        assert events[1].cancelled is True
        assert "cancelled" in events[0].text

    def test_logs_completion_when_completions_log_is_configured(self, make_state, no_esc_watcher, tmp_path):
        server = FakeServer(script=[{"content": "logged answer"}])
        log_path = tmp_path / "completions.jsonl"
        state = with_server(make_state, server, completions_log=Logger(str(log_path)))
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req).execute(state)
        assert any(isinstance(e, LogCompletion) for e in events)
        assert len(events) == 2  # AppendTurn + LogCompletion

    def test_does_not_emit_log_completion_when_completions_log_is_none(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "unlogged answer"}])
        state = with_server(make_state, server, completions_log=None)
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req).execute(state)
        assert len(events) == 1
        assert isinstance(events[0], TurnEnd)


# ---------------------
# TurnEnd
# ---------------------

class TestTurnEnd:
    def test_appends_the_pending_turn_and_returns_to_the_loop_head(self, make_state):
        """No compaction on this path: whether the next request needs one is NextRound's call."""
        state = make_state(pending=PendingTurn("q"))
        new_state, events = TurnEnd(assistant="a", cancelled=False).execute(state)
        assert len(new_state.history) == 1
        assert new_state.history.turns[0].user == "q"
        assert new_state.history.turns[0].assistant == "a"
        assert new_state.pending is None
        assert events == [MaybeRegenerate()]

    def test_original_state_is_untouched_immutability(self, make_state):
        state = make_state(pending=PendingTurn("q"))
        TurnEnd(assistant="a", cancelled=False).execute(state)
        assert len(state.history) == 0
        assert state.pending == PendingTurn("q")

    def test_cancelled_flag_lands_on_the_turn_not_on_tokens(self, make_state):
        """Regression: Turn(user, assistant, self.cancelled) once landed the
        cancelled bool in the positional `tokens` slot instead of `cancelled`.
        Masked when cancelled=False (False==0 lets __post_init__ recompute
        tokens correctly); broke silently whenever cancelled=True (tokens
        frozen at 1, cancelled flag never actually set).
        """
        long_user = "a reasonably long user message that is not four characters"
        long_assistant = "a reasonably long assistant reply that is not four characters"
        new_state, _ = TurnEnd(assistant=long_assistant, cancelled=True).execute(make_state(pending=PendingTurn(long_user)))
        turn = new_state.history.turns[0]
        assert turn.cancelled is True
        assert turn.tokens == estimate_tokens(long_user) + estimate_tokens(long_assistant)
        assert turn.tokens != 1

    def test_non_cancelled_turn_tokens_are_also_correct(self, make_state):
        state = make_state(pending=PendingTurn("a decent length user message here"))
        new_state, _ = TurnEnd(assistant="a decent length assistant reply here", cancelled=False).execute(state)
        turn = new_state.history.turns[0]
        assert turn.cancelled is False
        assert turn.tokens == estimate_tokens(turn.user) + estimate_tokens(turn.assistant)


# ---------------------
# NextRound: when a request compacts first
# ---------------------

class TestNextRoundCompaction:
    """Compaction is paid only when a request is about to go out and the window leaves less room
    than min_gen_tokens() = (1 - compaction_threshold) * context. With context=1000 and a 0.5
    threshold the floor is 500; the system prompt is 7 tokens and "hi" one, so a window of W
    leaves 992 - W."""
    SETTINGS = Settings(model=MODELS[0], temperature=0.3, think=False,
                        context=1000, max_turn_tokens=100, compaction_threshold=0.5)

    def test_the_floor_is_derived_from_the_threshold(self, make_state):
        assert make_state(settings=self.SETTINGS).min_gen_tokens() == 500

    def test_enough_room_streams_without_compacting(self, make_state):
        history = ChatHistory().append(Turn("short", "reply", tokens=100))    # leaves 892
        _, events = open_turn(make_state(settings=self.SETTINGS, history=history), "hi")
        assert [type(e) for e in events] == [StreamCompletion]

    def test_too_little_room_compacts_then_retries_the_same_round(self, make_state):
        history = ChatHistory().append(Turn("long", "reply", tokens=500))     # leaves 492
        new_state, events = open_turn(make_state(settings=self.SETTINGS, history=history), "hi")
        assert [type(e) for e in events] == [Info, CompactHistory, NextRound]
        assert events[2] == NextRound(compacted=True)
        assert new_state.pending is not None and new_state.pending.rounds == ()    # the turn is untouched

    def test_the_retry_is_built_from_the_compacted_history(self, make_state):
        """CompactHistory rewrites history and schedules nothing; NextRound(compacted=True) reads the
        state as it is by then, so the request carries the summary and not the turns behind it."""
        server = FakeServer(script=[{"content": "a tidy summary"}, {"content": "unused"}])
        history = ChatHistory().append(Turn("long question", "long reply", tokens=500))
        state = with_server(make_state, server, settings=self.SETTINGS, history=history)
        state, events = open_turn(state, "hi")
        state, _ = events[1].execute(state)                  # CompactHistory
        _, events = events[2].execute(state)                 # NextRound(compacted=True)
        assert [type(e) for e in events] == [StreamCompletion]
        contents = [m["content"] for m in events[0].request.messages]
        assert ChatHistory.SUMMARY_PREFIX + "a tidy summary" in contents
        assert "long question" not in contents

    def test_still_short_after_compacting_ends_the_turn_as_an_overflow(self, make_state):
        history = ChatHistory().append(Turn("long", "reply", tokens=500))
        state, _ = open_turn(make_state(settings=self.SETTINGS, history=history), "hi")
        _, events = NextRound(compacted=True).execute(state)
        assert [type(e) for e in events] == [Error, TurnEnd]
        assert events[1].cancelled is True and events[1].stop == "overflow"

    def test_a_window_that_is_only_a_summary_is_not_compacted_again(self, make_state):
        """Summarising the summary cannot free room: a message too large for what is left goes
        straight to the overflow record without paying for a compaction."""
        history = ChatHistory().compact("s", tokens=500)
        _, events = open_turn(make_state(settings=self.SETTINGS, history=history), "hi")
        assert [type(e) for e in events] == [Error, TurnEnd]

    def test_the_window_is_measured_as_it_is(self, make_state):
        """Regression from the post-turn design: the just-finished turn is already in the window,
        so nothing may add it a second time. A 450 window leaves 542, above the 500 floor."""
        history = (ChatHistory()
                   .append(Turn("earlier", "reply", tokens=350))
                   .append(Turn("latest", "reply", tokens=100)))
        state = make_state(settings=self.SETTINGS, history=history)
        assert state.history.window_tokens() == 450
        _, events = open_turn(state, "hi")
        assert [type(e) for e in events] == [StreamCompletion], "compaction fired early — the last turn was double-counted"

    def test_cancelled_turns_do_not_count(self, make_state):
        """window_tokens() excludes cancelled turns at the source (since_last_summary()'s own
        filter), so a cancelled turn cannot move this check either way."""
        history = ChatHistory().append(Turn("cancelled", "partial", tokens=900, cancelled=True))
        _, events = open_turn(make_state(settings=self.SETTINGS, history=history), "hi")
        assert [type(e) for e in events] == [StreamCompletion]

    def test_the_floor_is_on_the_room_not_on_the_capped_budget(self, make_state):
        """max_turn_tokens (100 here) is below the 500 floor: gen_budget can never reach the floor,
        but the window leaves plenty, so no compaction. Comparing the budget instead of the room
        would compact on every request and overflow on every retry."""
        _, events = open_turn(make_state(settings=self.SETTINGS), "hi")
        assert [type(e) for e in events] == [StreamCompletion]
        assert events[0].request.max_tokens == 100


# ---------------------
# CompactHistory
# ---------------------

class TestCompactHistory:
    def test_happy_path_replaces_history_and_logs(self, make_state, capsys):
        server = FakeServer(script=[{"content": "a tidy summary"}])
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
        history = ChatHistory().append(Turn("what happened", "some stuff happened"))
        state = with_server(make_state, server, settings=settings, history=history)

        new_state, events = CompactHistory().execute(state)

        assert len(new_state.history.turns) == 2  # original turn stays in the tuple (lifetime record)...
        assert new_state.history.since_last_summary() == [new_state.history.turns[-1]]  # ...but is unreachable via the window now
        assert new_state.history.turns[-1].summary is True
        assert "a tidy summary" in new_state.history.turns[-1].user

        # no successor: the caller (NextRound, /compact) sequences what follows the compaction
        assert [type(e) for e in events] == [Info, LogCompletion]

        events[0].execute(new_state)  # the summary is now surfaced to the user, not just logged
        assert "a tidy summary" in capsys.readouterr().out

    def test_the_instruction_comes_from_settings(self, make_state):
        """The compaction prompt is a setting with a default written for tool sessions, so a run
        (an eval of compaction strategies) can be built with another without touching the event."""
        from desh_chat.state import COMPACTION_PROMPT
        server = FakeServer(script=[{"content": "summary"}, {"content": "summary"}])
        history = ChatHistory().append(Turn("q", "a", tokens=100))
        CompactHistory().execute(with_server(make_state, server, history=history))
        assert server.calls[0][1].messages[0] == {"role": "system", "content": COMPACTION_PROMPT}
        for must_keep in ("file path", "line numbers", "verbatim", "next steps"):
            assert must_keep in COMPACTION_PROMPT
        custom = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192,
                          compaction_prompt="Summarise for a robot.")
        CompactHistory().execute(with_server(make_state, server, settings=custom, history=history))
        assert server.calls[1][1].messages[0] == {"role": "system", "content": "Summarise for a robot."}

    def test_compaction_request_is_non_streaming_and_deterministic(self, make_state):
        server = FakeServer(script=[{"content": "summary"}])
        history = ChatHistory().append(Turn("q", "a", tokens=100))
        state = with_server(make_state, server, history=history)
        CompactHistory().execute(state)
        assert server.calls[0][0] == "complete"  # not stream()
        req = server.calls[0][1]
        assert req.stream is False
        assert req.temperature == 0.0
        assert req.think is False

    def test_floors_and_warns_when_computed_budget_is_too_tight(self, make_state, capsys):
        """Reproduces the small-context edge case (originally hit at -c 512):
        instruction + transcript alone can exceed a small context, driving
        the computed gen_budget negative. min_compaction_tokens is the floor.
        """
        server = FakeServer(script=[{"content": "summary"}])
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=100, max_turn_tokens=100, turn_token_cap=1.0,
                             min_compaction_tokens=64)
        long_transcript_turn = Turn("a long user turn " * 10, "a long assistant reply " * 10, tokens=200)
        history = ChatHistory().append(long_transcript_turn)
        state = with_server(make_state, server, settings=settings, history=history)

        CompactHistory().execute(state)

        assert server.calls[0][1].max_tokens == 64
        out = capsys.readouterr().out
        assert "tight on room" in out

    def test_does_not_warn_when_there_is_enough_room(self, make_state, capsys):
        server = FakeServer(script=[{"content": "summary"}])
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192)
        history = ChatHistory().append(Turn("q", "a", tokens=100))
        state = with_server(make_state, server, settings=settings, history=history)
        CompactHistory().execute(state)
        assert "tight on room" not in capsys.readouterr().out


# ---------------------
# Full chain, real Engine
# ---------------------

class TestFullEngineRun:
    def test_prompt_completion_and_eof_terminate_cleanly_no_livelock(self, make_state, no_esc_watcher, monkeypatch):
        """End-to-end regression: MaybeRegenerate -> TurnStart -> [DisplayStats, PromptUser]
        -> UserMessage -> NextRound -> StreamCompletion -> TurnEnd -> MaybeRegenerate
        -> TurnStart -> PromptUser -> (EOF) -> Exit -> running=False -> [Info] -> []
        -> queue drains. Exit never touches MaybeRegenerate — it settles directly
        through its own Info sink. Engine.run() must return, not hang.
        """
        inputs = iter(["hello there"])

        def fake_input(prompt=""):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr("builtins.input", fake_input)
        server = FakeServer(script=[{"content": "hi yourself"}])
        state = with_server(make_state, server)

        final_state = Engine[type(state)]().run(state, seed=[MaybeRegenerate()])

        assert final_state.running is False
        assert len(final_state.history.turns) == 1
        assert final_state.history.turns[0].user == "hello there"
        assert final_state.history.turns[0].assistant == "hi yourself"

    def test_keyboard_interrupt_at_prompt_exits_cleanly(self, make_state, monkeypatch, capsys):
        """Ctrl+C while blocked on input() at the prompt must route through
        the real on_interrupt policy (desh_chat.handlers — the same function
        cli.py wires into its Engine) to a clean Exit -> Info("Goodbye!") ->
        [], not propagate raw and crash the run. PromptUser itself only
        catches EOFError; the graceful handling here is an Engine-level
        property (_step()'s except KeyboardInterrupt -> on_interrupt), so
        this has to go through a real Engine, not a bare PromptUser call.
        """
        def raise_keyboard_interrupt(prompt=""):
            raise KeyboardInterrupt
        monkeypatch.setattr("builtins.input", raise_keyboard_interrupt)
        state = make_state()

        final_state = Engine[type(state)](on_interrupt=on_interrupt).run(state, seed=[MaybeRegenerate()])

        assert final_state.running is False
        assert "Goodbye!" in capsys.readouterr().out

    def test_cancelled_turn_never_reaches_the_model_on_the_next_request(self, make_state, no_esc_watcher):
        """A cancelled turn is stored (DisplayHistory can still show it) but
        must never surface in the messages sent for a later turn.
        """
        server = FakeServer(script=[{"content": "partial", "finish_reason": "cancelled"}])
        state = with_server(make_state, server, pending=PendingTurn("cancel me"))
        req = Request(messages=[{"role": "system", "content": state.system_prompt},
                                 {"role": "user", "content": "cancel me"}], model=MODELS[0], stream=True)
        state, events = StreamCompletion(request=req).execute(state)
        state, events = events[1].execute(state)  # TurnEnd
        assert state.history.turns[0].cancelled is True

        _, events = open_turn(state, "a follow-up question")
        contents = [m["content"] for m in events[0].request.messages]
        assert "cancel me" not in contents
        assert not any("partial" in c for c in contents)
