"""Unit tests for the ported desh_chat turn chain:

  UserMessage -> StreamCompletion -> [AppendTurn, LogCompletion] -> AppendTurn
    -> MaybeCompact -> (CompactHistory -> [Info, LogCompletion] ->) MaybeRegenerate
    -> [DisplayStats, PromptUser]

MaybeRegenerate is a 2-fan, not a chain link: it schedules DisplayStats and
PromptUser directly, as siblings. DisplayStats/DisplayHistory/Info/Warn are
pure sinks (execute() returns [] — no further events); MaybeRegenerate is
the only thing that reschedules PromptUser.

Each event is driven directly (state, events = Event(...).execute(state)),
same style as test_command_surface.py's Command tests, plus one full
Engine.run() test at the end for the end-to-end termination guarantee.

FakeServer/FakeESCWatcher (conftest.py) stand in for the network and the
real terminal-raw-mode watcher — no live router, no real stdin required.
"""
from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.engine import Engine
from desh.llama.client import Logger, Request
from desh.llama.tokens import estimate_tokens
from desh_chat.events import (
    AppendTurn, CompactHistory, DisplayStats, Exit, Info, LogCompletion,
    MaybeCompact, MaybeRegenerate, PromptUser, StreamCompletion, UserMessage,
)
from desh_chat.state import ChatHistory, InferenceEngine, Settings, Turn


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


# ---------------------
# UserMessage
# ---------------------

class TestUserMessageBudget:
    """Hand-verified gen_budget under each of the three ceilings in the min()."""

    def test_max_turn_tokens_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, turn_token_cap=0.5)
        state = make_state(settings=settings)
        _, events = UserMessage("hi").execute(state)
        assert isinstance(events[0], StreamCompletion)
        assert events[0].request.max_tokens == 100

    def test_turn_token_cap_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=10_000, turn_token_cap=0.5)
        state = make_state(settings=settings)
        _, events = UserMessage("hi").execute(state)
        assert events[0].request.max_tokens == 500  # 0.5 * 1000

    def test_remaining_context_is_the_binding_ceiling(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=50, max_turn_tokens=10_000, turn_token_cap=0.9)
        state = make_state(settings=settings)
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        msg_tokens = estimate_tokens("hi")
        expected = 50 - sys_prompt_tokens - msg_tokens  # window_tokens() is 0, no prior history
        _, events = UserMessage("hi").execute(state)
        assert events[0].request.max_tokens == expected
        assert expected < 45  # confirms this is genuinely the tightest of the three ceilings

    def test_gen_budget_leq_zero_is_rejected_before_any_request_is_built(self, make_state, capsys):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1, max_turn_tokens=100, turn_token_cap=1.0)
        state = make_state(settings=settings)
        _, events = UserMessage("a message long enough to blow a context of 1 token").execute(state)
        assert len(events) == 1
        assert isinstance(events[0], MaybeRegenerate)
        assert "exceeds context window" in capsys.readouterr().out


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
        _, events = UserMessage("a new question").execute(state)
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
        _, events = UserMessage(message).execute(state)

        req = events[0].request
        assert req.max_tokens == 100 - sys_prompt_tokens - msg_tokens - prior_turn.tokens  # context-headroom ceiling bound
        contents = [m["content"] for m in req.messages]
        assert "previous question" in contents
        assert "previous answer" in contents
        assert message in contents

    def test_request_message_order_is_system_then_history_then_new_user_message(self, make_state):
        history = ChatHistory().append(Turn("q1", "a1"))
        state = make_state(history=history)
        _, events = UserMessage("q2").execute(state)
        roles_and_last = [(m["role"], m["content"]) for m in events[0].request.messages]
        assert roles_and_last[0] == ("system", state.system_prompt)
        assert roles_and_last[-1] == ("user", "q2")


class TestUserMessageRequestShape:
    def test_request_carries_current_settings(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.9, think=True, context=16384, max_turn_tokens=8192)
        state = make_state(settings=settings)
        _, events = UserMessage("hello").execute(state)
        req = events[0].request
        assert req.model == MODELS[0]
        assert req.temperature == 0.9
        assert req.think is True
        assert req.stream is True


# ---------------------
# StreamCompletion
# ---------------------

class TestStreamCompletion:
    def test_happy_path_emits_append_turn_with_the_streamed_content(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "the answer", "finish_reason": "stop"}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "the question"}],
                      model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req).execute(state)
        assert len(events) == 1
        assert isinstance(events[0], AppendTurn)
        assert events[0].user_msg == "the question"
        assert events[0].assistant_msg == "the answer"
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
        assert events[0].cancelled is True
        assert "cancelled" in capsys.readouterr().out.lower()

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
        assert isinstance(events[0], AppendTurn)


# ---------------------
# AppendTurn
# ---------------------

class TestAppendTurn:
    def test_appends_a_turn_and_always_emits_maybe_compact(self, make_state):
        state = make_state()
        new_state, events = AppendTurn(user_msg="q", assistant_msg="a", cancelled=False).execute(state)
        assert len(new_state.history) == 1
        assert new_state.history.turns[0].user == "q"
        assert new_state.history.turns[0].assistant == "a"
        assert len(events) == 1 and isinstance(events[0], MaybeCompact)

    def test_original_state_is_untouched_immutability(self, make_state):
        state = make_state()
        AppendTurn(user_msg="q", assistant_msg="a", cancelled=False).execute(state)
        assert len(state.history) == 0

    def test_cancelled_flag_lands_on_the_turn_not_on_tokens(self, make_state):
        """Regression: Turn(user, assistant, self.cancelled) once landed the
        cancelled bool in the positional `tokens` slot instead of `cancelled`.
        Masked when cancelled=False (False==0 lets __post_init__ recompute
        tokens correctly); broke silently whenever cancelled=True (tokens
        frozen at 1, cancelled flag never actually set).
        """
        long_user = "a reasonably long user message that is not four characters"
        long_assistant = "a reasonably long assistant reply that is not four characters"
        new_state, _ = AppendTurn(user_msg=long_user, assistant_msg=long_assistant, cancelled=True).execute(make_state())
        turn = new_state.history.turns[0]
        assert turn.cancelled is True
        assert turn.tokens == estimate_tokens(long_user) + estimate_tokens(long_assistant)
        assert turn.tokens != 1

    def test_non_cancelled_turn_tokens_are_also_correct(self, make_state):
        new_state, _ = AppendTurn(user_msg="a decent length user message here", assistant_msg="a decent length assistant reply here", cancelled=False).execute(make_state())
        turn = new_state.history.turns[0]
        assert turn.cancelled is False
        assert turn.tokens == estimate_tokens(turn.user) + estimate_tokens(turn.assistant)


# ---------------------
# MaybeCompact
# ---------------------

class TestMaybeCompact:
    def test_below_threshold_regenerates(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, compaction_threshold=0.5)
        history = ChatHistory().append(Turn("short", "reply", tokens=100))  # well under 500
        state = make_state(settings=settings, history=history)
        _, events = MaybeCompact().execute(state)
        assert len(events) == 1 and isinstance(events[0], MaybeRegenerate)

    def test_at_or_above_threshold_compacts(self, make_state):
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, compaction_threshold=0.5)
        history = ChatHistory().append(Turn("long", "reply", tokens=500))  # exactly at threshold
        state = make_state(settings=settings, history=history)
        _, events = MaybeCompact().execute(state)
        assert len(events) == 1 and isinstance(events[0], CompactHistory)

    def test_does_not_double_count_the_just_appended_turn(self, make_state):
        """Regression: MaybeCompact used to receive `last_turn` and check
        window_tokens() + last_turn.tokens — but by the time MaybeCompact
        runs, AppendTurn has already appended that turn, so window_tokens()
        already includes it. The old check double-counted it and could fire
        before the real window actually reached compaction_threshold.

        Built so the true window (450) sits just under a 500 threshold, but
        old_window + last_turn.tokens (450 + 100 = 550) would have wrongly
        cleared it.
        """
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, compaction_threshold=0.5)
        history = (ChatHistory()
                   .append(Turn("earlier", "reply", tokens=350))
                   .append(Turn("latest", "reply", tokens=100)))  # window = 450, under 500
        state = make_state(settings=settings, history=history)
        assert state.history.window_tokens() == 450
        _, events = MaybeCompact().execute(state)
        assert isinstance(events[0], MaybeRegenerate), "compaction fired early — the just-appended turn was double-counted"

    def test_cancelled_status_of_the_last_turn_no_longer_needs_special_casing(self, make_state):
        """window_tokens() already excludes cancelled turns at the source
        (since_last_summary()'s own filter) — a cancelled turn simply cannot
        move this check either way, with no cancelled-specific logic needed
        in MaybeCompact itself.
        """
        settings = Settings(model=MODELS[0], temperature=0.3, think=False,
                             context=1000, max_turn_tokens=100, compaction_threshold=0.5)
        history = ChatHistory().append(Turn("cancelled", "partial", tokens=900, cancelled=True))
        state = make_state(settings=settings, history=history)
        _, events = MaybeCompact().execute(state)
        assert isinstance(events[0], MaybeRegenerate)


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

        assert len(events) == 3
        assert isinstance(events[0], Info)
        assert isinstance(events[1], LogCompletion)
        assert isinstance(events[2], MaybeRegenerate)

        events[0].execute(new_state)  # the summary is now surfaced to the user, not just logged
        assert "a tidy summary" in capsys.readouterr().out

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
# DisplayStats
# ---------------------

class TestDisplayStats:
    def test_prints_context_and_session_figures(self, make_state, capsys):
        history = ChatHistory().append(Turn("q", "a"))
        state = make_state(history=history)
        sys_tokens = estimate_tokens(state.system_prompt)
        expected_window = sys_tokens + state.history.window_tokens()
        expected_total = sys_tokens + state.history.get_total_tokens()

        _, events = DisplayStats().execute(state)

        out = capsys.readouterr().out
        assert f"Context: {expected_window} / {state.settings.context} tokens" in out
        assert f"Session: {expected_total}" in out
        assert events == []  # pure sink now — MaybeRegenerate (not DisplayStats) schedules PromptUser


# ---------------------
# LogCompletion
# ---------------------

class TestLogCompletion:
    def test_records_when_a_logger_is_configured(self, make_state, tmp_path):
        import json
        log_path = tmp_path / "completions.jsonl"
        state = make_state(completions_log=Logger(str(log_path)))
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0])
        from desh.llama.client import Completion
        completion = Completion(id="x", model=MODELS[0], created=0, system_fingerprint="",
                                 content="hello", reasoning="", finish_reason="stop",
                                 usage=None, timings=None, streamed=False)
        new_state, events = LogCompletion(request=req, completion=completion, port=PORT).execute(state)
        assert events == []
        assert new_state is state  # no state mutation — pure side effect
        record = json.loads(log_path.read_text().strip())
        assert record["port"] == PORT
        assert record["response"]["choices"][0]["message"]["content"] == "hello"

    def test_no_op_when_no_logger_configured(self, make_state):
        state = make_state(completions_log=None)
        req = Request(messages=[{"role": "user", "content": "hi"}], model=MODELS[0])
        from desh.llama.client import Completion
        completion = Completion(id="x", model=MODELS[0], created=0, system_fingerprint="",
                                 content="hello", reasoning="", finish_reason="stop",
                                 usage=None, timings=None, streamed=False)
        new_state, events = LogCompletion(request=req, completion=completion, port=PORT).execute(state)
        assert events == []
        assert new_state is state


# ---------------------
# Full chain, real Engine
# ---------------------

class TestFullEngineRun:
    def test_prompt_completion_and_eof_terminate_cleanly_no_livelock(self, make_state, no_esc_watcher, monkeypatch):
        """End-to-end regression: PromptUser -> UserMessage -> StreamCompletion
        -> AppendTurn -> MaybeCompact -> MaybeRegenerate -> [DisplayStats,
        PromptUser] -> (EOF) -> Exit -> running=False -> [Info] -> [] ->
        queue drains. Exit never touches MaybeRegenerate — it settles
        directly through its own Info sink. Engine.run() must return, not
        hang.
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

        final_state = Engine[type(state)]().run(state, seed=[PromptUser()])

        assert final_state.running is False
        assert len(final_state.history.turns) == 1
        assert final_state.history.turns[0].user == "hello there"
        assert final_state.history.turns[0].assistant == "hi yourself"

    def test_cancelled_turn_never_reaches_the_model_on_the_next_request(self, make_state, no_esc_watcher):
        """A cancelled turn is stored (DisplayHistory can still show it) but
        must never surface in the messages sent for a later turn.
        """
        server = FakeServer(script=[{"content": "partial", "finish_reason": "cancelled"}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "system", "content": state.system_prompt},
                                 {"role": "user", "content": "cancel me"}], model=MODELS[0], stream=True)
        state, events = StreamCompletion(request=req).execute(state)
        state, events = events[0].execute(state)  # AppendTurn
        assert state.history.turns[0].cancelled is True

        _, events = UserMessage("a follow-up question").execute(state)
        contents = [m["content"] for m in events[0].request.messages]
        assert "cancel me" not in contents
        assert not any("partial" in c for c in contents)
