"""
Token pricing (backlog #932): turn_tokens() policy and the plumbing that
carries the server's usage frame from StreamCompletion into Turn.tokens.

Policy under test (desh/llama/tokens.py):
  - usage None            -> 0 (Turn falls back to the len//4 heuristic)
  - user half             -> prompt_tokens - prior_tokens (residual: includes template overhead
                             and absorbs system-prompt estimate error); heuristic if <= 0
  - assistant half        -> completion_tokens, scaled by content share when reasoning is present;
                             heuristic if <= 0
"""
from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.llama.client import Request
from desh.llama.tokens import estimate_tokens, turn_tokens
from desh_chat.events import AppendTurn, CompactHistory, StreamCompletion, UserMessage
from desh_chat.state import ChatHistory, InferenceEngine, Turn


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return make_state(inference=inference, **overrides)


USAGE = {"prompt_tokens": 380, "completion_tokens": 205, "total_tokens": 585}


# ---------------------
# turn_tokens policy
# ---------------------

class TestTurnTokensPolicy:
    def test_no_usage_returns_zero_so_turn_falls_back_to_heuristic(self):
        assert turn_tokens(None, "hello there", "general kenobi", "", prior_tokens=100) == 0
        turn = Turn("hello there", "general kenobi", tokens=turn_tokens(None, "hello there", "general kenobi", "", 100))
        assert turn.tokens == estimate_tokens("hello there") + estimate_tokens("general kenobi")

    def test_residual_user_half_plus_completion_tokens(self):
        # prior=300 -> user half is 380-300=80 (template overhead included); assistant half is 205
        assert turn_tokens(USAGE, "q", "a", "", prior_tokens=300) == 80 + 205

    def test_prior_plus_turn_reproduces_the_servers_total(self):
        # The point of the residual: what we account for equals what the server tokenized + generated.
        prior = 300
        assert prior + turn_tokens(USAGE, "q", "a", "", prior) == USAGE["total_tokens"]

    def test_overestimated_prior_falls_back_to_heuristic_for_user_half(self):
        user = "a user message long enough to have some tokens"
        got = turn_tokens(USAGE, user, "a", "", prior_tokens=USAGE["prompt_tokens"] + 50)
        assert got == estimate_tokens(user) + USAGE["completion_tokens"]

    def test_reasoning_scales_completion_tokens_by_content_share(self):
        assistant, reasoning = "x" * 100, "y" * 300     # content is 1/4 of what was generated
        got = turn_tokens(USAGE, "q", assistant, reasoning, prior_tokens=300)
        assert got == 80 + round(205 * 0.25)

    def test_reasoning_only_reply_prices_assistant_half_by_heuristic(self):
        # content empty -> scaled share is 0 -> heuristic on "" is also 0; only the user half remains
        assert turn_tokens(USAGE, "q", "", "y" * 300, prior_tokens=300) == 80

    def test_result_is_an_int_even_with_reasoning(self):
        got = turn_tokens(USAGE, "q", "abc", "defg", prior_tokens=300)
        assert isinstance(got, int)

    def test_missing_completion_tokens_falls_back_to_heuristic_for_assistant_half(self):
        assistant = "an assistant reply of some length here"
        got = turn_tokens({"prompt_tokens": 380}, "q", assistant, "", prior_tokens=300)
        assert got == 80 + estimate_tokens(assistant)


# ---------------------
# Plumbing: UserMessage -> StreamCompletion -> AppendTurn -> Turn
# ---------------------

class TestUsagePlumbing:
    def test_user_message_reports_prior_tokens_as_sys_estimate_plus_view(self, make_state):
        history = ChatHistory().append(Turn("q1", "a1", tokens=40)).append(Turn("q2", "a2", tokens=60))
        state = make_state(history=history)
        _, events = UserMessage("q3").execute(state)
        sc = events[0]
        assert isinstance(sc, StreamCompletion)
        assert sc.prior_tokens == estimate_tokens(state.system_prompt) + 100

    def test_view_turns_is_the_turn_level_twin_of_view(self):
        # UserMessage sums prior_tokens over view_turns() and sends view()'s messages; the two
        # must agree on which turns are in, including eviction and the cancelled skip.
        history = (ChatHistory()
                   .append(Turn("old", "old", tokens=150))
                   .append(Turn("cancelled", "partial", tokens=5, cancelled=True))
                   .append(Turn("new", "new", tokens=20)))
        turns = history.view_turns(budget=100)
        assert [t.user for t in turns] == ["new"]
        assert history.view(budget=100) == [m for t in turns for m in t.messages()]
        assert sum(t.tokens for t in turns) == 20

    def test_stream_completion_prices_the_turn_from_usage(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "the answer", "usage": USAGE}])
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "user", "content": "the question"}], model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req, prior_tokens=300).execute(state)
        assert events[0].tokens == turn_tokens(USAGE, "the question", "the answer", "", 300)
        assert events[0].tokens == 285

    def test_stream_completion_without_usage_leaves_turn_unpriced(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "the answer"}])   # usage defaults to None
        state = with_server(make_state, server)
        req = Request(messages=[{"role": "user", "content": "the question"}], model=MODELS[0], stream=True)
        _, events = StreamCompletion(request=req, prior_tokens=300).execute(state)
        assert events[0].tokens == 0

    def test_append_turn_stores_the_priced_tokens(self, make_state):
        new_state, _ = AppendTurn(user_msg="q", assistant_msg="a", cancelled=False, tokens=285).execute(make_state())
        assert new_state.history.turns[-1].tokens == 285

    def test_append_turn_with_zero_tokens_keeps_heuristic(self, make_state):
        new_state, _ = AppendTurn(user_msg="hello there", assistant_msg="general kenobi", cancelled=False).execute(make_state())
        assert new_state.history.turns[-1].tokens == estimate_tokens("hello there") + estimate_tokens("general kenobi")

    def test_end_to_end_window_tracks_usage(self, make_state, no_esc_watcher):
        server = FakeServer(script=[{"content": "the answer", "usage": USAGE}])
        state = with_server(make_state, server)
        _, events = UserMessage("the question").execute(state)
        sc = events[0]
        _, events = sc.execute(state)
        state, _ = events[0].execute(state)  # AppendTurn
        sys_est = estimate_tokens(state.system_prompt)
        # residual policy: accounted total equals the server's own count
        assert sys_est + state.history.window_tokens() == USAGE["total_tokens"]


# ---------------------
# Compaction: summary turn pricing
# ---------------------

class TestCompactionSummaryPricing:
    def test_summary_turn_uses_completion_tokens_plus_wrapper_estimate(self, make_state):
        server = FakeServer(script=[{"content": "a tidy summary", "usage": {"prompt_tokens": 500, "completion_tokens": 42}}])
        history = ChatHistory().append(Turn("q", "a", tokens=100))
        state = with_server(make_state, server, history=history)
        new_state, _ = CompactHistory().execute(state)
        summary = new_state.history.turns[-1]
        assert summary.summary is True
        assert summary.tokens == 42 + estimate_tokens(ChatHistory.SUMMARY_PREFIX + ChatHistory.SUMMARY_ACK)

    def test_summary_turn_without_usage_keeps_heuristic(self, make_state):
        server = FakeServer(script=[{"content": "a tidy summary"}])
        history = ChatHistory().append(Turn("q", "a", tokens=100))
        state = with_server(make_state, server, history=history)
        new_state, _ = CompactHistory().execute(state)
        summary = new_state.history.turns[-1]
        assert summary.tokens == estimate_tokens(summary.user) + estimate_tokens(summary.assistant)
