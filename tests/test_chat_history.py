"""Unit tests for ChatHistory / Turn: token accounting, the view() budget
walk, and the cancelled/summary semantics that view()/since_last_summary()/
get_total_tokens() all depend on.

Deliberately decoupled from the Event/Engine layer — these are pure data
structure tests. Chain-level tests (UserMessage, AppendTurn, MaybeCompact,
...) live in test_chat_chain.py and exercise these same functions indirectly
through real turn-token math instead of hand-picked numbers.
"""
from desh.llama.tokens import estimate_tokens
from desh_chat.state import ChatHistory, Turn


def make_turn(user="u", assistant="a", **kw) -> Turn:
    return Turn(user, assistant, **kw)


class TestTurnTokens:
    def test_tokens_autocomputed_from_user_and_assistant(self):
        turn = Turn("hello there", "general kenobi")
        assert turn.tokens == estimate_tokens("hello there") + estimate_tokens("general kenobi")

    def test_explicit_nonzero_tokens_is_not_recomputed(self):
        # __post_init__ only fills tokens when it is exactly 0 — an explicit
        # nonzero value (however it got there) is trusted as-is.
        turn = Turn("a very very long message well past four chars", "short", tokens=1)
        assert turn.tokens == 1

    def test_zero_length_strings_recompute_to_zero(self):
        turn = Turn("", "")
        assert turn.tokens == 0


class TestChatHistoryAppend:
    def test_append_is_immutable(self):
        h0 = ChatHistory()
        h1 = h0.append(make_turn())
        assert len(h0) == 0
        assert len(h1) == 1

    def test_append_preserves_order(self):
        h = ChatHistory().append(make_turn("first", "1")).append(make_turn("second", "2"))
        assert [t.user for t in h.turns] == ["first", "second"]


class TestView:
    def test_empty_history_returns_empty(self):
        assert ChatHistory().view(budget=10_000) == []

    def test_zero_budget_returns_empty(self):
        h = ChatHistory().append(make_turn("hello", "hi"))
        assert h.view(budget=0) == []

    def test_budget_is_room_available_not_room_consumed(self):
        # Regression: view()'s budget argument must mean "tokens of history
        # allowed", not "tokens already spent on something else". A caller
        # that (incorrectly) passes the small consumed-elsewhere figure
        # instead of the large remaining-room figure would see this fail —
        # a single turn's tokens must fit under a budget sized to hold it.
        turn = make_turn("a reasonably sized user message", "and a reasonably sized reply")
        h = ChatHistory().append(turn)
        assert h.view(budget=turn.tokens) == turn.messages()
        assert h.view(budget=turn.tokens - 1) == []

    def test_returns_longest_fitting_tail_newest_first_order_preserved(self):
        # estimate_tokens is len(text)//4 — strings need real length or every
        # turn rounds down to 0 tokens and "fits" under any budget, silently
        # defeating this test's premise.
        t1 = make_turn("turn one user message padded out", "turn one assistant reply padded out")
        t2 = make_turn("turn two user message padded out", "turn two assistant reply padded out")
        t3 = make_turn("turn three user message padded out", "turn three assistant reply padded out")
        h = ChatHistory().append(t1).append(t2).append(t3)
        assert t1.tokens > 0 and t2.tokens > 0 and t3.tokens > 0
        # budget for exactly the newest two turns
        budget = t2.tokens + t3.tokens
        msgs = h.view(budget=budget)
        assert msgs == t2.messages() + t3.messages()

    def test_a_turn_that_would_overflow_the_budget_is_excluded(self):
        t1 = make_turn("small", "ok")
        t2 = make_turn("a much longer message that costs many more tokens than t1 does", "and a longer reply too, to push the total well past the small budget")
        h = ChatHistory().append(t1).append(t2)
        # budget fits t2 alone but not both
        msgs = h.view(budget=t2.tokens)
        assert msgs == t2.messages()

    def test_stops_at_a_summary_turn_and_includes_it(self):
        h = ChatHistory().append(make_turn("old, pre-summary", "should not surface"))
        h = h.compact("what happened before")
        h = h.append(make_turn("new question", "new answer"))
        msgs = h.view(budget=1_000_000)
        # summary turn included, but nothing from before it
        assert "old, pre-summary" not in [m["content"] for m in msgs]
        assert any("what happened before" in m["content"] for m in msgs)
        assert any(m["content"] == "new question" for m in msgs)

    def test_cancelled_turns_are_skipped_not_counted_and_do_not_stop_the_walk(self):
        real_turn = make_turn("real question", "real answer")
        cancelled_turn = make_turn("cancelled question", "partial", cancelled=True)
        h = ChatHistory().append(real_turn).append(cancelled_turn)
        # budget sized for the real turn only — if the cancelled turn were
        # counted, it would eat the budget and exclude the real one too.
        msgs = h.view(budget=real_turn.tokens)
        assert msgs == real_turn.messages()
        assert "cancelled question" not in [m["content"] for m in msgs]


class TestGetTotalTokens:
    def test_sums_all_non_cancelled_turns(self):
        t1 = make_turn("a", "b")
        t2 = make_turn("c", "d")
        h = ChatHistory().append(t1).append(t2)
        assert h.get_total_tokens() == t1.tokens + t2.tokens

    def test_excludes_cancelled_turns(self):
        real = make_turn("kept", "kept reply")
        cancelled = make_turn("dropped", "dropped reply", cancelled=True)
        h = ChatHistory().append(real).append(cancelled)
        assert h.get_total_tokens() == real.tokens

    def test_survives_compaction_as_a_lifetime_counter(self):
        # compact() appends a summary turn rather than truncating the tuple,
        # so pre-compaction turns still count toward the lifetime total even
        # though they're no longer reachable via view()/window_tokens().
        t1 = make_turn("pre-summary", "reply")
        h = ChatHistory().append(t1)
        total_before = h.get_total_tokens()
        h = h.compact("summary text")
        assert h.get_total_tokens() > total_before


class TestSinceLastSummaryAndWindowTokens:
    def test_no_summary_returns_all_non_cancelled_turns(self):
        t1 = make_turn("a", "b")
        t2 = make_turn("c", "d")
        h = ChatHistory().append(t1).append(t2)
        assert h.since_last_summary() == [t1, t2]
        assert h.window_tokens() == t1.tokens + t2.tokens

    def test_stops_at_and_includes_the_most_recent_summary(self):
        h = ChatHistory().append(make_turn("old", "reply"))
        h = h.compact("summary of old stuff")
        new_turn = make_turn("new", "reply2")
        h = h.append(new_turn)
        since = h.since_last_summary()
        assert len(since) == 2  # [summary_turn, new_turn]
        assert since[0].summary is True
        assert since[1] is new_turn

    def test_window_tokens_matches_since_last_summary_sum(self):
        h = ChatHistory().append(make_turn("a", "b")).append(make_turn("c", "d"))
        assert h.window_tokens() == sum(t.tokens for t in h.since_last_summary())

    def test_window_after_compaction_is_exactly_the_summary_turns_own_tokens(self):
        # Not a magnitude comparison against the pre-compaction window (fragile
        # — depends on how the summary text's length happens to compare to the
        # original turns'). The real invariant: since_last_summary() reaches
        # exactly the summary turn and nothing older, so window_tokens() must
        # equal that one turn's own token count, exactly.
        h = ChatHistory()
        for i in range(5):
            h = h.append(make_turn(f"question {i}", f"answer {i}"))
        h = h.compact("a summary of the five turns above")
        summary_turn = h.turns[-1]
        assert summary_turn.summary is True
        assert h.since_last_summary() == [summary_turn]
        assert h.window_tokens() == summary_turn.tokens

    def test_cancelled_turns_excluded_but_walk_continues_past_them(self):
        t1 = make_turn("kept one", "reply one")
        cancelled = make_turn("dropped", "dropped reply", cancelled=True)
        t2 = make_turn("kept two", "reply two")
        h = ChatHistory().append(t1).append(cancelled).append(t2)
        since = h.since_last_summary()
        assert cancelled not in since
        assert since == [t1, t2]
        assert h.window_tokens() == t1.tokens + t2.tokens


class TestMessages:
    def test_returns_all_turns_unfiltered_including_cancelled_and_summary(self):
        h = ChatHistory().append(make_turn("a", "b", cancelled=True)).append(make_turn("c", "d"))
        msgs = h.messages()
        assert len(msgs) == 4  # both turns, 2 messages each — messages() does not filter
