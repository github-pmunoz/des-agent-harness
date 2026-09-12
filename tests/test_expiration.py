"""
Tool result expiration: results age by ROUND, counted back from the latest completed round of the
pending turn (distance 0). With expire_after = k a request shows three bands —

  active     distance <  k-1   results whole
  expiring   distance == k-1   results whole, named in the scratchpad block as leaving next round
  stubbed    distance >= k     EXPIRED_RESULT in place of each result; the calls stay

— and a history turn is always stubbed (its answer is what the results led to). Nothing is
recorded: the bands are a rendering of (rounds, k) at request time, so the session file, the
repeat detector and the compaction transcript always read the whole results, and a changed k
re-renders. k <= 0 turns expiration off.
"""
import json

from conftest import MODELS

from desh.llama.tokens import estimate_tokens
from desh.llama.wire import ToolCall
from desh_chat.events import NextRound, StreamCompletion
from desh_chat.scratchpad import Scratchpad
from desh_chat.state import EXPIRED_RESULT, ChatHistory, PendingTurn, Round, Settings, ToolResult, Turn


def call(name: str, index: int = 0, **arguments) -> ToolCall:
    return ToolCall(index=index, id=f"call_{index}", type="function", name=name, arguments=json.dumps(arguments))


def round_(n: int, *names: str) -> Round:
    """Round number n of a turn, one call per name, each answered 'result n'."""
    calls = tuple(call(name, i) for i, name in enumerate(names or ("Read",)))
    return Round(f"round {n}", calls, tuple(ToolResult(tc.id, tc.name, f"result {n}") for tc in calls), tokens=10 * n)


def pending(n: int) -> PendingTurn:
    """A pending turn with n completed rounds, numbered 1..n."""
    p = PendingTurn("q")
    for i in range(1, n + 1):
        p = p.add_round(round_(i))
    return p


def settings(k: int) -> Settings:
    return Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, tool_expiration=k)


def results_in(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


# ---------------------
# The bands, as pure functions of (rounds, k)
# ---------------------

class TestBands:
    def test_distance_is_counted_back_from_the_latest_round(self):
        p = pending(5)                      # rounds 1..5, distances 4..0
        assert [p.stubbed(i, 2) for i in range(5)] == [True, True, True, False, False]
        assert [p.stubbed(i, 1) for i in range(5)] == [True, True, True, True, False]

    def test_the_latest_round_is_never_stubbed(self):
        for k in (1, 2, 6):
            assert not pending(8).stubbed(7, k)

    def test_off_when_none_or_not_positive(self):
        p = pending(5)
        for k in (None, 0, -1):
            assert not any(p.stubbed(i, k) for i in range(5))
            assert p.expiring(k) == ()
            assert results_in(p.messages(expire_after=k)) == [f"result {n}" for n in range(1, 6)]

    def test_messages_render_the_three_bands(self):
        p = pending(5)
        msgs = p.messages(expire_after=2)   # round 4 expiring (d=1), rounds 1..3 stubbed (d>=2)
        assert results_in(msgs) == [EXPIRED_RESULT, EXPIRED_RESULT, EXPIRED_RESULT, "result 4", "result 5"]
        assert p.messages(expire_after=3)[1:] != p.messages(expire_after=2)[1:]     # k matters
        assert p.messages() == p.messages(expire_after=None)

    def test_stubbing_keeps_the_calls_and_the_ids(self):
        p = PendingTurn("q").add_round(round_(1, "Read", "Bash")).add_round(round_(2))
        msgs = p.messages(expire_after=1)
        assert msgs[1]["tool_calls"] == p.rounds[0].messages()[0]["tool_calls"]
        assert [(m["tool_call_id"], m["content"]) for m in msgs[2:4]] == [("call_0", EXPIRED_RESULT), ("call_1", EXPIRED_RESULT)]
        assert p.rounds[0].results[0].content == "result 1"

    def test_expiring_names_the_round_at_distance_k_minus_1(self):
        p = PendingTurn("q").add_round(round_(1, "Read", "Bash")).add_round(round_(2, "Edit")).add_round(round_(3))
        assert p.expiring(1) == ("round 3: Read",)             # the latest round itself, gone next time
        assert p.expiring(2) == ("round 2: Edit",)
        assert p.expiring(3) == ("round 1: Read, Bash",)
        assert p.expiring(4) == ()                              # nothing old enough yet
        assert PendingTurn("q").expiring(1) == ()

    def test_expiring_names_only_never_arguments_or_results(self):
        p = PendingTurn("q").add_round(round_(1, "Bash")).add_round(round_(2))
        (line,) = p.expiring(2)
        assert "call_0" not in line and "result" not in line and "{" not in line


# ---------------------
# NextRound: what the request looks like
# ---------------------

class TestRequest:
    def stream_event(self, make_state, p: PendingTurn, k: int, scratchpad=None) -> StreamCompletion:
        state = make_state(pending=p, settings=settings(k), scratchpad=scratchpad)
        _, events = NextRound().execute(state)
        (ev,) = events
        assert isinstance(ev, StreamCompletion)
        return ev

    def test_the_pending_turn_renders_its_bands(self, make_state):
        ev = self.stream_event(make_state, pending(4), k=2)
        assert results_in(ev.request.messages) == [EXPIRED_RESULT, EXPIRED_RESULT, "result 3", "result 4"]

    def test_the_expiring_line_is_in_the_scratchpad_block(self, make_state):
        ev = self.stream_event(make_state, pending(4), k=2, scratchpad=Scratchpad().with_entry("k", "v"))
        block = ev.request.messages[-1]["content"]
        assert block.startswith("<scratchpad>") and "Expiring next round" in block and "round 3: Read" in block
        assert not any("Expiring" in m["content"] for m in ev.request.messages[:-1])

    def test_no_scratchpad_means_no_warning_anywhere(self, make_state):
        ev = self.stream_event(make_state, pending(4), k=2)
        assert not any("Expiring" in m["content"] for m in ev.request.messages)

    def test_no_line_when_nothing_expires_next_round(self, make_state):
        ev = self.stream_event(make_state, pending(1), k=3, scratchpad=Scratchpad())
        assert "Expiring" not in ev.request.messages[-1]["content"]

    def test_the_block_is_priced_with_its_expiring_line(self, make_state):
        state = make_state(pending=pending(4), settings=settings(2), scratchpad=Scratchpad())
        assert state.scratchpad_tokens() == estimate_tokens(state.scratchpad_block()["content"])
        assert state.scratchpad_tokens() > estimate_tokens(Scratchpad().message(2))

    def test_history_turns_are_always_stubbed_in_the_view(self, make_state):
        turn = Turn("q0", "a0", rounds=(round_(1),), tokens=5)
        state = make_state(pending=PendingTurn("q").add_round(round_(1)), settings=settings(6), history=ChatHistory().append(turn))
        _, events = NextRound().execute(state)
        results = results_in(events[0].request.messages)
        assert results == [EXPIRED_RESULT, "result 1"]     # history round stubbed, pending round (d=0) whole

    def test_expiration_off_sends_everything_whole(self, make_state):
        ev = self.stream_event(make_state, pending(4), k=0)
        assert results_in(ev.request.messages) == [f"result {n}" for n in range(1, 5)]


# ---------------------
# Pricing follows the rendering
# ---------------------

class TestPricing:
    def test_active_rounds_keep_their_frame_count_and_stubbed_ones_are_estimated(self):
        p = pending(4)      # tokens 10, 20, 30, 40; with k=2 rounds 1 and 2 are stubbed
        stubbed = sum(estimate_tokens(p.rounds[i].text(stubbed=True)) for i in (0, 1))
        assert p.priced_tokens(2) == stubbed + 30 + 40
        assert p.priced_tokens(None) == 100
        assert p.priced_tokens(4) == 100        # nothing old enough yet

    def test_a_stubbed_round_costs_less_than_its_frame_priced(self):
        big = Round("", (call("Read"),), (ToolResult("call_0", "Read", "x" * 4000),), tokens=1000)
        p = PendingTurn("q").add_round(big).add_round(round_(2))
        assert p.priced_tokens(1) < p.priced_tokens(None)
        assert p.priced_tokens(1) == estimate_tokens(big.text(stubbed=True)) + 20

    def test_an_unpriced_active_round_still_contributes_nothing(self):
        """tokens == 0 means no frame priced it; the latest results are priced by unpriced_text()
        instead, so a 0 here is not a bug to paper over."""
        p = PendingTurn("q").add_round(Round("", (call("Read"),), (ToolResult("call_0", "Read", "r"),), tokens=0))
        assert p.priced_tokens(6) == 0

    def test_prior_tokens_drop_when_a_round_expires(self, make_state):
        """The same pending turn, one more round: the oldest result leaves the request and
        prior_tokens follows, so the frame's prompt count minus prior stays the new text."""
        big = Round("", (call("Read"),), (ToolResult("call_0", "Read", "x" * 4000),), tokens=1000)
        before = make_state(pending=PendingTurn("q").add_round(big), settings=settings(1))     # big at distance 0: whole
        after = make_state(pending=PendingTurn("q").add_round(big).add_round(round_(2)), settings=settings(1))   # distance 1: stubbed
        _, (ev_before,) = NextRound().execute(before)
        _, (ev_after,) = NextRound().execute(after)
        assert ev_before.prior_tokens - estimate_tokens(before.system_prompt) == 1000
        assert ev_after.prior_tokens - estimate_tokens(after.system_prompt) == estimate_tokens(big.text(stubbed=True)) + 20

    def test_compaction_check_sees_the_stubbed_cost(self, make_state):
        """A turn whose whole results would overflow fits once they are stubbed: no compaction,
        no overflow, the request goes out."""
        huge = Round("", (call("Read"),), (ToolResult("call_0", "Read", "x" * 40000),), tokens=10_000)
        p = PendingTurn("q").add_round(huge).add_round(round_(2))
        state = make_state(pending=p, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=8000, max_turn_tokens=2000, tool_expiration=1))
        _, events = NextRound().execute(state)
        assert isinstance(events[0], StreamCompletion)
        assert state.gen_room(state.pending_tokens()) >= state.min_gen_tokens()
