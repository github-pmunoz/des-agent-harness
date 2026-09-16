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
from desh.tools import ToolRegistry
from desh_chat.events import NextRound, StreamCompletion
from desh_chat.scratchpad import Scratchpad
from desh_chat.state import CHECKPOINT_PREFIX, EXPIRED_RESULT, MENTION_CHARS, ChatHistory, PendingTurn, Round, Settings, ToolResult, Turn


def call(name: str, index: int = 0, **arguments) -> ToolCall:
    return ToolCall(index=index, id=f"call_{index}", type="function", name=name, arguments=json.dumps(arguments))


def bash(reason: str, command: str) -> str:
    """A Bash-shaped tool: the command is what the call was about."""
    return command


def read(file_path: str, offset: int = 1) -> str:
    """A Read-shaped tool: the path is what the call was about."""
    return file_path


# Read and Bash as the coding toolset declares them (coding_registry): a Read is about its path, a
# Bash about its command.
REGISTRY = ToolRegistry().add(read, name="Read", target="file_path").add(bash, name="Bash", identity=("command",), target="command")


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


class TestExpiringMentions:
    """The expiring line says what each call was ABOUT, not just which tool ran: 'Read
    tests/conftest.py' lets the model decide what to persist without remembering what round 3 read.
    The target comes from the registry (ToolRegistry.target) through `describe`; the line is still
    names only when no describe is given, when the registry does not know the tool, or when the
    call has no target. A target is folded onto one line and cut at MENTION_CHARS: a mention, never
    a dump."""

    def turn(self, *rounds: Round) -> PendingTurn:
        p = PendingTurn("q")
        for r in rounds:
            p = p.add_round(r)
        return p

    def test_a_call_is_mentioned_with_its_target(self):
        first = Round("", (call("Read", 0, file_path="tests/conftest.py"),
                           call("Bash", 1, reason="find the answer", command='grep -n "def answer" src/x.py')))
        p = self.turn(first, round_(2))
        assert p.expiring(2, describe=REGISTRY.target) == ('round 1: Read tests/conftest.py, Bash grep -n "def answer" src/x.py',)

    def test_without_describe_the_line_is_names_only(self):
        first = Round("", (call("Read", 0, file_path="tests/conftest.py"), call("Bash", 1, reason="r", command="ls")))
        assert self.turn(first, round_(2)).expiring(2) == ("round 1: Read, Bash",)

    def test_a_call_without_a_target_keeps_its_name(self):
        first = Round("", (call("nope", 0, anything="at all"), call("Read", 1)))     # unknown tool; Read with no path
        assert self.turn(first, round_(2)).expiring(2, describe=REGISTRY.target) == ("round 1: nope, Read",)

    def test_a_long_target_is_cut_at_mention_chars(self):
        command = "x" * (MENTION_CHARS + 40)
        (line,) = self.turn(Round("", (call("Bash", 0, reason="r", command=command),)), round_(2)).expiring(2, describe=REGISTRY.target)
        assert line == f"round 1: Bash {'x' * MENTION_CHARS}..."

    def test_a_multiline_target_is_folded_onto_one_line(self):
        command = "ls   src/\n\tgrep -c def\n"
        (line,) = self.turn(Round("", (call("Bash", 0, reason="r", command=command),)), round_(2)).expiring(2, describe=REGISTRY.target)
        assert line == "round 1: Bash ls src/ grep -c def"


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

    def test_the_expiring_line_mentions_the_targets_the_registry_knows(self, make_state):
        first = Round("", (call("Read", 0, file_path="tests/conftest.py"), call("Bash", 1, reason="r", command="ls")),
                      (ToolResult("call_0", "Read", "r"), ToolResult("call_1", "Bash", "r")), tokens=10)
        p = PendingTurn("q").add_round(first).add_round(round_(2))
        state = make_state(pending=p, settings=settings(2), scratchpad=Scratchpad(), tools=REGISTRY)
        _, (ev,) = NextRound().execute(state)
        assert "Expiring next round, persist what you still need from: round 1: Read tests/conftest.py, Bash ls" in ev.request.messages[-1]["content"]

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


# ---------------------
# A checkpoint in the turn: the bands and the pricing run over the view
# ---------------------

def checkpointed(folded: int, after: int) -> PendingTurn:
    """A turn of `folded` rounds checkpointed, then `after` more rounds: the view is the checkpoint
    plus rounds folded+1 .. folded+after."""
    p = pending(folded + 1).compact("c")        # rounds 1..folded fold, round folded+1 stays whole
    for n in range(folded + 2, folded + after + 1):
        p = p.add_round(round_(n))
    return p


class TestCheckpoint:
    def test_compact_places_the_checkpoint_before_the_last_round(self):
        p = pending(3).compact("c", tokens=100)
        assert [r.summary for r in p.rounds] == [False, False, True, False]     # the record keeps every round
        assert p.since_last_summary() == p.rounds[-2:]                          # the view is (checkpoint, round 3)
        assert p.since_last_summary()[0].assistant == CHECKPOINT_PREFIX + "c"
        assert p.non_summary_rounds() == 3

    def test_a_second_checkpoint_subsumes_the_first(self):
        p = checkpointed(2, 3).compact("two")          # view was (one, 3, 4, 5): 3 and 4 fold with `one`
        view = p.since_last_summary()
        assert [r.assistant for r in view] == [CHECKPOINT_PREFIX + "two", "round 5"]
        assert sum(1 for r in p.rounds if r.summary) == 2
        transcript = checkpointed(2, 3).transcript(checkpointed(2, 3).since_last_summary()[:-1])
        assert f"USER: {CHECKPOINT_PREFIX}c" in transcript and "round 4" in transcript and "round 5" not in transcript

    def test_compact_needs_two_model_rounds_in_the_view(self):
        import pytest
        with pytest.raises(AssertionError):
            pending(1).compact("c")
        with pytest.raises(AssertionError):
            checkpointed(2, 1).compact("again")        # view (checkpoint, round 3): nothing to fold

    def test_the_checkpoint_renders_as_one_user_message_and_is_never_stubbed(self):
        p = checkpointed(2, 3)                         # view: checkpoint, 3, 4, 5
        for k in (1, 2, 3, 4):
            assert not p.stubbed(0, k)
        msgs = p.messages(expire_after=1)
        assert msgs[1] == {"role": "user", "content": CHECKPOINT_PREFIX + "c"}
        assert results_in(msgs) == [EXPIRED_RESULT, EXPIRED_RESULT, "result 5"]
        assert "result 1" not in json.dumps(msgs) and "result 2" not in json.dumps(msgs)

    def test_distances_and_round_numbers_are_the_models(self):
        p = checkpointed(2, 3)                         # view: checkpoint, 3, 4, 5 (distances -, 2, 1, 0)
        assert p.expiring(2) == ("round 4: Read",)
        assert p.expiring(3) == ("round 3: Read",)
        assert p.expiring(4) == ()                     # the checkpoint never leaves
        assert p.expiring(5) == ()

    def test_pricing_counts_the_checkpoint_and_the_view_only(self):
        p = pending(3).compact("c", tokens=100)        # rounds 1, 2 (10, 20) folded; view: checkpoint 100, round 3 re-priced
        own = estimate_tokens(round_(3).own_text())
        assert p.priced_tokens(None) == 100 + own
        assert p.priced_tokens(6) == 100 + own

    def test_the_kept_round_is_repriced_to_its_own_text(self):
        """A round's frame priced the results of the round before it. Once those fold into the
        checkpoint, keeping the frame count would charge the view for text no longer in the request
        (a 2K pytest output priced on the round after it overflowed an 8k run that had room)."""
        big = Round("", (call("Read"),), (ToolResult("call_0", "Read", "x" * 8000),), tokens=10)
        last = Round("r", (call("Read"),), (ToolResult("call_0", "Read", "small"),), tokens=2000)    # 2000 = big's results, priced here
        p = PendingTurn("q").add_round(big).add_round(last).compact("c", tokens=50)
        kept = p.since_last_summary()[1]
        assert kept.tokens == estimate_tokens(last.own_text()) and kept.tokens < 10
        assert kept.results == last.results                              # the results stay: they are the unpriced text

    def test_an_unpriced_checkpoint_is_estimated(self):
        p = pending(2).compact("c")
        assert p.since_last_summary()[0].tokens == estimate_tokens(CHECKPOINT_PREFIX + "c")

    def test_the_finished_turn_keeps_the_record_and_renders_the_view(self):
        turn = pending(3).compact("c", tokens=100).finish("done", tokens=0, cancelled=False)
        assert len(turn.rounds) == 4
        assert [m["role"] for m in turn.messages()] == ["user", "user", "assistant", "tool", "assistant"]
        assert "round 1" not in turn.transcript() and f"USER: {CHECKPOINT_PREFIX}c" in turn.transcript()
        assert Round.from_dict(turn.rounds[2].to_dict()) == turn.rounds[2]       # the flag round-trips


# ---------------------
# The tool schemas are prior, like the system prompt
# ---------------------

class TestToolSchemasArePrior:
    """The schemas go on every request whole. Priced as prior they cost the same on every request;
    left to the usage frames they would sit on whichever round's frame first absorbed them (round
    one, in practice) and leave the estimate with that round when a checkpoint folds it."""

    def test_the_schemas_are_priced_by_estimate(self, make_state):
        bare = make_state(pending=PendingTurn("q"), settings=settings(6))
        with_tools = make_state(pending=PendingTurn("q"), settings=settings(6), tools=REGISTRY)
        cost = estimate_tokens(json.dumps(REGISTRY.schemas()))
        assert cost > 0 and bare.tools_tokens() == 0 and with_tools.tools_tokens() == cost
        assert with_tools.prompt_tokens(with_tools.pending_tokens()) - bare.prompt_tokens(bare.pending_tokens()) == cost
        assert with_tools.session_tokens(0) - bare.session_tokens(0) == cost

    def test_the_stream_counts_them_as_prior(self, make_state):
        bare = make_state(pending=PendingTurn("q"), settings=settings(6))
        with_tools = make_state(pending=PendingTurn("q"), settings=settings(6), tools=REGISTRY)
        _, (ev_bare,) = NextRound().execute(bare)
        _, (ev_tools,) = NextRound().execute(with_tools)
        assert ev_tools.prior_tokens - ev_bare.prior_tokens == with_tools.tools_tokens()

    def test_the_price_does_not_move_when_round_one_folds(self, make_state):
        """Before: round one's frame carried the schemas, so checkpointing it dropped them from the
        estimate for one request. Now the prior carries them on either side of the fold."""
        state = make_state(pending=pending(3), settings=settings(6), tools=REGISTRY)
        folded = make_state(pending=pending(3).compact("c", tokens=5), settings=settings(6), tools=REGISTRY)
        _, (before,) = NextRound().execute(state)
        _, (after,) = NextRound().execute(folded)
        assert before.prior_tokens >= state.tools_tokens() and after.prior_tokens >= folded.tools_tokens()
