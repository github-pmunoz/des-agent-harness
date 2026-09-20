"""
Where tool results leave the context. A pending turn renders WHOLE: every round of its view goes
out with its results, so each request appends to the previous one. Results leave the request at
two points only — when a checkpoint folds their round (PendingTurn.compact), and when the turn
ends: a history turn is always stubbed, EXPIRED_RESULT in place of each result, the calls kept
(its answer is what the results led to). Nothing is recorded: stubbing is a rendering, so the
session file, the repeat detector and the compaction transcript always read the whole results.
The scratchpad block paces the model towards the round cap, the one expiry event it can see coming.
"""
import json

from conftest import MODELS

from desh.llama.tokens import estimate_result_tokens, estimate_tokens
from desh.llama.wire import ToolCall
from desh.tools import ToolRegistry
from desh_chat.events import NextRound, StreamCompletion
from desh_chat.scratchpad import CAP_REACHED_LINE, LAST_ROUND_LINE, Scratchpad
from desh_chat.state import CHECKPOINT_PREFIX, EXPIRED_RESULT, MENTION_CHARS, ChatHistory, PendingTurn, Round, Settings, ToolResult, Turn, StopReason


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


def settings(max_tool_rounds: int = 10) -> Settings:
    return Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=max_tool_rounds)


def results_in(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


# ---------------------
# The pending turn renders whole
# ---------------------

class TestWhole:
    def test_every_result_of_the_view_is_sent_however_old(self):
        p = pending(12)
        assert results_in(p.messages()) == [f"result {n}" for n in range(1, 13)]

    def test_one_more_round_appends_to_the_request(self):
        """The prefix of a request is the previous request: what the server has cached stays valid."""
        before = pending(5).messages()
        after = pending(6).messages()
        assert after[:len(before)] == before

    def test_stubbing_keeps_the_calls_and_the_ids(self):
        r = round_(1, "Read", "Bash")
        msgs = r.messages(stubbed=True)
        assert msgs[0]["tool_calls"] == r.messages()[0]["tool_calls"]
        assert [(m["tool_call_id"], m["content"]) for m in msgs[1:]] == [("call_0", EXPIRED_RESULT), ("call_1", EXPIRED_RESULT)]
        assert r.results[0].content == "result 1"


class TestMentions:
    """A digest line says what each call was ABOUT, not just which tool ran: 'Read
    tests/conftest.py' keeps the shape of a folded round without its results. The target comes
    from the registry (ToolRegistry.target) through `describe`; the line is names only when no
    describe is given, when the registry does not know the tool, or when the call has no target.
    A target is folded onto one line and cut at MENTION_CHARS: a mention, never a dump."""

    def line(self, first: Round, describe=None) -> str:
        p = PendingTurn("q").add_round(first).add_round(round_(2))
        return p.digest(p.rounds[:1], describe=describe)

    def test_a_call_is_mentioned_with_its_target(self):
        first = Round("", (call("Read", 0, file_path="tests/conftest.py"),
                           call("Bash", 1, reason="find the answer", command='grep -n "def answer" src/x.py')))
        assert self.line(first, REGISTRY.target) == 'round 1: Read tests/conftest.py, Bash grep -n "def answer" src/x.py'

    def test_without_describe_the_line_is_names_only(self):
        first = Round("", (call("Read", 0, file_path="tests/conftest.py"), call("Bash", 1, reason="r", command="ls")))
        assert self.line(first) == "round 1: Read, Bash"

    def test_a_call_without_a_target_keeps_its_name(self):
        first = Round("", (call("nope", 0, anything="at all"), call("Read", 1)))     # unknown tool; Read with no path
        assert self.line(first, REGISTRY.target) == "round 1: nope, Read"

    def test_a_long_target_is_cut_at_mention_chars(self):
        command = "x" * (MENTION_CHARS + 40)
        assert self.line(Round("", (call("Bash", 0, reason="r", command=command),)), REGISTRY.target) == f"round 1: Bash {'x' * MENTION_CHARS}..."

    def test_a_multiline_target_is_folded_onto_one_line(self):
        command = "ls   src/\n\tgrep -c def\n"
        assert self.line(Round("", (call("Bash", 0, reason="r", command=command),)), REGISTRY.target) == "round 1: Bash ls src/ grep -c def"


# ---------------------
# NextRound: what the request looks like
# ---------------------

class TestRequest:
    def stream_event(self, make_state, p: PendingTurn, cap: int = 10, scratchpad=None) -> StreamCompletion:
        state = make_state(pending=p, settings=settings(cap), scratchpad=scratchpad)
        _, events = NextRound().execute(state)
        (ev,) = events
        assert isinstance(ev, StreamCompletion)
        return ev

    def test_the_pending_turn_is_sent_whole(self, make_state):
        ev = self.stream_event(make_state, pending(8))
        assert results_in(ev.request.messages) == [f"result {n}" for n in range(1, 9)]

    def test_no_closing_line_before_the_cap(self, make_state):
        ev = self.stream_event(make_state, pending(3), cap=5, scratchpad=Scratchpad())
        block = ev.request.messages[-1]["content"]
        assert block.startswith("<scratchpad>Round 4 of 5") and LAST_ROUND_LINE not in block and CAP_REACHED_LINE not in block

    def test_the_last_round_is_announced_in_the_scratchpad_block(self, make_state):
        """Round cap of cap: the last reply whose calls run, and the last request that shows the
        turn's results."""
        ev = self.stream_event(make_state, pending(4), cap=5, scratchpad=Scratchpad().with_entry("k", "fact", "v"))
        block = ev.request.messages[-1]["content"]
        assert block.startswith("<scratchpad>Round 5 of 5") and LAST_ROUND_LINE in block and CAP_REACHED_LINE not in block
        assert not any(LAST_ROUND_LINE in m["content"] for m in ev.request.messages[:-1])

    def test_the_capped_reply_is_told_only_scratchpad_calls_run(self, make_state):
        ev = self.stream_event(make_state, pending(5), cap=5, scratchpad=Scratchpad())
        block = ev.request.messages[-1]["content"]
        assert CAP_REACHED_LINE in block and LAST_ROUND_LINE not in block

    def test_no_scratchpad_means_no_line_anywhere(self, make_state):
        ev = self.stream_event(make_state, pending(4), cap=5)
        assert not any(LAST_ROUND_LINE in m["content"] for m in ev.request.messages)

    def test_the_block_is_priced_with_its_closing_line(self, make_state):
        state = make_state(pending=pending(4), settings=settings(5), scratchpad=Scratchpad())
        assert state.scratchpad_tokens() == estimate_tokens(state.scratchpad_block()["content"])
        assert state.scratchpad_tokens() > estimate_tokens(Scratchpad().message((4, 5)))

    def test_history_turns_are_always_stubbed_in_the_view(self, make_state):
        turn = Turn("q0", "a0", rounds=(round_(1),), tokens=5, stop=StopReason.ANSWER)
        state = make_state(pending=PendingTurn("q").add_round(round_(1)), settings=settings(), history=ChatHistory().append(turn))
        _, events = NextRound().execute(state)
        results = results_in(events[0].request.messages)
        assert results == [EXPIRED_RESULT, "result 1"]     # history round stubbed, pending round whole


# ---------------------
# Pricing follows the rendering
# ---------------------

class TestPricing:
    def test_the_rounds_cost_what_their_frames_priced(self):
        assert pending(4).priced_tokens() == 10 + 20 + 30 + 40

    def test_an_unpriced_round_still_contributes_nothing(self):
        """tokens == 0 means no frame priced it; the latest results are priced by unpriced_text()
        instead, so a 0 here is not a bug to paper over."""
        p = PendingTurn("q").add_round(Round("", (call("Read"),), (ToolResult("call_0", "Read", "r"),), tokens=0))
        assert p.priced_tokens() == 0

    def test_prior_tokens_grow_by_the_round_added(self, make_state):
        before = make_state(pending=pending(1), settings=settings())
        after = make_state(pending=pending(2), settings=settings())
        _, (ev_before,) = NextRound().execute(before)
        _, (ev_after,) = NextRound().execute(after)
        assert ev_after.prior_tokens - ev_before.prior_tokens == 20

    def test_a_turn_that_outgrows_the_window_is_checkpointed(self, make_state):
        """Nothing ages out of a pending turn, so the checkpoint is what makes room inside it."""
        from desh_chat.events import CompactPendingTurn
        huge = Round("", (call("Read"),), (ToolResult("call_0", "Read", "x" * 40000),), tokens=10_000)
        p = PendingTurn("q").add_round(huge).add_round(round_(2))
        state = make_state(pending=p, settings=Settings(model=MODELS[0], temperature=0.3, think=False, context=8000, max_turn_tokens=2000))
        _, events = NextRound().execute(state)
        assert any(isinstance(e, CompactPendingTurn) for e in events)


# ---------------------
# A checkpoint in the turn: the request and the pricing run over the view
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
        assert "EARLIER CHECKPOINT: c" in transcript and "round 4" in transcript and "round 5" not in transcript

    def test_compact_needs_two_model_rounds_in_the_view(self):
        import pytest
        with pytest.raises(AssertionError):
            pending(1).compact("c")
        with pytest.raises(AssertionError):
            checkpointed(2, 1).compact("again")        # view (checkpoint, round 3): nothing to fold

    def test_the_checkpoint_renders_as_one_user_message_in_place_of_the_folded_rounds(self):
        p = checkpointed(2, 3)                         # view: checkpoint, 3, 4, 5
        msgs = p.messages()
        assert msgs[1] == {"role": "user", "content": CHECKPOINT_PREFIX + "c"}
        assert results_in(msgs) == ["result 3", "result 4", "result 5"]
        assert "result 1" not in json.dumps(msgs) and "result 2" not in json.dumps(msgs)

    def test_the_block_counts_the_models_rounds_across_a_checkpoint(self, make_state):
        state = make_state(pending=checkpointed(2, 3), settings=settings(6), scratchpad=Scratchpad())     # 5 model rounds, one checkpoint
        block = state.scratchpad_block()["content"]
        assert block.startswith("<scratchpad>Round 6 of 6") and LAST_ROUND_LINE in block

    def test_pricing_counts_the_checkpoint_and_the_view_only(self):
        p = pending(3).compact("c", tokens=100)        # rounds 1, 2 (10, 20) folded; view: checkpoint 100, round 3 re-priced
        own = estimate_tokens(round_(3).own_text())
        assert p.priced_tokens() == 100 + own

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
        turn = pending(3).compact("c", tokens=100).finish("done", tokens=0, stop=StopReason.ANSWER)
        assert len(turn.rounds) == 4
        assert [m["role"] for m in turn.messages()] == ["user", "user", "assistant", "tool", "assistant"]
        assert "round 1" not in turn.transcript() and "EARLIER CHECKPOINT: c" in turn.transcript()
        assert Round.from_dict(turn.rounds[2].to_dict()) == turn.rounds[2]       # the flag round-trips


# ---------------------
# The tool schemas are prior, like the system prompt
# ---------------------

class TestToolSchemasArePrior:
    """The schemas go on every request whole. Priced as prior they cost the same on every request;
    left to the usage frames they would sit on whichever round's frame first absorbed them (round
    one, in practice) and leave the estimate with that round when a checkpoint folds it."""

    def test_the_schemas_are_priced_by_estimate(self, make_state):
        bare = make_state(pending=PendingTurn("q"), settings=settings())
        with_tools = make_state(pending=PendingTurn("q"), settings=settings(), tools=REGISTRY)
        cost = estimate_tokens(json.dumps(REGISTRY.schemas()))
        assert cost > 0 and bare.tools_tokens() == 0 and with_tools.tools_tokens() == cost
        assert with_tools.prompt_tokens(with_tools.pending_tokens()) - bare.prompt_tokens(bare.pending_tokens()) == cost
        assert with_tools.session_tokens(0) - bare.session_tokens(0) == cost

    def test_the_stream_counts_them_as_prior(self, make_state):
        bare = make_state(pending=PendingTurn("q"), settings=settings())
        with_tools = make_state(pending=PendingTurn("q"), settings=settings(), tools=REGISTRY)
        _, (ev_bare,) = NextRound().execute(bare)
        _, (ev_tools,) = NextRound().execute(with_tools)
        assert ev_tools.prior_tokens - ev_bare.prior_tokens == with_tools.tools_tokens()

    def test_the_price_does_not_move_when_round_one_folds(self, make_state):
        """Before: round one's frame carried the schemas, so checkpointing it dropped them from the
        estimate for one request. Now the prior carries them on either side of the fold."""
        state = make_state(pending=pending(3), settings=settings(), tools=REGISTRY)
        folded = make_state(pending=pending(3).compact("c", tokens=5), settings=settings(), tools=REGISTRY)
        _, (before,) = NextRound().execute(state)
        _, (after,) = NextRound().execute(folded)
        assert before.prior_tokens >= state.tools_tokens() and after.prior_tokens >= folded.tools_tokens()


# ---------------------
# The compaction transcript: the view, fitted to a budget
# ---------------------

class TestTranscriptFit:
    """fit_transcript gives up the least first: the oldest section is reduced to its last rendering,
    then the next; then the oldest non-fixed sections are left out with one line saying so; a
    fixed section (the task message, a checkpoint, a summary) is never left out; and if the fixed
    sections alone do not fit, the head of the text is cut."""

    HEADER = "[2 earlier rounds left out of this transcript]"

    def sections(self):
        from desh_chat.state import Section
        return [Section(("TASK",), fixed=True),
                Section(("A whole " + "a" * 200, "A stubbed " + "a" * 40)),
                Section(("B whole " + "b" * 200, "B stubbed " + "b" * 40)),
                Section(("C whole " + "c" * 200, "C stubbed " + "c" * 40))]

    def test_no_budget_renders_everything_whole(self):
        from desh_chat.state import fit_transcript
        text = fit_transcript(self.sections(), None)
        assert text.startswith("TASK\nA whole") and "C whole" in text

    def test_the_oldest_is_reduced_first(self):
        from desh_chat.state import fit_transcript
        want = "TASK\nA stubbed " + "a" * 40 + "\nB whole " + "b" * 200 + "\nC whole " + "c" * 200
        text = fit_transcript(self.sections(), estimate_result_tokens(want) + 1)
        assert text == want

    def test_then_the_oldest_are_left_out_with_a_note_and_fixed_ones_stay(self):
        from desh_chat.state import fit_transcript
        want = self.HEADER + "\nTASK\nC stubbed " + "c" * 40
        text = fit_transcript(self.sections(), estimate_result_tokens(want) + 1)
        assert text == want

    def test_the_head_is_cut_as_a_last_resort(self):
        from desh_chat.state import fit_transcript, Section
        text = fit_transcript([Section(("x" * 1000,), fixed=True)], 30)
        assert text.startswith("[transcript cut to fit]\n") and len(text) < 200

    def long_pending(self, n: int) -> PendingTurn:
        """Like pending(n), with results longer than the expiry stub, as real ones are."""
        p = PendingTurn("q")
        for i in range(1, n + 1):
            tc = call("Read")
            p = p.add_round(Round(f"round {i}", (tc,), (ToolResult(tc.id, tc.name, f"result {i} " + "x" * 200),), tokens=10 * i))
        return p

    def test_the_pending_transcript_is_whole_and_follows_the_budget(self):
        """The folded rounds 1..3 of a 4-round turn render whole, as the model saw them. A budget
        too small for that stubs the oldest result first."""
        p = self.long_pending(4)
        folded = p.since_last_summary()[:-1]
        whole = p.transcript(folded)
        lines = whole.splitlines()
        assert lines[0] == "USER: q"
        assert [lines[i].startswith(f"TOOL Read: result {n} x") for i, n in ((2, 1), (4, 2), (6, 3))] == [True] * 3
        fitted = p.transcript(folded, budget_tokens=estimate_result_tokens(whole) - estimate_result_tokens("USER: q") - 5).splitlines()     # the budget is the rounds'
        assert fitted[2] == f"TOOL Read: {EXPIRED_RESULT}" and fitted[4].startswith("TOOL Read: result 2 x")

    def test_a_checkpoint_in_the_transcript_is_fixed(self):
        p = checkpointed(2, 3)                                  # view: checkpoint, 3, 4, 5
        want = "[2 earlier rounds left out of this transcript]\nUSER: q\nEARLIER CHECKPOINT: c"
        assert p.transcript(p.since_last_summary()[:-1], budget_tokens=estimate_result_tokens(want) + 1) == want

    def test_the_history_transcript_stubs_rounds_and_reduces_oldest_turns_first(self):
        turns = [Turn(f"q{i}", f"a{i}", rounds=(round_(1),), tokens=5, stop=StopReason.ANSWER) for i in range(3)]
        h = ChatHistory()
        for t in turns:
            h = h.append(t)
        whole = h.transcript()
        assert "result 1" not in whole and whole.count(EXPIRED_RESULT) == 3      # as the model last saw them
        reduced = h.transcript(budget_tokens=estimate_result_tokens(whole) - 1)
        assert reduced.startswith("USER: q0\nASSISTANT: a0") and reduced.count(EXPIRED_RESULT) == 2
        h = ChatHistory().compact("s", tokens=5).append(Turn("q " + "x" * 300, "a", tokens=5, stop=StopReason.ANSWER))   # too long even without rounds
        summary_only = "[1 earlier turns left out of this transcript]\n" + h.turns[0].transcript()
        assert h.transcript(budget_tokens=estimate_result_tokens(summary_only) + 1) == summary_only     # a summary is fixed


class TestResultPricing:
    def test_the_latest_results_are_priced_denser_than_prose(self, make_state):
        """The unpriced text is tool output after round one: code and paths tokenize at ~3.3
        chars per token, not 4. The first request's unpriced text is the user message: prose."""
        from desh.llama.tokens import estimate_result_tokens
        p = pending(1)
        state = make_state(pending=p, settings=settings())
        assert state.pending_tokens() == p.priced_tokens() + estimate_result_tokens("result 1")
        first = make_state(pending=PendingTurn("q" * 40), settings=settings())
        assert first.pending_tokens() == estimate_tokens("q" * 40)
