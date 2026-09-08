"""
Per-call confirmation inside the round: ExecuteToolCalls(i) asks the operator about call i when its
tool wants confirmation, then runs it or records the denial, and steps to i+1, NextRound or TurnEnd.

Policy is data on the Tool (confirm=True asks, confirm=False is read-only). Unknown tools and
read-only tools are never asked about. A "no" short-circuits the round: the later calls are answered
"not run" without asking. A "no" with a message puts that message on the wire instead of DENIED_TEXT.
A "cancel" ends the turn cancelled.

The prompt dialogue itself (ask) is tested on its own in TestAsk; the step tests stub it.
"""
import pytest

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer

from desh.engine import Engine
from desh.llama.wire import ToolCall
from desh.tools import Tool, ToolRegistry
from desh_chat import gate
from desh_chat.display import DisplayStats, Info, Warn
from desh_chat.events import ExecuteToolCalls, NextRound, PromptUser, TurnEnd, UserMessage
from desh_chat.gate import DENIED_TEXT, SKIPPED_TEXT, Answer
from desh_chat.state import InferenceEngine, PendingTurn, Round


RAN: list[str] = []     # what actually executed, to prove denied/skipped calls never do


def read_file(path: str) -> str:
    """Contents of a file."""
    RAN.append(f"read {path}")
    return f"<{path}>"


def write_file(path: str, text: str) -> str:
    """Write text to a file."""
    RAN.append(f"write {path}")
    return "ok"


def delete_file(path: str) -> str:
    """Delete a file."""
    RAN.append(f"delete {path}")
    return "gone"


REGISTRY = ToolRegistry().add(read_file, confirm=False).add(write_file).add(delete_file)

READ = ToolCall(index=0, id="call_r", type="function", name="read_file", arguments='{"path": "a"}')
WRITE = ToolCall(index=1, id="call_w", type="function", name="write_file", arguments='{"path": "a", "text": "x"}')
DELETE = ToolCall(index=2, id="call_d", type="function", name="delete_file", arguments='{"path": "a"}')
UNKNOWN = ToolCall(index=3, id="call_u", type="function", name="format_disk", arguments='{}')


def pending_with(*calls: ToolCall) -> PendingTurn:
    return PendingTurn("q").add_round(Round("", calls))


@pytest.fixture(autouse=True)
def clear_ran():
    RAN.clear()


@pytest.fixture
def answers(monkeypatch):
    """Script ask(): one Answer per prompt, in order; a prompt past the script is a test failure."""
    def _set(*script: Answer):
        it = iter(script)
        asked: list[str] = []

        def fake_ask(tc):
            asked.append(tc.name)
            try:
                return next(it)
            except StopIteration:
                pytest.fail(f"ask() called for {tc.name} but the script is exhausted")
        monkeypatch.setattr(gate, "ask", fake_ask)
        return asked
    return _set


def run_round(state, index=0):
    """Drive ExecuteToolCalls steps until they hand off to something else. Returns (state, final events)."""
    ev = ExecuteToolCalls(index)
    while True:
        state, evs = ev.execute(state)
        nxt = [e for e in evs if isinstance(e, ExecuteToolCalls)]
        if not nxt:
            return state, evs
        ev = nxt[0]


# ---------------------
# Policy on the Tool
# ---------------------

class TestPolicy:
    def test_tools_ask_unless_declared_read_only(self):
        assert Tool.define(write_file).confirm is True
        assert Tool.define(read_file, confirm=False).confirm is False

    def test_registry_add_passes_the_policy_through(self):
        assert REGISTRY.get("read_file").confirm is False
        assert REGISTRY.get("write_file").confirm is True


# ---------------------
# Who gets asked
# ---------------------

class TestWhoIsAsked:
    def test_read_only_and_unknown_calls_run_without_asking(self, make_state, answers):
        asked = answers()
        state, evs = run_round(make_state(pending=pending_with(READ, UNKNOWN), tools=REGISTRY))
        assert asked == []
        assert RAN == ["read a"]
        assert [r.name for r in state.pending.rounds[-1].results] == ["read_file", "format_disk"]
        assert [type(e) for e in evs] == [Info, DisplayStats, NextRound]

    def test_empty_registry_never_asks(self, make_state, answers):
        asked = answers()
        run_round(make_state(pending=pending_with(WRITE)))
        assert asked == []

    def test_only_asking_tools_are_asked_and_in_call_order(self, make_state, answers):
        asked = answers(Answer("yes"), Answer("yes"))
        run_round(make_state(pending=pending_with(READ, WRITE, UNKNOWN, DELETE), tools=REGISTRY))
        assert asked == ["write_file", "delete_file"]
        assert RAN == ["read a", "write a", "delete a"]


# ---------------------
# Answers -> results and events
# ---------------------

class TestAnswers:
    def test_yes_runs_the_call_and_steps_to_the_next(self, make_state, answers):
        answers(Answer("yes"))
        mid, evs = ExecuteToolCalls(0).execute(make_state(pending=pending_with(WRITE, READ), tools=REGISTRY))
        assert RAN == ["write a"]
        assert mid.pending.rounds[-1].results[0].content == "ok"
        assert [type(e) for e in evs] == [Info, DisplayStats, ExecuteToolCalls] and evs[2].index == 1

    def test_last_call_hands_off_to_next_round(self, make_state, answers):
        answers(Answer("yes"))
        _, evs = ExecuteToolCalls(0).execute(make_state(pending=pending_with(WRITE), tools=REGISTRY))
        assert [type(e) for e in evs] == [Info, DisplayStats, NextRound]

    def test_no_records_the_denial_and_skips_the_rest_without_asking(self, make_state, answers):
        asked = answers(Answer("no"))
        state, evs = run_round(make_state(pending=pending_with(READ, WRITE, DELETE, READ), tools=REGISTRY))
        assert asked == ["write_file"]                  # delete_file was never asked about
        assert RAN == ["read a"]                        # only the call before the "no" ran
        results = state.pending.rounds[-1].results
        assert [r.tool_call_id for r in results] == [READ.id, WRITE.id, DELETE.id, READ.id]   # every call answered
        assert results[1].content == DENIED_TEXT
        assert results[2].content == SKIPPED_TEXT and results[3].content == SKIPPED_TEXT
        assert [type(e) for e in evs] == [Warn, Warn, DisplayStats, NextRound]
        assert "write_file" in evs[0].text and "delete_file" in evs[1].text

    def test_no_on_the_last_call_has_nothing_to_skip(self, make_state, answers):
        answers(Answer("no"))
        _, evs = run_round(make_state(pending=pending_with(WRITE), tools=REGISTRY))
        assert [type(e) for e in evs] == [Warn, DisplayStats, NextRound]

    def test_message_replaces_the_denial_text_on_the_wire(self, make_state, answers):
        answers(Answer("no", message="never touch a; use b instead"))
        state, evs = run_round(make_state(pending=pending_with(WRITE), tools=REGISTRY))
        msg = state.pending.messages()[-1]
        assert msg == {"role": "tool", "tool_call_id": WRITE.id, "name": "write_file", "content": "never touch a; use b instead"}
        assert "use b instead" in evs[0].text

    def test_cancel_ends_the_turn_cancelled_keeping_what_ran(self, make_state, answers):
        answers(Answer("cancel"))
        state, evs = run_round(make_state(pending=pending_with(READ, WRITE, DELETE), tools=REGISTRY))
        assert RAN == ["read a"]
        assert len(state.pending.rounds[-1].results) == 1          # partial round stays on pending; TurnEnd freezes it
        assert [type(e) for e in evs] == [Warn, TurnEnd]
        assert evs[1] == TurnEnd(assistant="", tokens=0, cancelled=True)

    def test_no_step_ever_schedules_prompt_user(self, make_state, answers):
        for a in (Answer("yes"), Answer("no"), Answer("no", "why"), Answer("cancel")):
            answers(a)
            _, evs = run_round(make_state(pending=pending_with(WRITE), tools=REGISTRY))
            assert not any(isinstance(e, PromptUser) for e in evs)


# ---------------------
# Through the engine: the model sees the denial and adapts
# ---------------------

class TestFullLoop:
    def test_denied_round_is_echoed_to_the_model_and_the_turn_completes(self, make_state, answers, no_esc_watcher):
        answers(Answer("no", message="read only today"))
        server = FakeServer(script=[
            {"tool_calls": [{"name": "read_file", "arguments": '{"path": "a"}'}, {"name": "delete_file", "arguments": '{"path": "a"}'}]},
            {"content": "Understood, I only read it."},
        ])
        state = make_state(inference=InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT),
                           tools=REGISTRY, running=False)
        final = Engine[type(state)]().run(state, seed=[UserMessage("clean up a")])
        turn = final.history.turns[0]
        assert turn.cancelled is False
        assert [r.content for r in turn.rounds[0].results] == ["<a>", "read only today"]
        _, second = server.calls[1]
        assert [m["role"] for m in second.messages][-3:] == ["assistant", "tool", "tool"]
        assert second.messages[-1]["content"] == "read only today"

    def test_cancelled_round_becomes_a_cancelled_turn(self, make_state, answers, no_esc_watcher):
        answers(Answer("cancel"))
        server = FakeServer(script=[{"tool_calls": [{"name": "delete_file", "arguments": '{"path": "a"}'}]}, {"content": "never"}])
        state = make_state(inference=InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT),
                           tools=REGISTRY, running=False)
        final = Engine[type(state)]().run(state, seed=[UserMessage("rm a")])
        assert final.pending is None
        assert final.history.turns[0].cancelled is True
        assert len(server.calls) == 1 and RAN == []


# ---------------------
# The prompt dialogue: one key decides, "m" opens a line for the message
# ---------------------

import builtins
import io
import sys

from desh_chat.gate import ask


@pytest.fixture
def keys(monkeypatch):
    """Feed sys.stdin (not a TTY, so no termios) and script input() for the message line.
    Returns the list of input() prompts seen, to prove the message line is only opened on 'm'."""
    def _set(typed: str, lines: tuple[str, ...] = ()):
        monkeypatch.setattr(sys, "stdin", io.StringIO(typed))
        it = iter(lines)
        seen: list[str] = []

        def fake_input(prompt=""):
            seen.append(prompt)
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        monkeypatch.setattr(builtins, "input", fake_input)
        return seen
    return _set


class TestAsk:
    @pytest.mark.parametrize("typed, kind", [("y", "yes"), ("Y", "yes"), ("\n", "yes"), ("\r", "yes"),
                                             ("n", "no"), ("N", "no"),
                                             ("c", "cancel"), ("C", "cancel"), ("\x1b", "cancel")])
    def test_single_key_decides_without_enter(self, keys, typed, kind):
        seen = keys(typed)
        assert ask(WRITE) == Answer(kind)
        assert seen == []                       # no line-mode prompt was opened

    def test_unknown_keys_ask_again_until_a_decision(self, keys):
        keys("x?7n")
        assert ask(WRITE) == Answer("no")

    def test_m_opens_a_message_line_and_denies_with_it(self, keys):
        seen = keys("m", lines=("use b, a is read-only",))
        assert ask(WRITE) == Answer("no", message="use b, a is read-only")
        assert len(seen) == 1

    def test_m_with_an_empty_message_is_a_plain_no(self, keys):
        keys("m", lines=("",))
        assert ask(WRITE) == Answer("no", message="")

    def test_m_then_eof_on_the_message_line_is_a_plain_no(self, keys):
        keys("m")                                # input() raises EOFError
        assert ask(WRITE) == Answer("no", message="")

    def test_eof_at_the_key_prompt_is_cancel_not_yes(self, keys):
        # A closed or broken stdin must never approve a call: absence of input is not consent.
        keys("")
        assert ask(WRITE) == Answer("cancel")

    def test_message_line_is_scrubbed_from_readline_history(self, keys, monkeypatch):
        keys("m", lines=("why",))
        length = {"n": 5}
        removed: list[int] = []
        monkeypatch.setattr(gate.readline, "get_current_history_length", lambda: length["n"])
        monkeypatch.setattr(gate.readline, "remove_history_item", lambda i: removed.append(i))
        real_input = builtins.input

        def input_that_adds_history(prompt=""):
            length["n"] += 1                     # what readline does under a TTY
            return real_input(prompt)
        monkeypatch.setattr(builtins, "input", input_that_adds_history)
        assert ask(WRITE) == Answer("no", message="why")
        assert removed == [5]                    # exactly the entry input() added, nothing older
