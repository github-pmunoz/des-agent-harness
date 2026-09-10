"""
The delegate tool: a subagent is a nested Engine run whose final turn becomes the parent's tool
result. Three layers are covered: what the parent reads for each way a child run can end
(answer()), what a child is built from (settings, session file, tools), and the full loop —
parent asks for delegate, child runs to drain on the same FakeServer script, parent answers —
including the two exits that must NOT become tool text: a harness bug does, Ctrl+C does not.
"""
import os
from dataclasses import replace

import pytest

from desh.engine import Engine
from desh.tools import ToolRegistry
from desh_chat import gate
from desh_chat.delegate import DELEGATE_SYSTEM_PROMPT, Delegate, answer, child_session_file, child_settings
from desh_chat.events import UserMessage
from desh_chat.gate import Answer
from desh_chat.state import ChatHistory, InferenceEngine, Round, Settings, ToolResult, Turn
from desh.llama.wire import ToolCall

from conftest import FakeServer, MAX_CONTEXT, MODELS, PORT


SETTINGS = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, max_tool_rounds=3)


def turn(assistant: str, rounds: int = 0, cancelled: bool = False, stop: str = "") -> Turn:
    tc = ToolCall(index=0, id="c", type="function", name="Read", arguments="{}")
    return Turn("task", assistant, cancelled=cancelled, stop=stop,
                rounds=tuple(Round("", (tc,), results=(ToolResult("c", "Read", "x"),)) for _ in range(rounds)))


def child_of(make_state, *turns: Turn):
    return make_state(settings=SETTINGS, history=ChatHistory(tuple(turns)), running=False)


@pytest.fixture
def always_yes(monkeypatch):
    monkeypatch.setattr(gate, "ask", lambda tc: Answer("yes"))


def with_server(make_state, server, **overrides):
    inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
    return inference, make_state(inference=inference, settings=SETTINGS, **overrides)


def with_delegate(tools: ToolRegistry, *, inference, settings, session_file=None, root=".") -> ToolRegistry:
    """`tools` plus the delegate tool, whose subagents get `tools` as given — without delegate.
    What cli.build_tools does for the real run, minus the per-flag toolsets: registered with
    inject=("settings",) so every call carries the parent's current settings."""
    d = Delegate(root=root, inference=inference, settings=settings, tools=tools, session_file=session_file)
    return tools.add(d.delegate, name="delegate", inject=("settings",))


def parent_with_delegate(make_state, server, tools=ToolRegistry(), delegate_root=".", **overrides):
    """A parent state whose registry is `tools` plus delegate; subagents get `tools`. Not running,
    so the parent drains after its turn instead of prompting for another."""
    inference, state = with_server(make_state, server, running=False, **overrides)
    registry = with_delegate(tools, inference=inference, settings=state.settings,
                             session_file=state.session_file, root=delegate_root)
    return state.__class__(**{**state.__dict__, "tools": registry})


class RecordingEngine:
    """Stands in for Engine inside delegate.py: records the child state it was asked to run and
    returns it with one answered turn, so a test can look at what the child was BUILT from
    without driving a server script."""
    states: list = []

    def __class_getitem__(cls, item):
        return cls

    def __init__(self, **kwargs):
        pass

    def run(self, state, seed, **kwargs):
        RecordingEngine.states.append(state)
        return replace(state, history=ChatHistory((Turn("t", "42"),)))


# ---------------------
# CLI wiring: the delegate is built from the resolved objects, not the raw flags
# ---------------------

class TestCliWiring:
    def test_build_tools_hands_the_delegate_the_resolved_logs_and_session(self, tmp_path):
        """The flags are strings; the delegate needs what main() resolves them into — the parent's
        session PATH (derived from --sessions-folder), the completions Logger and the open DES log
        file — or the child engine writes to a str and its session file has no parent stem."""
        import argparse
        from desh.llama.logger import Logger
        from desh_chat.cli import build_tools
        args = argparse.Namespace(workspace=str(tmp_path), read=True, write=False, edit=False, bash=False,
                                  current_time=False, delegate=True, system_prompt="sp", debug=False,
                                  session="", sessions_folder=str(tmp_path), completions_log="c.jsonl", des_log="d.jsonl")
        inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=FakeServer(script=[]), port=PORT)
        session_file = str(tmp_path / "run1.json")
        logger = Logger(str(tmp_path / "c.jsonl"))
        with open(tmp_path / "d.jsonl", "a") as des_log:
            tools = build_tools(args, inference, SETTINGS, session_file=session_file, completions_log=logger, des_log=des_log)
            delegate = tools.get("delegate").fn.__self__
            assert delegate.session_file == session_file
            assert delegate.completions_log is logger
            assert delegate.des_log is des_log
            assert [t.name for t in tools.tools] == ["Read", "delegate"]
            # the parent's settings go in untouched; the child derives its own per call (item below)
            assert delegate.settings == SETTINGS
            assert tools.get("delegate").inject == ("settings",)


# ---------------------
# answer(): the child's final state as tool text
# ---------------------

class TestAnswer:
    def test_no_turn_means_the_request_never_fit(self, make_state):
        assert "context window" in answer(child_of(make_state))

    def test_cancelled_turn_is_reported_not_passed_on(self, make_state):
        text = answer(child_of(make_state, turn("partial...", cancelled=True)))
        assert "cancelled" in text and "partial" not in text

    def test_plain_answer_is_returned_verbatim(self, make_state):
        assert answer(child_of(make_state, turn("42", rounds=1))) == "42"

    def test_empty_answer_is_named(self, make_state):
        assert answer(child_of(make_state, turn(""))) == "(no answer)"

    def test_a_capped_turn_qualifies_the_answer(self, make_state):
        text = answer(child_of(make_state, turn("so far: x", rounds=SETTINGS.max_tool_rounds, stop="cap")))
        assert text == "so far: x\n[Subagent hit the tool round cap]"

    def test_round_cap_with_no_text_composes_both_notes(self, make_state):
        text = answer(child_of(make_state, turn("", rounds=SETTINGS.max_tool_rounds, stop="cap")))
        assert text == "(no answer)\n[Subagent hit the tool round cap]"

    def test_finishing_on_the_last_allowed_round_is_not_a_cap(self, make_state):
        """The reason lives on the turn, not in the round count: a model that answers exactly at the
        limit answered, and the parent must not be told otherwise."""
        assert answer(child_of(make_state, turn("42", rounds=SETTINGS.max_tool_rounds))) == "42"

    def test_overflow_is_reported_as_overflow_not_as_a_cancel(self, make_state):
        text = answer(child_of(make_state, turn("", rounds=2, cancelled=True, stop="overflow")))
        assert text == "The subagent ran out of context window before finishing."

    def test_a_pending_turn_at_drain_is_a_harness_bug(self, make_state):
        from desh_chat.state import PendingTurn
        with pytest.raises(AssertionError):
            answer(make_state(settings=SETTINGS, running=False, pending=PendingTurn("t")))


# ---------------------
# What a child is built from
# ---------------------

class TestChildSetup:
    def test_child_settings_inherit_everything_but_compaction(self):
        child = child_settings(SETTINGS)
        assert child.compaction_threshold == float("inf")
        assert {k: v for k, v in child.__dict__.items() if k != "compaction_threshold"} == \
               {k: v for k, v in SETTINGS.__dict__.items() if k != "compaction_threshold"}

    def test_child_session_file_sits_beside_the_parents(self):
        path = child_session_file("/tmp/runs/abc.json")
        assert os.path.dirname(path) == "/tmp/runs"
        assert os.path.basename(path).startswith("abc.delegate-") and path.endswith(".json")
        assert child_session_file(None) is None

    def test_delegate_is_added_asking_and_subagents_do_not_get_it(self, make_state):
        inference, state = with_server(make_state, FakeServer())
        base = ToolRegistry().add(lambda: "x", name="Read", confirm=False)
        registry = with_delegate(base, inference=inference, settings=SETTINGS)
        tool = registry.get("delegate")
        assert tool is not None and tool.confirm is True
        assert [t.name for t in registry.tools] == ["Read", "delegate"]
        assert [t.name for t in tool.fn.__self__.tools.tools] == ["Read"]

    def test_the_model_sees_the_full_signature(self, make_state):
        inference, _ = with_server(make_state, FakeServer())
        params = with_delegate(ToolRegistry(), inference=inference, settings=SETTINGS).get("delegate").parameters
        assert set(params["properties"]) == {"task", "context", "gate", "check"} and params["required"] == ["task"]

    def test_child_is_built_from_child_settings_of_the_fallback(self, make_state, monkeypatch, capsys):
        """No settings injected: the Delegate's own (startup) settings, with compaction off."""
        monkeypatch.setattr("desh_chat.delegate.Engine", RecordingEngine)
        RecordingEngine.states.clear()
        inference, _ = with_server(make_state, FakeServer())
        Delegate(root=".", inference=inference, settings=SETTINGS).delegate("task")
        child, = RecordingEngine.states
        assert child.settings == child_settings(SETTINGS)
        assert child.settings.compaction_threshold == float("inf")

    def test_injected_settings_replace_the_startup_ones(self, make_state, monkeypatch, capsys):
        """The parent flipped auto mode and switched model after the registry was built: the child
        must be built from the CURRENT settings, still with compaction off."""
        monkeypatch.setattr("desh_chat.delegate.Engine", RecordingEngine)
        RecordingEngine.states.clear()
        inference, _ = with_server(make_state, FakeServer())
        current = replace(SETTINGS, auto=True, model=MODELS[1], temperature=0.9)
        Delegate(root=".", inference=inference, settings=SETTINGS).delegate("task", settings=current)
        child, = RecordingEngine.states
        assert child.settings == child_settings(current)
        assert (child.settings.auto, child.settings.model, child.settings.temperature) == (True, MODELS[1], 0.9)


# ---------------------
# The full loop: parent -> child -> parent, one server script
# ---------------------

class TestFullLoop:
    DELEGATION = {"tool_calls": [{"name": "delegate", "arguments": '{"task": "count the files", "context": "src only"}'}]}

    def test_child_answer_is_the_parents_tool_result(self, make_state, always_yes, no_esc_watcher):
        server = FakeServer(script=[self.DELEGATION, {"content": "There are 12 files."}, {"content": "Twelve."}])
        state = parent_with_delegate(make_state, server)
        final = Engine[type(state)]().run(state, seed=[UserMessage("how many files?")])
        assert len(final.history.turns) == 1 and final.pending is None
        parent_turn = final.history.turns[0]
        assert parent_turn.assistant == "Twelve."
        assert [r.content for r in parent_turn.rounds[0].results] == ["There are 12 files."]

    def test_child_keeps_its_own_session_file_beside_the_parents(self, make_state, always_yes, no_esc_watcher, tmp_path):
        parent_file = str(tmp_path / "run.json")
        server = FakeServer(script=[self.DELEGATION, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server, session_file=parent_file)
        Engine[type(state)]().run(state, seed=[UserMessage("how many files?")])
        files = {p.name for p in tmp_path.iterdir()}
        assert "run.json" in files and len(files) == 2
        child_file, = files - {"run.json"}
        assert child_file.startswith("run.delegate-") and child_file.endswith(".json")
        import json
        child_doc = json.load(open(tmp_path / child_file))
        assert [t["assistant"] for t in child_doc["turns"]] == ["12"]
        assert DELEGATE_SYSTEM_PROMPT in child_doc["meta"]["system_prompt"]

    def test_a_bug_in_the_child_becomes_tool_text(self, make_state, always_yes, no_esc_watcher):
        class Broken(FakeServer):
            def stream(self, req, renderer, cancelled=lambda: False):
                if len(self.calls) == 1:          # the child's first completion
                    self.calls.append(("stream", req))
                    raise RuntimeError("boom")
                return super().stream(req, renderer, cancelled)
        server = Broken(script=[self.DELEGATION, {"content": "ok, no child"}])
        state = parent_with_delegate(make_state, server)
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        result = final.history.turns[0].rounds[0].results[0].content
        assert result.startswith("Tool 'delegate' raised RuntimeError") and "boom" in result

    def test_the_parents_current_settings_reach_the_child_through_execute_tool_calls(self, make_state, monkeypatch, no_esc_watcher):
        """End to end: the Delegate was built with auto OFF, the parent state has auto ON (the
        operator pressed "a" after startup). ExecuteToolCalls injects state.settings, so neither
        the parent's delegate call nor the child's confirmed tool ever reaches gate.ask."""
        asked = []
        monkeypatch.setattr(gate, "ask", lambda tc: asked.append(tc.name) or Answer("yes"))
        child_tools = ToolRegistry().add(lambda: "touched", name="Touch")        # confirm=True by default
        server = FakeServer(script=[self.DELEGATION, {"tool_calls": [{"name": "Touch"}]}, {"content": "12"}, {"content": "12"}])
        inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT)
        registry = with_delegate(child_tools, inference=inference, settings=SETTINGS)      # startup: auto off
        state = make_state(inference=inference, settings=replace(SETTINGS, auto=True), running=False, tools=registry)
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        assert asked == []
        assert final.history.turns[0].rounds[0].results[0].content == "12"
        child_req = server.calls[2][1]
        assert child_req.model == SETTINGS.model

    def test_ctrl_c_in_the_child_reaches_the_parent(self, make_state, always_yes, no_esc_watcher):
        class Interrupted(FakeServer):
            def stream(self, req, renderer, cancelled=lambda: False):
                if len(self.calls) == 1:
                    raise KeyboardInterrupt
                return super().stream(req, renderer, cancelled)
        state = parent_with_delegate(make_state, Interrupted(script=[self.DELEGATION]))
        with pytest.raises(KeyboardInterrupt):
            Engine[type(state)]().run(state, seed=[UserMessage("go")])


# ---------------------
# gate and check: optional parameters on the delegate tool
# ---------------------

class TestGateAndCheck:
    def test_schema_pins_four_parameters(self, make_state):
        registry = with_delegate(ToolRegistry(), inference=InferenceEngine(models=MODELS, max_context=MAX_CONTEXT,
                                                                           server=FakeServer(script=[]), port=PORT),
                                 settings=SETTINGS)
        schema = registry.get("delegate").parameters
        assert set(schema["properties"]) == {"task", "context", "gate", "check"}
        assert schema["required"] == ["task"]
        assert schema["properties"]["gate"]["description"]
        assert schema["properties"]["check"]["description"]

    def test_gate_is_appended_to_the_child_system_prompt(self, make_state, always_yes, no_esc_watcher):
        script = {"tool_calls": [{"name": "delegate",
                                  "arguments": '{"task": "count the files", "context": "src only", "gate": "report a number"}'}]}
        server = FakeServer(script=[script, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server)
        Engine[type(state)]().run(state, seed=[UserMessage("go")])
        child_req = server.calls[1][1]
        system = child_req.messages[0]["content"]
        assert "Context from the delegating agent:\nsrc only" in system
        assert system.endswith("\n\nSuccess criterion:\nreport a number")

    def test_check_block_is_appended_on_success(self, make_state, always_yes, no_esc_watcher):
        script = {"tool_calls": [{"name": "delegate",
                                  "arguments": '{"task": "count the files", "check": "printf \'ok line1\\\\nok line2\\\\n\'"}'}]}
        server = FakeServer(script=[script, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server)
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        result = final.history.turns[0].rounds[0].results[0].content
        assert result.endswith(
            "[check `printf 'ok line1\\nok line2\\n'`: exit 0]\n"
            "ok line1\n"
            "ok line2"
        )

    def test_check_failure_is_a_result_not_an_error(self, make_state, always_yes, no_esc_watcher):
        script = {"tool_calls": [{"name": "delegate", "arguments": '{"task": "count the files", "check": "exit 3"}'}]}
        server = FakeServer(script=[script, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server)
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        result = final.history.turns[0].rounds[0].results[0].content
        assert "raised" not in result
        assert result.endswith("[check `exit 3`: exit 3]")

    def test_check_is_skipped_when_the_child_produced_no_answer(self, make_state, always_yes, no_esc_watcher):
        script = {"tool_calls": [{"name": "delegate", "arguments": '{"task": "count the files", "check": "echo ran"}'}]}
        server = FakeServer(script=[script, {"content": "partial...", "finish_reason": "cancelled"}, {"content": "done"}])
        state = parent_with_delegate(make_state, server)
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        result = final.history.turns[0].rounds[0].results[0].content
        assert "cancelled" in result and "[check" not in result

    def test_check_runs_in_the_specified_root(self, make_state, always_yes, no_esc_watcher, tmp_path):
        """The check command must run with cwd set to the delegate's root, not the harness's own
        directory: a command that only succeeds from the right place proves where it ran."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "marker.txt").write_text("here")
        script = {"tool_calls": [{"name": "delegate",
                                  "arguments": '{"task": "count the files", "check": "pwd && test -f marker.txt && echo found"}'}]}
        server = FakeServer(script=[script, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server, delegate_root=str(root))
        final = Engine[type(state)]().run(state, seed=[UserMessage("go")])
        result = final.history.turns[0].rounds[0].results[0].content
        assert result.endswith(f"[check `pwd && test -f marker.txt && echo found`: exit 0]\n{root}\nfound")

    def test_check_command_is_invisible_to_the_child(self, make_state, always_yes, no_esc_watcher):
        script = {"tool_calls": [{"name": "delegate",
                                  "arguments": '{"task": "count the files", "context": "src only", "gate": "report a number", "check": "echo secret-check"}'}]}
        server = FakeServer(script=[script, {"content": "12"}, {"content": "12"}])
        state = parent_with_delegate(make_state, server)
        Engine[type(state)]().run(state, seed=[UserMessage("go")])
        child_req = server.calls[1][1]
        blob = child_req.messages[0]["content"] + "".join(m["content"] for m in child_req.messages[1:])
        assert "secret-check" not in blob
