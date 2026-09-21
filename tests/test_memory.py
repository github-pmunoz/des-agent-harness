"""
The memory plugins and the harness frame around them (desh_chat.memory): a run registers any
number of memories, each owning its value, its tools and the prompt that says what it is for;
the harness owns the mechanics prompt, the one block that ends every request, the lines that
announce what is about to expire, and each memory's budget.
"""
import argparse
import json

import pytest

from conftest import MODELS
from desh.llama.tokens import estimate_tokens
from desh.llama.wire import ToolCall
from desh.tools import ToolRegistry
from desh_chat.cli import MEMORIES, build_tools, selected_memories
from desh_chat.events import ExecuteToolCalls, TurnEnd, run_memory_calls
from desh_chat.memory import CONTEXT_MECHANICS_PROMPT, FOLD_NEAR_LINE, Memories
from desh_chat.ontology import MAX_ENTITY_CHARS, ONTOLOGY, Ontology
from desh_chat.plan import PLAN, Plan
from desh_chat.scratchpad import SCRATCHPAD, Scratchpad
from desh_chat.session import LoadSession
from desh_chat.state import ChatHistory, PendingTurn, Round, Settings, StopReason, Turn


def call(name: str, index: int = 0, **arguments) -> ToolCall:
    return ToolCall(index=index, id=f"c{index}", type="function", name=name, arguments=json.dumps(arguments))


def registry(*memories) -> ToolRegistry:
    tools = ToolRegistry()
    for m in memories:
        tools = m.register(tools)
    return tools


BOTH = Memories.of(PLAN, ONTOLOGY)
BOTH_TOOLS = registry(PLAN, ONTOLOGY)


# ---------------------
# Registration
# ---------------------

class TestPlugin:
    def test_a_memorys_tools_are_unconfirmed_and_inject_its_slot(self):
        for m in MEMORIES.values():
            tools = m.register(ToolRegistry())
            assert [t.name for t in tools.tools] == list(m.tool_names())
            assert all(t.inject == (m.name,) and not t.confirm for t in tools.tools)
            assert all(m.name not in t.parameters["properties"] for t in tools.tools)     # the model cannot pass the slot

    def test_every_catalogue_value_round_trips_and_renders_empty_as_nothing(self):
        for m in MEMORIES.values():
            assert m.empty().render() == ""
            assert m.from_dict(m.empty().to_dict()) == m.empty()

    def test_a_plugin_prompt_never_explains_the_context(self):
        for m in MEMORIES.values():
            assert "round" not in m.prompt.lower() and "checkpoint" not in m.prompt.lower()

    def test_the_flags_select_memories_in_order_each_once(self):
        ns = argparse.Namespace(memory="plan, ontology,plan", scratchpad=True)
        assert selected_memories(ns) == (SCRATCHPAD, PLAN, ONTOLOGY)
        assert selected_memories(argparse.Namespace(memory="", scratchpad=False)) == ()
        with pytest.raises(SystemExit):
            selected_memories(argparse.Namespace(memory="nope", scratchpad=False))


# ---------------------
# The system prompt
# ---------------------

class TestSystemPrompt:
    def test_no_memory_means_the_base_prompt_alone(self):
        assert Memories().system_prompt("base", 10) == "base"

    def test_mechanics_come_once_then_each_memorys_own_prompt(self):
        text = BOTH.system_prompt("base", 7)
        assert text.startswith("base\n\nHow your context works.") and text.count("How your context works") == 1
        assert "at most 7 rounds" in text and "<plan>, <ontology>" in text
        assert "plan_write, plan_delete, ontology_write, ontology_delete, ontology_clear" in text
        assert text.index("How your context works") < text.index(PLAN.prompt) < text.index(ONTOLOGY.prompt)

    def test_the_mechanics_name_no_memory_themselves(self):
        assert not any(m in CONTEXT_MECHANICS_PROMPT for m in MEMORIES)


# ---------------------
# The block
# ---------------------

class TestBlock:
    def test_no_memory_means_no_block(self, make_state):
        assert Memories().block() is None and make_state(pending=PendingTurn("q")).memory_block() is None

    def test_one_user_message_holds_a_section_per_memory_in_order(self):
        memory = BOTH.with_value("plan", Plan().with_step("s1", "todo", "read it")).with_value(
            "ontology", Ontology().with_triplet("cli.py", "defines", "--memory"))
        block = memory.block()
        assert block["role"] == "user"
        text = block["content"]
        assert text.startswith("<memory>") and text.endswith("</memory>")
        assert "<plan>\n[ ] s1: read it\n</plan>\n<ontology>\ncli.py -[defines]-> --memory\n</ontology>" in text

    def test_each_section_shows_what_it_uses_of_its_budget(self, make_state):
        pad = Scratchpad().with_entry("k", "fact", "v" * 400)
        state = make_state(pending=PendingTurn("q"), scratchpad=pad)
        budget = int(state.settings.memory_target * state.settings.context)
        assert f'<scratchpad used="{estimate_tokens(pad.render())}/{budget} tokens">' in state.memory_block()["content"]


# ---------------------
# The checkpoint alert
# ---------------------

class TestFoldNear:
    SETTINGS = Settings(model=MODELS[0], temperature=0.3, think=False, context=4096, max_turn_tokens=2048, max_tool_rounds=30)

    def pending(self, rounds: int, tokens: int) -> PendingTurn:
        p = PendingTurn("q")
        for n in range(rounds):
            p = p.add_round(Round("", (call("Read", n),), tokens=tokens)).with_results(())
        return p

    def test_said_when_a_typical_round_more_would_leave_too_little_room(self, make_state):
        state = make_state(pending=self.pending(4, 540), settings=self.SETTINGS, scratchpad=Scratchpad())
        assert state.gen_room(state.pending_tokens()) >= state.min_gen_tokens()      # this request still fits
        assert state.fold_near() and FOLD_NEAR_LINE in state.memory_block()["content"]
        assert state.memory_tokens() == estimate_tokens(state.memory_block()["content"])     # priced as sent

    def test_said_once_on_the_edge_not_while_the_condition_holds(self, make_state):
        """A reply that only persists barely grows the prompt: the run is still one round from a
        fold, and saying so again is answered with the same write again."""
        p = self.pending(4, 540).add_round(Round("", (call("scratchpad_write", 9),), tokens=20)).with_results(())
        state = make_state(pending=p, settings=self.SETTINGS, scratchpad=Scratchpad())
        assert state._fold_within_a_round() and not state.fold_near()
        assert FOLD_NEAR_LINE not in state.memory_block()["content"]

    def test_not_said_with_room_to_spare(self, make_state):
        state = make_state(pending=self.pending(4, 50), settings=self.SETTINGS, scratchpad=Scratchpad())
        assert not state.fold_near() and FOLD_NEAR_LINE not in state.memory_block()["content"]

    def test_not_said_when_there_is_nothing_to_fold(self, make_state):
        state = make_state(pending=self.pending(1, 2000), settings=self.SETTINGS, scratchpad=Scratchpad())
        assert not state.fold_near()

    def test_not_said_while_the_history_goes_first(self, make_state):
        history = ChatHistory().append(Turn("q0", "a0", stop=StopReason.ANSWER))
        state = make_state(pending=self.pending(4, 500), settings=self.SETTINGS, scratchpad=Scratchpad(), history=history)
        assert not state.fold_near()


# ---------------------
# A call: commit, budget, the cap
# ---------------------

class TestCommit:
    def run(self, make_state, tc: ToolCall, memory: Memories = BOTH, **overrides):
        state = make_state(pending=PendingTurn("q").add_round(Round("", (tc,))), tools=BOTH_TOOLS, memory=memory, **overrides)
        new_state, _ = ExecuteToolCalls().execute(state)
        return new_state, new_state.pending.rounds[-1].results[0].content

    def test_a_call_commits_its_own_slot_and_leaves_the_others(self, make_state):
        new_state, content = self.run(make_state, call("plan_write", key="s1", status="todo", text="read it"))
        assert content == "created 's1' (todo)"
        assert new_state.memory.get("plan") == Plan().with_step("s1", "todo", "read it")
        assert new_state.memory.get("ontology") is BOTH.get("ontology")

    def test_a_write_past_the_budget_is_refused_and_nothing_is_committed(self, make_state):
        tight = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, memory_target=0.001)
        new_state, content = self.run(make_state, call("plan_write", key="s1", status="todo", text="x" * 400), settings=tight)
        assert content.startswith("Not recorded: <plan> would take") and "budget of 16" in content
        assert new_state.memory == BOTH

    def test_shrinking_an_over_budget_memory_is_allowed(self, make_state):
        tight = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, memory_target=0.001)
        big = BOTH.with_value("plan", Plan().with_step("a", "todo", "x" * 400).with_step("b", "todo", "y" * 400))
        new_state, content = self.run(make_state, call("plan_delete", key="a"), memory=big, settings=tight)
        assert content == "'a' deleted" and [s.key for s in new_state.memory.get("plan").steps] == ["b"]

    def test_at_the_cap_the_calls_of_every_registered_memory_run(self, make_state):
        calls = (call("plan_write", 0, key="s1", status="done", text="ok"),
                 call("Read", 1, file_path="a.py"),
                 call("ontology_write", 2, subject="a.py", predicate="imports", object="b.py"))
        state = make_state(pending=PendingTurn("q"), tools=BOTH_TOOLS, memory=BOTH)
        new_state, kept, rest = run_memory_calls(state, calls)
        assert kept == ["plan_write s1", "ontology_write a.py"] and [tc.name for tc in rest] == ["Read"]
        assert new_state.memory.get("ontology") == Ontology().with_triplet("a.py", "imports", "b.py")

    def test_a_memory_tool_whose_slot_is_not_registered_is_left_unrun(self, make_state):
        state = make_state(pending=PendingTurn("q"), tools=BOTH_TOOLS, memory=Memories.of(PLAN))
        _, kept, rest = run_memory_calls(state, (call("ontology_clear"),))
        assert kept == [] and [tc.name for tc in rest] == ["ontology_clear"]


# ---------------------
# The record: snapshot, restore, salvage
# ---------------------

class TestRecord:
    FULL = BOTH.with_value("plan", Plan().with_step("s1", "done", "ok")).with_value(
        "ontology", Ontology().with_triplet("a.py", "imports", "b.py"))

    def test_the_turn_records_every_slot_and_a_session_restores_the_registered_ones(self, make_state, tmp_path):
        path = str(tmp_path / "s.json")
        state = make_state(session_file=path, pending=PendingTurn("q"), memory=self.FULL)
        state, events = TurnEnd(assistant="a", stop=StopReason.ANSWER).execute(state)
        assert state.history.turns[-1].memory == self.FULL.snapshot()
        for e in events:
            if type(e).__name__ == "SaveSession":
                e.execute(state)
        restored, _ = LoadSession().execute(make_state(session_file=path, memory=Memories.of(ONTOLOGY)))
        assert restored.memory.names() == ("ontology",)      # the plan is in the file, not in this run
        assert restored.memory.get("ontology") == self.FULL.get("ontology")

    def test_a_salvaged_turn_carries_a_section_per_memory_that_holds_anything(self, make_state):
        memory = self.FULL.with_value("plan", Plan())
        text = PendingTurn("q").add_round(Round("", (call("Read"),))).salvage(StopReason.OVERFLOW, memory)
        assert "ONTOLOGY:\na.py -[imports]-> b.py" in text and "PLAN:" not in text


# ---------------------
# The plugins
# ---------------------

class TestPlan:
    def test_a_rewritten_step_keeps_its_place(self):
        d = Plan().with_step("a", "todo", "1").with_step("b", "todo", "2").to_dict()
        assert BOTH_TOOLS.invoke("plan_write", '{"key": "a", "status": "done", "text": "ok"}', plan=d) == "overwrote 'a' (todo -> done)"
        assert Plan.from_dict(d).render() == "[x] a: ok\n[ ] b: 2"

    def test_an_unknown_status_is_refused(self):
        d: dict = {}
        assert "unknown status" in BOTH_TOOLS.invoke("plan_write", '{"key": "a", "status": "fact", "text": "1"}', plan=d) and d == {}


class TestOntology:
    def write(self, d: dict, s: str, p: str, o: str) -> str:
        return BOTH_TOOLS.invoke("ontology_write", json.dumps({"subject": s, "predicate": p, "object": o}), ontology=d)

    def test_a_triplet_is_recorded_once_in_first_written_order(self):
        d: dict = {}
        assert self.write(d, "a", "imports", "b") == "recorded: a -[imports]-> b"
        assert self.write(d, "b", "imports", "c").startswith("recorded")
        assert self.write(d, "a", "imports", "b").startswith("already recorded")
        assert Ontology.from_dict(d).render() == "a -[imports]-> b\nb -[imports]-> c"

    def test_prose_in_place_of_an_entity_is_refused(self):
        d: dict = {}
        answer = self.write(d, "1acb78c", "changes", "x" * (MAX_ENTITY_CHARS + 1))
        assert answer.startswith("not recorded: the object is") and d == {}

    def test_one_triplet_can_be_deleted(self):
        d = Ontology().with_triplet("a", "imports", "b").with_triplet("a", "imports", "c").to_dict()
        args = json.dumps({"subject": "a", "predicate": "imports", "object": "b"})
        assert BOTH_TOOLS.invoke("ontology_delete", args, ontology=d) == "deleted: a -[imports]-> b"
        assert BOTH_TOOLS.invoke("ontology_delete", args, ontology=d).startswith("not found")
        assert Ontology.from_dict(d) == Ontology().with_triplet("a", "imports", "c")

    def test_parts_that_hold_a_separator_do_not_collide(self):
        assert Ontology().with_triplet("a\tb", "c", "d").to_dict().keys() != Ontology().with_triplet("a", "b\tc", "d").to_dict().keys()


# ---------------------
# Subagents
# ---------------------

class TestSubagentPolicy:
    def test_a_subagent_gets_the_selected_memories_whose_policy_is_fresh(self, make_state, tmp_path):
        from conftest import MAX_CONTEXT, PORT, FakeServer
        from desh_chat.state import InferenceEngine
        args = argparse.Namespace(workspace=str(tmp_path), read=True, write=False, edit=False, bash=False, current_time=False,
                                  delegate=True, debug=False, scratchpad=False, memory="plan,ontology", tool_cap=10.0)
        inference = InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=FakeServer(script=[]), port=PORT)
        tools = build_tools(args, inference, make_state().settings)
        delegate = tools.get("delegate").fn.__self__
        assert delegate.memories == (PLAN,)
        assert "plan_write" in delegate.tools and "ontology_write" not in delegate.tools
        assert "plan_write" in tools and "ontology_write" in tools
