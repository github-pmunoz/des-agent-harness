"""
--config and the prompt overrides (desh_chat.cli, desh_chat.prompts): one JSON file holds the
whole command line, and every instruction the model reads can be replaced from it, so an eval arm
is a settings file and not a change to the harness. An override is applied once, at construction,
to the value that owns the text; with no overrides every text is the default, byte for byte.
"""
import json

import pytest

from conftest import MAX_CONTEXT, MODELS, PORT, FakeServer
from desh.llama.wire import ToolCall
from desh_chat.cli import build_tools, effective_config, parse_args, settings_of
from desh_chat.delegate import CAP_CONTINUE_MSG, DELEGATE_SYSTEM_PROMPT, Delegate
from desh_chat.events import CompactPendingTurn
from desh_chat.memory import CONTEXT_MECHANICS_PROMPT, LAST_ROUND_LINE, Memories, MemoryFrame
from desh_chat.prompts import STATIC, PromptError, Prompts
from desh_chat.scratchpad import SCRATCHPAD, SCRATCHPAD_PROMPT
from desh_chat.state import InferenceEngine, PendingTurn, Round, Settings


def write_config(tmp_path, doc: dict, name: str = "c.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def tools_of(argv: list[str]):
    args, prompts = parse_args(argv)
    tools = build_tools(args, None, settings_of(args, prompts), prompts=prompts)
    return args, prompts, tools


# ---------------------
# --config
# ---------------------

class TestConfig:
    def test_a_flag_wins_over_the_file_and_the_file_over_the_default(self, tmp_path):
        path = write_config(tmp_path, {"context": 16384, "max_tool_rounds": 30, "read": True, "memory": "plan"})
        args, _ = parse_args(["--config", path, "-c", "8192"])
        assert (args.context, args.max_tool_rounds, args.read, args.memory) == (8192, 30, True, "plan")
        assert args.temperature == 0.3 and args.write is False       # untouched: the built-in defaults

    def test_an_unknown_key_is_an_error_not_a_run_on_the_default(self, tmp_path):
        with pytest.raises(SystemExit, match="unknown key 'tool-cap'"):
            parse_args(["--config", write_config(tmp_path, {"tool-cap": 2.5})])

    def test_the_flags_about_the_config_itself_are_not_config_keys(self, tmp_path):
        with pytest.raises(SystemExit, match="unknown key 'config'"):
            parse_args(["--config", write_config(tmp_path, {"config": "other.json"})])

    def test_values_are_typed_as_the_flags_are(self, tmp_path):
        args, _ = parse_args(["--config", write_config(tmp_path, {"tool_cap": 5, "task_timeout": 1800})])
        assert args.tool_cap == 5.0 and isinstance(args.tool_cap, float) and args.task_timeout == 1800.0
        with pytest.raises(SystemExit, match="'read' is a switch"):
            parse_args(["--config", write_config(tmp_path, {"read": "yes"})])
        with pytest.raises(SystemExit, match="'context'"):
            parse_args(["--config", write_config(tmp_path, {"context": "big"})])
        with pytest.raises(SystemExit, match="'model' takes a string"):
            parse_args(["--config", write_config(tmp_path, {"model": 7})])

    def test_an_at_value_is_the_text_of_a_file_beside_the_config(self, tmp_path):
        (tmp_path / "role.txt").write_text("You orchestrate.\n", encoding="utf-8")
        (tmp_path / "sub.txt").write_text("You are a subagent.\n", encoding="utf-8")
        path = write_config(tmp_path, {"system_prompt": "@role.txt", "prompts": {"delegate.system": "@sub.txt"}})
        args, prompts = parse_args(["--config", path])
        assert args.system_prompt == "You orchestrate." and prompts.get("delegate.system") == "You are a subagent."

    def test_a_missing_or_malformed_file_is_reported(self, tmp_path):
        with pytest.raises(SystemExit, match="cannot read config"):
            parse_args(["--config", str(tmp_path / "none.json")])
        (tmp_path / "bad.json").write_text("{", encoding="utf-8")
        with pytest.raises(SystemExit, match="cannot read config"):
            parse_args(["--config", str(tmp_path / "bad.json")])


# ---------------------
# Prompt keys
# ---------------------

class TestPromptKeys:
    def test_the_command_line_wins_over_the_configs_prompts(self, tmp_path):
        path = write_config(tmp_path, {"prompts": {"cap_continue": "from the file", "compaction.close": "\n\nNow."}})
        _, prompts = parse_args(["--config", path, "--prompt", "cap_continue=from the flag"])
        assert prompts.get("cap_continue") == "from the flag" and prompts.get("compaction.close") == "\n\nNow."

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(PromptError, match="unknown prompt key 'delegate.sytem'"):
            Prompts.resolve(({"delegate.sytem": "x"}, "."))
        with pytest.raises(SystemExit):
            parse_args(["--prompt", "nope=x"])
        with pytest.raises(SystemExit):
            parse_args(["--prompt", "no-equals-sign"])

    def test_a_template_must_carry_exactly_its_placeholders(self):
        with pytest.raises(PromptError, match="max_tool_rounds"):
            Prompts.resolve(({"memory.mechanics": "Rounds vanish. {tool_names} {sections}"}, "."))
        with pytest.raises(PromptError, match="found"):
            Prompts.resolve(({"memory.cap_reached": "Only {tool_names} run, {whatever}."}, "."))
        ok = Prompts.resolve(({"memory.cap_reached": "Only {tool_names} run."}, "."))
        assert ok.frame().cap_reached == "Only {tool_names} run."

    def test_a_text_that_is_not_a_template_may_hold_braces(self):
        assert Prompts.resolve(({"delegate.system": 'Reply as {"ok": true}.'}, ".")).get("delegate.system") == 'Reply as {"ok": true}.'

    def test_no_overrides_means_every_default_byte_for_byte(self):
        p = Prompts()
        assert p.frame() == MemoryFrame() and p.memories((SCRATCHPAD,)) == (SCRATCHPAD,)
        assert all(p.get(k) == default for k, (default, _) in STATIC.items())
        base = Settings(model="m", temperature=0.3, think=False, context=1, max_turn_tokens=1)
        assert Settings(model="m", temperature=0.3, think=False, context=1, max_turn_tokens=1, **p.settings()) == base


# ---------------------
# Where each override lands
# ---------------------

class TestLanding:
    def test_memory_frame_and_plugin_prompt_reach_the_system_prompt_and_the_block(self):
        prompts = Prompts.resolve(({"memory.mechanics": "MECH {max_tool_rounds} {tool_names} {sections}",
                                    "memory.last_round": "LAST CALL", "memory.scratchpad.prompt": "PAD PROMPT"}, "."))
        memory = Memories.of(*prompts.memories((SCRATCHPAD,)), frame=prompts.frame())
        text = memory.system_prompt("base", 7)
        assert "MECH 7 scratchpad_write, scratchpad_delete, scratchpad_clear <scratchpad>" in text
        assert "PAD PROMPT" in text and SCRATCHPAD_PROMPT not in text and CONTEXT_MECHANICS_PROMPT[:30] not in text
        block = memory.block(round=(5, 5))["content"]
        assert "LAST CALL" in block and LAST_ROUND_LINE not in block

    def test_a_prompt_for_a_memory_the_run_did_not_select_is_an_error(self):
        with pytest.raises(PromptError, match="'plan' is not selected"):
            Prompts.resolve(({"memory.plan.prompt": "x"}, ".")).memories((SCRATCHPAD,))

    def test_tool_texts_reach_the_schema_in_both_registries(self, tmp_path):
        _, _, tools = tools_of(["-w", str(tmp_path), "--read", "--delegate", "--memory", "scratchpad",
                                "--prompt", "tool.Read.description=READ IT", "--prompt", "tool.delegate.param.task=THE TASK",
                                "--prompt", "tool.Bash.description=SUBAGENT SHELL"])
        child = tools.get("delegate").fn.__self__.tools
        assert tools.get("Read").description == "READ IT" and child.get("Read").description == "READ IT"
        assert tools.get("delegate").parameters["properties"]["task"]["description"] == "THE TASK"
        assert child.get("Bash").description == "SUBAGENT SHELL" and "Bash" not in tools     # only the subagent has it
        # the schema's shape is untouched: same parameters, same required ones
        _, _, plain = tools_of(["-w", str(tmp_path), "--read", "--delegate", "--memory", "scratchpad"])
        assert tools.get("delegate").parameters["required"] == plain.get("delegate").parameters["required"]
        assert set(tools.get("Read").parameters["properties"]) == set(plain.get("Read").parameters["properties"])

    def test_a_text_for_a_tool_or_a_parameter_the_run_does_not_have_is_an_error(self, tmp_path):
        with pytest.raises(PromptError, match="'Write' is not offered"):
            tools_of(["-w", str(tmp_path), "--read", "--prompt", "tool.Write.description=x"])
        with pytest.raises(PromptError, match="no parameter 'path'"):
            tools_of(["-w", str(tmp_path), "--read", "--prompt", "tool.Read.param.path=x"])

    def test_the_delegate_gets_its_system_prompt_cap_message_and_memory_frame(self, tmp_path):
        _, _, tools = tools_of(["-w", str(tmp_path), "--delegate", "--memory", "scratchpad", "--prompt", "delegate.system=SUB ROLE",
                                "--prompt", "cap_continue=GO ON", "--prompt", "memory.fold_near=FOLDING"])
        delegate: Delegate = tools.get("delegate").fn.__self__
        assert (delegate.system_prompt, delegate.cap_continue) == ("SUB ROLE", "GO ON")
        assert delegate.memory.frame.fold_near == "FOLDING" and delegate.memory.names() == ("scratchpad",)
        _, _, plain = tools_of(["-w", str(tmp_path), "--delegate"])
        default: Delegate = plain.get("delegate").fn.__self__
        assert (default.system_prompt, default.cap_continue) == (DELEGATE_SYSTEM_PROMPT, CAP_CONTINUE_MSG)

    def test_the_compaction_texts_reach_the_request(self, make_state):
        prompts = Prompts.resolve(({"checkpoint": "FOLD THESE", "checkpoint.close": "\n\nGO."}, "."))
        settings = Settings(model=MODELS[0], temperature=0.3, think=False, context=16384, max_turn_tokens=8192, **prompts.settings())
        server = FakeServer(script=[{"content": "a checkpoint"}])
        tc = ToolCall(index=0, id="c0", type="function", name="Read", arguments="{}")
        pending = PendingTurn("q").add_round(Round("", (tc,))).with_results(()).add_round(Round("", (tc,))).with_results(())
        state = make_state(settings=settings, pending=pending,
                           inference=InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT))
        CompactPendingTurn().execute(state)
        request = server.calls[0][1]
        assert request.messages[0]["content"] == "FOLD THESE" and request.messages[-1]["content"].endswith("\n\nGO.")


# ---------------------
# --print-config
# ---------------------

class TestPrintConfig:
    ARGV = ["--read", "--delegate", "--memory", "scratchpad,plan", "-mtr", "30", "--prompt", "delegate.system=SUB ROLE"]

    def test_it_lists_every_option_and_every_prompt_of_the_run(self, tmp_path):
        args, prompts = parse_args(["-w", str(tmp_path)] + self.ARGV)
        doc = effective_config(args, prompts)
        assert doc["max_tool_rounds"] == 30 and doc["memory"] == "scratchpad,plan" and "config" not in doc and "prompt" not in doc
        p = doc["prompts"]
        assert p["delegate.system"] == "SUB ROLE" and p["cap_continue"] == CAP_CONTINUE_MSG
        assert p["memory.plan.prompt"] and "memory.ontology.prompt" not in p
        assert "tool.Read.description" in p and "tool.delegate.param.check" in p and "tool.Bash.param.command" in p

    def test_its_output_is_a_config_that_reproduces_itself(self, tmp_path):
        args, prompts = parse_args(["-w", str(tmp_path)] + self.ARGV)
        doc = effective_config(args, prompts)
        again_args, again_prompts = parse_args(["--config", write_config(tmp_path, doc)])
        assert effective_config(again_args, again_prompts) == doc
