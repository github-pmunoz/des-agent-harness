"""
desh.tools: the registry the model is offered (schemas) and the engine runs (callables).

Schema derivation is the contract llama-server compiles into a grammar, so the tests here pin the
wire shape, not just "it returns a dict": property types, required vs optional, docstring
descriptions, the override slots, and the strict rule that an underivable parameter fails at
definition time.
"""
from typing import Literal, Optional

import pytest

from conftest import MAX_CONTEXT, MODELS, PORT
from desh.llama.wire import ToolCall
from desh.tools import Tool, ToolRegistry, json_type, parameters_schema, parse_docstring
from desh_chat.display import DisplayStats, Info
from desh_chat.events import ExecuteToolCalls, NextRound
from desh_chat.state import InferenceEngine, PendingTurn, Round, ToolResult


def get_weather(city: str, units: Literal["C", "F"] = "C", days: int = 1) -> str:
    """Current weather for a city.

    Args:
        city: City name, as the user wrote it.
        units (str): Temperature scale.
        days: Forecast horizon.

    Returns:
        A one-line report.
    """
    return f"{city}: sunny"


# ---------------------
# json_type
# ---------------------

class TestJsonType:
    @pytest.mark.parametrize("annotation, expected", [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list, {"type": "array"}),
        (dict, {"type": "object"}),
        (list[int], {"type": "array", "items": {"type": "integer"}}),
        (dict[str, int], {"type": "object"}),
        (Optional[str], {"type": "string"}),
        (str | None, {"type": "string"}),
        (int | str, {"anyOf": [{"type": "integer"}, {"type": "string"}]}),
        (Literal["C", "F"], {"enum": ["C", "F"]}),
    ])
    def test_maps_python_annotations_to_json_schema(self, annotation, expected):
        assert json_type(annotation) == expected

    def test_unsupported_annotation_raises(self):
        class Thing: ...
        with pytest.raises(TypeError):
            json_type(Thing)


# ---------------------
# parse_docstring (Google style)
# ---------------------

class TestParseDocstring:
    def test_summary_is_the_first_paragraph_and_args_block_gives_param_descriptions(self):
        summary, params = parse_docstring(get_weather.__doc__)
        assert summary == "Current weather for a city."
        assert params == {"city": "City name, as the user wrote it.", "units": "Temperature scale.", "days": "Forecast horizon."}

    def test_multiline_summary_is_joined(self):
        summary, params = parse_docstring("""First line
        continues here.

        Args:
            x: an x.
        """)
        assert summary == "First line continues here."
        assert params == {"x": "an x."}

    def test_returns_section_does_not_leak_into_params(self):
        _, params = parse_docstring("""Summary.

        Args:
            x: an x.
        Returns:
            nothing: really.
        """)
        assert params == {"x": "an x."}

    def test_no_docstring(self):
        assert parse_docstring(None) == ("", {})


# ---------------------
# parameters_schema / Tool.define
# ---------------------

class TestParametersSchema:
    def test_properties_required_and_descriptions_follow_the_signature(self):
        assert parameters_schema(get_weather) == {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, as the user wrote it."},
                "units": {"enum": ["C", "F"], "description": "Temperature scale."},
                "days": {"type": "integer", "description": "Forecast horizon."},
            },
            "required": ["city"],
        }

    def test_no_parameters(self):
        def now() -> str:
            return "10:00"
        assert parameters_schema(now) == {"type": "object", "properties": {}}

    def test_unannotated_parameter_raises_at_definition_time(self):
        def bad(x):
            return x
        with pytest.raises(TypeError, match="'x'"):
            parameters_schema(bad)

    def test_var_positional_and_var_keyword_raise(self):
        def splat(*args: int):
            return args
        def kw(**kwargs: int):
            return kwargs
        with pytest.raises(TypeError, match="keyword"):
            parameters_schema(splat)
        with pytest.raises(TypeError, match="keyword"):
            parameters_schema(kw)


class TestToolDefine:
    def test_schema_is_the_openai_wire_shape(self):
        tool = Tool.define(get_weather)
        assert tool.name == "get_weather"
        assert tool.schema == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a city.",
                "parameters": parameters_schema(get_weather),
            },
        }
        assert tool.fn is get_weather

    def test_override_slots_replace_the_derived_parts_verbatim(self):
        hand = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        tool = Tool.define(get_weather, name="weather", description="Hand-written.", parameters=hand)
        assert tool.name == "weather"
        assert tool.schema["function"] == {"name": "weather", "description": "Hand-written.", "parameters": hand}

    def test_parameters_override_bypasses_derivation_entirely(self):
        def bad(x):
            return x
        tool = Tool.define(bad, parameters={"type": "object", "properties": {}})
        assert tool.parameters == {"type": "object", "properties": {}}


# ---------------------
# ToolRegistry
# ---------------------

class TestToolRegistry:
    def test_register_returns_a_new_registry_and_leaves_the_old_one_alone(self):
        r0 = ToolRegistry()
        r1 = r0.add(get_weather)
        assert len(r0) == 0 and len(r1) == 1
        assert "get_weather" in r1 and "get_weather" not in r0
        assert r1.get("get_weather").fn is get_weather
        assert r1.get("nope") is None

    def test_duplicate_name_is_rejected(self):
        with pytest.raises(ValueError, match="get_weather"):
            ToolRegistry().add(get_weather).add(get_weather)

    def test_schemas_are_the_tools_in_registration_order(self):
        def now() -> str:
            return "10:00"
        r = ToolRegistry().add(get_weather).add(now)
        assert [s["function"]["name"] for s in r.schemas()] == ["get_weather", "now"]
        assert ToolRegistry().schemas() == []


# ---------------------
# ToolRegistry.invoke: the boundary where tool-side failures become text
# ---------------------

def divide(a: float, b: float) -> float:
    """a / b."""
    return a / b


def now() -> str:
    """Current time."""
    return "10:00"


def inventory() -> dict:
    """Stock levels."""
    return {"apples": 3, "pears": 0}


def raises_type_error() -> str:
    """A tool whose OWN body raises TypeError — must read as raised, not as rejected arguments."""
    raise TypeError("inside the tool")


def unserializable() -> object:
    """Returns something json.dumps cannot handle."""
    return object()


# All read-only: these are doubles for invoke() and the request wiring, so nothing here should stop
# at the confirmation gate. The gate has its own tests in test_confirm.py.
REGISTRY = ToolRegistry()
for _fn in (get_weather, divide, now, inventory, raises_type_error, unserializable):
    REGISTRY = REGISTRY.add(_fn, confirm=False)


class TestInvoke:
    def test_success_returns_the_string_result(self):
        assert REGISTRY.invoke("get_weather", '{"city": "Santiago"}') == "Santiago: sunny"

    def test_non_string_results_are_json(self):
        assert REGISTRY.invoke("divide", '{"a": 1, "b": 4}') == "0.25"
        assert REGISTRY.invoke("inventory", "{}") == '{"apples": 3, "pears": 0}'

    def test_blank_arguments_mean_no_arguments(self):
        assert REGISTRY.invoke("now", "") == "10:00"
        assert REGISTRY.invoke("now", "  ") == "10:00"

    def test_unknown_tool_is_not_available(self):
        content = ToolRegistry().invoke("get_weather", "{}")
        assert "get_weather" in content and "not available" in content

    def test_malformed_json_is_reported_not_raised(self):
        content = REGISTRY.invoke("now", "{not json")
        assert "JSON" in content

    @pytest.mark.parametrize("arguments, word", [("[1, 2]", "array"), ('"x"', "string"), ("3", "integer"), ("null", "null")])
    def test_non_object_arguments_name_the_json_type(self, arguments, word):
        content = REGISTRY.invoke("now", arguments)
        assert "JSON object" in content and word in content

    def test_missing_extra_or_unknown_arguments_are_rejected_before_the_call(self):
        assert "rejected" in REGISTRY.invoke("divide", '{"a": 1}')
        assert "rejected" in REGISTRY.invoke("divide", '{"a": 1, "b": 2, "c": 3}')
        assert "rejected" in REGISTRY.invoke("now", '{"tz": "CLT"}')

    def test_exception_inside_the_tool_is_reported_with_its_type(self):
        content = REGISTRY.invoke("divide", '{"a": 1, "b": 0}')
        assert "raised" in content and "ZeroDivisionError" in content

    def test_type_error_inside_the_tool_reads_as_raised_not_rejected(self):
        content = REGISTRY.invoke("raises_type_error", "{}")
        assert "raised" in content and "inside the tool" in content and "rejected" not in content

    def test_unserializable_result_is_reported(self):
        assert "unserializable" in REGISTRY.invoke("unserializable", "{}")

    def test_nothing_tool_side_ever_raises(self):
        for name, args in [("nope", "{}"), ("now", "{"), ("now", "[]"), ("divide", "{}"), ("divide", '{"a":1,"b":0}'), ("unserializable", "{}")]:
            assert isinstance(REGISTRY.invoke(name, args), str)


# ---------------------
# Tool.inject: parameters the harness supplies, which the model never sees
# ---------------------

def greet(name: str, *, settings: dict) -> str:
    """Greet someone in the configured tone.

    Args:
        name: Who to greet.
        settings: Supplied by the harness.
    """
    return f"{name}:{settings['tone']}"


class TestInject:
    INJECTING = ToolRegistry().add(greet, confirm=False, inject=("settings",)).add(now, confirm=False)

    def test_injected_parameters_are_left_out_of_the_schema(self):
        params = self.INJECTING.get("greet").parameters
        assert set(params["properties"]) == {"name"} and params["required"] == ["name"]
        assert Tool.define(greet, inject=("settings",)).inject == ("settings",)

    def test_declared_names_are_passed_through_and_the_rest_dropped(self):
        assert self.INJECTING.invoke("greet", '{"name": "ana"}', settings={"tone": "warm"}, other=1) == "ana:warm"

    def test_a_tool_that_declared_nothing_gets_nothing(self):
        assert self.INJECTING.invoke("now", "", settings={"tone": "warm"}) == "10:00"

    def test_the_model_cannot_pass_an_injected_name(self):
        content = self.INJECTING.invoke("greet", '{"name": "ana", "settings": {"tone": "rude"}}', settings={"tone": "warm"})
        assert "rejected" in content and "settings" in content

    def test_a_missing_injection_is_rejected_not_raised(self):
        assert "rejected" in self.INJECTING.invoke("greet", '{"name": "ana"}')

    def test_an_undeclared_parameter_without_a_hint_still_fails_at_definition(self):
        def bad(name, *, settings: dict) -> str:
            return name
        with pytest.raises(TypeError):
            Tool.define(bad, inject=("settings",))


# ---------------------
# Wiring: NextRound offers the registry's schemas
# ---------------------

class TestNextRoundOffersTools:
    def test_request_tools_are_the_registry_schemas(self, make_state):
        state = make_state(pending=PendingTurn("q"), tools=ToolRegistry().add(get_weather))
        _, events = NextRound().execute(state)
        req = events[0].request
        assert req.tools == [Tool.define(get_weather).schema]
        assert "tools" in req.payload()

    def test_empty_registry_puts_no_tools_key_on_the_wire(self, make_state):
        state = make_state(pending=PendingTurn("q"))
        _, events = NextRound().execute(state)
        req = events[0].request
        assert req.tools == []
        assert "tools" not in req.payload()


class TestExecuteToolCallsWithRegistry:
    def test_results_come_from_the_registry_and_keep_the_call_ids(self, make_state):
        weather = ToolCall(index=0, id="call_a", type="function", name="get_weather", arguments='{"city": "Santiago"}')
        broken = ToolCall(index=1, id="call_b", type="function", name="divide", arguments='{"a": 1, "b": 0}')
        state = make_state(pending=PendingTurn("q").add_round(Round("", (weather, broken))), tools=REGISTRY)
        mid, events = ExecuteToolCalls().execute(state)          # one call per step
        assert [type(e) for e in events] == [Info, DisplayStats, ExecuteToolCalls]
        new_state, events = events[2].execute(mid)
        results = new_state.pending.rounds[-1].results
        assert results[0] == ToolResult("call_a", "get_weather", "Santiago: sunny")
        assert results[1].tool_call_id == "call_b" and "ZeroDivisionError" in results[1].content
        assert [type(e) for e in events] == [Info, DisplayStats, NextRound]

    def test_full_turn_with_a_real_tool(self, make_state, no_esc_watcher):
        from conftest import FakeServer
        from desh.engine import Engine
        from desh_chat.events import UserMessage
        server = FakeServer(script=[
            {"tool_calls": [{"name": "get_weather", "arguments": '{"city": "Santiago"}'}]},
            {"content": "Sunny in Santiago."},
        ])
        state = make_state(inference=InferenceEngine(models=MODELS, max_context=MAX_CONTEXT, server=server, port=PORT),
                           tools=REGISTRY, running=False)
        final = Engine[type(state)]().run(state, seed=[UserMessage("weather?")])
        turn = final.history.turns[0]
        assert turn.rounds[0].results[0].content == "Santiago: sunny"
        assert turn.assistant == "Sunny in Santiago."
        # the second request echoed the round AND still offered the tools
        _, second = server.calls[1]
        assert [m["role"] for m in second.messages][-2:] == ["assistant", "tool"]
        assert [s["function"]["name"] for s in second.tools] == [t.name for t in REGISTRY.tools]
