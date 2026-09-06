"""
Tool registry: what the model is offered (JSON schemas on Request.tools) and what runs when it
asks (Python callables). Lives in the engine package, not desh_chat: the transport needs the
schemas, the engine needs the callables, neither is chat-specific.

    Tool          one callable + the OpenAI-shaped schema the model sees for it + its confirm policy
    Tool.define   derives the schema from signature + type hints + Google-style docstring; each
                  part has a hand-written override slot (name / description / parameters)
    ToolRegistry  frozen tuple of Tools; schemas() for the request, invoke() for the engine

Policy is data on the Tool, not logic in the harness: `confirm` says whether the operator is asked
before the call runs. It defaults to True — a tool that was never thought about asks; only a tool
declared read-only (confirm=False) runs unprompted. The harness asks, the OS enforces: sandboxing
and permissions stay outside the app.

Why derive: llama-server builds its tool-call grammar from `parameters`, so a schema that drifts
from the signature constrains the model to arguments the function cannot accept. Derivation makes
the signature the single source of truth. The rule is strict — derive from a hint, or override by
hand, never guess: an unannotated or unsupported parameter raises at definition time, not at the
first call the model makes.
"""
from __future__ import annotations

import inspect
import json
import re
import types
import typing
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
_JSON_NAMES = {**_JSON_TYPES, type(None): "null"}    # for messages back to the model: JSON words, not Python ones


def json_type(annotation: Any) -> dict:
    """JSON Schema fragment for one Python annotation. Raises TypeError when it cannot be derived."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return json_type(members[0]) if len(members) == 1 else {"anyOf": [json_type(a) for a in members]}
    if origin is typing.Literal:
        return {"enum": list(typing.get_args(annotation))}
    if origin in (list, tuple, set, frozenset):
        items = typing.get_args(annotation)
        return {"type": "array", "items": json_type(items[0])} if items else {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    if annotation is inspect.Parameter.empty:
        raise TypeError("parameter has no type hint")
    raise TypeError(f"no JSON schema derivation for annotation {annotation!r}")


_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments)\s*:\s*$")
_SECTION_HEADER = re.compile(r"^\s*\w+\s*:\s*$")                          # Returns: / Raises: / ... ends the block
_ARG_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+?)\s*$")     # name (type): text


def parse_docstring(doc: Optional[str]) -> tuple[str, dict[str, str]]:
    """(summary, {param: description}) from a Google-style docstring. Summary is the first paragraph;
    parameter descriptions come from the 'Args:' block. Both are optional."""
    if not doc:
        return "", {}
    text = inspect.cleandoc(doc)
    summary = text.split("\n\n", 1)[0].replace("\n", " ").strip()
    params: dict[str, str] = {}
    in_args = False
    for line in text.splitlines():
        if _ARGS_HEADER.match(line):
            in_args = True
        elif in_args and (not line.strip() or _SECTION_HEADER.match(line)):
            in_args = False
        elif in_args and (m := _ARG_LINE.match(line)):
            params[m.group(1)] = m.group(2)
    return summary, params


def parameters_schema(fn: Callable[..., Any]) -> dict:
    """The `parameters` object for fn: one property per parameter, required when it has no default.
    Only parameters that can be passed by keyword are expressible (the engine calls fn(**arguments))."""
    _, param_docs = parse_docstring(fn.__doc__)
    hints = typing.get_type_hints(fn)
    properties: dict[str, dict] = {}
    required: list[str] = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            raise TypeError(f"{fn.__name__}: parameter {p.name!r} cannot be passed by keyword; tools take keyword arguments only")
        try:
            prop = json_type(hints.get(p.name, p.annotation))
        except TypeError as e:
            raise TypeError(f"{fn.__name__}: parameter {p.name!r}: {e}. Add a type hint or pass parameters= by hand.") from e
        if p.name in param_docs:
            prop = {**prop, "description": param_docs[p.name]}
        properties[p.name] = prop
        if p.default is p.empty:
            required.append(p.name)
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Tool / ToolRegistry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """One callable and the schema the model sees for it. `schema` is the OpenAI wire shape:
    {type: "function", function: {name, description, parameters}} — exactly what goes on Request.tools."""
    name: str
    fn: Callable[..., Any] = field(repr=False, compare=False)
    schema: dict = field(repr=False)
    confirm: bool = True    # ask the operator before running; False only for a tool declared read-only
    # how the call is shown at the confirmation prompt: decoded arguments -> text. None -> generic rendering.
    preview: Optional[Callable[[dict], str]] = field(default=None, repr=False, compare=False)

    @classmethod
    def define(cls, fn: Callable[..., Any], *, name: Optional[str] = None, description: Optional[str] = None,
               parameters: Optional[dict] = None, confirm: bool = True,
               preview: Optional[Callable[[dict], str]] = None) -> Tool:
        """Derive the schema from fn's signature, type hints and docstring. Each keyword is an override
        slot that replaces the derived part verbatim — `parameters` is the hand-written JSON Schema escape
        hatch for a signature the derivation cannot express. `confirm=False` declares the tool read-only;
        `preview` renders the call for the operator (an Edit as a diff) instead of the generic listing."""
        summary, _ = parse_docstring(fn.__doc__)
        name = name or fn.__name__
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description if description is not None else summary,
                "parameters": parameters if parameters is not None else parameters_schema(fn),
            },
        }
        return cls(name=name, fn=fn, schema=schema, confirm=confirm, preview=preview)

    @property
    def description(self) -> str:
        return self.schema["function"]["description"]

    @property
    def parameters(self) -> dict:
        return self.schema["function"]["parameters"]


@dataclass(frozen=True)
class ToolRegistry:
    """The tools a session offers. Frozen: register() returns a new registry, so the registry on a
    state is a value like everything else on it. An empty registry offers nothing — Request.tools
    stays absent and the model cannot call anything."""
    tools: tuple[Tool, ...] = ()
    max_result_chars: int = 8000    # every result is bounded here so no tool can flood the context window

    def register(self, tool: Tool) -> ToolRegistry:
        if tool.name in self:
            raise ValueError(f"tool {tool.name!r} is already registered")
        return replace(self, tools=self.tools + (tool,))

    def add(self, fn: Callable[..., Any], **overrides) -> ToolRegistry:
        """register(Tool.define(fn, **overrides))."""
        return self.register(Tool.define(fn, **overrides))

    def get(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)

    def schemas(self) -> list[dict]:
        """What goes on Request.tools."""
        return [t.schema for t in self.tools]

    def invoke(self, name: str, arguments: str) -> str:
        """Run the tool the model asked for and return the content of its role:tool message.

        `arguments` is the wire string from ToolCall.arguments — json.loads happens here, nowhere
        earlier. This method never raises for anything the MODEL can cause or the TOOL can do:
        an unknown tool, malformed JSON, arguments the function rejects, or an exception inside the
        tool all come back as text the model reads and can recover from (the same rule CommandError
        follows). Engine.on_error is reserved for bugs in the harness, so nothing tool-side may
        propagate past this boundary. The result is bounded to max_result_chars (head and tail kept).
        """
        return self.bound(self._invoke(name, arguments))

    def bound(self, text: str) -> str:
        """text cut to max_result_chars: the head and the tail survive, the middle is replaced by a
        marker saying how much was dropped. Errors usually sit at the end, so the tail matters."""
        limit = self.max_result_chars
        if len(text) <= limit:
            return text
        head = limit * 3 // 4
        tail = limit - head
        dropped = len(text) - head - tail
        return text[:head] + f"\n[... {dropped} characters truncated ...]\n" + text[-tail:]

    def _invoke(self, name: str, arguments: str) -> str:
        tool = self.get(name)
        if tool is None:
            return f"Tool {name!r} is not available."
        try:
            decoded = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            return f"Could not decode arguments as JSON: {e}"
        if not isinstance(decoded, dict):
            return f"Arguments must be a JSON object, got {_JSON_NAMES.get(type(decoded), type(decoded).__name__)}."

        try:
            inspect.signature(tool.fn).bind(**decoded)
        except TypeError as e:
            return f"Tool {name!r} rejected the arguments: {e}"

        try:
            result = tool.fn(**decoded)
        except Exception as e:
            return f"Tool {name!r} raised {type(e).__name__}: {e}"

        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Tool {name!r} returned an unserializable result: {e}"


    def __contains__(self, name: object) -> bool:
        return any(t.name == name for t in self.tools)

    def __len__(self) -> int:
        return len(self.tools)
