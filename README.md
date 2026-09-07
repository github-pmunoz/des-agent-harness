# des-agent-harness

A discrete-event-simulation engine with a chat and coding agent built on it, running against a local llama-server instance.

# Overview

**Status**: early stages of development, APIs are subject to change.

**Why this is not "yet another agent harness"**: The main property that distinguishes this project from others is its focus on a pure-state discrete-event machine architecture: `event.execute(state)->(state, events)` returns a new state and future events. State is frozen, and every step is transactional: a step that raises leaves the last committed state in place. Side effects (printing, tool execution, HTTP calls to the model) happen only inside `execute`, so they are explicit and visible in the log, and `trace()` gives you every intermediate state if needed.

**Design choices:**
- *Zero deps*: no external libraries, only the Python standard library, POSIX only.
- *Extensibility*: control flow is data. The chat loop, or any app built on the engine, is expressed entirely in what its events return, so behavior is changed by editing an event's return value and added by returning a new event. The turn diagram below is a literal reading of the code, not a summary of it.
- *Observability*: every sim step can be logged with its duration, outcome, queue depth and emitted events (`--debug` to stderr, `--des-log` to a JSONL file); when configured, sessions are written to disk after every turn and recoverable. The engine runs a simulation three ways with one signature: `run()` returns the final state, `iter()` yields the state after each step, `trace()` returns every intermediate state.
- *Tools as data*: tools are plain functions whose schema is derived from signatures and docstrings. A derived schema cannot drift; overrides are explicit. Confirm policy and preview defined at the tool itself.
- *Tool confirmation gate*: per-call approval, denials can carry a message, the coding toolset's Bash requires a stated reason.
- *What this is not*: at this stage, one model at a time (`/model` switches between turns), the engine is single-threaded (a daemon thread only watches for ESC during streaming), and there are no parallel agents.

# Quick Start

**Prerequisites**
- llama-server running locally, in single-model or router mode (default port is 8012; tested with build b10643)
- Python >= 3.12

**Installation**
```bash
git clone https://github.com/github-pmunoz/des-agent-harness.git
cd des-agent-harness
pip install -e .
```


## Usage
Start a chat using a model name exposed by your llama-server:

```bash
chat-des --model Qwen3.8-27B-UD-Q4_K_M
```

Use `chat-des --help` to see the complete, current argument options.


### CLI options

A fully specified coding-agent session:

```bash
chat-des \
  --debug \
  --des-log logs/des.jsonl \
  --completions-log logs/completions.jsonl \
  --session sessions/project.json \
  --port 8012 \
  --model Qwen3.8-27B-UD-Q4_K_M \
  --temperature 0.5 \
  --context 16384 \
  --max-turn-tokens 8192 \
  --max-tool-rounds 10 \
  --toolset coding \
  --workspace . \
  --think
```

The most useful options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--port` | `8012` | Local llama-server port. |
| `--model` | `Qwen3.8-27B-UD-Q4_K_M` | Model ID, which must match the server/router configuration. |
| `--temperature` | `0.3` | Sampling temperature. |
| `--context` | `16384` | Context-window size used for budgeting and compaction. |
| `--max-turn-tokens` | `8192` | Completion-token limit for one turn. |
| `--max-tool-rounds` | `10` | Maximum number of tool-call rounds in a single turn. |
| `--think` | off | Ask the server to enable reasoning/thinking. |
| `--toolset` | `basic` | Tools offered to the model: `none`, `basic` (current time), or `coding`. |
| `--workspace` | `.` | Root directory available to the `coding` toolset. |
| `--session PATH` | none | Load an existing session or save the current session to `PATH`. |
| `--sessions-folder DIR` | none | Create a uniquely named session file in `DIR`; ignored when `--session` is supplied. |
| `--completions-log PATH` | none | Append successful model requests and completions as JSONL. |
| `--des-log PATH` | none | Append engine run/step records as JSONL. |
| `--debug` | off | Print event execution records to stderr, including queue depth and duration. |
| `--timeout SECONDS` | derived from context | Per-request HTTP timeout; `0` derives one from `--context`. |
| `--system-prompt TEXT` | toolset-specific | Override the default system prompt. |

With `--toolset coding`, the model can `Read` without confirmation. `Write`, `Edit`, and `Bash` are shown for approval before they run, one call at a time. File paths are confined to `--workspace` (no escaping the root, no symlinks); `Bash` runs with the workspace as its working directory and must state a `reason` alongside the command. An `Edit` is shown as a diff.

At the approval prompt one key decides: `y` runs the call, `n` declines it, `m` declines it with a message the model reads as the tool result, `c` (or ESC) cancels the turn. Enter is `y`. A declined call short-circuits the rest of that round: the model sees the denial and adapts on its next round.

### In-chat commands

Type these at the `You:` prompt. Tab completes canonical command names.

| Command | Effect |
| --- | --- |
| `/compact` | Compact the conversation history now. |
| `/context [tokens]` | Show or set the context-window size. |
| `/exit` or `/quit` | Exit. (The session file, when configured, is written after every turn, not at exit.) |
| `/history` | Print the conversation history. |
| `/max_turn_tokens [tokens]` | Show or set the completion-token limit. |
| `/max_tool_rounds [rounds]` | Show or set the maximum tool-call rounds in a turn. |
| `/models` | List models reported by the server. |
| `/model [name]` | Show or select a reported model. |
| `/temperature [0.0–2.0]` | Show or set sampling temperature. |
| `/think` | Enable thinking. |
| `/nothink` | Disable thinking. |

Commands with an optional argument show the current setting when invoked without one.

## Architecture

The repository structure is described in more detail in [INDEX.md](INDEX.md).

| Package | Responsibility |
| --- | --- |
| `desh.engine` | Generic priority-queue DES runtime, transactional execution, tracing, debug output, and JSONL step logs. |
| `desh.tools` | Tool definitions, schema derivation from Python signatures/docstrings, confirmation policy, bounded results, and tool-side errors returned as text. |
| `desh.render` | Terminal colour palette and readline-safe prompts. |
| `desh.llama` | llama-server HTTP/SSE protocol (`wire`, `server`), completions logging, token accounting, ESC watcher, and streaming render stages. |
| `desh_chat` | CLI, immutable chat state, turn events, commands, sessions, terminal display, confirmation gate, and toolsets. |

A normal turn is an event loop, rather than a single blocking completion:

```text
PromptUser
  ├─ "/command" ─► Command ─► MaybeRegenerate
  └─ text ──────► UserMessage ─► NextRound ─► StreamCompletion
                                    ▲              ├─ final reply ─► TurnEnd
                                    │              └─ tool calls ──► AppendRound ─► ExecuteToolCalls(i)
                                    │                                                 ├─ yes: run, attach result ─► ExecuteToolCalls(i+1), or after the last call ─┐
                                    │                                                 ├─ no:  denial result, later calls skipped ──────────────────────────────────┤
                                    │                                                 └─ cancel ─► TurnEnd (cancelled)                                             │
                                    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

TurnEnd ─┬─► MaybeCompact ─┬─ window over threshold ─► CompactHistory ─► MaybeRegenerate
         │                 └─ otherwise ──────────────────────────────► MaybeRegenerate ─► DisplayStats, PromptUser
         └─► SaveSession (when a session file is configured)
```

`TurnEnd` is the sole point that commits a completed turn to history; until then the turn in progress lives on `state.pending`, so compaction and persistence never see a half-finished turn. Tool calls are answered one per step, and every result, including an operator denial, becomes an input to the next `NextRound`, so the model can inspect it and continue. `MaybeRegenerate` is the single point that returns to the prompt: commands and compaction resolve through it rather than scheduling `PromptUser` themselves.

## Development

Run the test suite with:

```bash
pytest
```

### Add an event

Define an immutable state and an `Event` whose `execute` method returns a new state plus follow-up events. Use `Priority.HIGH` for display-like work that must run before normal queued events.

```python
from dataclasses import dataclass, replace

from desh.engine import Event, State

@dataclass(frozen=True)
class Counter(State):
    value: int = 0

@dataclass(frozen=True)
class Increment(Event):
    def execute(self, state: Counter) -> tuple[Counter, list[Event]]:
        updated = replace(state, value=state.value + 1)
        return updated, [Report(updated.value)]

@dataclass(frozen=True)
class Report(Event):
    value: int
    def execute(self, state: Counter) -> tuple[Counter, list[Event]]:
        print(self.value)  # explicit side-effect boundary
        return state, []
```

Seed it with `Engine[Counter]().run(Counter(), [Increment()])`. An event should not mutate the state it receives; schedule follow-up work by returning events instead.

### Add a tool

Tools are ordinary typed functions. `ToolRegistry.add()` derives the JSON schema from type hints and a Google-style docstring. Tools require confirmation by default; use `confirm=False` only for a read-only operation.

```python
from desh.tools import ToolRegistry

def search_notes(query: str, limit: int = 10) -> str:
    """Search the local note index.

    Args:
        query: Text to search for.
        limit: Maximum number of matches.
    """
    return "no matches"

registry = ToolRegistry().add(search_notes, confirm=False)
```

Pass the registry into your `ChatState`, or add it to a toolset builder in `desh_chat.cli.TOOLSETS`. Functions must accept keyword arguments and have supported type hints; a parameter without a usable hint fails at registration, not at the model's first call. `add()` also takes `name=` and `description=` overrides, `parameters={...}` as a JSON Schema escape hatch for unusual signatures, and `preview=` to render a call for approval in a custom way (the coding toolset's `Edit` shows a diff). Bound methods work too, which is how the coding toolset keeps its workspace root out of the schema.

### Add a command

Commands belong in `desh_chat.commands.Command`: add a handler returning `(state, events)`, then add a `CommandSpec` entry to `COMMANDS`. The registry powers dispatch and tab completion.

```python
class Command(Event):
    ...
    def _cmd_greet(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Info("Hello!"), MaybeRegenerate()]

COMMANDS: dict[str, CommandSpec] = {
    ...
    "greet": CommandSpec("print a greeting", Command._cmd_greet, aliases=("hello",)),
}
```

The entry must go in the `COMMANDS` literal itself: the dispatch index that resolves names and aliases is built from it once, at import, so assigning to the dict afterwards would not register the command. Return `MaybeRegenerate()` for commands that should return to the prompt, or `Exit()` for commands that end the session.

### Add a display event or streaming stage

`desh_chat.display` contains display events: terminal sinks that return the unchanged state and no events. Subclass `DisplayEvent` to inherit high priority.

```python
from dataclasses import dataclass
from desh_chat.display import DisplayEvent
from desh_chat.state import ChatState

@dataclass(frozen=True)
class Notice(DisplayEvent):
    text: str
    def execute(self, state: ChatState):
        print(self.text)
        return state, []
```

For transforming streamed model output instead, add a `Stage` in `desh.llama.stages` and compose it around `Terminal` in `StreamCompletion`. A stage receives stream events, transforms or consumes them, then forwards them to the next stage; it is distinct from a DES event.

## License

[MIT](LICENSE)
