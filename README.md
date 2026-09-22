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
chat-des --model Qwen3.8-27B-UD-Q4_K_M-64K
```

Use `chat-des --help` to see the complete, current argument options. `chat-des.sh` is a thin wrapper that `cd`s to its own directory, creates `.sessions/`, and runs `chat-des -sf .sessions -cl .completions.log -dl .des.log "$@"`. `examples/desh-fibonacci.py` demonstrates the minimal engine contract: a frozen state, an event that re-queues itself, run via `engine.iter()`.


### CLI options

A fully specified coding-agent session:

```bash
chat-des \
  --debug \
  --des-log logs/des.jsonl \
  --completions-log logs/completions.jsonl \
  --session sessions/project.json \
  --port 8012 \
  --model Qwen3.8-27B-UD-Q4_K_M-64K \
  --temperature 0.5 \
  --context 65536 \
  --max-turn-tokens 65536 \
  --max-tool-rounds 10 \
  --read --write --edit --bash --delegate \
  --workspace . \
  --think
```

The most useful options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--port` | `8012` | Local llama-server port. |
| `--model` | `Qwen3.8-27B-UD-Q4_K_M-64K` | Model ID, which must match the server/router configuration. |
| `--temperature` | `0.3` | Sampling temperature. |
| `--context` | `65536` | Context-window size used for budgeting and compaction. |
| `--max-turn-tokens` | `65536` | Completion-token limit for one turn. |
| `--max-tool-rounds` | `10` | Maximum number of tool-call rounds in a single turn. |
| `--tool-cap` | `10.0` | Cap on one tool result, as a percentage of the context window (in chars, 4 per token); the rest is reachable by Read. |
| `--checkpoint-target` | `0.15` | Share of the context a mid-turn checkpoint summary may take. |
| `--memory NAMES` | none | Memory tools to offer, comma-separated: `scratchpad`, `plan`, `ontology`. Registering any memory also appends the context-mechanics prompt. |
| `--scratchpad` | off | Same as `--memory scratchpad`. |
| `--memory-target` | `0.10` | Share of the context one memory may take; a write past it is refused. |
| `--think` | off | Ask the server to enable reasoning/thinking. |
| `--tree` | off | Append the workspace's file tree to the system prompt, generated at launch (venv, .git, caches, logs and JSON records left out): the map the model names paths from. |
| `--read` | off | Offer the `Read` tool. Toolset flags are additive; none of them means no tools. |
| `--write` | off | Offer the `Write` tool. |
| `--edit` | off | Offer the `Edit` tool. |
| `--bash` | off | Offer the `Bash` tool. |
| `--current_time` | off | Offer the current time. |
| `--delegate` | off | Offer `delegate`: subagents with the same tools and settings, minus `delegate` itself. |
| `--task` | none | Run a task non-interactively. |
| `--task-timeout` | `0` | Wall-clock budget in seconds for a `--task` run, 0 = none; checked before each tool round, so a run overshoots by at most one completion. |
| `--workspace` | `.` | Root directory available to the coding tools. |
| `--session PATH` | none | Load an existing session or save the current session to `PATH`. |
| `--sessions-folder DIR` | none | Create a uniquely named session file in `DIR`; ignored when `--session` is supplied. |
| `--completions-log PATH` | none | Append successful model requests and completions as JSONL. |
| `--des-log PATH` | none | Append engine run/step records as JSONL. |
| `--debug` | off | Print event execution records to stderr, including queue depth and duration. |
| `--auto` | off | Enable auto mode for the session. |
| `--cont` | off | Enable auto-continue prompt on tool round cap of orchestrator. |
| `--timeout SECONDS` | derived from context | Per-request HTTP timeout; `0` derives one from `--context`. |
| `--system-prompt TEXT` | `You are a helpful assistant. Reply concisely.` | Override the default system prompt. |
| `--config FILE` | none | JSON file of these options, keyed by long flag name with underscores, plus a `prompts` object. A flag on the command line wins over the file. |
| `--prompt KEY=TEXT` | none | Override one instruction prompt; `TEXT` may be `@file`. Repeatable; wins over the config's `prompts`. |
| `--print-config` | off | Print the effective configuration as JSON, every prompt included, and exit. |

With `--read`, the model can `Read` without confirmation. `Write`, `Edit`, and `Bash` are shown for approval before they run, one call at a time. File paths are confined to `--workspace` (no escaping the root, no symlinks); `Bash` runs with the workspace as its working directory and must state a `reason` alongside the command. An `Edit` is shown as a diff.

At the approval prompt one key decides: `y` runs the call, `n` declines it, `m` declines it with a message the model reads as the tool result, `c` (or ESC) cancels the turn, `a` enables auto mode for the session. Enter is `y`. A declined call short-circuits the rest of that round: the model sees the denial and adapts on its next round.

Auto mode turns the confirmation gate off: confirmed tools run without asking. It is on for the rest of the session — `Ctrl+C` turns it off (in auto mode `Ctrl+C` does not exit; it turns auto mode off and returns to the prompt), and `/noauto` does the same. Delegate subagents inherit it, so their confirmed tools run unconfirmed too.

With `--delegate`, the model can hand a self-contained task to a subagent and read back only its final answer, which keeps the reads and tool rounds of a subtask out of the main context window. The subagent is a nested engine run: it has its own system prompt, always gets `Read`, `Write`, `Edit` and `Bash` (never `delegate` itself, so there is no nesting), and takes the main agent's settings as they are at the moment of the call, so a `/model`, `/temperature` or auto-mode change reaches the next subagent. It streams to the terminal between two banner lines, and its confirmed tools ask for approval exactly as the main agent's do. Each delegation asks for approval, with the task and context shown. ESC or cancel inside the subagent ends only the subagent, and the main agent reads that it was cancelled; a subagent that hits the round cap or runs out of context window reports that instead, so the main agent can tell the three apart. When a session file is set, every subagent run keeps its own beside it, named `<session stem>.delegate-<timestamp>_<hash>.json`. The delegate's `root` attribute (set from `--workspace`) is the working directory for the subagent's tools and for any `check` command the orchestrator supplies, so validation runs in the project root rather than the harness's own directory. Subagents get the run's memories whose subagent policy is `fresh` (the scratchpad and the plan; the ontology is `off`) and start with them empty; a subagent's memory is its own and is never handed back to the main agent, and a run launched without memories gives its subagents none. The `delegate` tool takes a `check` parameter — a shell command the harness runs in the project root *after* the child finishes (60s timeout); its exit code and last 20 output lines (capped at 2000 chars) are appended to the answer as `[check <cmd>: exit N]`, so the child never sees it. A brief over 400 chars is echoed as the 400-char task head + "..." with the note "context, gate and check omitted; see the result". A subagent that hits the round cap is continued with a checkpoint message rather than returned, and its stop reason maps to a note the parent reads: overflow → "[Subagent ran out of context window]", deadline → "[Subagent hit the task deadline]", error → "[Subagent hit an error]", cap → "[Subagent hit the tool round cap]", repeat → "[Subagent ran into a repeat loop]", length → "[Subagent's reply was cut at the token limit]", cancelled → "The operator cancelled the request.", no turn → "The request didn't fit the context window."

Tool results do not accumulate forever. Within a turn every result stays in the context, so each request appends to the previous one and the server's prompt cache keeps its prefix; when the window runs short, the turn's earlier rounds are folded into a checkpoint. The round cap is the one expiry event: when a turn ends there and the task continues in a new turn, every result of the previous turn is replaced by an expired stub. History turns are always stubbed. The `<memory>` block, appended to the end of every request as one user message, is the only working memory that survives both: it holds one section per registered memory (`<scratchpad>`, `<plan>`, `<ontology>`, each tagged with what it uses of its budget), shows which round the run is on, says when the cap or a checkpoint is one round away so the model can persist what it still needs, and says at the cap that only memory calls will run. At the tool-round cap, memory calls still run before the turn ends; other tool calls in the same reply are not run.

### Configuration and prompts

`--config FILE` reads the whole command line from a JSON object whose keys are the long flag names with underscores (`max_tool_rounds`, `tool_cap`, `current_time`). Precedence is flag, then file, then built-in default. The file is checked before anything runs: an unknown key is an error rather than a run on the flag's default, a switch takes a JSON bool, a typed option is passed through its type, and a string value spelled `@path` is the text of that file, relative to the config's directory. The eval settings files (`evals/*/baseline.json`) are config files, and `run_eval.sh` launches `chat-des --config` with only the per-run paths as flags.

Every instruction the model reads can be replaced without touching the harness, through the config's `prompts` object or `--prompt KEY=TEXT`. An override is applied once, at construction, to the value that owns the text; nothing looks a prompt up at run time, and with no overrides every text is the default byte for byte. `--print-config` prints every key of the run with its effective text, which is both the template for a new eval arm and what `run_eval.sh` records as `run_settings.json`.

| Key | Text |
| --- | --- |
| `delegate.system` | The subagent's system prompt. |
| `cap_continue` | The message a capped turn is continued with (the main agent's with `--cont`, and every subagent's). |
| `length_continue` | The message a turn cut at the token limit is continued with, once (same gate). |
| `compaction`, `checkpoint` | The system prompts of the history-compaction and the checkpoint requests. |
| `compaction.close`, `checkpoint.close`, `summary.retry` | What follows the transcript in those requests, and what an empty answer is asked again with. |
| `memory.mechanics` | The context-mechanics system prompt; template with `{max_tool_rounds}`, `{tool_names}`, `{sections}`. |
| `memory.block_head`, `memory.last_round`, `memory.cap_reached`, `memory.fold_near`, `memory.over_budget` | The lines the harness writes in the `<memory>` block and the budget refusal; `cap_reached` carries `{tool_names}`, `over_budget` carries `{name}`, `{tokens}`, `{budget}`. |
| `memory.<name>.prompt` | A selected memory's own prompt. |
| `tool.<Name>.description`, `tool.<Name>.param.<arg>` | A tool's description and a parameter's description, in the main and the subagent registry. The schema's shape is not overridable. |

A template must carry exactly its placeholders; a key for a tool the run does not offer, a parameter the tool does not have or a memory the run did not select is an error. The run's own system prompt is `--system-prompt` (`system_prompt` in a config). The strings the harness or an eval grader parses are deliberately not overridable: the re-read notice, the cut and spill markers, the checkpoint prefix, the repeat-stop note, the gate's declined texts and the notes a parent reads for a subagent's stop.

### Memory tools

Memory tools are plugins (`desh_chat.memory.Memory`): a name, an empty value, a `from_dict`, the tool functions and a prompt. The split of ownership is strict. A plugin owns its value (an immutable object with `to_dict` and `render`), its tools (registered unconfirmed, with the slot name as their injected parameter) and the prompt that says what the memory is *for*. The harness owns everything about *when* memory matters: the context-mechanics system prompt (rounds, the cap, stubs, checkpoints, folded echoes), sent only when a memory is registered; the `<memory>` block and its pricing; the expiry lines; the per-memory budget (`--memory-target`, a write that grows a memory past it is answered with a refusal and not committed); the snapshot on every finished turn, restored per registered slot by `LoadSession` (session format 7, which still reads format 6); the salvage sections; and what a subagent gets (`Memory.subagent`: `fresh` or `off`). No event knows a memory by name, so adding one is a module plus an entry in `cli.MEMORIES`.

- `scratchpad`: `scratchpad_write`, `scratchpad_delete`, `scratchpad_clear`. Entries carry a `kind` field (`todo`, `done`, `fact`, `hypothesis`, `block`) and render grouped by kind in the order DONE, FACTS, HYPOTHESES, BLOCKS, TO DO.
- `plan`: `plan_write`, `plan_delete`. Steps are `todo` or `done`, keep their place when rewritten, and render as a checklist.
- `ontology`: `ontology_write`, `ontology_delete`, `ontology_clear`. Subject-predicate-object triplets, one per triple, in first-written order; a subject or object over 80 characters (a predicate over 40) is refused, because an entity is a name and prose belongs elsewhere. Not offered to subagents.

### Context management

- **Two-rung compaction ladder.** When generation room drops below the minimum, `CompactHistory` summarizes history into a synthetic summary turn. When history is already all summaries but the pending turn has ≥2 non-summary rounds, `CompactPendingTurn` (the dedicated checkpoint) folds all pending rounds except the last into one summary round. Transcripts are fitted to a budget via `fit_transcript` against `transcript_budget(context, instruction, target_tokens)`.
- **Empty-summary retry / fallback digest.** If the model returns an empty summary, it is asked once more with a nudge; if still empty, a deterministic digest of the previous summary/checkpoint text plus one line per turn/round naming the tool calls is used instead.
- **Salvage.** A turn that ends by overflow, deadline, or error has its record salvaged — "TURN ENDED: {stop}" + checkpoint body + one section per non-empty memory + remaining rounds, fitted to `checkpoint_target * context` tokens — so the model still gets what it wrote down.
- **Token-limit cut.** A completion that ends with `finish_reason: length` hit `max_tokens` mid-reply: whatever tool call it was writing is unclosed and never ran, and the text is not an answer. The turn ends with stop `length`, its text followed by a note (`[cut at the token limit after N tokens while writing a call to <tools>]`) and the salvaged record. With `--cont` (always, for subagents) the turn is continued once with the `length_continue` prompt, which says a reply must fit and a brief must carry instructions rather than content; two cuts in a row end the run on the record.
- **Re-read guard.** A read-only (non-confirming) call answered 2+ times in a turn with nothing written/edited/run since gets a corrective notice instead of the result. The count resets at any confirming call.
- **Tool result cap** (`--tool-cap`). Default 10% of the context window (chars, 4 chars/token). Read, Bash and delegate emit a pointer to the rest (Read: `[showing lines X-Y of Z (cap N chars); continue with offset=M]`; Bash: `[output is N characters, cut to M; the whole of it is saved at .desh/out/bash-...txt — Read it with offset and limit]`; delegate: the same line for a subagent's answer, saved whole at `.desh/out/delegate-...txt`).
- **Fold functions.** Once a call ran, the round echoes it with its long argument removed and a `folded` note in its place, a key the schema does not have: a Write's content over 120 chars becomes `folded: "N characters written; Read the file to see it"`, an Edit's old and new strings over 240 chars together become their sizes, a `scratchpad_write` value becomes `folded: "N characters, shown in the scratchpad block"`, and a delegate brief over 400 chars keeps the head of its task. The argument is removed rather than replaced because the model copies the shape of its own echoed calls: a copied fold is a rejected call, never a marker written as a value.

### Orchestrator mode (`coding.sh`)

`coding.sh` is a ready-made orchestrator entry point that wraps `chat-des` with a delegation-focused system prompt. It launches the agent with `--delegate` and a dynamically generated project file tree appended to the system prompt, so the orchestrator always has an up-to-date map of the codebase.

```bash
./coding.sh [extra chat-des args...]
```

- `coding.sh` no longer takes a `MODEL` positional argument: it runs `./src/desh_chat/cli.py` with `-sf .sessions -cl .completions.log -dl .des.log -sp <orchestrator system prompt> --delegate --scratchpad --auto --cont "$@"`, and the CLI default model applies. All remaining positional arguments are passed through to the CLI.
- The orchestrator prompt instructs the agent to decompose work into gated stages (investigation → tests → implementation) and to delegate each stage to a subagent.
- The file tree is generated at launch via `find`, excluding `venv`, `evals`, `.git`, `sandbox`, `.sessions`, `__pycache__`, `*.egg-info`, `.pytest_cache`, and `.log`/`.jsonl`/`.json`/`.bad` files.
- `README.md` and `INDEX.md` are **not** included in the orchestrator's prompt; they are available to subagents on demand.

### Evals: wc-like harness

`evals/wc-like/` is a repeatable eval that has chat-des implement a `wc`-like CLI from `task.txt` + `train/` samples, then grades it against held-out goldens in `test/` that the run never sees.

- `./run_eval.sh <EVAL_SETTINGS.json>` — one run. The settings is flat JSON whose keys map 1:1 to chat-des flags (underscores); unknown keys are refused. It erases the llama-server KV cache for a cold start, runs in its own `runs/run-<ts>-<hash6>/` (workspace, session, des/completions logs, stdout/stderr, manifest), then auto-grades.
- `./sweep.sh <SWEEP.json>` or `./sweep.sh <SETTINGS.json> <REPEATS> [NAME]` — repeated runs. A sweep set names a baseline plus named sweeps (reps + override, validated before any inference is spent); results land in `sweeps/<name>-<ts>/` with `settings.json`, `manifest.json`, `summary.json` (per-metric mean/min/max/stdev over 18 metrics incl. compactions, checkpoints, salvaged_turns, rereads_refused, cut_results, first_green_round) and, for a set, `compare.json` across sweeps.
- `grade.py <run dir>` — checks: held-out scores, CLI contract, the agent's own pytest suite, stdlib-only imports, no tool call touching `test/` or `../`, and run stats → `result.json`.
- `summarise.py <SWEEP_DIR>` / `compare.py <SET_DIR>` — pure functions of the graded files; rerunnable on any old sweep.

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
| `/auto` | Enable auto mode: confirmed tools run without asking. |
| `/noauto` | Disable auto mode. |
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
| `desh.llama.send_direct` | Standalone CLI for direct llama-server requests (python port of an earlier bash prototype). |
| `desh.llama.pyscan` | Streaming Python syntax scanner for the renderer; classifies fence-body text into coloured spans while fragments are still arriving. |
| `desh_chat` | CLI, immutable chat state, turn events, commands, sessions, terminal display, confirmation gate, and toolsets. |
| `desh_chat.handlers` | Error/interrupt handlers for the chat loop: `AutoOff` event, `on_error` (mid-turn error closes the turn with `stop=ERROR`, a hidden stop), `on_interrupt` (in auto mode Ctrl+C turns auto off and cancels the in-flight turn; in manual mode it prints "~ Interrupted" and exits). |

A normal turn is an event loop, rather than a single blocking completion:

```text
MaybeRegenerate ─ running ─► TurnStart (opens an empty pending turn)
                               ├─ last turn capped, auto_prompt set ─► UserMessage(auto_prompt) ─┐
                               ├─ operator ─► DisplayStats, PromptUser                            │
                               │               ├─ "/command" ─► Command ─► MaybeRegenerate        │
                               │               └─ text ──────► UserMessage ───────────────────────┤
                               └─ neither ─► (queue drains, the run returns)                      │
                                                                                                  ▼
NextRound ─┬─ room ≥ min_gen ─► StreamCompletion ─┬─ final reply ─► TurnEnd
    ▲      │                                       └─ tool calls ──► AppendRound ─► ExecuteToolCalls(i)
    │      ├─ room short, first time ─► CompactHistory ─► NextRound(compacted)       ├─ yes: run, attach result ─► ExecuteToolCalls(i+1), or after the last call ─┐
    │      └─ room short, retried ────► TurnEnd (stop=OVERFLOW)                      ├─ no:  denial result, later calls skipped ──────────────────────────────────┤
    │                                                                                └─ cancel ─► TurnEnd (cancelled)                                             │
    └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

TurnEnd ─┬─► SaveSession (when a session file is configured)
         └─► MaybeRegenerate
```

`TurnStart` is the sole creator of `state.pending`: the turn exists, empty, before its message is known, and the message source fills it. A command is a harness instruction rather than turn content, so it leaves the placeholder untouched and the loop head reuses it. `TurnEnd` is the sole point that commits a completed turn to history. Compaction is decided by `NextRound`, once per request, when the window leaves less room than `(1 - compaction_threshold) * context` for the completion: it is paid only when a request is about to go out and needs it, never after a turn that nothing follows. `CompactHistory` schedules no successor of its own, so the same event serves `NextRound` mid-turn and the `/compact` command between turns. What still does not fit after one compaction ends the turn as a recorded overflow, which is what lets an auto prompt that cannot fit stop instead of being issued again. Tool calls are answered one per step, and every result, including an operator denial, becomes an input to the next `NextRound`, so the model can inspect it and continue.

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

Pass the registry into your `ChatState`, or give it a flag in `desh_chat.cli.build_tools`. Functions must accept keyword arguments and have supported type hints; a parameter without a usable hint fails at registration, not at the model's first call. `add()` also takes `name=` and `description=` overrides, `parameters={...}` as a JSON Schema escape hatch for unusual signatures, and `preview=` to render a call for approval in a custom way (the coding toolset's `Edit` shows a diff). Bound methods work too, which is how the coding toolset keeps its workspace root out of the schema.

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
