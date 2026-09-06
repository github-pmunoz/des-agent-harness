from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional
from desh.engine import Event, Priority
from desh.render import Palette, c_out, rl_prompt
from desh.llama.client import Completion, Request, Seam, CodeFence, Terminal, ToolCall, ToolProgress
from desh.tools import Tool
from desh.llama.esc_watcher import ESCWatcher
from desh.llama.tokens import estimate_tokens, turn_tokens
from desh_chat.state import ChatState, ChatHistory, PendingTurn, Round, ToolResult
import sys
import readline
import os
import glob
import json
import time
import termios
import tty

_TTY = sys.stdout.isatty()


@dataclass(frozen=True)
class MaybeRegenerate(Event):
    """Schedules PromptUser if running."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not state.running:
            return state, []
        return state, [DisplayStats(), PromptUser()]


@dataclass(frozen=True)
class Exit(Event):
    """Exit the simulation."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        return replace(state, running=False), [Info(f"Goodbye!" + (c_out(Palette.DIM_CHROME, f"\nsession saved to {state.session_file}") if state.session_file else ""))]


@dataclass(frozen=True)
class PromptUser(Event):
    """Event to prompt user for input."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            user_input = input(rl_prompt(Palette.CHROME_USER, "You: "))
        except EOFError:
            return state, [Exit()]
        if not user_input:
            return state, [MaybeRegenerate()]
        if user_input[0] == "/":
            # command names are case-sensitive by design
            c = s = ""
            if " " in user_input:
                c, s = user_input[1:].split(" ", 1)
            else:
                c, s = user_input[1:], ""
            return state, [Command(command=c.strip(), args=s.strip())]
        return state, [UserMessage(user_input)]

    @staticmethod
    def command_auto_complete(text, state):
        buffer = readline.get_line_buffer()
        candidates = []
        if buffer.startswith("/") and " " not in buffer:
            # canonical names only: aliases stay out of completion and help
            candidates = [f"/{name}" for name in COMMANDS if f"/{name}".startswith(buffer)]
        return candidates[state] if state < len(candidates) else None

readline.set_completer_delims(readline.get_completer_delims().replace("/", ""))
readline.set_completer(PromptUser.command_auto_complete)
readline.parse_and_bind("tab: complete")

# ---------------------
# Display
# ---------------------

class DisplayEvent(Event):
    priority: int = Priority.HIGH


@dataclass(frozen=True)
class Info(DisplayEvent):
    text: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, self.text))
        return state, []


@dataclass(frozen=True)
class Warn(DisplayEvent):
    text: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.WARNING, self.text))
        return state, []


@dataclass(frozen=True)
class Error(DisplayEvent):
    text: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.ERROR, self.text))
        return state, []
    

@dataclass(frozen=True)
class DisplayHistory(DisplayEvent):
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, "History:"))
        for turn in state.history.turns:
            if turn.summary:
                print(c_out(Palette.HISTORY_SUMMARY, f"{turn.user}"))
                continue
            for line in turn.transcript().splitlines():
                colour = Palette.HISTORY_USER if line.startswith("USER: ") else Palette.HISTORY_ASSISTANT
                print(c_out(colour, line))
        return state, []


@dataclass(frozen=True)
class DisplayStats(DisplayEvent):
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        window_tokens = sys_prompt_tokens + state.history.window_tokens()
        total_tokens = sys_prompt_tokens + state.history.get_total_tokens()
        print(c_out(Palette.STATS_LINE, f"Context: {window_tokens} / {state.settings.context} tokens ({window_tokens/state.settings.context*100.0:.1f}%) \t Session: {total_tokens}"))
        return state, []


# ---------------------
# Commands
# ---------------------

class CommandError(Exception):
    """User facing command problem; never a bug"""


CommandHandler = Callable[["Command", ChatState], tuple[ChatState, list[Event]]]


@dataclass(frozen=True)
class CommandSpec:
    """One registry row: what a command does, and the handler that does it.

    `aliases` resolve on dispatch only; completion and any help listing show
    the canonical name alone.
    """
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Command(Event):
    """Event to process a command."""
    command: str
    args: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        try:
            return self._dispatch(state)
        except CommandError as e:
            return state, [Warn(str(e)), MaybeRegenerate()]

    def _dispatch(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        # command names are case-sensitive by design (see PromptUser)
        spec = _COMMAND_INDEX.get(self.command)
        if spec is None:
            raise CommandError(f"Unknown command: /{self.command}")
        return spec.handler(self, state)

    # --- handlers: one per registry row, in COMMANDS order ---

    def _cmd_compact(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Info(f"compacting conversation history..."), CompactHistory()]

    def _cmd_context(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"context: {state.settings.context}"), MaybeRegenerate()]
        value = self._int(0, state.inference.max_context[state.settings.model])
        return state.change_setting("context", value), [Info(f"\u21aa context set to: {value}"), MaybeRegenerate()]

    def _cmd_exit(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Exit()]

    def _cmd_history(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [DisplayHistory(), MaybeRegenerate()]

    def _cmd_max_turn_tokens(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"max_turn_tokens: {state.settings.max_turn_tokens}"), MaybeRegenerate()]
        value = self._int(0, state.settings.context)
        return state.change_setting("max_turn_tokens", value), [Info(f"\u21aa max_turn_tokens set to: {value}"), MaybeRegenerate()]

    def _cmd_max_tool_rounds(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"max_tool_rounds: {state.settings.max_tool_rounds}"), MaybeRegenerate()]
        value = self._int(lo=1)
        return state.change_setting("max_tool_rounds", value), [Info(f"\u21aa max_tool_rounds set to: {value}"), MaybeRegenerate()]
    
    def _cmd_models(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state, [Info("\n".join(state.inference.models)), MaybeRegenerate()]

    def _cmd_model(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"model: {state.settings.model}"), MaybeRegenerate()]
        name = self._single_arg()
        if name not in state.inference.models:
            raise CommandError(f"Model {name} not found.")
        return state.change_setting("model", name), [Info(f"\u21aa model set to: {name}"), MaybeRegenerate()]

    def _cmd_temperature(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not self.args:
            return state, [Info(f"temperature: {state.settings.temperature}"), MaybeRegenerate()]
        value = self._float(lo=0.0, hi=2.0)
        return state.change_setting("temperature", value), [Info(f"\u21aa temperature set to: {value}"), MaybeRegenerate()]

    def _cmd_think(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state.change_setting("think", True), [Info(f"\u21aa thinking mode enabled"), MaybeRegenerate()]

    def _cmd_nothink(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        self._no_args()
        return state.change_setting("think", False), [Info(f"\u21aa thinking mode disabled"), MaybeRegenerate()]

    # --- argument parsing ---

    def _no_args(self):
        if self.args:
            raise CommandError(f"/{self.command} takes no argument: {self.args}")

    def _single_arg(self) -> str:
        if not self.args:
            raise CommandError(f"/{self.command} needs an argument")
        if " " in self.args:
            raise CommandError(f"/{self.command} needs a single argument, got {len(self.args.split(" "))}")
        return self.args

    def _float(self, lo, hi) -> float:
        try:
            v = float(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not a float: {self.args}")
        if not lo <= v <= hi:
            raise CommandError(f"/{self.command}: must be in [{lo}, {hi}]")
        return v

    def _int(self, lo=None, hi=None) -> int:
        try:
            v = int(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not an int: {self.args}")
        if (lo is not None and not lo <= v):
            raise CommandError(f"/{self.command}: must be at least {lo}")
        if (hi is not None and not v <= hi):
            raise CommandError(f"/{self.command}: must be at most {hi}")
        return v


# Single source of truth for the command surface: completion, dispatch and
# any help listing all read from here. Keys are canonical names without the
# leading slash. Must follow the Command class so the handlers resolve.
COMMANDS: dict[str, CommandSpec] = {
    "compact":         CommandSpec("compact the conversation history",      Command._cmd_compact),
    "context":         CommandSpec("set the context window size",           Command._cmd_context),
    "exit":            CommandSpec("exit the chat",                         Command._cmd_exit, aliases=("quit",)), 
    "history":         CommandSpec("show the conversation history",         Command._cmd_history),
    "max_turn_tokens": CommandSpec("set the max number of tokens per turn", Command._cmd_max_turn_tokens),
    "max_tool_rounds": CommandSpec("set the max number of tool rounds",     Command._cmd_max_tool_rounds),
    "models":          CommandSpec("list available models",                 Command._cmd_models),
    "model":           CommandSpec("set the model to use",                  Command._cmd_model),
    "temperature":     CommandSpec("set the temperature",                   Command._cmd_temperature),
    "think":           CommandSpec("enable thinking",                       Command._cmd_think),
    "nothink":         CommandSpec("disable thinking",                      Command._cmd_nothink),
}

# Dispatch index: canonical names plus aliases, all pointing at the same spec.
_COMMAND_INDEX: dict[str, CommandSpec] = {
    name: spec
    for canonical, spec in COMMANDS.items()
    for name in (canonical, *spec.aliases)
}


# ---------------------
# Session persistence
# ---------------------

@dataclass(frozen=True)
class LoadSession(Event):
    """Seed event: restore history from state.session_file, if any.

    Missing file  -> new session, nothing to restore.
    Corrupt file  -> moved aside to <file>.bad so it is never overwritten; session starts empty
                     and keeps saving to the original path.
    """
    priority: int = Priority.HIGH   # must run before the first PromptUser

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        path = state.session_file
        if path is None:
            return state, []
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = ChatHistory.from_dict(json.load(f))
            for turn in history.turns:
                if not turn.summary:
                    readline.add_history(turn.user)
        except FileNotFoundError:
            return state, [Info(c_out(Palette.DIM_CHROME, f"New session: {path}"))]
        except (ValueError, KeyError, TypeError) as e:     # ValueError covers json.JSONDecodeError
            bad = path + ".bad"
            os.replace(path, bad)
            return state, [Warn(f"Session file {path} is unreadable ({e}); moved to {bad}, starting fresh.")]
        return replace(state, history=history), [Info(c_out(Palette.DIM_CHROME, f"Restored {len(history)} turns from {path}")), DisplayStats()]


@dataclass(frozen=True)
class SaveSession(Event):
    """Write the whole history to state.session_file. Atomic: temp file + os.replace, so a crash
    mid-write can never leave a truncated session behind. No-op without a session file."""
    priority: int = Priority.HIGH   # persist right after the history change, before the next prompt

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        path = state.session_file
        if path is None:
            return state, []
        doc = {
            **state.history.to_dict(),
            # informational only — LoadSession restores turns; settings stay with the CLI flags
            "meta": {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model": state.settings.model,
                "context": state.settings.context,
                "system_prompt": state.system_prompt,
            },
        }
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            return state, [Error(f"Could not save session to {path}: {e}")]
        return state, []


def _persist(state: ChatState) -> list[Event]:
    """The events a history-changing step appends so the session file tracks the change."""
    return [SaveSession()] if state.session_file is not None else []


# ---------------------
# Turn logic
# ---------------------

@dataclass(frozen=True)
class LogCompletion(Event):
    request: Request
    completion: Completion
    port: int
    priority: int = Priority.HIGH
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if state.completions_log is not None:
            state.completions_log.record(self.request, self.completion, self.port)
        return state, []


# A turn is a loop, not one completion:
#
#   UserMessage(msg)         opens state.pending = PendingTurn(msg)          -> NextRound
#   NextRound                budgets + builds the request from history view
#                            + pending.messages()                            -> StreamCompletion
#   StreamCompletion         streams one round; routes on finish_reason:
#                              cancelled            -> TurnEnd(cancelled=True)
#                              tool_calls           -> AppendRound
#                              anything else        -> TurnEnd
#   AppendRound              round cap check; records the calls on pending  -> ExecuteToolCalls(0) | Warn + TurnEnd
#   ExecuteToolCalls(i)      ONE call per step, in order: asks the operator
#                            if the tool wants confirmation, runs it or
#                            records the denial, attaches the result       -> ExecuteToolCalls(i+1) | NextRound | TurnEnd(cancelled)
#   TurnEnd                  freezes pending into a history Turn, clears it  -> MaybeCompact (+ SaveSession)
#
# Today's plain chat is the one-round case: NextRound -> StreamCompletion -> TurnEnd.
# Only TurnEnd touches history, so view(), compaction and persistence never see a turn in progress;
# an interrupt mid-loop (Ctrl+C -> Exit) simply drops state.pending.
#
# Confirmation happens inside the round, per call, not as a verdict over the whole round: the
# operator answers yes / no / no-with-guidance / cancel as each call comes up. A "no" short-circuits
# the round — the calls after it are answered "not run" without asking, since the model will need
# to rethink them anyway — and a denial is not an error: it is a tool message the model reads and
# adapts to on the next round. The prompt pauses the way PromptUser does but resolves only through
# logged events (Info/Warn, then the next step, NextRound or TurnEnd); it never schedules
# PromptUser itself, so the regeneration point stays where MaybeRegenerate puts it.


@dataclass(frozen=True)
class StreamCompletion(Event):
    request: Request
    prior_tokens: int = 0   # tokens already accounted for in request.messages (system prompt estimate + history view + priced rounds)
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        watcher = ESCWatcher()
        term = Terminal(out=sys.stdout, colour=_TTY)
        new_events: list[Event] = []
        print(c_out(Palette.CHROME_ASSISTANT, "Assistant: "), end="", flush=True)
        try:
            watcher.start()
            completion = state.inference.server.stream(self.request, Seam(CodeFence(ToolProgress(term))), cancelled=lambda: watcher.interrupted)
            cancelled = completion.finish_reason == "cancelled"
            if cancelled:
                new_events.append(Info("Response cancelled by user."))
        finally:
            watcher.stop()
        # The last message is what this round's usage frame prices as "new prompt": the user message on
        # round one, the tool results afterwards. Only used as the heuristic fallback.
        last_input = self.request.messages[-1]["content"]
        tokens = turn_tokens(completion.usage, last_input, completion.content, completion.reasoning, self.prior_tokens)
        if cancelled:
            new_events.append(TurnEnd(assistant=completion.content, tokens=tokens, cancelled=True))
        elif completion.finish_reason == "tool_calls" and completion.tool_calls:
            new_events.append(AppendRound(assistant=completion.content, tool_calls=tuple(completion.tool_calls), tokens=tokens))
        else:
            new_events.append(TurnEnd(assistant=completion.content, tokens=tokens, cancelled=False))
        if state.completions_log is not None:
            new_events.append(LogCompletion(request=self.request, completion=completion, port=state.inference.port))
        return state, new_events


@dataclass(frozen=True)
class AppendRound(Event):
    """The model asked for tools. Record the round on the pending turn and go run them — unless the
    turn has already used its round budget, in which case it ends here with whatever the model said."""
    assistant: str
    tool_calls: tuple[ToolCall, ...]
    tokens: int = 0
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None
        if len(state.pending.rounds) >= state.settings.max_tool_rounds:
            return state, [Warn(f"Tool-call round cap reached ({state.settings.max_tool_rounds}); ending the turn without running "
                                f"{', '.join(tc.name for tc in self.tool_calls)}."),
                           TurnEnd(assistant=self.assistant, tokens=self.tokens, cancelled=False)]
        pending = state.pending.add_round(Round(self.assistant, self.tool_calls, tokens=self.tokens))
        return replace(state, pending=pending), [ExecuteToolCalls()]


DENIED_TEXT = "The user declined this tool call. Do not retry it; ask the user or take another approach."
SKIPPED_TEXT = "Not run: the user declined an earlier tool call in this round. Reconsider the plan before retrying."


@dataclass(frozen=True)
class Answer:
    """One operator decision at the confirmation prompt. `message` is guidance the operator typed
    with a "no": it replaces DENIED_TEXT as the tool message, so the model learns why."""
    kind: Literal["yes", "no", "cancel"]
    message: str = ""


def ask(tc: ToolCall) -> Answer:
    """The operator's decision for one call whose tool wants confirmation. Never raises.
    ESC is cancel, Enter/EOF with no input is yes, [y/n/m/c] don't need Enter to be pressed,
    and any other key asks again."""
    def prompt(text: str) -> str | None:
        """Read one control line without adding it to the conversation history."""
        try:
            before = readline.get_current_history_length()
            try:
                value = input(rl_prompt(Palette.CHROME, text))
            finally:
                after = readline.get_current_history_length()
                for index in range(after - 1, before - 1, -1):
                    readline.remove_history_item(index)
            return value
        except (EOFError):
            return None
        except Exception:
            return None

    def key_prompt(text: str) -> str | None:
        """Read one decision key immediately, without putting the terminal in line mode."""
        old_settings = None
        try:
            sys.stdout.write(c_out(Palette.CHROME, text))    # plain write: readline's \001/\002 markers do not apply here
            sys.stdout.flush()
            if sys.stdin.isatty():
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            key = sys.stdin.read(1)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return key or None
        except Exception:
            return None
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    while True:
        choice = key_prompt("[y]es / [n]o / [m]essage / [c]ancel: (ESC to cancel, ENTER for yes)")
        if choice is None:
            return Answer(kind="cancel")
        choice = choice.lower()
        if choice in {"\r", "\n"}:
            return Answer(kind="yes")
        if choice == "\x1b":
            return Answer(kind="cancel")
        if choice == "y":
            return Answer(kind="yes")
        if choice == "c":
            return Answer(kind="cancel")
        if choice == "n":
            return Answer(kind="no")
        if choice == "m":
            message = prompt("Message for the model: ") or ""
            return Answer(kind="no", message=message)


@dataclass(frozen=True)
class ExecuteToolCalls(Event):
    """Answer the latest round's calls one per step, in order. This step handles call `index`:
    it asks the operator when the tool wants confirmation, then either runs the call through the
    registry or records the denial, and attaches the result to the round. The registry's invoke()
    turns everything tool-side (unknown tool, bad JSON, rejected arguments, a raising tool) into
    text, so a result always exists; only the RESULT enters state, never the side effect.

    A "no" short-circuits the round: the calls after it are answered SKIPPED_TEXT without asking,
    and the model gets its next round. A "cancel" ends the turn cancelled with whatever ran so far."""
    index: int = 0
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        round = state.pending.rounds[-1]
        tc = round.tool_calls[self.index]
        tool = state.tools.get(tc.name)
        if tool is not None and tool.confirm:
            print(c_out(Palette.CHROME, describe_call(tc, tool)))
            answer = ask(tc)
        else:
            answer = Answer("yes")

        if answer.kind == "cancel":
            return state, [Warn("Turn cancelled at the confirmation prompt."),
                           TurnEnd(assistant="", tokens=0, cancelled=True)]
        if answer.kind == "no":
            denied = ToolResult(tc.id, tc.name, answer.message or DENIED_TEXT)
            skipped = tuple(ToolResult(o.id, o.name, SKIPPED_TEXT) for o in round.tool_calls[self.index + 1:])
            shown: list[Event] = [Warn(f"✗ {tc.name} declined" + (f": {answer.message}" if answer.message else ""))]
            if skipped:
                shown.append(Warn(f"  {len(skipped)} later call(s) not run: {', '.join(r.name for r in skipped)}"))
            return replace(state, pending=state.pending.add_results(denied, *skipped)), shown + [NextRound()]

        result = ToolResult(tc.id, tc.name, state.tools.invoke(tc.name, tc.arguments))
        last = self.index + 1 == len(round.tool_calls)
        # the echo is for the operator's eye, so it is short; the model gets the full result
        echo = f"← {shorten(result.content)}" if tool is not None and tool.confirm else f"→ {tc.name}({shorten(tc.arguments)}) ← {shorten(result.content)}"
        return (replace(state, pending=state.pending.add_results(result)),
                [Info(c_out(Palette.DIM_CHROME, echo)),
                 NextRound() if last else ExecuteToolCalls(self.index + 1)])


def shorten(text: str, limit: int = 200) -> str:
    """One line, at most `limit` characters, for terminal echoes of calls and results."""
    flat = text.replace("\n", "⏎")
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def describe_call(tc: ToolCall, tool: Optional[Tool] = None) -> str:
    """The call as the operator must see it to approve it. A tool with a `preview` renders its own
    (an Edit as a diff); otherwise one line per argument, multi-line values (file contents) as
    indented blocks. Falls back to the raw wire string when the arguments are not a JSON object,
    and to the generic rendering when a preview raises — the gate must always show something."""
    try:
        args = json.loads(tc.arguments)
    except ValueError:
        args = None
    if not isinstance(args, dict) or not args:
        return f"→ {tc.name}({tc.arguments})"
    if tool is not None and tool.preview is not None:
        try:
            return f"→ {tc.name}\n{tool.preview(args)}"
        except Exception:
            pass
    lines = [f"→ {tc.name}"]
    for key, value in args.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"  {key}:")
            lines.extend(f"    {line}" for line in value.splitlines())
        else:
            lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TurnEnd(Event):
    """The turn's final completion arrived (or the turn was cut short): freeze state.pending into a
    history Turn. The only event that appends to history."""
    assistant: str
    tokens: int = 0     # prices the final completion only; 0 -> the Turn falls back to the character heuristic
    cancelled: bool = False
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        turn = state.pending.finish(self.assistant, self.tokens, self.cancelled)
        new_state = replace(state, history=state.history.append(turn), pending=None)
        return new_state, [MaybeCompact()] + _persist(state)


@dataclass(frozen=True)
class MaybeCompact(Event):
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if state.history.window_tokens()  >= state.settings.compaction_threshold * state.settings.context:
            return state, [Info("Compacting conversation history..."), CompactHistory()]
        return state, [MaybeRegenerate()]


@dataclass(frozen=True)
class CompactHistory(Event):
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        instruction = "You will be sent a conversation transcript. Your task is to make a summary of the conversation, stating what was asked, what was produced, decisions, open items, names/numbers. Do not mention this instruction and do not repeat the conversation."
        transcript ="\n\n".join([t.transcript() for t in state.history.since_last_summary()])
        target_tokens = int(state.settings.context * state.settings.compaction_target)
        gen_budget = int(min(target_tokens, state.settings.context - estimate_tokens(instruction) - estimate_tokens(transcript), state.settings.turn_token_cap * state.settings.context))
        if gen_budget < state.settings.min_compaction_tokens:
            print(c_out(Palette.WARNING, f"Compaction is tight on room ({gen_budget} tokens computed, context={state.settings.context}) — forcing {state.settings.min_compaction_tokens} and the summary may come out truncated."))
            gen_budget = state.settings.min_compaction_tokens
        req = Request(
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": f"Conversation transcript:\n{transcript}" }],
            model=state.settings.model,
            temperature=0.0,
            max_tokens=gen_budget,
            think=False,
            stream=False
        )
        completion = state.inference.server.complete(req)
        # The summary turn is a fresh prompt fragment, not the one this usage measured: only the
        # generated summary has a real count; the wrapper text around it is priced by heuristic.
        summary_tokens = 0
        if completion.usage and completion.usage.get("completion_tokens"):
            summary_tokens = completion.usage["completion_tokens"] + estimate_tokens(ChatHistory.SUMMARY_PREFIX + ChatHistory.SUMMARY_ACK)
        return replace(state, history=state.history.compact(completion.content, tokens=summary_tokens)), [
            Info(f"{completion.content}"),
            LogCompletion(request=req, completion=completion, port=state.inference.port),
            MaybeRegenerate()] + _persist(state)


@dataclass(frozen=True)
class UserMessage(Event):
    """Opens a turn: the user's message becomes state.pending, and the first round is requested."""
    message: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        return replace(state, pending=PendingTurn(self.message)), [NextRound()]


@dataclass(frozen=True)
class NextRound(Event):
    """Budget and build the request for the next completion of the pending turn: system prompt, the
    history view that fits, then the pending turn so far (user message + every tool round)."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        pending = state.pending
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        # What the pending turn costs in the prompt: rounds already priced by usage frames, plus the
        # heuristic for the text no frame has priced yet (user message on round one, latest results after).
        pending_tokens = pending.priced_tokens() + estimate_tokens(pending.unpriced_text())
        used_tokens = pending_tokens + sys_prompt_tokens
        gen_budget = int(min(
            state.settings.max_turn_tokens,
            state.settings.context - used_tokens - state.history.window_tokens(),
            state.settings.turn_token_cap * state.settings.context))
        reserved = sys_prompt_tokens + pending_tokens + gen_budget
        if gen_budget <= 0:
            if not pending.rounds:      # nothing happened yet: reject the message, no turn recorded
                return replace(state, pending=None), [Error("Request exceeds context window."), MaybeRegenerate()]
            # mid-loop: the rounds so far are a real exchange; keep them as a cancelled turn
            return state, [Error("Request exceeds context window; ending the turn."),
                           TurnEnd(assistant="", tokens=0, cancelled=True)]
        view = state.history.view_turns(state.settings.context - reserved)
        return state, [StreamCompletion(
            request=Request(
                messages=[{"role": "system", "content": state.system_prompt}] + [m for t in view for m in t.messages()] + pending.messages(),
                model=state.settings.model,
                temperature=state.settings.temperature,
                max_tokens=gen_budget,
                think=state.settings.think,
                stream=True,
                tools=state.tools.schemas(),
                ),
            prior_tokens=sys_prompt_tokens + sum(t.tokens for t in view) + pending.priced_tokens(),
        )]
