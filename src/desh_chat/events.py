from dataclasses import dataclass, replace
from typing import Callable
from desh.engine import Event, Priority
from desh.render import Palette, c_out, rl_prompt
from desh.llama.client import Completion, Request, Seam, CodeFence, Terminal
from desh.llama.esc_watcher import ESCWatcher
from desh.llama.tokens import estimate_tokens, turn_tokens
from desh_chat.state import ChatState, ChatHistory, Turn
import sys
import readline
import os
import glob
import json
import time

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
        return replace(state, running=False), [Info(f"Goodbye!" + c_out(Palette.DIM_CHROME, f"\nsession saved to {state.session_file}") if state.session_file else "")]


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
    def comman_dauto_complete(text, state):
        buffer = readline.get_line_buffer()
        candidates = []
        if buffer.startswith("/") and " " not in buffer:
            # canonical names only: aliases stay out of completion and help
            candidates = [f"/{name}" for name in COMMANDS if f"/{name}".startswith(buffer)]
        return candidates[state] if state < len(candidates) else None

readline.set_completer_delims(readline.get_completer_delims().replace("/", ""))
readline.set_completer(PromptUser.comman_dauto_complete)
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
            else:
                print(c_out(Palette.HISTORY_USER, f"USER: {turn.user}"))
                print(c_out(Palette.HISTORY_ASSISTANT, f"ASSISTANT: {turn.assistant}"))
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
        value = self._int(0, state.settings.max_turn_tokens)
        return state.change_setting("max_turn_tokens", value), [Info(f"\u21aa max_turn_tokens set to: {value}"), MaybeRegenerate()]

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

    def _int(self, lo, hi) -> int:
        try:
            v = int(self._single_arg())
        except ValueError:
            raise CommandError(f"/{self.command}: not an int: {self.args}")
        if not lo <= v <= hi:
            raise CommandError(f"/{self.command}: must be in [{lo}, {hi}]")
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


@dataclass(frozen=True)
class StreamCompletion(Event):
    request: Request
    prior_tokens: int = 0   # tokens already accounted for in request.messages (system prompt estimate + history view)
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        watcher = ESCWatcher()
        term = Terminal(out=sys.stdout, colour=_TTY)
        new_events: list[Event] = []
        print(c_out(Palette.CHROME_ASSISTANT, "Assistant: "), end="", flush=True)
        try:
            watcher.start()
            completion = state.inference.server.stream(self.request, Seam(CodeFence(term)), cancelled=lambda: watcher.interrupted)
            cancelled = completion.finish_reason == "cancelled"
            if cancelled:
                new_events.append(Info("Response cancelled by user."))
        finally:
            watcher.stop()
        user_msg = self.request.messages[-1]["content"]
        tokens = turn_tokens(completion.usage, user_msg, completion.content, completion.reasoning, self.prior_tokens)
        new_events.append(AppendTurn(user_msg=user_msg, assistant_msg=completion.content, cancelled=cancelled, tokens=tokens))
        if state.completions_log is not None:
            new_events.append(LogCompletion(request=self.request, completion=completion, port=state.inference.port))
        return state, new_events


@dataclass(frozen=True)
class AppendTurn(Event):
    user_msg: str
    assistant_msg: str
    cancelled: bool
    tokens: int = 0     # 0 -> Turn falls back to the character heuristic
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        turn = Turn(self.user_msg, self.assistant_msg, tokens=self.tokens, cancelled=self.cancelled)
        return replace(state, history=state.history.append(turn)), [MaybeCompact()] + _persist(state)


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
        transcript ="\n\n".join([f"USER: {t.user}\nASSISTANT: {t.assistant}" for t in state.history.since_last_summary()])
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
    message: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        user_message = {"role": "user", "content": self.message}
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        msg_tokens = estimate_tokens(self.message)
        used_tokens = msg_tokens + sys_prompt_tokens
        gen_budget = int(min(
            state.settings.max_turn_tokens,
            state.settings.context - used_tokens - state.history.window_tokens(),
            state.settings.turn_token_cap * state.settings.context))
        reserved = sys_prompt_tokens + msg_tokens + gen_budget
        if gen_budget <= 0:
            return state, [Error("Request exceeds context window."), MaybeRegenerate()]
        view = state.history.view_turns(state.settings.context - reserved)
        return state, [StreamCompletion(
            request=Request(
                messages=[{"role": "system", "content": state.system_prompt}] + [m for t in view for m in t.messages()] + [user_message],
                model=state.settings.model,
                temperature=state.settings.temperature,
                max_tokens=gen_budget,
                think=state.settings.think,
                stream=True
                ),
            prior_tokens=sys_prompt_tokens + sum(t.tokens for t in view),
        )]
