"""
The chat loop: prompt, turn, compaction, regeneration. Everything here is one strongly connected
component of the event graph — MaybeRegenerate -> PromptUser -> Command -> ... -> MaybeRegenerate —
which is why it is one module. The leaves live elsewhere: display.py (what gets printed),
session.py (load/save), gate.py (the operator's side of a tool call). commands.py is the one
member of the loop kept in its own file: it imports this module at the top, and this module
reaches it only at call time (PromptUser, the completer), so the cycle never enters the import
graph.
"""
import readline
import sys
from dataclasses import dataclass, replace

from desh.engine import Event, Priority
from desh.render import Palette, c_out, rl_prompt
from desh.llama.stages import Seam, CodeFence, Terminal, ToolProgress
from desh.llama.wire import Completion, Request, ToolCall
from desh.llama.esc_watcher import ESCWatcher
from desh.llama.tokens import estimate_tokens, turn_tokens
from desh_chat.state import ChatState, ChatHistory, PendingTurn, Round, ToolResult
from desh_chat.display import DisplayStats, Error, Info, Warn
from desh_chat.session import persist
from desh_chat import gate
from desh_chat.gate import Answer, DENIED_TEXT, SKIPPED_TEXT, describe_call, shorten

_TTY = sys.stdout.isatty()


# ---------------------
# Prompt and regeneration
# ---------------------

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
        info_events: list[Event] = [Info("Goodbye!")]
        if state.session_file:
            info_events.append(Info(f"session saved to {state.session_file}", colour=Palette.DIM_CHROME))
        return replace(state, running=False), info_events


@dataclass(frozen=True)
class PromptUser(Event):
    """Event to prompt user for input."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        from desh_chat.commands import Command     # the loop's back edge: bound at call time, see module docstring
        try:
            user_input = input(rl_prompt(Palette.CHROME_USER, "You: "))
        except EOFError:
            return state, [Exit()]
        if not user_input:
            return state, [MaybeRegenerate()]
        if user_input.lstrip().startswith("/"):
            # command names are case-sensitive by design
            user_input = user_input.lstrip()
            c = s = ""
            if " " in user_input:
                c, s = user_input[1:].split(" ", 1)
            else:
                c, s = user_input[1:], ""
            return state, [Command(command=c.strip(), args=s.strip())]
        return state, [UserMessage(user_input)]

    @staticmethod
    def command_auto_complete(text, state):
        from desh_chat.commands import COMMANDS    # same back edge as execute()
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
        print(describe_call(tc, tool))
        if tool is not None and tool.confirm:
            answer = gate.ask(tc)       # through the module so a test can script the prompt
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
        return (replace(state, pending=state.pending.add_results(result)),
                [Info(shorten(result.content), colour=Palette.TOOL_RESULT),
                 NextRound() if last else ExecuteToolCalls(self.index + 1)])


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
        return new_state, [MaybeCompact()] + persist(state)


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
            MaybeRegenerate()] + persist(state)


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
