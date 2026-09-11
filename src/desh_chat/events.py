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
import json
from dataclasses import dataclass, replace
from typing import Any

from desh.engine import Event, Priority
from desh.render import Palette, c_out, rl_prompt
from desh.llama.stages import Seam, CodeFence, PyHighlight, Terminal, ToolProgress
from desh.llama.wire import Completion, Request, ToolCall
from desh.llama.esc_watcher import ESCWatcher
from desh.llama.tokens import estimate_tokens, turn_tokens
from desh.tools import Tool
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
    """The loop head: where every path that is not inside a turn comes back to. It owns one
    question — is the run still going? — and nothing else: off (Exit) means the queue drains and
    Engine.run returns; on means a turn begins, and how it begins is TurnStart's business."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not state.running:
            return state, []
        return state, [TurnStart()]


@dataclass(frozen=True)
class TurnStart(Event):
    """Opens the next turn. The only creator of state.pending: the turn exists, empty, before its
    message is known, and the message source fills it (UserMessage). Which source is this event's
    policy, read from the state: a capped turn is continued with `auto_prompt` when one is set;
    otherwise an operator is prompted, or a run without one returns.

    `message` is the seed form: a caller that already has the first message (a delegated task, a
    queue-fed harness) opens the turn and delivers it in one step, skipping the policy.

    A command is a harness instruction, not turn content: it never touches pending, and the loop
    comes back here with the placeholder still empty. So this event opens a placeholder only when
    there is none, applies the policy over an empty one, and must never see a filled one — nothing
    reaches the loop head mid-turn. The drain branch opens nothing: a run that returns has no
    turn open (Delegate.answer relies on it)."""
    message: str | None = None
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is None or state.pending.user is None, "TurnStart reached mid-turn"
        opened = state if state.pending is not None else replace(state, pending=PendingTurn())
        if self.message is not None:
            return opened, [UserMessage(self.message)]
        last = state.history.last_non_summary()
        if last is not None and not last.cancelled and last.stop == "cap" and state.auto_prompt is not None:
            return opened, [Info("Checkpoint: round cap reached, continuing the task."), UserMessage(state.auto_prompt)]
        if state.operator:
            return opened, [DisplayStats(), PromptUser()]
        return state, []

@dataclass(frozen=True)
class Exit(Event):
    """Exit the simulation. The turn open at that moment — the empty placeholder behind the prompt,
    or a turn cut short by Ctrl+C — is dropped: a run that returns has no turn open."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        info_events: list[Event] = [Info("Goodbye!")]
        if state.session_file:
            info_events.append(Info(f"session saved to {state.session_file}", colour=Palette.DIM_CHROME))
        return replace(state, running=False, pending=None), info_events


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
#   TurnStart                opens state.pending = PendingTurn() (empty)      -> UserMessage | PromptUser | drain
#   UserMessage(msg)         fills pending.user                               -> NextRound
#   NextRound                budgets + builds the request from history view
#                            + pending.messages(); compacts first when the
#                            window leaves less than min_gen_tokens           -> StreamCompletion
#                                                                             | CompactHistory + NextRound(compacted=True)
#                                                                             | TurnEnd(cancelled, stop="overflow")
#   StreamCompletion         streams one round; routes on finish_reason:
#                              cancelled            -> TurnEnd(cancelled=True)
#                              tool_calls           -> AppendRound
#                              anything else        -> TurnEnd
#   AppendRound              round cap check; records the calls on pending  -> ExecuteToolCalls(0) | Warn + TurnEnd(stop="cap")
#   ExecuteToolCalls(i)      ONE call per step, in order: asks the operator
#                            if the tool wants confirmation, runs it or
#                            records the denial, attaches the result       -> ExecuteToolCalls(i+1) | NextRound | TurnEnd(cancelled)
#   TurnEnd                  freezes pending into a history Turn, clears it  -> SaveSession + MaybeRegenerate
#
# Today's plain chat is the one-round case: NextRound -> StreamCompletion -> TurnEnd.
# Only TurnEnd appends to history. Compaction rewrites history, never pending: it runs from
# NextRound with the turn's rounds intact, or from /compact against an empty placeholder, which
# is not a turn in progress. An interrupt mid-loop (Ctrl+C -> Exit) simply drops state.pending.
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
            completion = state.inference.server.stream(self.request, Seam(CodeFence(PyHighlight(ToolProgress(term)))), cancelled=lambda: watcher.interrupted)
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
    """The model asked for tools. Record the round on the pending turn and go run them.. Emits a 
    warning and ends the turn if:
        - turn has already used its round budget, in which case it ends here with whatever the model said.
        - a third repeated request, after two identical rounds with identical results"""
    assistant: str
    tool_calls: tuple[ToolCall, ...]
    tokens: int = 0
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None
        if len(state.pending.rounds) >= state.settings.max_tool_rounds:
            # This event only knows the cap was hit and the calls were not run. Whether the turn is
            # over or a checkpoint is the idle event's business (Continue says so when it goes on).
            names = ", ".join(tc.name for tc in self.tool_calls)
            return state, [Warn(f"Tool-call round cap reached ({state.settings.max_tool_rounds}); {names} not run."),
                           TurnEnd(assistant=self.assistant, tokens=self.tokens, cancelled=False, stop="cap")]
        
        # A model that asks for the same calls a third time, having twice seen the same results, is
        # looping: the third round is not recorded and the turn ends the way the round cap ends it.
        # Calls are compared by the registry's identity (Tool.identity), not the wire string: a call
        # that differs only in an argument the tool does not count (Bash's reason) is the same call.
        # OBS: The results comparison relies on every recorded round having one result per call
        def shape(calls: tuple[ToolCall, ...]) -> tuple[tuple[str, str], ...]:
            return tuple(sorted(state.tools.identity(tc.name, tc.arguments) for tc in calls))
        has_tail = len(state.pending.rounds) >= 2
        if has_tail:
            last, before = state.pending.rounds[-1:], state.pending.rounds[-2:-1]
            if last and before and shape(self.tool_calls) == shape(last[0].tool_calls) == shape(before[0].tool_calls) \
                    and [r.content for r in last[0].results] == [r.content for r in before[0].results]:
                names = ", ".join(tc.name for tc in self.tool_calls)
                return state, [Warn(f"Repeated round: {names} asked for a third time with identical results; ending the turn."),
                            TurnEnd(assistant=f"{self.assistant}\n[stopped: {names} repeated three times with identical results]",
                                    tokens=self.tokens, cancelled=False)]

            
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
        assert state.pending is not None
        round = state.pending.rounds[-1]
        tc = round.tool_calls[self.index]
        tool = state.tools.get(tc.name)
        print(describe_call(tc, tool))
        if state.settings.auto:
            answer = Answer("yes")      # auto mode: confirmed tools run without asking
        elif tool is not None and tool.confirm:
            answer = gate.ask(tc)       # through the module so a test can script the prompt
        else:
            answer = Answer("yes")

        if answer.kind == "auto":
            # "a" neither runs nor denies the call: it turns auto mode on and re-asks the SAME
            # call; on the re-ask auto is on, so the call runs without asking.
            return state.change_setting("auto", True), [
                Info("auto mode on: confirmed tools run without asking; Ctrl+C turns it off"),
                ExecuteToolCalls(self.index)]
        if answer.kind == "cancel":
            return state, [Warn("Turn cancelled at the confirmation prompt."),
                           TurnEnd(assistant="", tokens=0, cancelled=True)]
        if answer.kind == "no":
            denied = ToolResult(tc.id, tc.name, answer.message or DENIED_TEXT)
            skipped = tuple(ToolResult(o.id, o.name, SKIPPED_TEXT) for o in round.tool_calls[self.index + 1:])
            shown: list[Event] = [Warn(f"✗ {tc.name} declined" + (f": {answer.message}" if answer.message else ""))]
            if skipped:
                shown.append(Warn(f"  {len(skipped)} later call(s) not run: {', '.join(r.name for r in skipped)}"))
            return replace(state, pending=state.pending.add_results(denied, *skipped)), shown + [DisplayStats(colour=Palette.TOOL_STATS), NextRound()]

        # the settings go along for tools that declared them (delegate): a subagent inherits the
        # parent's CURRENT settings, not the ones captured when the registry was built
        result = ToolResult(tc.id, tc.name, state.tools.invoke(tc.name, tc.arguments, settings=state.settings))
        last = self.index + 1 == len(round.tool_calls)
        pending = state.pending.add_results(result)

        # The call ran with its full arguments; what the round echoes back from now on is the tool's
        # folded form of them (a delegate brief shrinks to its head once the answer supersedes it).
        # Recorded here, before the round is ever sent, so the prefix cache only sees the fold.
        if tool is not None and tool.fold is not None:
            pending = pending.fold_call(self.index, folded_arguments(tool, tc.arguments))

        # the echo is for the operator's eye, so it is short; the model gets the full result
        return (replace(state, pending=pending),
                [Info(shorten(result.content), colour=Palette.TOOL_RESULT),
                 DisplayStats(colour=Palette.TOOL_STATS),
                 NextRound() if last else ExecuteToolCalls(self.index + 1)])


def folded_arguments(tool: Tool, arguments: str) -> str:
    """The wire string a call is recorded with after it ran: tool.fold applied to the decoded
    arguments, re-serialised. Arguments that are not a JSON object are kept as they are — the
    registry already answered such a call with an error, and there is nothing to fold."""
    try:
        args = json.loads(arguments)
    except ValueError:
        return arguments
    if not isinstance(args, dict) or tool.fold is None:
        return arguments
    return json.dumps(tool.fold(args), ensure_ascii=False)


@dataclass(frozen=True)
class TurnEnd(Event):
    """The turn's final completion arrived (or the turn was cut short): freeze state.pending into a
    history Turn. The only event that appends to history."""
    assistant: str
    tokens: int = 0     # prices the final completion only; 0 -> the Turn falls back to the character heuristic
    cancelled: bool = False
    stop: str = ""      # recorded on the Turn: "cap" | "overflow" | "" (see Turn.stop)
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None
        turn = state.pending.finish(self.assistant, self.tokens, self.cancelled, self.stop)
        new_state = replace(state, history=state.history.append(turn), pending=None)
        return new_state, persist(state) + [MaybeRegenerate()]


@dataclass(frozen=True)
class CompactHistory(Event):
    """Replace the window since the last summary with a summary turn. Rewrites history only —
    a pending turn, empty or mid-loop, is left as it is. Schedules no successor: the caller
    sequences what follows (NextRound retries the request, /compact returns to the loop head),
    which is what lets one event serve both."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        instruction = state.settings.compaction_prompt     # a setting, so a run can be built with another (see COMPACTION_PROMPT)
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
            LogCompletion(request=req, completion=completion, port=state.inference.port)] + persist(state)


@dataclass(frozen=True)
class UserMessage(Event):
    """The turn's message arrived: fill the placeholder TurnStart opened and request the first round."""
    message: str
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None, "UserMessage before TurnStart"
        return replace(state, pending=state.pending.with_user(self.message)), [NextRound()]


@dataclass(frozen=True)
class NextRound(Event):
    """Budget and build the request for the next completion of the pending turn: system prompt, the
    history view that fits, then the pending turn so far (user message + every tool round).

    Compaction is decided here, once per request, because this is the only point that knows what
    the request needs: when the window leaves less than min_gen_tokens for the completion, and
    there is a window to summarise, the history is compacted and the request rebuilt from the
    summary. Once is the limit — a turn that is itself too large for the context would otherwise
    summarise the summary forever — and what still does not fit ends the turn as an overflow,
    recorded as a cancelled turn so the loop head can see it (a dropped message would be issued
    again by an auto prompt, forever)."""
    compacted: bool = False     # True on the retry after a compaction: no second one
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None and state.pending.user is not None
        pending = state.pending
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        # What the pending turn costs in the prompt: rounds already priced by usage frames, plus the
        # heuristic for the text no frame has priced yet (user message on round one, latest results after).
        pending_tokens = state.pending_tokens()
        if state.gen_room(pending_tokens) < state.min_gen_tokens():
            # Compaction can only help while the window holds something other than a summary.
            summarisable = any(not t.summary for t in state.history.since_last_summary())
            if not self.compacted and summarisable:
                return state, [Info("Compacting conversation history..."), CompactHistory(), NextRound(compacted=True)]
            return state, [Error("Request exceeds context window; ending the turn."),
                           TurnEnd(assistant="", tokens=0, cancelled=True, stop="overflow")]
        gen_budget = state.gen_budget(pending_tokens)
        reserved = sys_prompt_tokens + pending_tokens + gen_budget
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
