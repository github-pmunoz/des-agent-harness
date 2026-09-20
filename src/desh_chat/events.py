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
from desh.llama.tokens import estimate_result_tokens, estimate_tokens, turn_tokens
from desh.tools import Tool
from desh_chat.state import CHECKPOINT_PREFIX, SALVAGE_STOPS, ChatState, ChatHistory, PendingTurn, Round, ToolResult, StopReason
from desh_chat.scratchpad import Scratchpad
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
    Engine.run returns; on means a turn begins, and how it begins is TurnStart's business.
    The policy is read from state.idle_policy: 'prompt' begins a turn, 'exit' ends the run."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        if not state.running:
            return state, []
        if state.idle_policy == "exit":
            return state, [Exit(on_exit=None)]
        elif state.idle_policy == "prompt":
            return state, [TurnStart()]
        else:
            raise ValueError(f"Unknown idle_policy: {state.idle_policy}")

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
        if last is not None and last.stop == StopReason.CAP and state.auto_prompt is not None:
            return opened, [Info("Checkpoint: round cap reached, continuing the task."), UserMessage(state.auto_prompt)]
        if state.operator:
            return opened, [DisplayStats(), PromptUser()]
        return state, []

@dataclass(frozen=True)
class Exit(Event):
    """Exit the simulation. The turn open at that moment — the empty placeholder behind the prompt,
    or a turn cut short by Ctrl+C — is dropped: a run that returns has no turn open."""
    on_exit: str | None = "Goodbye!"
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        info_events: list[Event] = [Info(self.on_exit)] if self.on_exit is not None else []
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
#                                                                             | CompactHistory + NextRound
#                                                                             | CompactPendingTurn + NextRound
#                                                                             | TurnEnd(stop=OVERFLOW)
#   StreamCompletion         streams one round; routes on finish_reason:
#                              cancelled            -> TurnEnd(stop=CANCELLED)
#                              tool_calls           -> AppendRound
#                              anything else        -> TurnEnd
#   AppendRound              round cap check; records the calls on pending  -> ExecuteToolCalls(0) | Warn + TurnEnd(stop=CAP)
#   ExecuteToolCalls(i)      ONE call per step, in order: asks the operator
#                            if the tool wants confirmation, runs it or
#                            records the denial, attaches the result       -> ExecuteToolCalls(i+1) | NextRound | TurnEnd(stop=CANCELLED)
#   TurnEnd                  freezes pending into a history Turn, clears it  -> SaveSession + MaybeRegenerate
#
# Today's plain chat is the one-round case: NextRound -> StreamCompletion -> TurnEnd.
# Only TurnEnd appends to history. Compaction is a ladder NextRound climbs one rung per request:
# CompactHistory rewrites history with the turn's rounds intact (it also runs from /compact
# against an empty placeholder, which is not a turn in progress); CompactPendingTurn folds the
# turn's own rounds but the last into a checkpoint round, history untouched. Each rung removes
# the condition that allowed it, so the ladder ends. An interrupt mid-loop (Ctrl+C -> Exit)
# simply drops state.pending.
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
    prior_tokens: int = 0   # tokens already accounted for in request.messages (system prompt estimate + history view + priced rounds + scratchpad block)
    # the text this round's usage frame prices as "new prompt": the user message on round one, the
    # tool results afterwards. Only the heuristic fallback reads it. None -> the request's last
    # message, which is right when nothing follows the pending turn in the request.
    unpriced: str | None = None
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
        last_input = self.unpriced if self.unpriced is not None else self.request.messages[-1]["content"]
        tokens = turn_tokens(completion.usage, last_input, completion.content, completion.reasoning, self.prior_tokens)
        if cancelled:
            new_events.append(TurnEnd(assistant=completion.content, tokens=tokens, stop=StopReason.CANCELLED))
        elif completion.finish_reason == "tool_calls" and completion.tool_calls:
            new_events.append(AppendRound(assistant=completion.content, tool_calls=tuple(completion.tool_calls), tokens=tokens))
        else:
            new_events.append(TurnEnd(assistant=completion.content, tokens=tokens, stop=StopReason.ANSWER))
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
        names = ", ".join(tc.name for tc in self.tool_calls)
        # Two budgets are checked here, before the calls run; both end the turn with the model's text
        # so far as the answer and the calls named in the warning. The deadline comes first: a run
        # out of time must not be continued, and a capped turn would be (TurnStart's policy).
        if state.deadline is not None and state.deadline.passed():
            return state, [Warn(f"Task deadline reached ({state.deadline.budget:g}s); {names} not run."),
                           TurnEnd(assistant=self.assistant, tokens=self.tokens, stop=StopReason.DEADLINE)]
        if state.pending.non_summary_rounds() >= state.settings.max_tool_rounds:
            # This event only knows the cap was hit and the calls were not run. Whether the turn is
            # over or a checkpoint is the idle event's business (Continue says so when it goes on).
            # The round's scratchpad calls are the exception: they are the persistence the cap
            # message asks for, side-effect free on the workspace, so they run before the turn
            # ends. The round itself is still not recorded — its acks would be read by nobody; the
            # next turn's block shows the entries — and TurnEnd snapshots the value onto the Turn.
            state, kept, rest = run_scratchpad_calls(state, self.tool_calls)
            cap = state.settings.max_tool_rounds
            shown: list[Event] = []
            if rest:
                shown.append(Warn(f"Tool-call round cap reached ({cap}); {', '.join(tc.name for tc in rest)} not run."))
                if kept:
                    shown.append(Info("Ran at the round cap: " + ", ".join(kept)))
            else:
                shown.append(Warn(f"Tool-call round cap reached ({cap}); only the scratchpad calls ran: {', '.join(kept)}."))
            return state, shown + [TurnEnd(assistant=self.assistant, tokens=self.tokens, stop=StopReason.CAP)]

        # A model that asks for the same calls a third time, having twice seen the same results, is
        # looping: the third round is not recorded and the turn ends the way the round cap ends it.
        # Calls are compared by the registry's identity (Tool.identity), not the wire string: a call
        # that differs only in an argument the tool does not count (Bash's reason) is the same call.
        # OBS: The results comparison relies on every recorded round having one result per call
        def shape(calls: tuple[ToolCall, ...]) -> tuple[tuple[str, str], ...]:
            return tuple(sorted(state.tools.identity(tc.name, tc.arguments) for tc in calls))
        has_tail = sum(1 for r in state.pending.since_last_summary() if not r.summary) >= 2
        if has_tail:
            last, before = state.pending.rounds[-1:], state.pending.rounds[-2:-1]
            if last and before and shape(self.tool_calls) == shape(last[0].tool_calls) == shape(before[0].tool_calls) \
                    and [r.content for r in last[0].results] == [r.content for r in before[0].results]:
                names = ", ".join(tc.name for tc in self.tool_calls)
                return state, [Warn(f"Repeated round: {names} asked for a third time with identical results; ending the turn."),
                            TurnEnd(assistant=f"{self.assistant}\n[stopped: {names} repeated three times with identical results]",
                                    tokens=self.tokens, stop=StopReason.REPEAT)]
            
        pending = state.pending.add_round(Round(self.assistant, self.tool_calls, tokens=self.tokens))
        return replace(state, pending=pending), [ExecuteToolCalls()]


# A read-only call answered this many times in a turn, with nothing written, edited or run
# since, is a loop whatever its period: the model is re-reading what it already had and lost
# (to expiry, or to a checkpoint that dropped file contents). Seen at 4k: two 30-round turns of
# nothing but the same three reads, which the two-round repeat guard could not see.
REREAD_LIMIT = 2
REREAD_TEXT = ("Not run: {call} has already been answered {n} times this turn with nothing written, edited or run since, "
               "and its result has not changed. Save what you need from it to the scratchpad, or act on it.")


def rereads(state: ChatState, tc: ToolCall) -> int:
    """How many times this turn has already answered the same read-only call — same registry
    identity — since the model last acted. A round with a confirming call (Write, Edit, Bash, a
    delegate) is where the count stops: what was read before it was acted on. A confirming tool,
    or an unknown one, is never counted: re-running a test is polling, not a loop."""
    assert state.pending is not None
    tools = state.tools
    tool = tools.get(tc.name)
    if tool is None or tool.confirm:
        return 0
    key = tools.identity(tc.name, tc.arguments)
    n = 0
    for round in reversed(state.pending.rounds):
        if round.summary:
            continue
        if any((t := tools.get(c.name)) is not None and t.confirm for c in round.tool_calls):
            break
        n += sum(1 for c, _ in zip(round.tool_calls, round.results) if tools.identity(c.name, c.arguments) == key)
    return n


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
                           TurnEnd(assistant="", tokens=0, stop=StopReason.CANCELLED)]
        if answer.kind == "no":
            denied = ToolResult(tc.id, tc.name, answer.message or DENIED_TEXT)
            skipped = tuple(ToolResult(o.id, o.name, SKIPPED_TEXT) for o in round.tool_calls[self.index + 1:])
            shown: list[Event] = [Warn(f"✗ {tc.name} declined" + (f": {answer.message}" if answer.message else ""))]
            if skipped:
                shown.append(Warn(f"  {len(skipped)} later call(s) not run: {', '.join(r.name for r in skipped)}"))
            return replace(state, pending=state.pending.add_results(denied, *skipped)), shown + [DisplayStats(colour=Palette.TOOL_STATS), NextRound()]

        last = self.index + 1 == len(round.tool_calls)
        if (n := rereads(state, tc)) >= REREAD_LIMIT:
            # answered with the notice instead of the result: the turn goes on, and the model reads
            # why. Two such notices and a third identical round are the repeat guard's business.
            what = f"{tc.name} {state.tools.target(tc.name, tc.arguments)}".strip()
            refused = ToolResult(tc.id, tc.name, REREAD_TEXT.format(call=what, n=n))
            return (replace(state, pending=state.pending.add_results(refused)),
                    [Warn(f"✗ {what} not run: answered {n} times already with nothing acted on since"),
                     DisplayStats(colour=Palette.TOOL_STATS),
                     NextRound() if last else ExecuteToolCalls(self.index + 1)])

        content, scratchpad = run_call(state, tc)
        result = ToolResult(tc.id, tc.name, content)
        pending = state.pending.add_results(result)

        # The call ran with its full arguments; what the round echoes back from now on is the tool's
        # folded form of them (a delegate brief shrinks to its head once the answer supersedes it).
        # Recorded here, before the round is ever sent, so the prefix cache only sees the fold.
        if tool is not None and tool.fold is not None:
            pending = pending.fold_call(self.index, folded_arguments(tool, tc.arguments))

        # the echo is for the operator's eye, so it is short; the model gets the full result
        return (replace(state, pending=pending, scratchpad=scratchpad),
                [Info(shorten(result.content), colour=Palette.TOOL_RESULT),
                 DisplayStats(colour=Palette.TOOL_STATS),
                 NextRound() if last else ExecuteToolCalls(self.index + 1)])


def run_call(state: ChatState, tc: ToolCall) -> tuple[str, Scratchpad | None]:
    """Run one call through the registry: the text that answers it, and the scratchpad value after
    it. What the harness supplies to tools that declared it (Tool.inject):
    - the settings, so a subagent inherits the parent's CURRENT settings, not the ones captured
      when the registry was built
    - the deadline, so a subagent stops when the run that spawned it must
    - the scratchpad as a dict built from the state value for this one call.
    A tool that asked for the scratchpad may have changed it: the dict it wrote to is read back
    into a value here, the one way a call becomes a state transition. Nothing mutable survives
    the step, so a step the engine rolls back leaves the working memory untouched."""
    tool = state.tools.get(tc.name)
    provided: dict[str, Any] = {"settings": state.settings, "deadline": state.deadline}
    if state.scratchpad is not None:
        provided["scratchpad"] = state.scratchpad.to_dict()
    content = state.tools.invoke(tc.name, tc.arguments, **provided)
    scratchpad = state.scratchpad
    if tool is not None and "scratchpad" in tool.inject and scratchpad is not None:
        scratchpad = Scratchpad.from_dict(provided["scratchpad"])
    return content, scratchpad


def run_scratchpad_calls(state: ChatState, calls: tuple[ToolCall, ...]) -> tuple[ChatState, list[str], list[ToolCall]]:
    """Run the calls that only touch the scratchpad, in order, and return the state after them,
    a mention of each one that ran (name and key), and the calls left unrun. A scratchpad tool is
    one that declared the injection; without a working memory on the state, its calls are left
    unrun like any other, rather than answered with an error nobody reads."""
    kept: list[str] = []
    rest: list[ToolCall] = []
    for tc in calls:
        tool = state.tools.get(tc.name)
        if tool is None or "scratchpad" not in tool.inject or state.scratchpad is None:
            rest.append(tc)
            continue
        _, scratchpad = run_call(state, tc)
        state = replace(state, scratchpad=scratchpad)
        kept.append(f"{tc.name} {state.tools.target(tc.name, tc.arguments)}".strip())
    return state, kept, rest


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
    stop: StopReason
    tokens: int = 0     # prices the final completion only; 0 -> the Turn falls back to the character heuristic

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None
        assistant, shown = self.assistant, []
        if self.stop in SALVAGE_STOPS:
            # The turn ends without an answer: what it got down stands as the answer, below any
            # text the model did produce (a deadline keeps the model's text so far). Bounded like
            # a checkpoint — it is one, written by the harness — so history and a parent's tool
            # result stay within their shares.
            budget = int(state.settings.checkpoint_target * state.settings.context)
            salvaged = state.pending.salvage(self.stop, state.scratchpad, budget_tokens=budget)
            assistant = f"{assistant.rstrip()}\n\n{salvaged}" if assistant.strip() else salvaged
            shown = [Warn(f"Turn ended by {self.stop}; its record stands as the answer ({len(salvaged)} chars salvaged).")]
        turn = state.pending.finish(assistant, self.tokens, self.stop, scratchpad=state.scratchpad)
        new_state = replace(state, history=state.history.append(turn), pending=None)
        return new_state, shown + persist(state) + [MaybeRegenerate()]


TRANSCRIPT_LEAD = "Conversation transcript:\n"
TRANSCRIPT_MARGIN = 64      # tokens kept free in a compaction request for template overhead and estimate error


# The user message of a summary request must not END with the transcript: a model reading raw
# tool output up to the last token takes it for the end of a document and stops at once
# (replayed: 0 of 8 such requests answered at temperature 0, 0.1 or 0.3; 8 of 8 with a closing
# line). So the transcript is followed by the instruction to write, and an empty answer is
# asked once more with a firmer one.
SUMMARY_CLOSE = "\n\nWrite the summary now."
CHECKPOINT_CLOSE = "\n\nWrite the checkpoint now."
RETRY_NUDGE = "\n\nAn empty reply is not an answer: write it now."


def complete_summary(state: ChatState, req: Request) -> tuple[list[tuple[Request, Completion]], str, list[Event]]:
    """One summary request, asked a second time with a nudge when the model answers with nothing.
    Returns every attempt (request, completion) for the log, the text, and the notes for the log."""
    completion = state.inference.server.complete(req)
    attempts = [(req, completion)]
    summary = completion.content.strip()
    notes: list[Event] = []
    if not summary:
        notes.append(Warn("The model returned no summary; asking again."))
        last = req.messages[-1]
        req = replace(req, messages=req.messages[:-1] + [{**last, "content": last["content"] + RETRY_NUDGE}])
        completion = state.inference.server.complete(req)
        attempts.append((req, completion))
        summary = completion.content.strip()
    return attempts, summary, notes


def transcript_budget(context: int, instruction: str, target_tokens: int) -> int:
    """What a compaction request can spend on its transcript: the window minus the instruction,
    the lead, the summary it must leave room for, and a margin."""
    return max(context - estimate_tokens(instruction) - estimate_tokens(TRANSCRIPT_LEAD) - target_tokens - TRANSCRIPT_MARGIN, 0)


@dataclass(frozen=True)
class CompactHistory(Event):
    """Replace the window since the last summary with a summary turn. Rewrites history only —
    a pending turn, empty or mid-loop, is left as it is. Schedules no successor: the caller
    sequences what follows (NextRound retries the request, /compact returns to the loop head),
    which is what lets one event serve both."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        s = state.settings
        instruction = s.compaction_prompt     # a setting, so a run can be built with another (see COMPACTION_PROMPT)
        target_tokens = int(s.context * s.compaction_target)
        # The transcript is fitted to what the window leaves next to the instruction and the
        # summary (ChatHistory.transcript): a request larger than the context would fail outright.
        transcript = state.history.transcript(transcript_budget(s.context, instruction, target_tokens))
        gen_budget = int(min(target_tokens, s.context - estimate_tokens(instruction) - estimate_result_tokens(transcript), s.turn_token_cap * s.context))
        if gen_budget < s.min_compaction_tokens:
            print(c_out(Palette.WARNING, f"Compaction is tight on room ({gen_budget} tokens computed, context={s.context}) — forcing {s.min_compaction_tokens} and the summary may come out truncated."))
            gen_budget = s.min_compaction_tokens
        req = Request(
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": f"{TRANSCRIPT_LEAD}{transcript}{SUMMARY_CLOSE}" }],
            model=state.settings.model,
            temperature=0.0,
            max_tokens=gen_budget,
            think=False,
            stream=False
        )
        attempts, summary, notes = complete_summary(state, req)
        completion = attempts[-1][1]
        digested = not summary
        if digested:
            # Twice nothing: the window must not fold into nothing, so the digest stands in.
            summary = state.history.digest(target_tokens)
            notes.append(Warn("Still no summary; the window is folded with a digest of its turns instead."))
        # The summary turn is a fresh prompt fragment, not the one this usage measured: only the
        # generated summary has a real count; the wrapper text around it is priced by heuristic.
        summary_tokens = 0
        if not digested and completion.usage and completion.usage.get("completion_tokens"):
            summary_tokens = completion.usage["completion_tokens"] + estimate_tokens(ChatHistory.SUMMARY_PREFIX + ChatHistory.SUMMARY_ACK)
        return replace(state, history=state.history.compact(summary, tokens=summary_tokens)), notes + [Info(f"{summary}")] + [
            LogCompletion(request=r, completion=c, port=state.inference.port) for r, c in attempts] + persist(state)


@dataclass(frozen=True)
class CompactPendingTurn(Event):
    """Checkpoint the pending turn: fold every round of its view but the last into one summary
    round (PendingTurn.compact). The second rung of NextRound's ladder, taken when the history has
    nothing left to fold and the turn itself is what fills the window. Rewrites pending only —
    history is left as it is — and, like CompactHistory, schedules no successor."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        pending = state.pending
        assert pending is not None and pending.user is not None
        s = state.settings
        instruction = s.checkpoint_prompt      # its own instruction and share of the context (see CHECKPOINT_PROMPT)
        target_tokens = int(s.context * s.checkpoint_target)
        # The rounds render as the model last saw them (the request's bands) and are fitted to what
        # the window leaves next to the instruction and the summary (PendingTurn.transcript).
        transcript = pending.transcript(pending.since_last_summary()[:-1], state.expire_after(),
                                        transcript_budget(s.context, instruction, target_tokens))
        gen_budget = int(min(target_tokens, s.context - estimate_tokens(instruction) - estimate_result_tokens(transcript), s.turn_token_cap * s.context))
        if gen_budget < s.min_compaction_tokens:
            print(c_out(Palette.WARNING, f"Checkpoint is tight on room ({gen_budget} tokens computed, context={s.context}) — forcing {s.min_compaction_tokens} and the summary may come out truncated."))
            gen_budget = s.min_compaction_tokens
        req = Request(
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": f"{TRANSCRIPT_LEAD}{transcript}{CHECKPOINT_CLOSE}"}],
            model=s.model,
            temperature=0.0,
            max_tokens=gen_budget,
            think=False,
            stream=False
        )
        attempts, summary, notes = complete_summary(state, req)
        completion = attempts[-1][1]
        digested = not summary
        if digested:
            # Same guard as CompactHistory: the digest keeps the previous checkpoint and names what
            # each folded round did, so the model keeps at least the shape of its own work.
            summary = pending.digest(pending.since_last_summary()[:-1], describe=state.tools.target, budget_tokens=target_tokens)
            notes.append(Warn("Still no checkpoint; the rounds are folded with a digest of their calls instead."))
        # As for a summary turn: the generated text has a real count, the prefix around it is heuristic.
        summary_tokens = 0
        if not digested and completion.usage and completion.usage.get("completion_tokens"):
            summary_tokens = completion.usage["completion_tokens"] + estimate_tokens(CHECKPOINT_PREFIX)
        return replace(state, pending=pending.compact(summary, tokens=summary_tokens)), notes + [Info(f"{summary}")] + [
            LogCompletion(request=r, completion=c, port=state.inference.port) for r, c in attempts]


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

    Compaction is decided here because this is the only point that knows what the request needs.
    When the window leaves less than min_gen_tokens for the completion, the ladder is climbed one
    rung and the request rebuilt: history first, while it holds anything but a summary (the colder
    content, and the turn is the live task); then the pending turn, while its view holds two rounds
    the model made (one to fold, the last kept whole). No flag counts the rungs: each compaction
    removes the condition that allowed it — a summarised window is not summarisable, a checkpointed
    view holds one model round — so the ladder ends by itself. What still does not fit ends the
    turn as an overflow, recorded as a hidden turn so the loop head can see it (a dropped
    message would be issued again by an auto prompt, forever)."""
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        assert state.pending is not None and state.pending.user is not None
        pending = state.pending
        sys_prompt_tokens = estimate_tokens(state.system_prompt)
        # What the pending turn costs in the prompt: rounds already priced by usage frames, plus the
        # heuristic for the text no frame has priced yet (user message on round one, latest results after).
        pending_tokens = state.pending_tokens()
        gen_room = state.gen_room(pending_tokens)
        if gen_room < state.min_gen_tokens():
            non_compacted_history = any(not t.summary for t in state.history.since_last_summary())
            if non_compacted_history:
                return state, [Info("Compacting conversation history..."), CompactHistory(), NextRound()]
            non_compacted_pending = sum(1 for r in pending.since_last_summary() if not r.summary) >= 2
            if non_compacted_pending:
                return state, [Info("Compacting pending turn..."), CompactPendingTurn(), NextRound()]
            # The request never goes out, so no usage frame prices it: these estimates are the only
            # record of how far the turn grew (the eval grader reads them from the log).
            return state, [Error(f"Request exceeds context window; ending the turn "
                                 f"(prompt≈{state.prompt_tokens(pending_tokens)} room={gen_room} need={state.min_gen_tokens()})."),
                           TurnEnd(assistant="", tokens=0, stop=StopReason.OVERFLOW)]
        gen_budget = state.gen_budget(pending_tokens)
        # The scratchpad block goes LAST: it changes whenever the model writes, and everything before
        # it is a stable prefix the server can keep cached. It is not part of the turn — pending
        # and history never hold it — so it is priced here as the current value and counted as
        # prior, the way the system prompt is, and never as the round's own text. It also carries
        # the expiring line: the block is re-sent every request anyway, so announcing there which
        # rounds lose their results next costs no cache, where a note inside the round would.
        #
        # Tool results expire by round distance (PendingTurn.stubbed): the pending turn renders its
        # bands for this request, history turns are always stubbed (Turn.messages). The pending
        # turn is never left out of the request, however many rounds it holds — only its results age.
        k = state.expire_after()
        block = state.scratchpad_block()
        scratchpad_tokens = state.scratchpad_tokens()
        prior = sys_prompt_tokens + state.tools_tokens() + scratchpad_tokens     # re-sent whole every request, never a round's own text
        reserved = prior + pending_tokens + gen_budget
        view = state.history.view_turns(state.settings.context - reserved)
        return state, [StreamCompletion(
            request=Request(
                messages=([{"role": "system", "content": state.system_prompt}]
                          + [m for t in view for m in t.messages()]
                          + pending.messages(expire_after=k)
                          + ([block] if block is not None else [])),
                model=state.settings.model,
                temperature=state.settings.temperature,
                max_tokens=gen_budget,
                think=state.settings.think,
                stream=True,
                tools=state.tools.schemas(),
                ),
            prior_tokens=prior + sum(t.tokens for t in view) + pending.priced_tokens(k),
            unpriced=pending.unpriced_text(),
        )]
