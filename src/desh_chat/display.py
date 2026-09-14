"""
Display events: sinks that print and emit nothing. High priority so what a step wants shown
appears before the next step runs. Leaves of the event graph — every other module may import
these, none of these import back.
"""
from dataclasses import dataclass

from desh.engine import Event, Priority
from desh.render import Palette, c_out
from desh.llama.tokens import estimate_tokens
from desh_chat.state import ChatState


class DisplayEvent(Event):
    priority: int = Priority.HIGH


@dataclass(frozen=True)
class Info(DisplayEvent):
    text: str
    colour: str = Palette.CHROME
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(self.colour, self.text))
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
    colour: str = Palette.STATS_LINE
    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        # This event also accounts for tokens during a pending turn
        pending_tokens = state.pending_tokens()
        window_tokens = state.prompt_tokens(pending_tokens)
        print(c_out(self.colour, f"Context: {window_tokens} / {state.settings.context} tokens ({window_tokens/state.settings.context*100.0:.1f}%) \t Session: {state.session_tokens(pending_tokens)}"))
        return state, []


@dataclass(frozen=True)
class DisplayBanner(DisplayEvent):
    banner_format = """DES Chat v0.1
    Server:       http://127.0.0.1:{port}
    Model:        {model}
    Temperature:  {temperature}
    Think mode:   {think}
    Auto mode:    {auto}
    Context:      {context}
    Turn tokens:  {max_turn_tokens}
    Tool rounds:  {max_tool_rounds}
    Compl log:    {completions_log}
    DES log:      {des_log}
    Debug:        {debug}
    Timeout:      {timeout}s
    Session:      {session_file}
    Tools:        {tools}
    Workspace:    {workspace}
    System prompt:{system_prompt_head}"""
    debug: bool
    des_log: str
    timeout: int
    workspace: str
    completions_log: str

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        print(c_out(Palette.CHROME, f"{"═"*50}"))
        print(c_out(Palette.CHROME, self.banner_format.format(
            port=state.inference.port,
            model=state.settings.model,
            temperature=state.settings.temperature,
            think="enabled" if state.settings.think else "disabled",
            auto="enabled" if state.settings.auto else "disabled",
            context=state.settings.context,
            max_turn_tokens=state.settings.max_turn_tokens,
            max_tool_rounds=state.settings.max_tool_rounds,
            completions_log=self.completions_log,
            des_log=self.des_log,
            debug="enabled" if self.debug else "disabled",
            timeout=self.timeout,
            session_file=state.session_file,
            tools=", ".join(t.name for t in state.tools.tools),
            workspace=self.workspace,
            system_prompt_head=state.system_prompt[:40] if state.system_prompt else "none",
        )))
        print(c_out(Palette.CHROME, f"{"═"*50}"))

        return state, []
