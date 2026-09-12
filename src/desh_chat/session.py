"""
Session persistence: the history goes to and comes from state.session_file. Both events are
high priority — LoadSession must run before the first prompt, SaveSession right after the history
change it follows. Depends on display only.
"""
import json
import os
import readline
import time
from dataclasses import dataclass, replace

from desh.engine import Event, Priority
from desh.render import Palette, c_out
from desh_chat.state import ChatState, ChatHistory
from desh_chat.display import DisplayStats, Error, Info, Warn


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
        # The working memory the file carries is the newest snapshot on a turn (history.last_scratchpad());
        # what the run starts with is state.scratchpad: an empty value when the tool is offered, None
        # when it is not.
        # When the tool is offered, the scratchpad is the one from the last turn that recorded it if any, otherwise stays empty.
        # When the tool is not offered, the scratchpad is not loaded even if the session file carried one.
        scratchpad = state.scratchpad
        if scratchpad is not None: # tool was offered
            loaded_scratchpad = history.last_scratchpad()
            if loaded_scratchpad is not None:
                scratchpad = loaded_scratchpad
        return replace(state, history=history, scratchpad=scratchpad), [Info(c_out(Palette.DIM_CHROME, f"Restored {len(history)} turns from {path}")), DisplayStats()]


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


def persist(state: ChatState) -> list[Event]:
    """The events a history-changing step appends so the session file tracks the change."""
    return [SaveSession()] if state.session_file is not None else []
