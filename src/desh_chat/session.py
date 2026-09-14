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
from desh_chat.state import ChatState, ChatHistory, Settings
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
                doc = json.load(f)
            history = ChatHistory.from_dict(doc)
            for turn in history.turns:
                if not turn.summary:
                    readline.add_history(turn.user)
        except FileNotFoundError:
            return state, [Info(c_out(Palette.DIM_CHROME, f"New session: {path}")), SaveSession()]
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
        # format 4: the document's settings seed the restored settings (v1-3 files carry no
        # "settings" key and load unchanged); settings turns then replay in order on top, so
        # the last change wins.
        settings = Settings.from_dict(doc["settings"]) if "settings" in doc else state.settings
        for turn in history.turns:
            if turn.type == "settings" and turn.delta is not None:
                for setting, value in turn.delta.items():
                    settings = replace(settings, **{setting: value})
        return replace(state, history=history, scratchpad=scratchpad, settings=settings), [Info(c_out(Palette.DIM_CHROME, f"Restored {len(history)} turns from {path}")), DisplayStats()]


@dataclass(frozen=True)
class SaveSession(Event):
    """Write the whole history to state.session_file. Atomic: temp file + os.replace, so a crash
    mid-write can never leave a truncated session behind. No-op without a session file."""
    priority: int = Priority.HIGH   # persist right after the history change, before the next prompt

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        path = state.session_file
        if path is None:
            return state, []
        # format 4: the document's "settings" key is the INITIAL seed — the settings in force
        # when the file was first created. A later save must not overwrite it: LoadSession
        # starts from the seed and replays the settings turns on top (last change wins).
        seed = state.settings.to_dict()
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "settings" in existing:
                seed = existing["settings"]
        except (OSError, ValueError, TypeError):
            pass   # no file yet, or unreadable/corrupt: fall back to the current settings
        doc = {
            **state.history.to_dict(),
            "settings": seed,
            # informational only — LoadSession restores turns and settings
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
