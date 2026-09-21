"""
Session persistence: the history goes to and comes from state.session_file. Both events are
high priority — LoadSession must run before the first prompt, SaveSession right after the history
change it follows. Depends on display only.
"""
import json
import os
import readline
import time
from dataclasses import dataclass, fields, replace

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
                session_file_content = json.load(f)
            history = ChatHistory.from_dict(session_file_content)
            for turn in history.turns:
                if not turn.summary:
                    readline.add_history(turn.user)
        except FileNotFoundError:
            return state, [Info(c_out(Palette.DIM_CHROME, f"New session: {path}")), SaveSession()]
        except (ValueError, KeyError, TypeError) as e:     # ValueError covers json.JSONDecodeError
            bad = path + ".bad"
            os.replace(path, bad)
            return state, [Warn(f"Session file {path} is unreadable ({e}); moved to {bad}, starting fresh.")]
        # The working memory the file carries is, per slot, the newest snapshot on a turn
        # (history.last_memory); what the run starts with is state.memory: every registered memory,
        # empty. A registered slot takes the snapshot when the file has one and stays empty
        # otherwise; a slot the file carries but the run did not register is not loaded.
        memory = state.memory.restored(history.last_memory)
        # format 4: the session_file_contentument's settings seed the restored settings (v1-3 files carry no
        # "settings" key and load unchanged); settings turns then replay in order on top, so
        # the last change wins. A delta naming a setting that no longer exists (a file written
        # by an older version) is left out: the turn stays on the record, the setting has no effect.
        settings = Settings.from_dict(session_file_content["settings"]) if "settings" in session_file_content else state.settings
        known = {f.name for f in fields(Settings)}
        for turn in history.turns:
            if turn.type == "settings" and turn.delta is not None:
                for setting, value in turn.delta.items():
                    if setting in known:
                        settings = replace(settings, **{setting: value})
        return replace(state, history=history, memory=memory, settings=settings), [Info(c_out(Palette.DIM_CHROME, f"Restored {len(history)} turns from {path}")), DisplayStats()]


@dataclass(frozen=True)
class SaveSession(Event):
    """Write the whole history to state.session_file. Atomic: temp file + os.replace, so a crash
    mid-write can never leave a truncated session behind. No-op without a session file."""
    priority: int = Priority.HIGH   # persist right after the history change, before the next prompt

    def execute(self, state: ChatState) -> tuple[ChatState, list[Event]]:
        path = state.session_file
        if path is None:
            return state, []
        # format 4: the session_file_contentument's "settings" key is the INITIAL seed — the settings in force
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
        session_file_content = {
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
                json.dump(session_file_content, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            return state, [Error(f"Could not save session to {path}: {e}")]
        return state, []


def persist(state: ChatState) -> list[Event]:
    """The events a history-changing step appends so the session file tracks the change."""
    return [SaveSession()] if state.session_file is not None else []
