"""
The coding-agent toolset: Read, Write, Edit, Bash — the surface a coding agent is usually given.

Every tool is a method of one Workspace, so the schema the model sees (derived from the method's
signature and docstring) never mentions the root, and no call can reach outside it: paths resolve
against the root and must stay under it, no path component may be a symlink, and commands run with
the root as their working directory. Read is read-only and runs unprompted; Write, Edit and Bash
ask the operator first. Confinement is a property of the tools, not of the harness — the gate
stays generic.

Bash takes a mandatory `reason` BEFORE the command: the model states in words what the command is
for, and the operator sees both at the confirmation prompt. Nothing guarantees the command does what
the reason says, but a stated intent next to a dense command line is far better than the command
alone — and asking for the reason first conditions the command on the intent, not the other way
round. The reason is free text the model can reword at will, so it is NOT part of the call's
identity (Tool.identity): the repetition guard counts the command alone, or a model that re-runs
one command with a fresh reason each time is never caught looping.
"""
from __future__ import annotations

import difflib
import hashlib
import os
import subprocess
import time
from dataclasses import dataclass

from desh.render import Palette, c_out
from desh.tools import DEFAULT_RESULT_CHARS, ToolRegistry


SPILL_DIR = ".desh/out"     # under the root, so Read can reach it; ignored by git


@dataclass(frozen=True)
class Workspace:
    """One project root. The tool methods below are what the model calls.

    result_chars is the cap on one result, the registry's bound made known to the tools that can
    do better than a blind cut: Read returns whole lines up to it and says where to continue, Bash
    saves the whole output to a file under the root and says where it is."""
    root: str
    result_chars: int = DEFAULT_RESULT_CHARS

    def __post_init__(self):
        object.__setattr__(self, "root", os.path.realpath(os.path.expanduser(self.root)))

    def path(self, file_path: str) -> str:
        """Absolute path for a file the model named. ValueError when it would leave the root, or when
        any component under the root is a symlink — links are refused outright, wherever they point."""
        full = os.path.normpath(os.path.join(self.root, os.path.expanduser(file_path)))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ValueError(f"{file_path!r} is outside the project root")
        current = self.root
        for part in os.path.relpath(full, self.root).split(os.sep) if full != self.root else ():
            current = os.path.join(current, part)
            if os.path.islink(current):
                raise ValueError(f"{file_path!r} goes through a symlink ({os.path.relpath(current, self.root)}); symlinks are not allowed")
        return full

    # --- the tools ---

    def read(self, file_path: str, offset: int = 1, limit: int = 0) -> str:
        """Read a text file. Returns its contents, or the lines from `offset` (1-based) when given.

        Args:
            file_path: Path of the file to read, relative to the project root.
            offset: First line to return, 1-based. Default 1 (the start).
            limit: How many lines to return from offset. 0 means all remaining lines.
        """
        with open(self.path(file_path), "r", encoding="utf-8") as f:
            lines = f.read().splitlines(keepends=True)
        start = max(offset, 1) - 1
        end = start + limit if limit > 0 else len(lines)
        wanted = lines[start:end]
        if sum(len(line) for line in wanted) <= self.result_chars:
            return "".join(wanted)
        # The slice is over the cap: return whole lines from `start` that fit within
        # self.result_chars TOGETHER with a trailer line that tells the model how many lines the
        # file has, which lines it is looking at, and the offset to continue from. The whole return
        # value must stay within self.result_chars, or the registry's blind head-and-tail cut fires
        # on top of it and the trailer is lost in the middle.
        def trailer(last: int, cut: bool) -> str:
            note = " — line truncated" if cut else ""
            return (f"\n[showing lines {start + 1}-{last + 1} of {len(lines)} "
                    f"(cap {self.result_chars} chars); continue with offset={last + 2}{note}]")
        if len(wanted[0]) + len(trailer(start, True)) > self.result_chars:
            # The very first line does not fit: cut it by characters so content + trailer fit.
            # Leave a margin below the cap so the registry's blind cut can never fire on top.
            cut = wanted[0][:max(self.result_chars - len(trailer(start, True)) - 100, 0)]
            return cut + trailer(start, True)
        content = ""
        last = start
        for i, line in enumerate(wanted, start=start):
            if len(content) + len(line) + len(trailer(i, False)) > self.result_chars:
                break
            content += line
            last = i
        return content + trailer(last, False)

    def write(self, file_path: str, content: str) -> str:
        """Create or overwrite a text file with the given content, creating parent directories.

        Args:
            file_path: Path of the file to write, relative to the project root.
            content: The full new contents of the file.
        """
        full = self.path(file_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} characters to {file_path}"

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace text in an existing file by exact string match.

        Args:
            file_path: Path of the file to edit, relative to the project root.
            old_string: The exact text to replace. Must match the file verbatim, including indentation.
            new_string: The text that replaces it.
            replace_all: Replace every occurrence. Default False: old_string must occur exactly once.
        """
        # Check if the string is empty
        if not old_string:
            return "old_string cannot be empty."

        if old_string == new_string:
            return "(no change: old_string and new_string are identical)"

        # Check if the file exists
        full = self.path(file_path)
        if not os.path.exists(full):
            return f"File {file_path} does not exist."
        
        # Check how many times the string appears in the file. Read straight from disk: read() is
        # the model's view, cut to the cap, and an edit written back through it would cut the file.
        with open(full, "r", encoding="utf-8") as f:
            file_content = f.read()
        old_string_matches = file_content.count(old_string)
        if old_string_matches == 0:
            return f"`old_string` not found in file."
        if old_string_matches == 1: # replace_all has no effect in this branch
            with open(full, "w", encoding="utf-8") as f:
                f.write(file_content.replace(old_string, new_string))
            return f"One occurrence of `old_string` replaced."
        elif old_string_matches > 1 and not replace_all:
            return f"`old_string` appears {old_string_matches} times in file. Use replace_all=True to replace all occurrences."
        else:
            with open(full, "w", encoding="utf-8") as f:
                f.write(file_content.replace(old_string, new_string, old_string_matches))
            return f"{old_string_matches} occurrences of string `old_string` replaced in file."

    def bash(self, reason: str, command: str, timeout: int = 60) -> str:
        """Run a shell command in the project root and return its output.

        Args:
            reason: One sentence, in plain words, saying what the command does and why. Shown to the user next to the command.
            command: The command line to run with the shell.
            timeout: Seconds to wait before killing it. Default 60.
        """
        try:
            proc = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"command timed out after {timeout}s"
        parts = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip("\n"))
        if proc.stderr:
            parts.append("stderr:\n" + proc.stderr.rstrip("\n"))
        if proc.returncode != 0:
            parts.append(f"exit code {proc.returncode}")
        output = "\n".join(parts) if parts else "(no output)"
        if len(output) <= self.result_chars:
            return output
        # Over the cap: the whole output goes to a file the model can Read by range, and the last
        # line says so. The registry's cut keeps the tail, so the pointer survives whatever it drops.
        spill = self.spill(command, output)
        return output + f"\n[output is {len(output)} characters, cut to {self.result_chars}; the whole of it is saved at {spill} — Read it with offset and limit]"

    def spill(self, command: str, output: str) -> str:
        """Save a command's whole output under SPILL_DIR and return the path relative to the root.
        Named by time and a hash of the command, so two runs of one command do not collide."""
        stamp = time.strftime("%H%M%S") + "-" + hashlib.sha1(f"{time.time_ns()}{command}".encode()).hexdigest()[:6]
        rel = os.path.join(SPILL_DIR, f"bash-{stamp}.txt")
        full = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(output)
        return rel


def edit_preview(args: dict) -> str:
    old = args.get("old_string")
    new = args.get("new_string")
    assert isinstance(old, str) and isinstance(new, str)
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", fromfile="old", tofile="new")
    lines = [c_out(Palette.DIFF_CTX, args.get("file_path", "") or "(no file)")]
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        elif line.startswith("+"):
            lines.append(c_out(Palette.DIFF_ADD, line))
        elif line.startswith("-"):
            lines.append(c_out(Palette.DIFF_DEL, line))
        else:
            lines.append(c_out(Palette.DIFF_CTX, line))
    if args.get("replace_all"):
        lines.append(c_out(Palette.DIFF_CTX, "(replace_all: true)")) 
    return "\n".join(lines)


def coding_registry(root: str = ".", result_chars: int = DEFAULT_RESULT_CHARS) -> ToolRegistry:
    """Read runs unprompted; Write, Edit and Bash ask. A mention of a call names its path, or its command."""
    ws = Workspace(root, result_chars=result_chars)
    return (ToolRegistry(max_result_chars=result_chars)
            .add(ws.read, name="Read", confirm=False, target="file_path")
            .add(ws.write, name="Write", target="file_path")
            .add(ws.edit, name="Edit", preview=edit_preview, target="file_path")
            .add(ws.bash, name="Bash", identity=("command",), target="command"))
