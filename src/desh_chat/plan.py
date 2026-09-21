from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

from desh_chat.memory import Memory

# The work plan, as a pluggable memory: the steps of the task and which are done. It holds no
# knowledge — what a step established goes in its outcome, a line, or in another memory.
STATUSES = ("todo", "done")
Status = Literal["todo", "done"]

PLAN_PROMPT = (
    "The <plan> is your work plan: the steps of the task and which are done. Write the steps with "
    "plan_write before you start on them. A step keeps its key for life: when it is done, write the "
    "SAME key again as done, with its outcome in a line. Never add a second key for a step that "
    "already has one."
)


def write(key: str, status: Status, text: str, plan: dict[str, dict]) -> str:
    """Write a step of the plan, creating a new key or overwriting an existing one.

    Args:
        key: The key of the step.
        status: todo, a step still to do; done, a finished step.
        text: What the step is, or, for a done step, its outcome.
    """
    if status not in STATUSES:
        return f"unknown status {status!r}; one of {', '.join(STATUSES)}"
    before = plan.get(key)
    plan[key] = {"status": status, "text": text}
    if before is None:
        return f"created {key!r} ({status})"
    return f"overwrote {key!r} ({before['status']} -> {status})" if before["status"] != status else f"overwrote {key!r} ({status})"


def fold_write(args: dict) -> dict:
    """The echoed form of an answered plan_write (Tool.fold): key and status, the text gone and its
    size noted under `folded` — removed rather than replaced, as the scratchpad's value is."""
    text = args.get("text")
    if not isinstance(text, str):
        return args
    return {**{k: v for k, v in args.items() if k != "text"}, "folded": f"{len(text)} characters, shown in the plan block"}


def delete(key: str, plan: dict[str, dict]) -> str:
    """Delete a step from the plan.

    Args:
        key: The key of the step to delete.
    """
    if key in plan:
        del plan[key]
        return f"{key!r} deleted"
    return f"{key!r} not found"


@dataclass(frozen=True)
class Step:
    """One step of the plan: the key the model addresses it by, its status and its text."""
    key: str
    status: str
    text: str


@dataclass(frozen=True)
class Plan:
    """The work plan, as a value: ordered steps, one per key. A rewritten step keeps its place, so
    the plan reads in the order it was laid out whatever order the steps were finished in."""
    steps: tuple[Step, ...] = ()

    def with_step(self, key: str, status: str, text: str) -> Plan:
        """Write a step: a new key goes last, an existing one is rewritten where it stands."""
        step = Step(key, status, text)
        if any(s.key == key for s in self.steps):
            return replace(self, steps=tuple(step if s.key == key else s for s in self.steps))
        return replace(self, steps=self.steps + (step,))

    @classmethod
    def from_dict(cls, d: dict[str, dict]) -> Plan:
        return cls(tuple(Step(k, v["status"], v["text"]) for k, v in d.items()))

    def to_dict(self) -> dict[str, dict]:
        """The dict the tools work on and the session file stores: {key: {"status", "text"}}."""
        return {s.key: {"status": s.status, "text": s.text} for s in self.steps}

    def render(self) -> str:
        """One `[x] key: text` line per step, in plan order (MemoryValue.render); "" when empty."""
        return "\n".join(f"[{'x' if s.status == 'done' else ' '}] {s.key}: {s.text}" for s in self.steps)


PLAN = Memory(
    name="plan",
    empty=Plan,
    from_dict=Plan.from_dict,
    tools=((write, {"name": "plan_write", "fold": fold_write, "target": "key"}),
           (delete, {"name": "plan_delete", "target": "key"})),
    prompt=PLAN_PROMPT,
)
