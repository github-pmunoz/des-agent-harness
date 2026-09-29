"""
The markup memory: the layout agent's geometric working memory, as a memory plugin.

What an operator of a graphical EDA tool draws while reading a layout: named points, segments and
boxes, in layout units. They are the agent's own claims (a corner it established, an edge it
suspects), not geometry: never measured, never snapped, never submitted. They live in the <markup>
section of the memory block, so they are listed in every request and outlive the turn, and
geo_render draws them over whatever it renders (it injects the slot to read it; it is not a
markup tool, Memories.owns).

    markup_point(name, x, y, note)            a named point
    markup_segment(name, x1, y1, x2, y2, note) a named segment
    markup_box(name, x, y, w, h, note)        a named box, (x, y) its corner with the smallest coordinates
    markup_delete(name), markup_clear()

A name is unique: placing a markup under a name that exists replaces it (it moves to the end).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from desh_chat.memory import Memory

MARKUP_PROMPT = (
    "The <markup> section is your layout working memory: named points, segments and boxes that you "
    "place in layout units with markup_point, markup_segment and markup_box. Use them to pin down "
    "corners and edges you have established, and to test a hypothesis against the picture: "
    "geo_render draws every markup in red over whatever it renders. The section lists them, so they "
    "outlive the turn. Markups are yours, not geometry: they are never measured, snapped or submitted. "
    "Reusing a name moves that markup."
)


@dataclass(frozen=True)
class Mark:
    name: str
    kind: str                       # point | segment | box
    coords: tuple[int, ...]         # (x, y) | (x1, y1, x2, y2) | (x, y, w, h)
    note: str = ""

    def describe(self) -> str:
        c = self.coords
        if self.kind == "point":
            where = f"point ({c[0]}, {c[1]})"
        elif self.kind == "segment":
            where = f"segment ({c[0]}, {c[1]})-({c[2]}, {c[3]})"
        else:
            where = f"box [{c[0]}, {c[1]}, {c[2]}, {c[3]}] (x {c[0]}..{c[0] + c[2]}, y {c[1]}..{c[1] + c[3]})"
        return f"{self.name}: {where}" + (f" — {self.note}" if self.note else "")


@dataclass(frozen=True)
class Markups:
    marks: tuple[Mark, ...] = ()

    def with_mark(self, mark: Mark) -> Markups:
        return replace(self, marks=tuple(m for m in self.marks if m.name != mark.name) + (mark,))

    @classmethod
    def from_dict(cls, d: dict) -> Markups:
        return cls(tuple(Mark(name, v["kind"], tuple(v["coords"]), v.get("note", "")) for name, v in d.items()))

    def to_dict(self) -> dict[str, dict]:
        return {m.name: {"kind": m.kind, "coords": list(m.coords), "note": m.note} for m in self.marks}

    def render(self) -> str:
        return "\n".join(m.describe() for m in self.marks)

    def overlay(self) -> list[tuple[str, str, tuple[int, ...]]]:
        """What desgeo.render draws: (kind, name, coords) per markup."""
        return [(m.kind, m.name, m.coords) for m in self.marks]


def _place(markup: dict, name: str, kind: str, coords: tuple[int, ...], note: str) -> str:
    if not name.strip():
        return "a markup needs a name"
    if kind == "box" and (coords[2] <= 0 or coords[3] <= 0):
        return f"box width and height must be positive, got {coords[2]} x {coords[3]}"
    moved = name in markup
    markup.pop(name, None)
    markup[name] = {"kind": kind, "coords": list(coords), "note": note}
    return ("moved " if moved else "placed ") + Mark(name, kind, coords, note).describe()


def point(name: str, x: int, y: int, markup: dict, note: str = "") -> str:
    """Place a named point on your markup layer.

    Args:
        name: A short name, e.g. A or top_left.
        x: x in layout units.
        y: y in layout units.
        note: What the point stands for (optional).
    """
    return _place(markup, name, "point", (x, y), note)


def segment(name: str, x1: int, y1: int, x2: int, y2: int, markup: dict, note: str = "") -> str:
    """Place a named segment on your markup layer.

    Args:
        name: A short name.
        x1: x of the first end, in layout units.
        y1: y of the first end.
        x2: x of the second end.
        y2: y of the second end.
        note: What the segment stands for (optional).
    """
    return _place(markup, name, "segment", (x1, y1, x2, y2), note)


def box(name: str, x: int, y: int, w: int, h: int, markup: dict, note: str = "") -> str:
    """Place a named box on your markup layer, given like a rect: corner (x, y) with the smallest coordinates, width, height.

    Args:
        name: A short name.
        x: x of the corner with the smallest coordinates, in layout units.
        y: y of that corner.
        w: Width.
        h: Height.
        note: What the box stands for (optional).
    """
    return _place(markup, name, "box", (x, y, w, h), note)


def delete(name: str, markup: dict) -> str:
    """Remove a markup by name.

    Args:
        name: The markup to remove.
    """
    return f"{name!r} removed" if markup.pop(name, None) is not None else f"{name!r} not found"


def clear(markup: dict) -> str:
    """Remove every markup."""
    n = len(markup)
    markup.clear()
    return f"cleared {n} markups" if n else "already empty"


MARKUP = Memory(
    name="markup",
    empty=Markups,
    from_dict=Markups.from_dict,
    tools=((point, {"name": "markup_point", "target": "name"}),
           (segment, {"name": "markup_segment", "target": "name"}),
           (box, {"name": "markup_box", "target": "name"}),
           (delete, {"name": "markup_delete", "target": "name"}),
           (clear, {"name": "markup_clear"})),
    prompt=MARKUP_PROMPT,
    subagent="off",
)
