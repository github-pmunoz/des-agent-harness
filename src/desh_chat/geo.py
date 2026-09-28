"""
The geo plugin: a layout task and the tools a layout agent works it with, over the desgeo library.

A task (GeoTask, a JSON file) is a frame, named layers with display colours, optional input layers
and a hidden target, both written as DSL programs, and the output layers the agent must produce.
The agent answers with DSL programs through geo_submit; the last submission is its answer. What
else it gets is the arm, chosen by which tools are registered (--geo-tools), never by the prompt:

    geo_submit        always: compile + evaluate a program, graded against the target at the
                      feedback level the run sets (--geo-feedback): iou, or mismatch (plus the
                      missing and extra parts as rects)
    geo_render        vision: a PNG of the target, the current submission or their diff, with a
                      zoom window, the run's rulers and optional shape-id labels
    geo_measure       instruments: a snapping ruler between two coarse points; it persists and
                      shows on later renders
    geo_auto_measure  instruments: width or space along x and y from one point
    geo_inspect       an arm of its own, since it gives geometry away: the exact shapes at a point

Instruments read the target by default (it is what the agent must perceive) or the current
submission. The grid and tick overlays are the operator's, per run (--geo-grid, --geo-ticks);
the model cannot turn them on. --geo-ruler-bias k is the causal-audit arm: every length an
instrument reports is off by k, so a final answer that follows the instrument can be told from
one that ignores it.

Everything the run produces lands in the session's out dir: renders (render-NNN-view.png, never
rewritten, since the request cites them by path) and submissions.jsonl, one line per geo_submit.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from desgeo import DslError, GeometryError, Layout, Metrology, RasterEngine, Rect, Ruler, render, render_diff, run
from desh.tools import ToolOutput, ToolRegistry

GEO_TOOLS = ("render", "measure", "auto_measure", "inspect")
FEEDBACK = ("iou", "mismatch")
MAX_MISMATCH_RECTS = 8

GEO_PROMPT = """You are working on a 2D layout of {width} x {height} layout units (1 unit = {resolution} nm). The origin (0, 0) is the bottom-left corner; x grows to the right and y grows upward. All geometry is axis-aligned on the integer grid.
Layers: {layers}.{inputs}
Your answer is a program in the layout language, submitted with geo_submit. It must assign the output layer{s} {outputs}. Submit whenever you want feedback; the last submission is your answer.

The layout language: one statement per line, `name = expression`. Only these forms exist:
  w = 40                          numbers: integers with + - * / (a division must be exact)
  a = rect(x, y, w, h)            lower-left corner (x, y), width w, height h
  p = poly((x, y), [(dx, dy), ...])   a rectilinear outline: a start point and axis-aligned steps, closing back to the start
  c = a | b                       union      c = a & b    intersection
  c = a - b                       difference c = a ^ b    xor
  d = size(a, 5)                  grow every edge outward by 5 (negative: shrink); size(a, dx, dy) per axis
  e = move(a, dx, dy)             translate
  f = scale(a, 2) or scale(a, 3, 2)   scale about the origin by an integer or a fraction
A layer name{input_note} is a region like any other name; assigning an output layer name writes that layer. Prefer the shortest program that is exact: fewer operations is better."""


@dataclass(frozen=True)
class GeoTask:
    id: str
    width: int
    height: int
    layers: dict[str, tuple[int, int, int]]
    target: str                             # DSL program assigning every output layer
    outputs: tuple[str, ...]
    inputs: str = ""                        # DSL program assigning the input layers
    resolution: float = 0.1
    prompt: str = ""

    @classmethod
    def load(cls, path: str) -> GeoTask:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            d = json.load(f)
        base = os.path.dirname(os.path.abspath(path))

        def text(v):                        # a program inline, or "@file" relative to the task
            if isinstance(v, str) and v.startswith("@"):
                with open(os.path.join(base, v[1:]), encoding="utf-8") as f:
                    return f.read()
            return v or ""

        return cls(id=d.get("id", os.path.splitext(os.path.basename(path))[0]), width=d["width"],
                   height=d["height"], layers={k: tuple(v) for k, v in d["layers"].items()},
                   target=text(d["target"]), outputs=tuple(d["outputs"]), inputs=text(d.get("inputs")),
                   resolution=d.get("resolution", 0.1), prompt=text(d.get("prompt")))

    def input_layers(self) -> tuple[str, ...]:
        return tuple(n for n in self.layers if n not in self.outputs and self.inputs and n in _assigned(self.inputs))

    def prompt_text(self) -> str:
        ins = self.input_layers()
        return GEO_PROMPT.format(
            width=self.width, height=self.height, resolution=self.resolution,
            layers=", ".join(self.layers),
            inputs=f" Input layer{'s' if len(ins) > 1 else ''} {', '.join(ins)} {'are' if len(ins) > 1 else 'is'} given and can be read by name." if ins else "",
            s="s" if len(self.outputs) > 1 else "", outputs=", ".join(self.outputs),
            input_note=" (input or output)" if ins else "")


@dataclass
class GeoSettings:
    feedback: str = "mismatch"
    snap: int = 15                          # snapping radius of the ruler, in layout units
    image_px: int = 800
    grid: int = 0                           # overlay spacing in units; 0 = none (the operator's choice)
    ticks: int = 0                          # labelled tick spacing in units; 0 = none
    ruler_bias: int = 0                     # causal-audit arm: added to every reported length


@dataclass
class GeoSession:
    """One task being worked: the frame with its inputs, the hidden target, the latest submission,
    the rulers placed so far. Mutable like a workspace on disk: the harness records the RESULTS
    of its tools, and the session is where those results come from."""
    task: GeoTask
    out_dir: str
    settings: GeoSettings = field(default_factory=GeoSettings)
    rulers: list[Ruler] = field(default_factory=list)
    current: Layout | None = None
    submissions: int = 0
    renders: int = 0

    def __post_init__(self):
        t = self.task
        blank = Layout(t.width, t.height, t.resolution)
        for name, colour in t.layers.items():
            blank.layer(name, colour)
        self.base = run(t.inputs, blank, outputs=self.task.input_layers()).layout if t.inputs else blank
        self.target = run(t.target, self.base, outputs=t.outputs).layout
        self.engine = RasterEngine(t.width, t.height)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "task.json"), "w", encoding="utf-8") as f:
            json.dump({**t.__dict__, "layers": {k: list(v) for k, v in t.layers.items()}}, f, indent=2)
        self._metrology: dict[str, Metrology] = {}

    # ---- tools ---------------------------------------------------------------------------------

    def submit(self, program: str) -> str:
        """Run a layout program and compare its output layers with the target. The last submission is your answer.

        Args:
            program: The whole program, one `name = expression` statement per line.
        """
        self.submissions += 1
        n = self.submissions
        record: dict = {"n": n, "time": time.time(), "program": program}
        try:
            res = run(program, self.base, outputs=self.task.outputs)
        except DslError as e:
            record.update(ok=False, error=str(e))
            self._log(record)
            return f"submission {n} rejected: {e}"
        self.current = res.layout
        self._metrology.pop("current", None)
        lines, layers, exact = [], {}, True
        for name in self.task.outputs:
            t = self.engine.region(self.target.layers[name].shapes)
            c = res.regions[name]
            missing, extra = self.engine.subtract(t, c), self.engine.subtract(c, t)
            m, x = self.engine.area(missing), self.engine.area(extra)
            union = self.engine.area(self.engine.union(t, c))
            iou = 1.0 if union == 0 else self.engine.area(self.engine.intersect(t, c)) / union
            layers[name] = {"iou": round(iou, 6), "missing": m, "extra": x, "exact": m == x == 0}
            exact &= m == x == 0
            if m == x == 0:
                lines.append(f"{name}: exact match")
                continue
            line = f"{name}: IoU {iou:.3f}, area {self.engine.area(c)} vs target {self.engine.area(t)}"
            if self.settings.feedback == "mismatch":
                line += (f"; missing {m}{_rect_list(self.engine.rects(missing))}"
                         f"; extra {x}{_rect_list(self.engine.rects(extra))}")
            lines.append(line)
        p = res.program
        tail = f"cost {p.cost} operation{'s' if p.cost != 1 else ''}" + (f"; unused statements: {', '.join(p.dead)}" if p.dead else "")
        record.update(ok=True, exact=exact, layers=layers, cost=p.cost, dead=p.dead, canonical=p.canonical())
        self._log(record)
        return f"submission {n}: " + ("ALL EXACT. " if exact else "") + "\n".join(lines) + f"\n{tail}"

    def render_view(self, view: Literal["target", "current", "diff"] = "target", window: list[int] | None = None,
                    labels: bool = False, layer: str = "") -> ToolOutput | str:
        """Render an image of the target layout, of your current submission, or of the diff between them.

        Args:
            view: target, current (your last submission) or diff (grey: both, orange: missing from yours, blue: extra in yours).
            window: Optional zoom [x0, y0, x1, y1] in layout units; omit for the whole layout.
            labels: Tag every shape with its layer and shape number, as the measuring tools name them.
            layer: For diff with several output layers, the layer to compare (default: the first output).
        """
        if view != "target" and self.current is None:
            return "Nothing submitted yet: geo_submit a program first."
        win = tuple(window) if window else None
        if win is not None and len(win) != 4:
            return "window must be [x0, y0, x1, y1]"
        s = self.settings
        rulers = [((r.a.x, r.a.y), (r.b.x, r.b.y), self._label(r)) for r in self.rulers]
        overlay = dict(window=win, image_px=s.image_px, grid=s.grid or None, ticks=s.ticks or None, rulers=rulers)
        try:
            if view == "diff":
                name = layer or self.task.outputs[0]
                if name not in self.task.outputs:
                    return f"no output layer {name!r}; outputs are {', '.join(self.task.outputs)}"
                im = render_diff(self.task.width, self.task.height, self.target.layers[name].shapes,
                                 self.current.layers[name].shapes, **overlay,
                                 labels=self._metro("target").labels() if labels else ())
            else:
                lay = self.target if view == "target" else self.current
                im = render(lay, **overlay, labels=self._metro(view).labels() if labels else ())
        except GeometryError as e:
            return str(e)
        self.renders += 1
        path = os.path.join(self.out_dir, f"render-{self.renders:03d}-{view}.png")
        im.save(path)
        x0, y0, x1, y1 = win or (0, 0, self.task.width, self.task.height)
        return ToolOutput(f"{view} rendered: x {x0}..{x1}, y {y0}..{y1}, {im.size[0]} x {im.size[1]} px"
                          + (f", {len(rulers)} ruler(s)" if rulers else ""), (path,))

    def measure(self, x1: int, y1: int, x2: int, y2: int, on: Literal["target", "current"] = "target",
                layer: str = "") -> str:
        """Place a ruler between two points; each end snaps to the nearest vertex or edge within range, so point roughly and read the exact numbers.

        Args:
            x1: x of the first point, in layout units.
            y1: y of the first point.
            x2: x of the second point.
            y2: y of the second point.
            on: Measure the target or your current submission.
            layer: Snap only to this layer (default: any layer).
        """
        m = self._metro(on)
        if isinstance(m, str):
            return m
        r = m.measure(x1, y1, x2, y2, self.settings.snap, layer or None)
        self.rulers.append(r)
        k = self.settings.ruler_bias
        bx = r.b.x + (k if r.b.x >= r.a.x else -k) if r.dx else r.b.x
        by = r.b.y + (k if r.b.y >= r.a.y else -k) if r.dy else r.b.y
        dx, dy = abs(bx - r.a.x), abs(by - r.a.y)
        lines = [f"ruler {len(self.rulers)} on {on}:",
                 f"  from {r.a.describe()}",
                 f"  to   {_describe_at(r.b, bx, by)}",
                 f"  dx = {dx}, dy = {dy}" + (f", distance = {(dx * dx + dy * dy) ** 0.5:.1f}" if dx and dy else "")]
        if r.others:
            lines.append("  ambiguous, also in range: " + "; ".join(o.describe() for o in r.others))
        return "\n".join(lines)

    def auto_measure(self, x: int, y: int, on: Literal["target", "current"] = "target", layer: str = "") -> str:
        """From one point, measure along x and y to the nearest edges: inside a shape this is its width and height there, in a gap the space to the neighbouring shapes.

        Args:
            x: x of the point, in layout units.
            y: y of the point.
            on: Measure the target or your current submission.
            layer: The layer to measure (default: the first output layer).
        """
        m = self._metro(on)
        if isinstance(m, str):
            return m
        name = layer or self.task.outputs[0]
        try:
            a = m.auto_measure(x, y, name)
        except (KeyError, ValueError) as e:
            return f"cannot measure: {e}"
        k = self.settings.ruler_bias
        where = f"inside {name} shape {a['shape']}" if a["inside"] else f"in a gap on {name}"
        lines = [f"({x}, {y}) is {where} on {on}:"]
        for axis in ("x", "y"):
            s = a[axis]
            what = ("width" if axis == "x" else "height") if a["inside"] else "space"
            text = f"  along {axis}: {what} {s['length'] + k} ({axis} {s['from']}..{s['to'] + k})"
            if not a["inside"]:
                ends = [f"shape {i}" if i else "nothing" for i in s["between"]]
                text += f", between {ends[0]} and {ends[1]}"
                if s["open"]:
                    text += f"; reaches the frame on the {' and '.join(s['open'])} side"
            lines.append(text)
        return "\n".join(lines)

    def inspect(self, x: int, y: int, on: Literal["target", "current"] = "target") -> str:
        """The exact geometry of every shape under a point: layer, shape number, bounding box, area, and the rectangles it is made of.

        Args:
            x: x of the point, in layout units.
            y: y of the point.
            on: Inspect the target or your current submission.
        """
        m = self._metro(on)
        if isinstance(m, str):
            return m
        hits = m.inspect(x, y)
        if not hits:
            return f"nothing at ({x}, {y}) on {on}"
        out = []
        for h in hits:
            b = h["bbox"]
            out.append(f"{h['layer']} shape {h['shape']}: bbox [{b.x}, {b.y}, {b.w}, {b.h}], area {h['area']}, "
                       f"{h['vertices']} vertices, rects{_rect_list(h['rects'], cap=16)}")
        return "\n".join(out)

    # ---- plumbing ------------------------------------------------------------------------------

    def _metro(self, on: str) -> Metrology | str:
        if on == "current" and self.current is None:
            return "Nothing submitted yet: geo_submit a program first."
        if on not in self._metrology:
            lay = self.target if on == "target" else self.current
            self._metrology[on] = Metrology(lay)
        return self._metrology[on]

    def _label(self, r: Ruler) -> str:
        k = self.settings.ruler_bias
        dx, dy = r.dx + (k if r.dx else 0), r.dy + (k if r.dy else 0)
        return str(dx) if not dy else str(dy) if not dx else f"{dx},{dy}"

    def _log(self, record: dict) -> None:
        with open(os.path.join(self.out_dir, "submissions.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def register(self, tools: ToolRegistry, arms: tuple[str, ...]) -> ToolRegistry:
        """The registry with geo_submit and the arm's tools. None asks the operator: they write only
        the session's own out dir. A submission is the agent acting on what it saw, so it resets
        the re-read count of the looking tools."""
        unknown = [a for a in arms if a not in GEO_TOOLS]
        if unknown:
            raise ValueError(f"unknown geo tool {', '.join(unknown)}; one of {', '.join(GEO_TOOLS)}")
        tools = tools.add(self.submit, name="geo_submit", confirm=False, acts=True, target="program")
        methods = {"render": (self.render_view, "view"), "measure": (self.measure, ""),
                   "auto_measure": (self.auto_measure, ""), "inspect": (self.inspect, "")}
        for arm in GEO_TOOLS:
            if arm in arms:
                fn, target = methods[arm]
                tools = tools.add(fn, name=f"geo_{arm}", confirm=False, target=target)
        return tools


def _assigned(program: str) -> set[str]:
    import ast
    try:
        return {t.id for s in ast.parse(program).body if isinstance(s, (ast.Assign, ast.AugAssign))
                for t in (s.targets if isinstance(s, ast.Assign) else [s.target]) if isinstance(t, ast.Name)}
    except SyntaxError:
        return set()


def _rect_list(rects: list[Rect], cap: int = MAX_MISMATCH_RECTS) -> str:
    if not rects:
        return ""
    shown = ", ".join(f"[{r.x}, {r.y}, {r.w}, {r.h}]" for r in rects[:cap])
    more = f" and {len(rects) - cap} more" if len(rects) > cap else ""
    return f" in {len(rects)} rect{'s' if len(rects) > 1 else ''} [x, y, w, h]: {shown}{more}"


def _describe_at(s, x: int, y: int) -> str:
    """A snap described at the (possibly biased) position the ruler reports."""
    d = s.describe()
    return d.replace(f"({s.x}, {s.y})", f"({x}, {y})", 1) if (x, y) != (s.x, s.y) else d


def session_from_args(args) -> GeoSession | None:
    """The geo session a run's flags ask for, or None without --geo."""
    path = getattr(args, "geo", "")
    if not path:
        return None
    task = GeoTask.load(path)
    out = args.geo_out or os.path.join(args.workspace, ".desh", "geo",
                                       f"{time.strftime('%Y%m%d-%H%M%S')}-{task.id}")
    if args.geo_feedback not in FEEDBACK:
        raise SystemExit(f"--geo-feedback must be one of {', '.join(FEEDBACK)}")
    settings = GeoSettings(feedback=args.geo_feedback, snap=args.geo_snap, image_px=args.geo_image_px,
                           grid=args.geo_grid, ticks=args.geo_ticks, ruler_bias=args.geo_ruler_bias)
    return GeoSession(task, os.path.abspath(os.path.expanduser(out)), settings)


def arms_of(args) -> tuple[str, ...]:
    return tuple(a.strip() for a in (args.geo_tools or "").split(",") if a.strip())
