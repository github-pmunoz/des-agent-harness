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
the model cannot turn them on. So is the origin (--geo-origin): top-left (y down, as image pixels
count; the default) or bottom-left (y up, the EDA convention). The geometry is the same under both,
and only renders, the prompt's orientation sentence and the words top and bottom follow it. The
agent faces top-left by default: it reads coordinates straight off the image rows, with no
arithmetic (measured: 17-31% fewer tokens at equal accuracy). Where layouts come from or go to an
EDA tool, the interface between the agent and the tool translates, not the agent. Every
render result states how its pixels map to layout units. The polygon syntax is the operator's
too (--geo-poly): steps from a start point (deltas, the OASIS form) or corners in order
(points). Only the agent's submissions follow it; the task's own programs are always steps.

--geo-ruler-bias k is the causal-audit arm: the instruments read the target translated by (k, k),
a miscalibrated probe. Everything they report about the target (coordinates, edges, vertices,
ambiguity alternatives, spans, shape bboxes, the rulers drawn on renders) is off by the same k
and agrees with itself; lengths are true, and reads of the current submission are true. Below the
eye's precision only the grading feedback can reveal the offset, so an answer that follows the
instrument can be told from one that corrects it. (A first version biased the reported lengths
alone; the true coordinates stayed in the edge text, and the model reconciled the contradiction
instead of being tested.)

Everything the run produces lands in the session's out dir: renders (render-NNN-view.png, never
rewritten, since the request cites them by path) and submissions.jsonl, one line per geo_submit.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal

from desgeo import DslError, GeometryError, Layout, Metrology, Polygon, RasterEngine, Rect, Ruler, render, render_diff, run
from desgeo.render import pixel_mapping
from desh_chat.markup import Markups
from desh.tools import ToolOutput, ToolRegistry

GEO_TOOLS = ("render", "measure", "auto_measure", "inspect")
# The display convention (--geo-origin): how y is drawn and named. Geometry is the same under both.
ORIENTATION = {
    "bottom-left": "The origin (0, 0) is the bottom-left corner; x grows to the right and y grows upward.",
    "top-left": "The origin (0, 0) is the top-left corner; x grows to the right and y grows downward.",
}
# How the agent writes a polygon (--geo-poly), and how its variables are counted in those terms.
# Both forms compile to the same polygon and the same costs (2n + 1 variables for n corners);
# the steps form is how OASIS stores it, the corners form is how the agent reads it off a picture.
POLY_FORMS = {
    "deltas": ("  p = poly((x, y), [(dx, dy), ...])   a rectilinear outline: a start point and axis-aligned steps, closing back to the start",
               "3 + 2 per step of a poly"),
    "points": ("  p = poly([(x0, y0), (x1, y1), ...])   a rectilinear outline through its corners in order, closing back to the first; consecutive corners share x or y",
               "1 + 2 per corner of a poly"),
}
FEEDBACK = ("iou", "mismatch")
MAX_MISMATCH_RECTS = 8
# What a run is told when it answers with no accepted submission (GeoSession.unfinished).
NOT_SUBMITTED = ("You have not submitted an accepted program: the task is only done once geo_submit accepts "
                 "one, and the last accepted submission is the answer. Submit your best program now.")

GEO_PROMPT = """You are working on a 2D layout of {width} x {height} layout units (1 unit = {resolution} nm). {orientation} All geometry is axis-aligned on the integer grid.
Layers: {layers}.{inputs}
Your answer is a program in the layout language, submitted with geo_submit. It must assign the output layer{s} {outputs}. Submit whenever you want feedback; the last submission is your answer.

The layout language: one statement per line, `name = expression`. Only these forms exist:
  w = 40                          numbers: integers with + - * / (a division must be exact)
  a = rect(x, y, w, h)            {corner} corner (x, y), width w, height h
{poly_line}
  c = a | b                       union      c = a & b    intersection
  c = a - b                       difference c = a ^ b    xor
  d = size(a, 5)                  grow every edge outward by 5 (negative: shrink); size(a, dx, dy) per axis
  e = move(a, dx, dy)             translate
  f = scale(a, 2) or scale(a, 3, 2)   scale about the origin by an integer or a fraction
A layer name{input_note} is a region like any other name; assigning an output layer name writes that layer.
Each submission reports two costs: operations (each rect, poly, input layer and operator) and variables (the numbers and names the program fixes: 5 per rect, {poly_cost}, 1 per operator plus its numeric arguments). Among exact programs, lower costs are better."""


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

    def prompt_text(self, origin: str = "top-left", poly: str = "deltas") -> str:
        ins = self.input_layers()
        return GEO_PROMPT.format(
            width=self.width, height=self.height, resolution=self.resolution,
            orientation=ORIENTATION[origin], corner="top-left" if origin == "top-left" else "lower-left",
            layers=", ".join(self.layers),
            inputs=f" Input layer{'s' if len(ins) > 1 else ''} {', '.join(ins)} {'are' if len(ins) > 1 else 'is'} given and can be read by name." if ins else "",
            s="s" if len(self.outputs) > 1 else "", outputs=", ".join(self.outputs),
            input_note=" (input or output)" if ins else "",
            poly_line=POLY_FORMS[poly][0], poly_cost=POLY_FORMS[poly][1])


@dataclass
class GeoSettings:
    feedback: str = "mismatch"
    snap: int = 15                          # snapping radius of the ruler, in layout units
    image_px: int = 800
    grid: int = 0                           # overlay spacing in units; 0 = none (the operator's choice)
    ticks: int = 0                          # labelled tick spacing in units; 0 = none
    ruler_bias: int = 0                     # causal-audit arm: the instruments read the target shifted by (k, k)
    origin: str = "top-left"                # display convention: top-left (y down) or bottom-left (y up)
    poly: str = "deltas"                    # how the agent writes a polygon: deltas (steps) or points (corners)

    @property
    def y_up(self) -> bool:
        return self.origin == "bottom-left"


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
        k = self.settings.ruler_bias
        if k and any(s.bbox.x + k < 0 or s.bbox.y + k < 0 or s.bbox.x1 + k > t.width or s.bbox.y1 + k > t.height
                     for layer in self.target.layers.values() for s in layer.shapes):
            raise ValueError(f"ruler bias {k} shifts the target of task {t.id!r} out of its frame")

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
            res = run(program, self.base, outputs=self.task.outputs, poly=self.settings.poly)
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
        tail = (f"cost: {p.ops} operation{'s' if p.ops != 1 else ''}, {p.variables} variables"
                + (f"; unused statements: {', '.join(p.dead)}" if p.dead else ""))
        record.update(ok=True, exact=exact, layers=layers, ops=p.ops, variables=p.variables, dead=p.dead,
                      canonical=p.canonical())
        self._log(record)
        return f"submission {n}: " + ("ALL EXACT. " if exact else "") + "\n".join(lines) + f"\n{tail}"

    def render_view(self, view: Literal["target", "current", "diff"] = "target", window: list[int] | None = None,
                    labels: bool = False, layer: str = "", markup: dict | None = None) -> ToolOutput | str:
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
        # the agent's markups (a memory slot the harness injects when the run registered it)
        marks = Markups.from_dict(markup).overlay() if markup else []
        overlay = dict(window=win, image_px=s.image_px, grid=s.grid or None, ticks=s.ticks or None, rulers=rulers,
                       origin=s.origin, markups=marks)
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
        m = pixel_mapping(self.task.width, self.task.height, window=win, image_px=s.image_px,
                          ticks=s.ticks or None, origin=s.origin)
        return ToolOutput(f"{view} rendered: x {x0}..{x1}, y {y0}..{y1}, {im.size[0]} x {im.size[1]} px"
                          + (f", {len(rulers)} ruler(s)" if rulers else "") + (f", {len(marks)} markup(s)" if marks else "")
                          + ".\n" + _mapping_text(m), (path,))

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
        dx, dy = r.dx, r.dy
        lines = [f"ruler {len(self.rulers)} on {on}:",
                 f"  from {r.a.describe(self.settings.y_up)}",
                 f"  to   {r.b.describe(self.settings.y_up)}",
                 f"  dx = {dx}, dy = {dy}" + (f", distance = {(dx * dx + dy * dy) ** 0.5:.1f}" if dx and dy else "")]
        if r.others:
            lines.append("  ambiguous, also in range: " + "; ".join(o.describe(self.settings.y_up) for o in r.others))
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
        where = f"inside {name} shape {a['shape']}" if a["inside"] else f"in a gap on {name}"
        lines = [f"({x}, {y}) is {where} on {on}:"]
        for axis in ("x", "y"):
            s = a[axis]
            what = ("width" if axis == "x" else "height") if a["inside"] else "space"
            text = f"  along {axis}: {what} {s['length']} ({axis} {s['from']}..{s['to']})"
            if not a["inside"]:
                ends = [f"shape {i}" if i else "nothing" for i in s["between"]]
                text += f", between {ends[0]} and {ends[1]}"
                if s["open"]:
                    size = self.task.width if axis == "x" else self.task.height
                    edges = [f"{axis}={0 if side == 'low' else size}" for side in s["open"]]
                    text += f"; reaches the frame edge at {' and '.join(edges)}"
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

    def unfinished(self) -> str | None:
        """The run's task check (ChatState.task_check): what is missing, or None when a submission
        was accepted. Whether it was exact is the grader's business, not the check's."""
        return None if self.current is not None else NOT_SUBMITTED

    # ---- plumbing ------------------------------------------------------------------------------

    def _metro(self, on: str) -> Metrology | str:
        if on == "current" and self.current is None:
            return "Nothing submitted yet: geo_submit a program first."
        if on not in self._metrology:
            if on == "target":
                # the audit arm's miscalibrated probe: the whole target, shifted, so every read agrees
                lay = _shifted(self.target, self.settings.ruler_bias)
            else:
                lay = self.current
            self._metrology[on] = Metrology(lay)
        return self._metrology[on]

    def _label(self, r: Ruler) -> str:
        return r.label()

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
                # a render reads the markup slot to draw it (supplied only when the run registered
                # the markup memory); reading does not make it a memory tool (Memories.owns)
                inject = ("markup",) if arm == "render" else ()
                tools = tools.add(fn, name=f"geo_{arm}", confirm=False, target=target, inject=inject)
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


def _mapping_text(m: dict) -> str:
    """The pixel -> layout rule for one image, in the fewest terms that state it exactly."""
    s = m["scale"]
    per = "" if s == 1 else f" / {s:.4g}"

    def rel(p: str, offset: int) -> str:
        if not offset:
            return f"{p}{per}"
        return f"{p} - {offset}" if not per else f"({p} - {offset}){per}"

    px, py = rel("px", m["left"]), rel("py", m["top"])
    x = px if not m["x_left"] else f"{m['x_left']} + {px}"
    if m["y_up"]:
        y = f"{m['y_top']} - {py}" if "(" in py or not m["top"] else f"{m['y_top']} - ({py})"
    else:
        y = py if not m["y_top"] else f"{m['y_top']} + {py}"
    scale = "" if s == 1 else f" 1 layout unit = {s:.4g} px."
    top = (f"The plot's top row (pixel row {m["top"]}) is layout y = {m['y_top']}."
           if m["top"] else f"The image's top edge is layout y = {m['y_top']}.")
    return (f"Image pixel (px, py), counted from the image's top-left corner, is layout x = {x}, y = {y}."
            f"{scale} {top}")


def _shifted(layout: Layout, k: int) -> Layout:
    """A copy of the layout with every shape translated by (k, k); the layout itself when k is 0.
    The frame is kept, so a shift that would push a shape out of it is refused at the first read."""
    if not k:
        return layout
    out = Layout(layout.width, layout.height, layout.resolution)
    for name, layer in layout.layers.items():
        out.layer(name, layer.colour)
        for s in layer.shapes:
            if isinstance(s, Rect):
                out.add(name, Rect(s.x + k, s.y + k, s.w, s.h))
            else:
                out.add(name, Polygon.of((s.origin.x + k, s.origin.y + k), s.deltas))
    return out


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
    if args.geo_poly not in POLY_FORMS:
        raise SystemExit(f"--geo-poly must be one of {', '.join(POLY_FORMS)}")
    if args.geo_origin not in ORIENTATION:
        raise SystemExit(f"--geo-origin must be one of {', '.join(ORIENTATION)}")
    settings = GeoSettings(feedback=args.geo_feedback, snap=args.geo_snap, image_px=args.geo_image_px,
                           grid=args.geo_grid, ticks=args.geo_ticks, ruler_bias=args.geo_ruler_bias,
                           origin=args.geo_origin, poly=args.geo_poly)
    return GeoSession(task, os.path.abspath(os.path.expanduser(out)), settings)


def arms_of(args) -> tuple[str, ...]:
    return tuple(a.strip() for a in (args.geo_tools or "").split(",") if a.strip())
