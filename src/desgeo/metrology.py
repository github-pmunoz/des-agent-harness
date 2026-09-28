"""
metrology.py — EDA-style instruments over a Layout: shapes numbered per layer, a snapping ruler,
auto-measure and inspect. The model points coarsely in layout units; the instrument snaps to the
exact geometry, the way a KLayout ruler does, and says what it snapped to.

    Metrology(layout)       indexes every layer: its shapes (edge-connected components, numbered
                            1.. bottom-up then left to right), their boundary edges and vertices
    snap(p, radius)         the nearest vertex within radius, else the nearest edge point; plus
                            every other distinct candidate in range (an ambiguity)
    measure(p1, p2)         snap both ends -> Ruler with exact dx, dy, distance
    auto_measure(p, layer)  from a point inside a shape: its width along x and y at that point;
                            from a point in a gap: the space to the nearest shape on each axis
    inspect(p)              the shapes under a point: layer, id, bbox, area, canonical rects

Everything is computed on the raster oracle (1 px = 1 dbu), so an instrument reads the same
geometry the grader does. Shape numbering is deterministic, so "M1 shape 3" means the same shape
in every tool result and every render label of a run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from desgeo.layout import Layout
from desgeo.raster import RasterEngine
from desgeo.shapes import Rect


@dataclass(frozen=True)
class Edge:
    layer: str
    shape: int
    axis: str                   # "x": a vertical edge at x = at; "y": a horizontal edge at y = at
    at: int
    lo: int                     # the span along the other axis
    hi: int
    side: str                   # where the shape is: left / right of a vertical edge (smaller / larger x),
                                # below / above a horizontal one (smaller / larger y, whichever way y is drawn)

    def describe(self, y_up: bool = True) -> str:
        """The edge as the shape's own side: its left/right edge, and its bottom/top edge in the
        display convention (with y down, the edge that has the shape at larger y is its top)."""
        if self.axis == "x":
            return f"{'right' if self.side == 'left' else 'left'} edge x={self.at} (y {self.lo}..{self.hi})"
        low_edge = self.side == "above"             # the shape lies at larger y
        name = ("bottom" if low_edge else "top") if y_up else ("top" if low_edge else "bottom")
        return f"{name} edge y={self.at} (x {self.lo}..{self.hi})"


@dataclass(frozen=True)
class Snap:
    x: int
    y: int
    kind: str                   # "vertex", "edge" or "free" (nothing in range)
    layer: str = ""
    shape: int = 0
    edge: Edge | None = None
    distance: float = 0.0       # from the pointed-at position

    def describe(self, y_up: bool = True) -> str:
        if self.kind == "free":
            return f"({self.x}, {self.y}) free, nothing within range"
        what = "vertex" if self.kind == "vertex" else self.edge.describe(y_up)
        return f"({self.x}, {self.y}) {self.layer} shape {self.shape} {what}"


@dataclass(frozen=True)
class Ruler:
    a: Snap
    b: Snap
    others: tuple[Snap, ...] = ()           # distinct candidates in range at either end

    @property
    def dx(self) -> int:
        return abs(self.b.x - self.a.x)

    @property
    def dy(self) -> int:
        return abs(self.b.y - self.a.y)

    @property
    def distance(self) -> float:
        return math.hypot(self.dx, self.dy)

    def label(self) -> str:
        if self.dy == 0:
            return str(self.dx)
        if self.dx == 0:
            return str(self.dy)
        return f"{self.dx},{self.dy}"


@dataclass
class _LayerIndex:
    bits: np.ndarray
    labels: np.ndarray                      # 0 outside, shape id inside
    shapes: dict[int, list[Rect]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    vertices: list[tuple[int, int, int]] = field(default_factory=list)   # (x, y, shape)


class Metrology:
    def __init__(self, layout: Layout, layers: list[str] | None = None):
        self.layout = layout
        self.engine = RasterEngine(layout.width, layout.height)
        self.index: dict[str, _LayerIndex] = {}
        for name in layers if layers is not None else list(layout.layers):
            self.index[name] = self._index(name)

    # ---- indexing ----------------------------------------------------------------------------

    def _index(self, name: str) -> _LayerIndex:
        region = self.engine.region(self.layout.layers[name].shapes)
        rects = self.engine.rects(region)
        comp = _components(rects)
        # number shapes bottom-up, then left to right, by their lowest-leftmost rect
        order = sorted(set(comp), key=lambda c: min((r.y, r.x) for r, k in zip(rects, comp) if k == c))
        ids = {c: i + 1 for i, c in enumerate(order)}
        idx = _LayerIndex(region.bits, np.zeros(region.bits.shape, dtype=np.int32))
        for r, c in zip(rects, comp):
            idx.labels[r.y:r.y1, r.x:r.x1] = ids[c]
            idx.shapes.setdefault(ids[c], []).append(r)
        idx.edges = _edges(name, idx.labels)
        seen = set()
        for e in idx.edges:
            for v in ((e.at, e.lo), (e.at, e.hi)) if e.axis == "x" else ((e.lo, e.at), (e.hi, e.at)):
                if (v, e.shape) not in seen:
                    seen.add((v, e.shape))
                    idx.vertices.append((v[0], v[1], e.shape))
        return idx

    def shape_count(self, layer: str) -> int:
        return len(self.index[layer].shapes)

    def labels(self) -> list[tuple[int, int, str, Rect]]:
        """A label for every shape, for Set-of-Mark renders: the centre of its largest rect, and
        that rect, so the renderer can put the tag outside a shape too small to hold it."""
        out = []
        for name, idx in self.index.items():
            for sid, rects in idx.shapes.items():
                r = max(rects, key=lambda r: (r.area, -r.y, -r.x))
                out.append((r.x + r.w // 2, r.y + r.h // 2, f"{name}:{sid}", r))
        return out

    # ---- instruments -------------------------------------------------------------------------

    def snap(self, x: int, y: int, radius: int, layer: str | None = None) -> tuple[Snap, tuple[Snap, ...]]:
        """The snap for a pointed-at position, and the other distinct candidates within radius.
        A vertex in range wins over an edge; within a kind, the nearest wins."""
        verts: list[Snap] = []
        edges: list[Snap] = []
        for name in self._layers(layer):
            idx = self.index[name]
            for vx, vy, sid in idx.vertices:
                d = math.hypot(vx - x, vy - y)
                if d <= radius:
                    verts.append(Snap(vx, vy, "vertex", name, sid, None, d))
            for e in idx.edges:
                if e.axis == "x":
                    px, py = e.at, min(max(y, e.lo), e.hi)
                else:
                    px, py = min(max(x, e.lo), e.hi), e.at
                d = math.hypot(px - x, py - y)
                if d <= radius:
                    edges.append(Snap(px, py, "edge", name, e.shape, e, d))
        ranked = sorted(verts, key=lambda s: s.distance) + sorted(edges, key=lambda s: s.distance)
        if not ranked:
            return Snap(x, y, "free"), ()
        best = ranked[0]
        # another candidate is an ambiguity when it would move the measurement: a different point
        others, seen = [], {(best.x, best.y)}
        for s in ranked[1:]:
            if (s.x, s.y) not in seen and not _same_line(best, s):
                seen.add((s.x, s.y))
                others.append(s)
        return best, tuple(others[:3])

    def measure(self, x1: int, y1: int, x2: int, y2: int, radius: int, layer: str | None = None) -> Ruler:
        a, oa = self.snap(x1, y1, radius, layer)
        b, ob = self.snap(x2, y2, radius, layer)
        # Both ends on parallel edges: measure straight across, the way a width or a space is read,
        # by sliding the second end along its edge to the first end's line when the edge reaches it.
        if a.kind == b.kind == "edge" and a.edge.axis == b.edge.axis:
            if a.edge.axis == "x" and b.edge.lo <= a.y <= b.edge.hi:
                b = Snap(b.x, a.y, "edge", b.layer, b.shape, b.edge, b.distance)
            elif a.edge.axis == "y" and b.edge.lo <= a.x <= b.edge.hi:
                b = Snap(a.x, b.y, "edge", b.layer, b.shape, b.edge, b.distance)
        return Ruler(a, b, oa + ob)

    def auto_measure(self, x: int, y: int, layer: str) -> dict:
        """Rays from the pixel at (x, y) along ±x and ±y to the first boundary. Inside a shape they
        measure its width; in a gap, the space to the nearest shape (None: open to the frame)."""
        idx = self.index[layer]
        if not (0 <= x < self.layout.width and 0 <= y < self.layout.height):
            raise ValueError(f"({x}, {y}) is outside the frame 0..{self.layout.width} x 0..{self.layout.height}")
        inside = bool(idx.bits[y, x])
        out = {"layer": layer, "inside": inside, "shape": int(idx.labels[y, x]) if inside else 0}
        for axis, line, pos in (("x", idx.bits[y, :], x), ("y", idx.bits[:, x], y)):
            lo, hi = _run(line, pos)
            open_lo, open_hi = lo == 0 and not inside, hi == len(line) and not inside
            span = {"from": lo, "to": hi, "length": hi - lo}
            if not inside:
                span["open"] = [s for s, o in (("low", open_lo), ("high", open_hi)) if o]
                other = idx.labels[y, :] if axis == "x" else idx.labels[:, x]
                span["between"] = [int(other[lo - 1]) if lo > 0 else 0, int(other[hi]) if hi < len(line) else 0]
            out[axis] = span
        return out

    def inspect(self, x: int, y: int) -> list[dict]:
        """Every shape whose pixel (x, y) is covered, over the indexed layers."""
        out = []
        if not (0 <= x < self.layout.width and 0 <= y < self.layout.height):
            return out
        for name, idx in self.index.items():
            sid = int(idx.labels[y, x])
            if sid:
                rects = idx.shapes[sid]
                bbox = Rect.corners(min(r.x for r in rects), min(r.y for r in rects),
                                    max(r.x1 for r in rects), max(r.y1 for r in rects))
                out.append({"layer": name, "shape": sid, "bbox": bbox, "area": sum(r.area for r in rects),
                            "rects": rects, "vertices": sum(1 for v in idx.vertices if v[2] == sid)})
        return out

    def _layers(self, layer: str | None) -> list[str]:
        if layer is None:
            return list(self.index)
        if layer not in self.index:
            raise ValueError(f"no layer {layer!r}; layers are {', '.join(self.index)}")
        return [layer]


def _components(rects: list[Rect]) -> list[int]:
    """Edge-connected components of disjoint rects (a corner touch does not connect): union-find
    over pairs that share a boundary segment of positive length."""
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            b = rects[j]
            share_x = min(a.x1, b.x1) - max(a.x, b.x) > 0 and (a.y1 == b.y or b.y1 == a.y)
            share_y = min(a.y1, b.y1) - max(a.y, b.y) > 0 and (a.x1 == b.x or b.x1 == a.x)
            if share_x or share_y:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(rects))]


def _edges(layer: str, labels: np.ndarray) -> list[Edge]:
    """Maximal boundary segments of every shape. A segment ends where the boundary turns, and
    where the side the shape lies on flips (two shapes of one layer meeting at a corner)."""
    h, w = labels.shape
    pad = np.pad(labels, 1)
    out: list[Edge] = []
    # vertical edges at x = k: between pixel column k-1 and k (padded: k and k+1)
    left, right = pad[1:-1, :-1], pad[1:-1, 1:]          # shape (h, w+1): left / right of line x=k
    for sidename, own, other in (("left", left, right), ("right", right, left)):
        # the shape lies to the `sidename` of the edge: own is the shape side
        mask = (own != 0) & (own != other)
        for k in range(w + 1):
            col = np.where(mask[:, k], own[:, k], 0)
            for lo, hi, sid in _label_runs(col):
                out.append(Edge(layer, sid, "x", k, lo, hi, "left" if sidename == "left" else "right"))
    below, above = pad[:-1, 1:-1], pad[1:, 1:-1]         # shape (h+1, w): below / above line y=k
    for sidename, own, other in (("below", below, above), ("above", above, below)):
        mask = (own != 0) & (own != other)
        for k in range(h + 1):
            row = np.where(mask[k, :], own[k, :], 0)
            for lo, hi, sid in _label_runs(row):
                out.append(Edge(layer, sid, "y", k, lo, hi, sidename))
    return out


def _label_runs(line: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal non-zero values: (start, end, value)."""
    out = []
    nz = np.flatnonzero(line)
    if not len(nz):
        return out
    start = prev = nz[0]
    for i in nz[1:]:
        if i != prev + 1 or line[i] != line[prev]:
            out.append((int(start), int(prev) + 1, int(line[prev])))
            start = i
        prev = i
    out.append((int(start), int(prev) + 1, int(line[prev])))
    return out


def _run(line: np.ndarray, pos: int) -> tuple[int, int]:
    """The run of equal values in a bool line that contains index pos: [lo, hi)."""
    v = line[pos]
    diff = np.flatnonzero(line != v)
    lo = int(diff[diff < pos].max()) + 1 if (diff < pos).any() else 0
    hi = int(diff[diff > pos].min()) if (diff > pos).any() else len(line)
    return lo, hi


def _same_line(a: Snap, b: Snap) -> bool:
    """Two edge snaps on one straight boundary line give the same measurement along its normal."""
    return (a.kind == b.kind == "edge" and a.edge.axis == b.edge.axis and a.edge.at == b.edge.at
            and a.layer == b.layer)
