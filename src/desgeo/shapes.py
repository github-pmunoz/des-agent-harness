"""
shapes.py — the Manhattan primitives: Point, Rect, Polygon, on an integer database-unit grid.

Coordinates are integers in database units (dbu), origin at the bottom-left, y up. Every shape is
axis-aligned. A Polygon is written as an origin plus axis-aligned deltas and is normalised on
construction: zero-length deltas dropped, collinear runs merged, the closing edge implied when it
is axis-aligned. It must be simple (no edge touches a non-adjacent edge), so no fill rule is needed.
Holes are not a shape; they appear only in regions, as the result of a boolean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class GeometryError(ValueError):
    """A shape or an operation the geometry does not allow; the message is model-facing."""


def _int(name: str, v) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise GeometryError(f"{name} must be an integer, got {v!r}")
    return v


class Point(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True, order=True)
class Rect:
    """An axis-aligned rectangle: lower-left corner (x, y), width w > 0, height h > 0."""
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self):
        for name in ("x", "y", "w", "h"):
            _int(name, getattr(self, name))
        if self.w <= 0 or self.h <= 0:
            raise GeometryError(f"rect width and height must be positive, got {self.w} x {self.h}")

    @classmethod
    def corners(cls, x0: int, y0: int, x1: int, y1: int) -> Rect:
        return cls(x0, y0, x1 - x0, y1 - y0)

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def bbox(self) -> Rect:
        return self

    @property
    def vertices(self) -> list[Point]:
        return [Point(self.x, self.y), Point(self.x1, self.y), Point(self.x1, self.y1),
                Point(self.x, self.y1)]


@dataclass(frozen=True)
class Polygon:
    """A simple rectilinear polygon: origin plus axis-aligned deltas, closed back to the origin.
    Use Polygon.of(origin, deltas) or Polygon.from_points(corners); the fields hold the normalised
    form, the same whichever way the polygon was written."""
    origin: Point
    deltas: tuple[tuple[int, int], ...]

    @classmethod
    def of(cls, origin, deltas) -> Polygon:
        ox, oy = (_int("origin x", origin[0]), _int("origin y", origin[1]))
        pts = [(ox, oy)]
        for i, d in enumerate(deltas):
            dx, dy = _int(f"delta {i} dx", d[0]), _int(f"delta {i} dy", d[1])
            if dx and dy:
                raise GeometryError(f"delta {i} ({dx}, {dy}) is not axis-aligned: one of dx, dy must be 0")
            if dx or dy:
                pts.append((pts[-1][0] + dx, pts[-1][1] + dy))
        if pts[-1] == pts[0]:
            pts.pop()
        elif pts[-1][0] != ox and pts[-1][1] != oy:
            raise GeometryError(f"the outline ends at {pts[-1]}, which is not axis-aligned with the "
                                f"origin {(ox, oy)}, so it cannot close")
        pts = _merge_collinear(pts)
        if len(pts) < 4:
            raise GeometryError("a polygon needs at least 4 vertices after normalisation")
        _check_simple(pts)
        # Counter-clockwise from the lowest-leftmost vertex: one polygon, one spelling.
        if sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1])) < 0:
            pts.reverse()
        k = pts.index(min(pts, key=lambda p: (p[1], p[0])))
        pts = pts[k:] + pts[:k]
        deltas = tuple((b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        return cls(Point(*pts[0]), deltas)

    @classmethod
    def from_points(cls, points) -> Polygon:
        """The polygon through these corners in order, closing back to the first (which may be
        repeated at the end). Consecutive corners, the closing pair included, must share x or y."""
        pts = [(_int(f"corner {i} x", p[0]), _int(f"corner {i} y", p[1])) for i, p in enumerate(points)]
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        if len(pts) < 4:
            raise GeometryError(f"a polygon needs at least 4 corners, got {len(pts)}")
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if a[0] != b[0] and a[1] != b[1]:
                j = (i + 1) % n
                raise GeometryError(f"corners {i} {a} and {j} {b} are not axis-aligned: consecutive "
                                    f"corners must share x or y" + (" (the outline closes back to corner 0)" if j == 0 else ""))
        deltas = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:] + pts[:1])]
        return cls.of(pts[0], deltas)

    @property
    def vertices(self) -> list[Point]:
        x, y = self.origin
        out = [Point(x, y)]
        for dx, dy in self.deltas:
            x, y = x + dx, y + dy
            out.append(Point(x, y))
        return out

    @property
    def area(self) -> int:
        v = self.vertices
        s = sum(a.x * b.y - b.x * a.y for a, b in zip(v, v[1:] + v[:1]))
        return abs(s) // 2

    @property
    def bbox(self) -> Rect:
        v = self.vertices
        return Rect.corners(min(p.x for p in v), min(p.y for p in v),
                            max(p.x for p in v), max(p.y for p in v))


Shape = Rect | Polygon


def _merge_collinear(pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Drop vertices whose two edges lie on one line. A reversal (a zero-width spike) is kept, so
    the simplicity check rejects it."""
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            same_x = a[0] == b[0] == c[0]
            same_y = a[1] == b[1] == c[1]
            forward = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) > 0
            if (same_x or same_y) and forward:
                pts.pop(i)
                changed = True
                break
    return pts


def _check_simple(pts: list[tuple[int, int]]) -> None:
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            (a0, a1), (b0, b1) = edges[i], edges[j]
            if adjacent:
                # Adjacent edges share one vertex; they may not fold back over each other.
                if _overlap_len(a0, a1, b0, b1) > 0:
                    raise GeometryError(f"edges {i} and {j} fold back over each other at {a1}")
            elif _touch(a0, a1, b0, b1):
                raise GeometryError(f"the outline touches itself: edge {i} {a0}->{a1} meets "
                                    f"edge {j} {b0}->{b1}")


def _touch(a0, a1, b0, b1) -> bool:
    """Whether two axis-aligned closed segments share any point."""
    return (max(min(a0[0], a1[0]), min(b0[0], b1[0])) <= min(max(a0[0], a1[0]), max(b0[0], b1[0]))
            and max(min(a0[1], a1[1]), min(b0[1], b1[1])) <= min(max(a0[1], a1[1]), max(b0[1], b1[1])))


def _overlap_len(a0, a1, b0, b1) -> int:
    """Length of the collinear overlap of two axis-aligned segments (0 if not collinear)."""
    if a0[0] == a1[0] == b0[0] == b1[0]:
        lo, hi = max(min(a0[1], a1[1]), min(b0[1], b1[1])), min(max(a0[1], a1[1]), max(b0[1], b1[1]))
        return max(0, hi - lo)
    if a0[1] == a1[1] == b0[1] == b1[1]:
        lo, hi = max(min(a0[0], a1[0]), min(b0[0], b1[0])), min(max(a0[0], a1[0]), max(b0[0], b1[0]))
        return max(0, hi - lo)
    return 0
