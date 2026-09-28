"""
raster.py — the oracle engine: regions as boolean masks, one pixel per database unit.

Dumb on purpose, so it is obviously right: every other engine is graded against it. A Mask covers
the fixed frame [0, width) x [0, height) and is indexed [y, x], row 0 at the bottom. Pixel (x, y)
is the unit cell [x, x+1) x [y, y+1), so a shape with integer vertices covers whole pixels and
pixel counts are exact areas. An operation whose result would leave the frame raises instead of
clipping, so the oracle either agrees with an unbounded engine or refuses.

Operations (the Engine interface the exact engine will share):
    region         shapes -> Mask (their union)
    union, intersect, subtract, xor
    size           grow (d > 0) or shrink (d < 0) per axis: dilation / erosion by a box, x pass
                   then y pass; for rectilinear shapes this is edge offsetting with square corners
    move           translate by (dx, dy)
    scale          about the origin by num/den; every vertex must land on the integer grid
    rects          the canonical form: maximal horizontal slabs, bottom-up, left to right
    area, bbox, equal
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np

from desgeo.shapes import GeometryError, Polygon, Rect, Shape


@dataclass(frozen=True, eq=False)
class Mask:
    bits: np.ndarray            # bool, shape (height, width)

    @property
    def width(self) -> int:
        return self.bits.shape[1]

    @property
    def height(self) -> int:
        return self.bits.shape[0]


class RasterEngine:
    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise GeometryError(f"frame must be positive, got {width} x {height}")
        self.width, self.height = width, height

    # ---- construction ----------------------------------------------------------------------

    def empty(self) -> Mask:
        return Mask(np.zeros((self.height, self.width), dtype=bool))

    def region(self, shapes: Iterable[Shape]) -> Mask:
        bits = np.zeros((self.height, self.width), dtype=bool)
        for s in shapes:
            self._paint(bits, s)
        return Mask(bits)

    def _paint(self, bits: np.ndarray, s: Shape) -> None:
        b = s.bbox
        if b.x < 0 or b.y < 0 or b.x1 > self.width or b.y1 > self.height:
            raise GeometryError(f"{_describe(s)} leaves the layout frame 0..{self.width} x 0..{self.height}")
        if isinstance(s, Rect):
            bits[s.y:s.y1, s.x:s.x1] = True
            return
        # Even-odd over pixel centres inside the bbox; each vertical edge toggles every pixel left
        # of it on the rows it spans. The polygon is simple, so even-odd is its interior.
        sub = np.zeros((b.h, b.w), dtype=bool)
        cx = np.arange(b.x, b.x1) + 0.5
        v = s.vertices
        for p, q in zip(v, v[1:] + v[:1]):
            if p.x != q.x:
                continue
            lo, hi = sorted((p.y, q.y))
            sub[lo - b.y:hi - b.y] ^= cx < p.x
        bits[b.y:b.y1, b.x:b.x1] |= sub

    # ---- booleans --------------------------------------------------------------------------

    def union(self, a: Mask, b: Mask) -> Mask:
        return Mask(a.bits | b.bits)

    def intersect(self, a: Mask, b: Mask) -> Mask:
        return Mask(a.bits & b.bits)

    def subtract(self, a: Mask, b: Mask) -> Mask:
        return Mask(a.bits & ~b.bits)

    def xor(self, a: Mask, b: Mask) -> Mask:
        return Mask(a.bits ^ b.bits)

    # ---- transforms ------------------------------------------------------------------------

    def size(self, a: Mask, dx: int, dy: int) -> Mask:
        bits = _box(a.bits, dx, axis=1)
        bits = _box(bits, dy, axis=0)
        return Mask(bits)

    def move(self, a: Mask, dx: int, dy: int) -> Mask:
        ys, xs = np.nonzero(a.bits)
        if len(xs) and (xs.min() + dx < 0 or ys.min() + dy < 0
                        or xs.max() + dx >= self.width or ys.max() + dy >= self.height):
            raise GeometryError(f"move by ({dx}, {dy}) leaves the layout frame")
        out = np.zeros_like(a.bits)
        out[ys + dy, xs + dx] = True
        return Mask(out)

    def scale(self, a: Mask, num: int, den: int = 1) -> Mask:
        if num <= 0 or den <= 0:
            raise GeometryError(f"scale factor must be positive, got {num}/{den}")
        f = Fraction(num, den)
        out = []
        for r in self.rects(a):
            c = [v * f for v in (r.x, r.y, r.x1, r.y1)]
            if any(v.denominator != 1 for v in c):
                raise GeometryError(f"scale by {f}: vertex ({r.x}, {r.y}) or ({r.x1}, {r.y1}) "
                                    f"does not land on the integer grid")
            out.append(Rect.corners(*(int(v) for v in c)))
        return self.region(out)

    # ---- queries ---------------------------------------------------------------------------

    def rects(self, a: Mask) -> list[Rect]:
        """Canonical decomposition: rows with identical runs merge into one band; each run of a
        band is one rect. Unique for a given point set, so equal regions give equal lists."""
        bits = a.bits
        out: list[Rect] = []
        y = 0
        while y < self.height:
            if not bits[y].any():
                y += 1
                continue
            y1 = y + 1
            while y1 < self.height and np.array_equal(bits[y1], bits[y]):
                y1 += 1
            for x0, x1 in _runs(bits[y]):
                out.append(Rect.corners(x0, y, x1, y1))
            y = y1
        return out

    def area(self, a: Mask) -> int:
        return int(np.count_nonzero(a.bits))

    def bbox(self, a: Mask) -> Rect | None:
        ys, xs = np.nonzero(a.bits)
        if not len(xs):
            return None
        return Rect.corners(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def equal(self, a: Mask, b: Mask) -> bool:
        return bool(np.array_equal(a.bits, b.bits))


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _box(bits: np.ndarray, d: int, axis: int) -> np.ndarray:
    """Dilate (d > 0) or erode (d < 0) along one axis by |d| cells. Outside the frame counts as
    empty, so erosion eats from the frame edge like from any edge; a dilation that would spill
    past the frame raises."""
    if d == 0:
        return bits
    k = abs(d)
    n = bits.shape[axis]
    if d > 0:
        edge = (bits[:, :k].any() or bits[:, n - k:].any()) if axis == 1 else \
               (bits[:k].any() or bits[n - k:].any())
        if edge:
            raise GeometryError(f"size by {d} along {'x' if axis == 1 else 'y'} leaves the layout frame")
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k, k)
    c = np.cumsum(np.pad(bits.astype(np.int32), pad), axis=axis)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate((zero, c), axis=axis)
    window = np.take(c, np.arange(2 * k + 1, 2 * k + 1 + n), axis=axis) - np.take(c, np.arange(n), axis=axis)
    return window > 0 if d > 0 else window == 2 * k + 1


def _describe(s: Shape) -> str:
    if isinstance(s, Rect):
        return f"rect({s.x}, {s.y}, {s.w}, {s.h})"
    return f"polygon at {tuple(s.origin)} with bbox {s.bbox.x}..{s.bbox.x1} x {s.bbox.y}..{s.bbox.y1}"
