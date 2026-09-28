"""desgeo shapes, the raster oracle and rendering. The property tests draw seeded random rect sets
and check each operation against an independent construction or a boolean identity."""
import random

import numpy as np
import pytest

from desgeo import GeometryError, Layout, Polygon, RasterEngine, Rect, png_bytes, render, render_diff

W = H = 120
SEEDS = range(60)
E = RasterEngine(W, H)


def rand_rects(rng: random.Random, n_max: int = 5, margin: int = 0) -> list[Rect]:
    out = []
    for _ in range(rng.randint(1, n_max)):
        w, h = rng.randint(1, 40), rng.randint(1, 40)
        out.append(Rect(rng.randint(margin, W - margin - w), rng.randint(margin, H - margin - h), w, h))
    return out


def staircase(rng: random.Random) -> Polygon:
    """A random rectilinear staircase: right along the bottom, then alternating up / left steps
    back to the y axis, then down to the origin."""
    x0, y0 = rng.randint(0, 20), rng.randint(0, 20)
    steps = rng.randint(1, 5)
    xs = sorted(rng.sample(range(1, 80), steps), reverse=True)     # x after each left step
    ys = sorted(rng.sample(range(1, 80), steps))                    # y after each up step
    width = xs[0] + rng.randint(1, 20)
    deltas = [(width, 0)]
    x, y = width, 0
    for sx, sy in zip(xs[1:] + [0], ys):
        deltas += [(0, sy - y), (sx - x, 0)]
        x, y = sx, sy
    return Polygon.of((x0, y0), deltas)


def column_mask(p: Polygon) -> np.ndarray:
    """Independent rasteriser: every pixel column filled between the outline's horizontal edges,
    even-odd over the crossings of the column's centre."""
    bits = np.zeros((H, W), dtype=bool)
    v = p.vertices
    for x in range(W):
        cx = x + 0.5
        ys = sorted(a.y for a, b in zip(v, v[1:] + v[:1]) if a.y == b.y and min(a.x, b.x) < cx < max(a.x, b.x))
        for lo, hi in zip(ys[::2], ys[1::2]):
            bits[lo:hi, x] = True
    return bits


# ---- shapes -----------------------------------------------------------------------------------

def test_rect_rejects_bad_sizes_and_types():
    with pytest.raises(GeometryError):
        Rect(0, 0, 0, 5)
    with pytest.raises(GeometryError):
        Rect(0, 0, 5.0, 5)
    with pytest.raises(GeometryError):
        Rect(0, 0, True, 5)


def test_polygon_normalises_to_one_spelling():
    ccw = Polygon.of((0, 0), [(40, 0), (0, 10), (-30, 0), (0, 20), (-10, 0)])
    cw = Polygon.of((0, 0), [(0, 30), (10, 0), (0, -20), (30, 0), (0, -10)])        # same L, other way round
    rotated = Polygon.of((40, 0), [(0, 10), (-30, 0), (0, 20), (-10, 0), (0, -30)])
    split = Polygon.of((0, 0), [(20, 0), (20, 0), (0, 0), (0, 10), (-30, 0), (0, 20), (-10, 0)])
    assert ccw == cw == rotated == split
    assert ccw.area == 40 * 10 + 10 * 20
    assert ccw.bbox == Rect(0, 0, 40, 30)


def test_polygon_rejects_bad_outlines():
    with pytest.raises(GeometryError, match="not axis-aligned"):
        Polygon.of((0, 0), [(10, 5), (0, 10)])
    with pytest.raises(GeometryError, match="cannot close"):
        Polygon.of((0, 0), [(10, 0), (0, 10), (-5, 0)])
    with pytest.raises(GeometryError, match="touches itself"):      # two squares meeting at a corner
        Polygon.of((0, 0), [(10, 0), (0, 10), (10, 0), (0, 10), (-10, 0), (0, -10), (-10, 0)])
    with pytest.raises(GeometryError):                               # zero-width spike
        Polygon.of((0, 0), [(10, 0), (0, 10), (5, 0), (-5, 0), (-10, 0)])


# ---- raster oracle ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_polygon_raster_matches_column_raster_and_area(seed):
    p = staircase(random.Random(seed))
    m = E.region([p])
    assert np.array_equal(m.bits, column_mask(p))
    assert E.area(m) == p.area


@pytest.mark.parametrize("seed", SEEDS)
def test_boolean_identities(seed):
    rng = random.Random(seed)
    a, b, c = (E.region(rand_rects(rng)) for _ in range(3))
    U, I, S, X = E.union, E.intersect, E.subtract, E.xor
    eq = E.equal
    assert eq(U(a, b), U(b, a)) and eq(I(a, b), I(b, a)) and eq(X(a, b), X(b, a))
    assert eq(U(U(a, b), c), U(a, U(b, c)))
    assert eq(X(a, b), U(S(a, b), S(b, a)))
    assert eq(S(a, b), X(a, I(a, b)))
    assert eq(I(a, U(b, c)), U(I(a, b), I(a, c)))
    assert eq(S(a, U(b, c)), I(S(a, b), S(a, c)))                    # De Morgan inside a
    assert eq(X(X(a, b), b), a)
    assert E.area(U(a, b)) == E.area(a) + E.area(b) - E.area(I(a, b))


@pytest.mark.parametrize("seed", SEEDS)
def test_canonical_rects_are_disjoint_exact_and_unique(seed):
    rng = random.Random(seed)
    rects = rand_rects(rng, n_max=8)
    m = E.region(rects)
    canon = E.rects(m)
    assert sum(r.area for r in canon) == E.area(m)                   # disjoint, given the next line
    assert E.equal(E.region(canon), m)
    shuffled = rects[:]
    rng.shuffle(shuffled)
    assert E.rects(E.region(shuffled)) == canon
    assert E.rects(E.region(canon)) == canon


@pytest.mark.parametrize("seed", SEEDS)
def test_size_grow_is_union_of_grown_rects(seed):
    rng = random.Random(seed)
    d = rng.randint(1, 8)
    dx, dy = d, rng.randint(1, 8)
    rects = rand_rects(rng, margin=8)
    grown = [Rect.corners(r.x - dx, r.y - dy, r.x1 + dx, r.y1 + dy) for r in rects]
    assert E.equal(E.size(E.region(rects), dx, dy), E.region(grown))


def erode_ref(bits: np.ndarray, d: int) -> np.ndarray:
    """Erosion by brute force over every shift in the (2d+1)^2 box; outside the frame is empty."""
    p = np.pad(bits, d, constant_values=False)
    out = np.ones_like(bits)
    for j in range(-d, d + 1):
        for i in range(-d, d + 1):
            out &= p[d + j:d + j + bits.shape[0], d + i:d + i + bits.shape[1]]
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_size_shrink_matches_brute_force_and_brackets_open_close(seed):
    rng = random.Random(seed)
    d = rng.randint(1, 6)
    a = E.region(rand_rects(rng, n_max=6))
    assert np.array_equal(E.size(a, -d, -d).bits, erode_ref(a.bits, d))
    inner = E.region(rand_rects(rng, margin=2 * d))
    opened = E.size(E.size(inner, -d, -d), d, d)
    closed = E.size(E.size(inner, d, d), -d, -d)
    assert E.equal(E.subtract(opened, inner), E.empty())
    assert E.equal(E.subtract(inner, closed), E.empty())


def test_size_shrinks_a_rect_and_removes_narrow_parts():
    a = E.region([Rect(10, 10, 50, 30), Rect(60, 20, 30, 6)])        # a 6-high stub on the right
    assert E.equal(E.size(a, -2, -2), E.region([Rect(12, 12, 46, 26), Rect(12, 22, 76, 2)]))
    assert E.rects(E.size(E.size(a, -4, -4), 4, 4)) == [Rect(10, 10, 50, 30)]


def test_frame_overflow_raises_instead_of_clipping():
    a = E.region([Rect(0, 10, 10, 10)])
    with pytest.raises(GeometryError, match="frame"):
        E.size(a, 1, 0)
    with pytest.raises(GeometryError, match="frame"):
        E.move(a, -1, 0)
    with pytest.raises(GeometryError, match="frame"):
        E.region([Rect(115, 0, 10, 10)])
    assert E.rects(E.size(a, -2, -2)) == [Rect(2, 12, 6, 6)]          # shrink eats the frame edge too


@pytest.mark.parametrize("seed", SEEDS)
def test_move_and_scale_match_transformed_rects(seed):
    rng = random.Random(seed)
    big = RasterEngine(2 * W + 20, 2 * H + 20)
    rects = rand_rects(rng)
    a = big.region(rects)
    up = big.scale(a, 2)
    assert big.equal(up, big.region([Rect(2 * r.x, 2 * r.y, 2 * r.w, 2 * r.h) for r in rects]))
    assert big.equal(big.scale(up, 1, 2), a)
    dx, dy = rng.randint(-5, 15), rng.randint(-5, 15)
    shifted = [Rect(r.x + 5, r.y + 5, r.w, r.h) for r in rects]
    assert big.equal(big.move(big.region(shifted), dx, dy),
                     big.region([Rect(r.x + dx, r.y + dy, r.w, r.h) for r in shifted]))


def test_scale_off_grid_raises():
    a = E.region([Rect(1, 0, 4, 4)])
    with pytest.raises(GeometryError, match="integer grid"):
        E.scale(a, 1, 2)


# ---- rendering --------------------------------------------------------------------------------

def test_render_flips_colours_and_windows():
    L = Layout(100, 50)
    L.layer("M1", (255, 0, 0))
    L.layer("VIA", (0, 0, 255))
    L.add("M1", Rect(0, 40, 10, 10))                                  # top-left in the image
    L.add("VIA", Rect(0, 40, 5, 5))
    im = render(L, image_px=200)
    assert im.size == (200, 100)
    px = np.array(im)
    assert tuple(px[0, 0]) == (255, 0, 0)                             # M1 only at the top edge
    assert tuple(px[19, 0]) == (128, 0, 128)                          # blend where VIA overlaps
    assert tuple(px[99, 199]) == (255, 255, 255)
    zoom = render(L, window=(0, 40, 10, 50), image_px=100)
    assert zoom.size == (100, 100)
    assert tuple(np.array(render(L, layers=["VIA"], image_px=100))[5, 0]) == (0, 0, 255)
    assert render(L, ticks=10).size[0] > 800
    assert png_bytes(im)[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(GeometryError, match="window"):
        render(L, window=(0, 0, 200, 10))


def test_render_diff_colours():
    im = np.array(render_diff(30, 10, [Rect(0, 0, 20, 10)], [Rect(10, 0, 20, 10)], image_px=30))
    assert tuple(im[5, 5]) == (230, 120, 0)                           # missing
    assert tuple(im[5, 15]) == (170, 170, 170)                        # match
    assert tuple(im[5, 25]) == (0, 90, 200)                           # extra


def test_origin_is_a_display_convention():
    L = Layout(100, 50)
    L.layer("M1", (255, 0, 0))
    L.add("M1", Rect(0, 0, 10, 10))                                   # at y = 0
    up = np.array(render(L, image_px=100))
    down = np.array(render(L, image_px=100, origin="top-left"))
    assert tuple(up[49, 0]) == (255, 0, 0) and tuple(up[0, 0]) == (255, 255, 255)     # bottom row
    assert tuple(down[0, 0]) == (255, 0, 0) and tuple(down[49, 0]) == (255, 255, 255)  # top row
    assert np.array_equal(np.flipud(up), down)
    with pytest.raises(GeometryError, match="origin"):
        render(L, origin="centre")


def test_pixel_mapping_matches_the_render():
    from desgeo.render import pixel_mapping
    L = Layout(800, 800)
    L.layer("M1", (0, 0, 0))
    L.add("M1", Rect(354, 143, 163, 192))
    for origin in ("bottom-left", "top-left"):
        for kw in ({}, {"window": (340, 130, 530, 350)}, {"ticks": 100}):
            im = np.array(render(L, origin=origin, **kw).convert("L")) < 128
            m = pixel_mapping(800, 800, origin=origin, **kw)
            x0, y0, x1, y1 = kw.get("window", (0, 0, 800, 800))
            ph = round((y1 - y0) * m["scale"])
            plot = im[m["top"]:m["top"] + ph, m["left"]:m["left"] + round((x1 - x0) * m["scale"])]
            rows = np.nonzero(plot.any(1))[0] + m["top"]
            # the edges of the shape's pixel rows, mapped back: the rect's y extent
            ys = [m["y_top"] + (-1 if m["y_up"] else 1) * (r - m["top"]) / m["scale"] for r in (rows.min(), rows.max() + 1)]
            assert round(min(ys)) == 143 and round(max(ys)) == 335


def test_labels_sit_outside_their_shape():
    from desgeo import Metrology
    L = Layout(400, 400)
    L.layer("M1", (0, 0, 0))
    L.add("M1", Rect(100, 100, 200, 200))
    im = np.array(render(L, image_px=400, labels=Metrology(L).labels()))
    inside = im[101:299, 101:299]
    assert (inside == 0).all()                                         # nothing drawn over the shape
    assert ((im[:, :, 0] == 255) & (im[:, :, 1] == 215) & (im[:, :, 2] == 0)).any()
