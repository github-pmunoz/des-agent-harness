#!/usr/bin/env python3
"""generate.py — write the Track A curriculum: image -> geometry, one task file per case.

Usage:
    ./generate.py            (re)writes cases/*.json; deterministic, so a rerun is a no-op

Every case is a GeoTask (desh_chat.geo) on an 800 x 800 frame with one output layer M1. Its
target is the reference program: the construction the generator used, whose two costs
(operations, variables) the grader compares a submission's with. A reference is a natural
construction, not a proven minimum, so a submission may beat it. Coordinates are off the 10-unit
snap on purpose: the perception probe showed a 10-unit prior, and an exact answer here needs
single-unit reads (by eye, by instrument, or from the feedback).

    L0 rect        one rectangle
    L1 disjoint    2-3 separate rectangles
    L2 overlap     2-3 overlapping rectangles: an L, a T, a plus
    L3 minus       a notch, a U, a ring
    L4 xor         two or three overlapping rectangles xor-ed
    L5 compound    4-6 rectangles with a cut: a comb, stairs with a slot, a frame with a bar
    L6 polygon     one rectilinear outline of 8-10 vertices, an H with a hole
    L7 dense       8-14 overlapping rects, an irregular union of 1-5 pieces
    L8 near        features 1-3 units apart (gaps, a slot, a jog): inside the ruler's snap radius
    L9 many        12-20 small separate rects
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from desgeo import Layout, Metrology, run

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
SIZE = 800
PER_LEVEL = 3
PROMPT = ("Reproduce the target layout exactly: write a program whose M1 layer equals the target's M1, "
          "and submit it with geo_submit. Among exact programs, lower costs are better.")


def off(rng: random.Random, lo: int, hi: int) -> int:
    """A coordinate in [lo, hi] that is not a multiple of 5."""
    while True:
        v = rng.randint(lo, hi)
        if v % 5:
            return v


def rect(r) -> str:
    return f"rect({r[0]}, {r[1]}, {r[2]}, {r[3]})"


def l0(rng, k):
    w, h = off(rng, 60, 420), off(rng, 60, 420)
    return f"M1 = {rect((off(rng, 30, SIZE - w - 30), off(rng, 30, SIZE - h - 30), w, h))}\n", {"shapes": 1}


def l1(rng, k):
    n = 2 + (k % 2)
    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    rng.shuffle(cells)
    rs = []
    for cx, cy in cells[:n]:              # one rect per quadrant: never touching
        w, h = off(rng, 60, 300), off(rng, 60, 300)
        x = cx * 400 + off(rng, 20, 380 - w)
        y = cy * 400 + off(rng, 20, 380 - h)
        rs.append((x, y, w, h))
    names = [chr(ord("a") + i) for i in range(n)]
    body = "".join(f"{nm} = {rect(r)}\n" for nm, r in zip(names, rs))
    return body + f"M1 = {' | '.join(names)}\n", {"shapes": n}


def l2(rng, k):
    x, y = off(rng, 120, 260), off(rng, 120, 260)
    if k == 0:                              # L: a bar and a leg sharing a corner block
        w, t = off(rng, 250, 420), off(rng, 50, 110)
        h = off(rng, 250, 420)
        rs = [(x, y, w, t), (x, y, t, h)]
    elif k == 1:                            # T: a bar and a stem through its middle
        w, t = off(rng, 300, 460), off(rng, 50, 100)
        s, h = off(rng, 40, 90), off(rng, 250, 380)
        rs = [(x, y + h - t, w, t), (x + (w - s) // 2 + off(rng, -30, 30), y, s, h)]
    else:                                   # plus: two bars and a square over the crossing
        w, t = off(rng, 300, 440), off(rng, 50, 90)
        rs = [(x, y + off(rng, 120, 200), w, t), (x + off(rng, 120, 200), y, t, off(rng, 300, 440)),
              (x + off(rng, 90, 150), y + off(rng, 90, 150), off(rng, 90, 130), off(rng, 90, 130))]
    names = [chr(ord("a") + i) for i in range(len(rs))]
    body = "".join(f"{nm} = {rect(r)}\n" for nm, r in zip(names, rs))
    return body + f"M1 = {' | '.join(names)}\n", {"shapes": 1}


def l3(rng, k):
    x, y = off(rng, 100, 220), off(rng, 100, 220)
    w, h = off(rng, 300, 480), off(rng, 300, 480)
    if k == 0:                              # notch into the right side
        nh, nd = off(rng, 60, h // 2), off(rng, 60, w // 2)
        cut = (x + w - nd, y + off(rng, 40, h - nh - 40), nd + off(rng, 10, 40), nh)
    elif k == 1:                            # U: a slot from the top edge
        sw = off(rng, 80, w // 2)
        cut = (x + off(rng, 50, w - sw - 50), y + off(rng, 60, h // 2), sw, h)
    else:                                   # ring: a hole strictly inside
        m = off(rng, 40, 90)
        cut = (x + m, y + off(rng, 40, 90), w - m - off(rng, 40, 90), h - 2 * off(rng, 40, 60))
    return f"outer = {rect((x, y, w, h))}\ncut = {rect(cut)}\nM1 = outer - cut\n", {"shapes": 1}


def l4(rng, k):
    x, y = off(rng, 100, 200), off(rng, 100, 200)
    a = (x, y, off(rng, 250, 380), off(rng, 250, 380))
    b = (x + off(rng, 90, 200), y + off(rng, 90, 200), off(rng, 250, 380), off(rng, 250, 380))
    if k < 2:
        return f"a = {rect(a)}\nb = {rect(b)}\nM1 = a ^ b\n", {}
    c = (x + off(rng, 20, 60), y + off(rng, 150, 250), off(rng, 400, 480), off(rng, 60, 110))
    return f"a = {rect(a)}\nb = {rect(b)}\nc = {rect(c)}\nM1 = a ^ b ^ c\n", {}


def l5(rng, k):
    x, y = off(rng, 80, 140), off(rng, 80, 140)
    if k == 0:                              # comb: a spine and 4 teeth
        sp, t = off(rng, 60, 100), off(rng, 30, 60)
        teeth = [(x + sp + 0, y + off(rng, 20, 40) + i * off(rng, 120, 140), off(rng, 250, 450), t) for i in range(4)]
        body = f"spine = {rect((x, y, sp, 560))}\n" + "".join(f"t{i} = {rect(r)}\n" for i, r in enumerate(teeth))
        return body + "M1 = spine | t0 | t1 | t2 | t3\n", {"shapes": 1}
    if k == 1:                              # stairs: 4 stacked blocks, a slot through them
        rs, cx, cy = [], x, y
        for i in range(4):
            w, h = off(rng, 380, 520) - i * 80, off(rng, 90, 130)
            rs.append((cx, cy, w, h))
            cy += h
        slot = (x + off(rng, 60, 120), y + off(rng, 30, 60), off(rng, 30, 50), off(rng, 200, 300))
        body = "".join(f"s{i} = {rect(r)}\n" for i, r in enumerate(rs)) + f"slot = {rect(slot)}\n"
        return body + "M1 = (s0 | s1 | s2 | s3) - slot\n", {"shapes": 1}
    # frame: an outer square with its middle cut out, then a bar across the hole
    w = off(rng, 450, 560)
    m = off(rng, 50, 80)
    bar = (x + m - off(rng, 5, 15), y + off(rng, 180, 260), w - 2 * m + off(rng, 10, 30), off(rng, 30, 50))
    body = f"outer = {rect((x, y, w, w))}\nhole = {rect((x + m, y + m, w - 2 * m, w - 2 * m))}\nbar = {rect(bar)}\n"
    return body + "M1 = (outer - hole) | bar\n", {"shapes": 1}


def l6(rng, k):
    x, y = off(rng, 100, 200), off(rng, 100, 200)
    if k < 2:                               # a staircase: along the bottom, then up-and-left steps home
        steps = 3 + k                       # 8 or 10 vertices
        width = off(rng, 460, 520)
        lefts = sorted({off(rng, 40, width - 40) for _ in range(steps - 1)}, reverse=True) + [0]
        ups = sorted({off(rng, 40, 480) for _ in range(len(lefts))})
        while len(ups) < len(lefts):
            ups = sorted(set(ups) | {off(rng, 40, 480)})
        deltas, cx, cy = [(width, 0)], width, 0
        for px, py in zip(lefts, ups):
            deltas += [(0, py - cy), (px - cx, 0)]
            cx, cy = px, py
        pts = ", ".join(f"({dx}, {dy})" for dx, dy in deltas)
        return f"M1 = poly(({x}, {y}), [{pts}])\n", {"shapes": 1}
    # an H outline with a hole in one leg
    lw, cw, h = off(rng, 80, 120), off(rng, 150, 220), off(rng, 380, 480)
    bh, by = off(rng, 60, 100), off(rng, 140, 200)
    deltas = [(lw, 0), (0, by), (cw, 0), (0, -by), (lw, 0), (0, h), (-lw, 0), (0, -(h - by - bh)),
              (-cw, 0), (0, h - by - bh), (-lw, 0)]
    pts = ", ".join(f"({dx}, {dy})" for dx, dy in deltas)
    hole = (x + off(rng, 20, 30), y + off(rng, 30, 60), off(rng, 25, 40), off(rng, 60, 90))
    return f"h = poly(({x}, {y}), [{pts}])\nM1 = h - {rect(hole)}\n", {"shapes": 1}


def l7(rng, k):
    """Dense: 8, 11 or 14 overlapping rects in a 520-unit square: an irregular union (1-5 pieces)."""
    n = 8 + 3 * k
    rs = []
    for _ in range(n):
        w, h = off(rng, 40, 200), off(rng, 40, 200)
        rs.append((off(rng, 140, 660 - w), off(rng, 140, 660 - h), w, h))
    names = [f"r{i}" for i in range(n)]
    body = "".join(f"{nm} = {rect(r)}\n" for nm, r in zip(names, rs))
    return body + f"M1 = {' | '.join(names)}\n", {}


def l8(rng, k):
    """Near-coincident: features 1-3 units apart, inside the ruler's snap radius."""
    x, y = off(rng, 120, 200), off(rng, 150, 220)
    if k == 0:                              # two blocks 2 apart, a third below with a 1-unit jog
        w, h = off(rng, 150, 220), off(rng, 120, 180)
        rs = [(x, y, w, h), (x + w + 2, y, off(rng, 150, 220), h), (x + 1, y + h + off(rng, 40, 70), 2 * w, off(rng, 80, 120))]
        names = ["a", "b", "c"]
        body = "".join(f"{nm} = {rect(r)}\n" for nm, r in zip(names, rs))
        return body + "M1 = a | b | c\n", {"shapes": 3}
    if k == 1:                              # a block with a 3-wide slot and a 2-wide notch
        w, h = off(rng, 380, 460), off(rng, 300, 380)
        slot = (x + off(rng, 100, 180), y + off(rng, 40, 80), 3, off(rng, 150, 220))
        notch = (x + w - off(rng, 60, 90), y - 5, 2, off(rng, 60, 110))
        body = f"outer = {rect((x, y, w, h))}\nslot = {rect(slot)}\nnotch = {rect(notch)}\n"
        return body + "M1 = outer - slot - notch\n", {"shapes": 1}
    # a comb whose teeth are 3 units apart
    tw, t = off(rng, 21, 29), 6
    teeth = [(x + i * (tw + 3), y, tw, off(rng, 180, 260)) for i in range(t)]
    spine = (x, y - off(rng, 36, 44), t * tw + (t - 1) * 3, 50)      # across every tooth's end
    body = f"spine = {rect(spine)}\n" + "".join(f"t{i} = {rect(r)}\n" for i, r in enumerate(teeth))
    return body + "M1 = spine | " + " | ".join(f"t{i}" for i in range(t)) + "\n", {"shapes": 1}


def l9(rng, k):
    """Many: 12, 16 or 20 small separate rects, one per cell of a jittered grid."""
    n = 12 + 4 * k
    cols = 4 + k
    rows = -(-n // cols)
    cw, ch = 600 // cols, 600 // rows
    rs = []
    for i in range(n):
        cx, cy = 100 + (i % cols) * cw, 100 + (i // cols) * ch
        w, h = off(rng, 15, cw - 25), off(rng, 15, ch - 25)
        rs.append((cx + off(rng, 5, cw - w - 8), cy + off(rng, 5, ch - h - 8), w, h))
    names = [f"r{i}" for i in range(n)]
    body = "".join(f"{nm} = {rect(r)}\n" for nm, r in zip(names, rs))
    return body + f"M1 = {' | '.join(names)}\n", {"shapes": n}


LEVELS = [("L0", "rect", l0), ("L1", "disjoint", l1), ("L2", "overlap", l2), ("L3", "minus", l3),
          ("L4", "xor", l4), ("L5", "compound", l5), ("L6", "polygon", l6),
          ("L7", "dense", l7), ("L8", "near", l8), ("L9", "many", l9)]


def main() -> int:
    CASES.mkdir(exist_ok=True)
    for lv, name, fn in LEVELS:
        for k in range(PER_LEVEL):
            rng = random.Random(f"track-A/{lv}/{k}")
            program, expect = fn(rng, k)
            layout = Layout(SIZE, SIZE)
            layout.layer("M1", (0, 0, 0))
            res = run(program, layout, outputs=["M1"])          # raises on a bad construction
            shapes = Metrology(res.layout).shape_count("M1")
            if "shapes" in expect and shapes != expect["shapes"]:
                raise SystemExit(f"{lv}/{k}: expected {expect['shapes']} shapes, got {shapes}\n{program}")
            cid = f"{lv}-{name}-{k}"
            task = {"id": cid, "level": lv, "width": SIZE, "height": SIZE, "layers": {"M1": [0, 0, 0]},
                    "outputs": ["M1"], "target": program, "prompt": PROMPT,
                    "reference": {"ops": res.program.ops, "variables": res.program.variables,
                                  "area": sum(r.area for r in res.layout.layers["M1"].shapes), "shapes": shapes}}
            (CASES / f"{cid}.json").write_text(json.dumps(task, indent=1) + "\n")
            print(f"{cid:16} ops {res.program.ops:2}  vars {res.program.variables:3}  shapes {shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
