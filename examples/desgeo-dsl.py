#!/usr/bin/env python
"""
desgeo DSL by example: named statements compiled to a canonical DAG and evaluated on the raster
oracle. Each example prints its canonical form, cost and area; PNGs go to the output directory.

    PYTHONPATH=src venv/bin/python examples/desgeo-dsl.py [OUT_DIR]      (default ./desgeo-out)

    1 u-shape      one shape, two spellings: minus vs a union of three rects; cost tells them apart
    2 canonical    commutative spellings collapse to one form; dead statements are reported
    3 numbers      named numbers, exact division, a ring and an L polygon
    4 layers       input layers read by name, output layers written, a VIA array by move
    5 drc          width, spacing and enclosure rules written as programs
    6 transforms   scale and move, and what the frame refuses
    7 diff         a wrong attempt against the target, as numbers and as a diff render
    8 errors       the model-facing messages for programs the language rejects
"""
import sys
from pathlib import Path

from desgeo import DslError, Layout, RasterEngine, Rect, compile, render, render_diff, run

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "desgeo-out")
W, H = 400, 300


def show(title: str, src: str, layout: Layout | None = None, outputs=None, png: str | None = None):
    """Compile and run one program, print what the grader would see, optionally save a render."""
    layout = layout or Layout(W, H)
    res = run(src, layout, outputs=outputs)
    eng = RasterEngine(layout.width, layout.height)
    print(f"\n=== {title}")
    print(src.strip())
    print("--- canonical")
    print(res.program.canonical())
    areas = ", ".join(f"{n}={eng.area(r)}" for n, r in res.regions.items())
    print(f"--- cost {res.program.cost}, area {areas}, dead {res.program.dead or 'none'}")
    if png:
        OUT.mkdir(parents=True, exist_ok=True)
        render(res.layout, ticks=50).save(OUT / png)          # inputs and outputs
    return res


def u_shape():
    minus = show("1a u-shape as a minus", """
outer = rect(100, 50, 200, 200)
slot  = rect(160, 120, 80, 130)
U = outer - slot
""", png="1-u-shape.png")
    union = show("1b the same u-shape as a union", """
left   = rect(100, 50, 60, 200)
right  = rect(240, 50, 60, 200)
bottom = rect(160, 50, 80, 70)
U = left | right | bottom
""")
    eng = RasterEngine(W, H)
    same = eng.equal(minus.regions["U"], union.regions["U"])
    print(f"--- same region: {same}; cost minus {minus.program.cost} vs union {union.program.cost}")


def canonical():
    a = compile("a = rect(0, 0, 50, 50)\nb = rect(25, 25, 50, 50)\nc = rect(60, 0, 20, 20)\n"
                "r = (a | b) | c")
    b = compile("c = rect(60, 0, 20, 20)\nb = rect(25, 25, 50, 50)\n"
                "r = c | (b | rect(0, 0, 50, 50)) | c")
    print("\n=== 2 canonical: (a | b) | c  vs  c | (b | a) | c")
    print(f"--- identical: {a.canonical() == b.canonical()}, costs {a.cost} and {b.cost}")
    show("2b dead statements", """
a = rect(0, 0, 50, 50)
b = rect(25, 25, 50, 50)
probe = a & b
r = a - b
""", outputs=["r"])


def numbers():
    show("3 named numbers, a ring and an L polygon", """
side = 120
wall = 20
x = 40
y = 60
outer = rect(x, y, side, side)
ring  = outer - size(outer, -wall)
L     = poly((220, 60), [(120, 0), (0, 40), (-80, 0), (0, 100), (-40, 0)])
M1    = ring | L
""", outputs=["M1"], png="3-ring-and-l.png")


def layers():
    L = Layout(W, H)
    L.layer("M1", (220, 40, 40))
    L.layer("VIA", (40, 70, 220))
    L.add("M1", Rect(40, 100, 320, 40), Rect(40, 200, 320, 40))
    show("4 layers: vias on both wires, M2 over the vias", """
pitch = 80
v0  = rect(60, 110, 20, 20)
row = v0 | move(v0, pitch, 0) | move(v0, 2 * pitch, 0) | move(v0, 3 * pitch, 0)
VIA = row | move(row, 0, 100)
M2  = size(VIA, 10) | rect(50, 100, 40, 140)
""", L, outputs=["VIA", "M2"], png="4-layers.png")


def drc():
    L = Layout(W, H)
    L.layer("M1", (220, 40, 40))
    L.layer("VIA", (40, 70, 220))
    L.add("M1", Rect(20, 20, 100, 100), Rect(120, 60, 60, 12),     # a 12-wide neck
          Rect(180, 20, 100, 100), Rect(295, 20, 80, 100))         # a 15-wide gap
    L.add("VIA", Rect(40, 40, 20, 20),                             # enclosed by 20
          Rect(262, 40, 16, 16))                                   # 2 from the M1 edge
    show("5 drc: width, spacing and enclosure as programs", """
w = 20
s = 20
e = 10
width_err = M1 - size(size(M1, -w/2), w/2)
space_err = size(size(M1, s/2), -s/2) - M1
encl_err  = VIA - size(M1, -e)
""", L, outputs=["width_err", "space_err", "encl_err"], png="5-drc.png")


def transforms():
    show("6 transforms: scale about the origin, then move", """
cell = rect(10, 10, 30, 20) | rect(10, 30, 10, 20)
big  = scale(cell, 3)
half = scale(big, 1, 2)
out  = move(half, 200, 100) | big
""", outputs=["out"], png="6-transforms.png")
    for src in ("r = scale(rect(1, 0, 4, 4), 1, 2)", "r = size(rect(0, 0, 10, 10), 5)"):
        try:
            run(src, Layout(W, H))
        except DslError as e:
            print(f"--- refused: {src}\n    {e}")


def diff():
    target = run("U = rect(100, 50, 200, 200) - rect(160, 120, 80, 130)", Layout(W, H))
    attempt = run("U = rect(100, 50, 200, 200) - rect(150, 110, 90, 140)", Layout(W, H))
    eng = RasterEngine(W, H)
    t, a = target.regions["U"], attempt.regions["U"]
    print("\n=== 7 diff: slot off by 10 on two sides")
    print(f"--- missing {eng.area(eng.subtract(t, a))}, extra {eng.area(eng.subtract(a, t))}, "
          f"iou {eng.area(eng.intersect(t, a)) / eng.area(eng.union(t, a)):.3f}")
    print(f"--- missing as rects: {eng.rects(eng.subtract(t, a))}")
    OUT.mkdir(parents=True, exist_ok=True)
    render_diff(W, H, target.layout.layers["U"].shapes, attempt.layout.layers["U"].shapes,
                ticks=50).save(OUT / "7-diff.png")


def errors():
    print("\n=== 8 errors")
    for src in ("r = rect(0, 0, 50, 50) + rect(10, 10, 5, 5)",
                "r = circle(0, 0, 5)",
                "w = 25\nr = size(rect(50, 50, 100, 100), -w/2)",
                "import os",
                "r = poly((0, 0), [(40, 0), (0, 40), (-20, 20)])",
                "w = 5\nw = rect(0, 0, 5, 5)"):
        try:
            compile(src)
        except DslError as e:
            print(f"{src!r:55} -> {e}")


if __name__ == "__main__":
    for example in (u_shape, canonical, numbers, layers, drc, transforms, diff, errors):
        example()
    print(f"\nrenders in {OUT}/")
