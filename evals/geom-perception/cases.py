"""
cases.py — the perception cases: what is drawn, what is asked, the answer format, the truth.

Levels climb from counting to reconstruction; each probes one perceptual skill the layout agent
will lean on. Every case is a single image and a single question, answered in one completion.

    1 count        separate shapes, arrays, edge-sharing vs corner-touching
    2 locate       read coordinates off the image, down to a small feature
    3 topology     a one-cell gap vs an abutment, holes and islands
    4 outline      vertex counts and vertex lists of rectilinear polygons
    5 measure      spacing, the narrowest width, area
    6 boolean      two-rectangle boolean decomposability, minimum rectangle cover
    7 layers       colour-coded layers and cover between them
    8 reconstruct  rectangles whose union reproduces the image

Tier 1 (c01-c22) sits on a 10-unit snap with generous features; tier 2 (c23+) spreads over the same
levels with off-snap coordinates (the prompt states each case's snap), a 6 x 9 feature, near-equal
distractor gaps, 20-vertex outlines, minus / XOR reconstructions and enclosure margins.

Grader kinds (grade.py): int, int_tol (linear credit within tol), num_rel (linear credit within a
relative tol), bool, fields_int (fraction of fields right), rects and polygon (IoU against the
truth mask, which the case carries instead of a value).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geom import poly, rect, union

BLACK = (0, 0, 0)
RED = (220, 40, 40)
BLUE = (40, 70, 220)

PREAMBLE = ("The image is a render of a 2D layout that is 800 x 800 layout units. The origin (0, 0) "
            "is the bottom-left corner; x grows to the right and y grows upward. All shapes are "
            "axis-aligned and {snap}.")
GRID_NOTE = ("Thin grey grid lines are drawn every 100 units, at x = 0, 100, ..., 800 and at "
             "y = 0, 100, ..., 800.")
MONO_LEGEND = "Shapes are black on a white background."
ANSWER_RULE = "End your reply with one final line of the form\nANSWER: <json>\nwhere <json> is {fmt}."

INT = "a single integer"
BOOL = "true or false"
RECT = "[x, y, width, height] of the rectangle"
RECTS = "a list of rectangles, each as [x, y, width, height]"
VERTICES = "a list of [x, y] vertices, in order around the outline"

COUNT_REGIONS = "How many connected black regions are there?"
CORNERS = "How many corners (vertices) does the outline of the black shape have?"
OUTLINE = "Give the outline of the black shape."
LOCATE_ONE = "Give the position and size of the black rectangle."
LOCATE_ALL = "Give the position and size of every black rectangle."
LOCATE_SMALL = ("There is one large and one very small black rectangle. Give the position and size of the "
                "SMALL one.")
TWO_LAYERS = ("Two layers are drawn on a white background: red is layer M1 (wires), blue is layer VIA "
              "(small squares). Where they overlap the colour is purple.")
RECONSTRUCT = "Give a list of axis-aligned rectangles whose union reproduces the black region exactly."
BOOLEAN_Q = ("Can the black shape be produced exactly by ONE boolean operation (union, intersection, "
             "difference or XOR) applied to TWO axis-aligned rectangles?")


@dataclass
class Case:
    id: str
    level: int
    topic: str
    question: str
    fmt: str
    kind: str
    layers: list[tuple[np.ndarray, tuple[int, int, int]]]
    truth: object = None                 # the value; None for rects / polygon, which grade on truth_mask
    truth_mask: np.ndarray | None = None
    truth_rects: list[list[int]] | None = None     # rects cases whose truth is disjoint rectangles: max_err
    tol: float | None = None
    legend: str = MONO_LEGEND
    snap: int = 10                       # every coordinate is a multiple of it; the prompt says so
    tier: int = 1                        # 1 base (c01-c22), 2 hard (c23+)

    def prompt(self, grid: bool) -> str:
        snap = ("every coordinate is a whole number of units" if self.snap == 1
                else f"every coordinate is a multiple of {self.snap} units")
        parts = [PREAMBLE.format(snap=snap), self.legend] + ([GRID_NOTE] if grid else []) + [self.question,
                                                                          ANSWER_RULE.format(fmt=self.fmt)]
        return "\n\n".join(parts)


def mono(mask: np.ndarray) -> list:
    return [(mask, BLACK)]


def boxes(rs: list[list[int]]) -> np.ndarray:
    return union(*[rect(*r) for r in rs])


# --- shapes shared by several cases ------------------------------------------------------------
STAIRCASE = poly((100, 100), [(600, 0), (0, 100), (-100, 0), (0, 100), (-100, 0), (0, 100),
                              (-100, 0), (0, 100), (-100, 0), (0, 100), (-200, 0)])        # 12 vertices
UNEVEN_U = poly((150, 150), [(500, 0), (0, 500), (-150, 0), (0, -350), (-200, 0), (0, 250),
                             (-150, 0)])                                                   # right arm taller
COMB = poly((100, 100), [(600, 0), (0, 460), (-140, 0), (0, -260), (-140, 0), (0, 350), (-100, 0),
                         (0, -350), (-120, 0), (0, 300), (-100, 0)])                       # 3 uneven teeth
NOTCHED = poly((100, 100), [(250, 0), (0, 80), (100, 0), (0, -80), (250, 0), (0, 300), (-80, 0), (0, 120),
                            (80, 0), (0, 180), (-250, 0), (0, -100), (-150, 0), (0, 100), (-200, 0),
                            (0, -200), (120, 0), (0, -150), (-120, 0)])                    # 20 vertices
PLUS = poly((300, 100), [(150, 0), (0, 250), (250, 0), (0, 150), (-250, 0), (0, 220), (-150, 0),
                         (0, -220), (-200, 0), (0, -150), (200, 0)])                       # uneven arms


def build() -> list[Case]:
    cases: list[Case] = []
    add = cases.append

    # 1 count ------------------------------------------------------------------------------------
    add(Case("c01_count_one", 1, "count", "How many separate black shapes are in the image?", INT, "int",
             mono(union(rect(300, 320, 200, 150))), truth=1))
    add(Case("c02_count_scattered", 1, "count", "How many separate black shapes are in the image?", INT, "int",
             mono(union(rect(80, 600, 120, 90), rect(350, 650, 60, 60), rect(560, 520, 180, 200),
                        rect(120, 120, 250, 140), rect(500, 90, 80, 280))), truth=5))
    add(Case("c03_count_array", 1, "count", "How many black squares are in the image?", INT, "int",
             mono(union(*[rect(110 + 110 * i, 150 + 110 * j, 40, 40) for i in range(6) for j in range(5)])),
             truth=30))
    add(Case("c04_count_corner_touch", 1, "count",
             COUNT_REGIONS + " Rectangles that share part of an edge belong to one region; shapes that "
             "meet only at a single corner point are separate regions.", INT, "int",
             mono(union(rect(100, 100, 200, 100), rect(300, 100, 100, 300),       # share the edge x = 300
                        rect(500, 500, 100, 100), rect(600, 600, 100, 100),       # corner at (600, 600)
                        rect(100, 500, 150, 150), rect(250, 550, 100, 50))),      # share the edge x = 250
             truth=4))

    # 2 locate -----------------------------------------------------------------------------------
    rs = [[230, 140, 310, 420]]
    add(Case("c05_locate_one", 2, "locate", LOCATE_ONE, RECT, "rects", mono(boxes(rs)),
             truth_mask=boxes(rs), truth_rects=rs))
    rs = [[80, 520, 200, 160], [380, 420, 140, 300], [560, 90, 170, 230]]
    add(Case("c06_locate_three", 2, "locate", LOCATE_ALL, RECTS, "rects", mono(boxes(rs)),
             truth_mask=boxes(rs), truth_rects=rs))
    rs = [[640, 610, 20, 30]]
    add(Case("c07_locate_small", 2, "locate", LOCATE_SMALL, RECT, "rects",
             mono(union(rect(100, 100, 450, 500)) | boxes(rs)), truth_mask=boxes(rs), truth_rects=rs))

    # 3 topology ---------------------------------------------------------------------------------
    add(Case("c08_gap_one_cell", 3, "topology", COUNT_REGIONS, INT, "int",
             mono(union(rect(100, 300, 300, 200), rect(410, 300, 250, 200))), truth=2))
    add(Case("c09_abutment", 3, "topology", COUNT_REGIONS, INT, "int",
             mono(union(rect(100, 300, 300, 200), rect(400, 250, 250, 150))), truth=1))
    frame = union(rect(100, 100, 600, 500)) & ~union(rect(200, 200, 150, 300), rect(450, 250, 150, 150))
    add(Case("c10_holes_island", 3, "topology",
             "How many connected black regions are there, and how many holes (white areas completely "
             "enclosed by black) do they have in total?", '{"regions": <int>, "holes": <int>}', "fields_int",
             mono(frame | union(rect(250, 300, 50, 50))), truth={"regions": 2, "holes": 2}))

    # 4 outline ----------------------------------------------------------------------------------
    add(Case("c11_vertices_L", 4, "outline", CORNERS, INT, "int",
             mono(union(poly((150, 150), [(500, 0), (0, 150), (-350, 0), (0, 400), (-150, 0)]))), truth=6))
    add(Case("c12_vertices_staircase", 4, "outline", CORNERS, INT, "int", mono(union(STAIRCASE)), truth=12))
    m = union(UNEVEN_U)
    add(Case("c13_outline_U", 4, "outline", OUTLINE, VERTICES, "polygon",
             mono(m), truth_mask=m))
    m = union(PLUS)
    add(Case("c14_outline_plus", 4, "outline", OUTLINE, VERTICES, "polygon",
             mono(m), truth_mask=m))

    # 5 measure ----------------------------------------------------------------------------------
    add(Case("c15_spacing", 5, "measure",
             "What is the horizontal distance, in layout units, between the two rectangles (the width of "
             "the white gap separating them)?", INT, "int_tol",
             mono(union(rect(120, 250, 280, 300), rect(470, 180, 200, 260))), truth=70, tol=50))
    add(Case("c16_neck_width", 5, "measure",
             "The shape is two blocks joined by a thin bar. How thick is the bar (its vertical extent), in "
             "layout units?", INT, "int_tol",
             mono(union(rect(100, 250, 200, 300), rect(500, 250, 200, 300), rect(300, 390, 200, 30))),
             truth=30, tol=30))
    add(Case("c17_area_L", 5, "measure", "What is the area of the black shape, in square layout units?", INT,
             "num_rel", mono(union(rect(100, 100, 500, 200), rect(100, 300, 200, 300))), truth=160000, tol=0.25))

    # 6 boolean ----------------------------------------------------------------------------------
    a, b = union(rect(150, 150, 400, 300)), union(rect(350, 300, 300, 350))
    add(Case("c18_boolean_xor_yes", 6, "boolean", BOOLEAN_Q, BOOL, "bool", mono(a ^ b), truth=True))
    add(Case("c19_boolean_U_no", 6, "boolean", BOOLEAN_Q, BOOL, "bool", mono(union(UNEVEN_U)), truth=False))
    add(Case("c20_min_cover", 6, "boolean",
             "What is the minimum number of axis-aligned rectangles whose union is exactly the black shape "
             "(the rectangles may overlap)?", INT, "int", mono(union(STAIRCASE)), truth=5))

    # 7 layers -----------------------------------------------------------------------------------
    metal = union(rect(80, 120, 640, 80), rect(80, 360, 640, 80), rect(80, 600, 640, 80))
    vias = union(*[rect(x, y, 40, 40) for x, y in
                   [(150, 140), (400, 140), (600, 380), (250, 620), (500, 620),    # covered
                    (330, 420),                                                   # half off a bar
                    (650, 250)]])                                                 # between bars
    add(Case("c21_layers_uncovered_vias", 7, "layers",
             "How many vias are NOT completely covered by M1?", INT, "int",
             [(metal, RED), (vias, BLUE)], truth=2,
             legend=TWO_LAYERS))

    # 8 reconstruct ------------------------------------------------------------------------------
    m = union(rect(100, 150, 400, 200), rect(300, 250, 150, 450), rect(380, 600, 320, 120))
    add(Case("c22_reconstruct_3", 8, "reconstruct",
             RECONSTRUCT, RECTS, "rects", mono(m), truth_mask=m))
    # --- tier 2: off-snap coordinates, tiny features, near-equal distractors, 20-vertex outlines,
    # --- minus / XOR reconstructions, enclosure margins ----------------------------------------
    rs = [[237, 143, 318, 411]]
    add(Case("c23_locate_offsnap", 2, "locate", LOCATE_ONE, RECT, "rects", mono(boxes(rs)),
             truth_mask=boxes(rs), truth_rects=rs, snap=1, tier=2))
    rs = [[45, 610, 135, 95], [260, 655, 70, 65], [515, 480, 215, 245], [95, 105, 285, 150], [470, 85, 75, 305]]
    add(Case("c24_locate_five_offsnap", 2, "locate", LOCATE_ALL, RECTS, "rects", mono(boxes(rs)),
             truth_mask=boxes(rs), truth_rects=rs, snap=5, tier=2))
    rs = [[655, 603, 6, 9]]
    add(Case("c25_locate_tiny", 2, "locate", LOCATE_SMALL, RECT, "rects",
             mono(union(rect(100, 100, 450, 500)) | boxes(rs)), truth_mask=boxes(rs), truth_rects=rs,
             snap=1, tier=2))
    add(Case("c26_spacing_offsnap", 5, "measure",
             "What is the horizontal distance, in layout units, between the two rectangles (the width of "
             "the white gap separating them)?", INT, "int_tol",
             mono(union(rect(120, 250, 283, 300), rect(440, 180, 200, 260))), truth=37, tol=20, snap=1, tier=2))
    add(Case("c27_min_spacing", 5, "measure",
             "What is the smallest white gap between any two of the rectangles, in layout units? Measure "
             "straight across, horizontally or vertically, between rectangles that face each other.", INT,
             "int_tol", mono(union(rect(100, 100, 200, 200), rect(335, 120, 150, 160),    # 35 across
                                   rect(110, 330, 180, 150), rect(530, 400, 170, 250))),  # 30 up from the first
             truth=30, tol=20, snap=5, tier=2))
    m = union(COMB)
    add(Case("c28_outline_comb", 4, "outline", OUTLINE, VERTICES, "polygon", mono(m), truth_mask=m, tier=2))
    add(Case("c29_vertices_notched", 4, "outline", CORNERS, INT, "int", mono(union(NOTCHED)), truth=20, tier=2))
    m = union(NOTCHED)
    add(Case("c30_outline_notched", 4, "outline", OUTLINE, VERTICES, "polygon", mono(m), truth_mask=m, tier=2))
    m = union(rect(100, 100, 600, 500)) & ~union(rect(250, 250, 300, 200), rect(100, 450, 100, 150))
    add(Case("c31_reconstruct_frame", 8, "reconstruct", RECONSTRUCT, RECTS, "rects", mono(m), truth_mask=m,
             tier=2))
    m = union(rect(80, 80, 300, 120), rect(200, 150, 120, 400), rect(260, 480, 380, 90),
              rect(560, 300, 120, 260), rect(420, 220, 200, 110), rect(80, 620, 200, 120))
    add(Case("c32_reconstruct_6", 8, "reconstruct", RECONSTRUCT, RECTS, "rects", mono(m), truth_mask=m, tier=2))
    m = union(rect(100, 100, 400, 400)) ^ union(rect(300, 300, 400, 400)) ^ union(rect(200, 200, 400, 120))
    add(Case("c33_reconstruct_xor3", 8, "reconstruct", RECONSTRUCT, RECTS, "rects", mono(m), truth_mask=m,
             tier=2))
    metal = union(rect(80, 120, 640, 100), rect(80, 380, 640, 100), rect(80, 620, 640, 100))
    vias = union(*[rect(x, y, 40, 40) for x, y in
                   [(150, 150), (400, 150), (600, 410), (250, 650), (450, 655),    # margin >= 25 all round
                    (300, 125), (500, 395),                                       # 5 and 15 below
                    (690, 650),                                                   # past the bar's end
                    (85, 410)]])                                                  # 5 from the bar's start
    add(Case("c34_layers_enclosure", 7, "layers",
             "A via is properly enclosed when M1 covers it and extends at least 20 units beyond it on all "
             "four sides. How many vias are NOT properly enclosed?", INT, "int",
             [(metal, RED), (vias, BLUE)], truth=4, snap=5, tier=2, legend=TWO_LAYERS))
    return cases
