"""desgeo statement DSL: the whitelist, the canonical DAG and its cost, evaluation on the oracle."""
import random

import pytest

from desgeo import DslError, Layout, RasterEngine, Rect, compile, run

E = RasterEngine(200, 200)
A, B, C = "rect(0, 0, 100, 50)", "rect(40, 20, 20, 30)", "rect(80, 0, 50, 80)"


def prog(body: str, **kw):
    return compile(f"a = {A}\nb = {B}\nc = {C}\n" + body, **kw)


# ---- canonical form ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y", [
    ("r = a | b", "r = b | a"),
    ("r = (a | b) | c", "r = a | (c | b)"),
    ("r = a & b & c", "r = c & (b & a)"),
    ("r = a ^ b ^ c", "r = (c ^ a) ^ b"),
    ("r = a | b", "t = b\nr = t | a | a"),
    ("r = size(a, 5)", "r = size(a, 5, 5)"),
    ("r = scale(a, 4, 2)", "r = scale(a, 2)"),
    ("r = a | size(b, 0)", "r = move(b, 0, 0) | scale(a, 3, 3)"),
    ("r = a - b", "r = rect(0, 0, 100, 50) - rect(40, 20, 20, 30)"),
])
def test_spellings_share_one_canonical_form(x, y):
    px, py = prog(x), prog(y)
    assert px.canonical() == py.canonical()
    assert (px.ops, px.variables) == (py.ops, py.variables)


def test_canonical_form_ignores_statement_order():
    x = compile("a = rect(0, 0, 50, 50)\nb = rect(25, 25, 50, 50)\nc = rect(60, 0, 20, 20)\n"
                "r = (a | b) - c")
    y = compile("c = rect(60, 0, 20, 20)\nb = rect(25, 25, 50, 50)\n"
                "r = (b | rect(0, 0, 50, 50)) - c")
    assert x.canonical() == y.canonical()


def test_polygon_spellings_are_one_node():
    cw = compile("r = poly((0, 0), [(0, 30), (10, 0), (0, -20), (30, 0), (0, -10)])")
    ccw = compile("r = poly([40, 0], [(0, 10), (-30, 0), (0, 20), (-10, 0), (0, -30)])")
    assert cw.canonical() == ccw.canonical()


def test_cost_counts_distinct_nodes_and_reports_dead_statements():
    p = prog("r = a - b\nunused = a & c\nout = r | size(r, 2)")
    assert p.ops == 5                         # a, b, a-b, size, |
    assert p.variables == 5 + 5 + 1 + 2 + 1
    assert list(p.outputs) == ["out"]
    assert p.dead == ["c (line 3)", "unused (line 5)"]
    assert prog("r = (a - b) | (a - b)").ops == 3


@pytest.mark.parametrize("src, ops, variables", [
    ("r = rect(0, 0, 10, 10) - rect(2, 2, 4, 4)", 3, 11),
    ("r = poly((0, 0), [(40, 0), (0, 40), (-10, 0), (0, -30), (-20, 0), (0, 30), (-10, 0)])", 1, 17),
    ("r = rect(0, 0, 5, 5) | rect(10, 0, 5, 5) | rect(20, 0, 5, 5)", 5, 17),
    ("a = rect(0, 0, 5, 5)\nr = a | move(a, 10, 0)", 3, 5 + 3 + 1),
    ("a = rect(10, 10, 5, 5)\nr = size(a, 2) | size(a, 2, 3) | scale(a, 2) | scale(a, 3, 2)", 8, 5 + 2 + 3 + 2 + 3 + 3),
])
def test_two_costs(src, ops, variables):
    p = compile(src)
    assert (p.ops, p.variables) == (ops, variables)


def test_subtraction_is_not_commutative():
    assert prog("r = a - b").canonical() != prog("r = b - a").canonical()


# ---- numbers ----------------------------------------------------------------------------------

def test_numbers_fold_and_division_must_be_exact():
    p = compile("w = 40\ne = 3 * w / 4 - -2\nr = size(rect(50, 50, 10, 10), w / 2, e - 30)")
    assert "size(n0, 20, 2)" in p.canonical()
    with pytest.raises(DslError, match=r"line 2: 41/2 = 20.5 is not an integer"):
        compile("w = 41\nr = size(rect(50, 50, 10, 10), w / 2)")


# ---- rejections -------------------------------------------------------------------------------

@pytest.mark.parametrize("src, msg", [
    ("import os", "line 1: every statement must be"),
    ("rect(0, 0, 1, 1)", "every statement must be"),
    ("a.b = rect(0, 0, 1, 1)", "plain name"),
    ("a, b = 1, 2", "plain name"),
    ("r = circle(0, 0, 5)", "unknown function 'circle'; the functions are rect, poly, size, move, scale"),
    ("r = os.system('x')", "unknown function 'os.system'"),
    ("r = rect(0, 0, 1.5, 1)", "only integer literals"),
    ("r = rect(0, 0, True, 1)", "only integer literals"),
    ("r = rect(x=0, y=0, w=1, h=1)", "positional arguments only"),
    ("r = rect(0, 0, 1)", "rect() takes 4 arguments, got 3"),
    ("r = rect(0, 0, 0, 1)", "rect: rect width and height must be positive"),
    ("r = rect(0, 0, 5, 5) + rect(1, 1, 5, 5)", "use | for union"),
    ("r = rect(0, 0, 5, 5) | 3", "cannot mix a region and a number"),
    ("r = -rect(0, 0, 5, 5)", "unary minus applies to numbers"),
    ("r = 2 ** 3", "operator ** does not apply to numbers"),
    ("r = lambda: 0", "Lambda is not allowed"),
    ("r = [rect(0, 0, 5, 5)][0]", "is not allowed"),
    ("r = q | rect(0, 0, 5, 5)", "undefined name 'q'"),
    ("r = rect | rect(0, 0, 5, 5)", "'rect' is a function"),
    ("rect = rect(0, 0, 5, 5)", "function name and cannot be assigned"),
    ("w = 5\nw = rect(0, 0, 5, 5)", "line 2: 'w' was a number and cannot become a region"),
    ("r = size(4, 2)", "size's first argument must be a region"),
    ("r = scale(rect(0, 0, 5, 5), 0)", "scale factor must be positive"),
    ("r = poly((0, 0), [(10, 5), (0, 10)])", "not axis-aligned"),
    ("r = poly(0, [(10, 0)])", "poly origin must be a pair"),
    ("w = 3", "defines no region"),
    ("r = rect(0, 0,", "line 1: syntax error"),
])
def test_rejections_name_the_line_and_the_rule(src, msg):
    with pytest.raises(DslError) as exc:
        compile(src)
    assert msg in str(exc.value)


def test_output_must_be_a_region():
    with pytest.raises(DslError, match="output 'w' is a number"):
        compile("w = 3\nr = rect(0, 0, 5, 5)", outputs=["w"])
    with pytest.raises(DslError, match="never assigned"):
        compile("r = rect(0, 0, 5, 5)", outputs=["M1"])


# ---- evaluation -------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_evaluation_matches_direct_engine_calls(seed):
    rng = random.Random(seed)
    rs = [Rect(rng.randint(20, 120), rng.randint(20, 120), rng.randint(1, 50), rng.randint(1, 50))
          for _ in range(4)]
    src = "\n".join(f"r{i} = rect({r.x}, {r.y}, {r.w}, {r.h})" for i, r in enumerate(rs))
    src += "\nout = move(size((r0 | r1) - r2, 3), 5, -4) ^ (r3 & scale(r1, 1))"
    got = compile(src).evaluate(E)["out"]
    m = [E.region([r]) for r in rs]
    want = E.xor(E.move(E.size(E.subtract(E.union(m[0], m[1]), m[2]), 3, 3), 5, -4),
                 E.intersect(m[3], m[1]))
    assert E.equal(got, want)


def test_layers_are_inputs_and_outputs():
    L = Layout(200, 200)
    L.add("M1", Rect(10, 10, 100, 20))
    res = run("M1 |= rect(10, 10, 20, 100)\nVIA = M1 & rect(0, 0, 40, 40)", L, outputs=["M1", "VIA"])
    assert res.layout.layers["M1"].shapes == [Rect(10, 10, 100, 20), Rect(10, 30, 20, 80)]
    assert res.layout.layers["VIA"].shapes == [Rect(10, 10, 30, 20), Rect(10, 30, 20, 10)]
    assert L.layers["M1"].shapes == [Rect(10, 10, 100, 20)]            # the input is not mutated
    assert compile("M1 = M1 | rect(0, 0, 5, 5)", layers=["M1"]).outputs.keys() == {"M1"}


def test_width_rule_as_a_program_flags_a_narrow_neck():
    """A DRC width check written in the DSL: whatever an opening by w removes is narrower than w."""
    rule = "w = 20\nnarrow = M1 - size(size(M1, -w/2), w/2)"
    for neck, flagged in ((10, True), (30, False)):
        L = Layout(200, 200)
        L.add("M1", Rect(10, 10, 60, 60), Rect(70, 30, 40, neck), Rect(110, 10, 60, 60))
        res = run(rule, L, outputs=["narrow"])
        assert bool(res.layout.layers["narrow"].shapes) is flagged
    L.layers["M1"].shapes = [Rect(10, 10, 60, 60), Rect(70, 30, 40, 10), Rect(110, 10, 60, 60)]
    assert run(rule, L, outputs=["narrow"]).layout.layers["narrow"].shapes == [Rect(70, 30, 40, 10)]


def test_run_turns_frame_errors_into_dsl_errors():
    with pytest.raises(DslError, match="frame"):
        run("r = size(rect(0, 0, 10, 10), 5)", Layout(100, 100))
