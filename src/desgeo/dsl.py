"""
dsl.py — the geometry language: named statements, a whitelisted subset of Python syntax,
compiled to a canonical expression DAG that any engine can evaluate.

    w = 40                              numbers: int literals, + - * /, unary minus; / must be exact
    a = rect(0, 0, 100, 50)             rect(x, y, w, h)
    p = poly((0, 0), [(100, 0), (0, 50), (-60, 0), (0, 30), (-40, 0)])   origin + deltas (poly="deltas")
    p = poly([(0, 0), (100, 0), (100, 50), (40, 50), (40, 80), (0, 80)])  corners (poly="points")
    c = a - b                           regions: | union, & intersect, - subtract, ^ xor
    d = size(c, w/2)                    size(r, d) or size(r, dx, dy): grow > 0, shrink < 0
    e = move(d, 10, 0)                  move(r, dx, dy)
    f = scale(e, 3, 2)                  scale(r, k) or scale(r, num, den), about the origin
    M1 = c | f                          assigning a layer name writes that layer
    a |= b                              augmented assignment is sugar for a = a | b

Only assignments to plain names; nothing is ever executed, the source is only parsed. A layer
name read before it is assigned is that layer's input geometry. A name may be rebound, but not
from a number to a region or back.

Canonical form: commutative operators are flattened and their operands sorted by structural
digest; | and & drop duplicate operands; size 0, move 0 and scale 1 vanish; identical
subexpressions are shared. Two costs are read off the DAG of what the outputs use, each shared
node counted once, so a spelling difference never changes them but a redundant operation does:

    ops         operations: 1 per leaf (rect, poly, input layer) and per operator; an n-ary
                | & ^ counts n - 1, as it is written with binary operators
    variables   the numbers and names a program must fix: a rect 5 (a name, the origin, the size),
                a poly 3 + 2 per delta (a name, the origin, each step), an input layer 1, a boolean
                1 per binary operator, size 1 + its amounts (1, or 2 when dx != dy), move 3,
                scale 1 + 1 (a whole factor) or 2 (a fraction)

So `rect - rect` is 3 ops and 11 variables, and an 8-vertex poly is 1 op and 17 variables.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

from desgeo.layout import Layout
from desgeo.raster import RasterEngine
from desgeo.shapes import GeometryError, Polygon, Rect

MAX_SOURCE = 100_000
MAX_STATEMENTS = 1000
FUNCTIONS = {"rect": (4,), "poly": (2,), "size": (2, 3), "move": (3,), "scale": (2, 3)}
# How a program writes a polygon, chosen per compile: a start point and axis-aligned steps (the
# OASIS form), or its corners in order. The two compile to the same node; only one is accepted.
POLY_SYNTAX = {
    "deltas": ("poly((x, y), [(dx, dy), ...]): a start point and axis-aligned steps", 2),
    "points": ("poly([(x0, y0), (x1, y1), ...]): the corners in order", 1),
}
REGION_OPS = {ast.BitOr: "or", ast.BitAnd: "and", ast.BitXor: "xor", ast.Sub: "sub"}
COMMUTATIVE = {"or", "and", "xor"}
IDEMPOTENT = {"or", "and"}
SYMBOL = {"or": "|", "and": "&", "xor": "^", "sub": "-"}


class DslError(ValueError):
    """A program the language rejects; the message is model-facing and names the line."""


class Node:
    """One canonical DAG node. op is a leaf (rect, poly, layer) or an operator; args holds child
    Nodes and int / tuple literals. Equality is structural, through the digest."""
    __slots__ = ("op", "args", "digest")

    def __init__(self, op: str, args: tuple):
        self.op, self.args = op, args
        parts = [op] + [a.digest if isinstance(a, Node) else repr(a) for a in args]
        self.digest = hashlib.sha1("\x1f".join(parts).encode()).hexdigest()

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and other.digest == self.digest

    def __hash__(self) -> int:
        return hash(self.digest)

    def __repr__(self) -> str:
        return f"Node({self.op}, {self.digest[:8]})"

    @property
    def children(self) -> list[Node]:
        return [a for a in self.args if isinstance(a, Node)]


@dataclass
class Program:
    outputs: dict[str, Node]
    order: list[Node]                       # every reachable node, children before parents,
                                            # in a spelling-independent order
    dead: list[str] = field(default_factory=list)   # "name (line n)" of statements no output uses

    @property
    def ops(self) -> int:
        return sum(_ops(n) for n in self.order)

    @property
    def variables(self) -> int:
        return sum(_variables(n) for n in self.order)

    def evaluate(self, engine, layout: Layout | None = None) -> dict[str, Any]:
        """Evaluate every output on the engine; each shared node is computed once."""
        val: dict[str, Any] = {}
        for n in self.order:
            val[n.digest] = _eval(n, val, engine, layout)
        return {name: val[n.digest] for name, n in self.outputs.items()}

    def canonical(self) -> str:
        """The DAG written back as numbered statements, outputs last: one text per program
        meaning-up-to-spelling."""
        ids = {n.digest: f"n{i}" for i, n in enumerate(self.order)}
        lines = [f"{ids[n.digest]} = {_text(n, ids)}" for n in self.order]
        lines += [f"{name} = {ids[n.digest]}" for name, n in self.outputs.items()]
        return "\n".join(lines)


@dataclass
class Result:
    program: Program
    regions: dict[str, Any]
    layout: Layout                          # a copy with every output written as a layer


def compile(src: str, layers: Iterable[str] = (), outputs: Iterable[str] | None = None,
            poly: str = "deltas") -> Program:
    """Parse and compile. layers are the names readable as input geometry. outputs defaults to
    the layer names the program assigns, else the last region statement. poly is the polygon
    syntax the program must use (POLY_SYNTAX)."""
    if poly not in POLY_SYNTAX:
        raise ValueError(f"poly syntax must be one of {', '.join(POLY_SYNTAX)}, got {poly!r}")
    return _Compiler(set(layers), poly).run(src, list(outputs) if outputs is not None else None)


def run(src: str, layout: Layout, outputs: Iterable[str] | None = None, engine=None,
        poly: str = "deltas") -> Result:
    """Compile against the layout's layers, evaluate (raster oracle by default), and return a
    copy of the layout with each output written as a layer of canonical rects."""
    program = compile(src, layout.layers, outputs, poly)
    engine = engine or RasterEngine(layout.width, layout.height)
    try:
        regions = program.evaluate(engine, layout)
    except GeometryError as e:
        raise DslError(str(e)) from None
    out = copy.deepcopy(layout)
    for name, region in regions.items():
        out.layer(name).shapes = list(engine.rects(region))
    return Result(program, regions, out)


# ---- compiler ----------------------------------------------------------------------------------

class _Compiler:
    def __init__(self, layers: set[str], poly: str = "deltas"):
        self.layers = layers
        self.poly = poly
        self.env: dict[str, int | Node] = {}
        self.interned: dict[str, Node] = {}
        self.statements: list[tuple[str, int, Node]] = []
        self.line = 0

    def run(self, src: str, outputs: list[str] | None) -> Program:
        if len(src) > MAX_SOURCE:
            raise DslError(f"program is {len(src)} characters; the limit is {MAX_SOURCE}")
        try:
            tree = ast.parse(src, mode="exec")
        except SyntaxError as e:
            raise DslError(f"line {e.lineno}: syntax error: {e.msg}") from None
        except RecursionError:
            raise DslError("expression nested too deeply") from None
        if len(tree.body) > MAX_STATEMENTS:
            raise DslError(f"{len(tree.body)} statements; the limit is {MAX_STATEMENTS}")
        for stmt in tree.body:
            self.line = stmt.lineno
            self.statement(stmt)
        return self.finish(outputs)

    def fail(self, msg: str):
        raise DslError(f"line {self.line}: {msg}")

    def statement(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                self.fail("assign to exactly one plain name: `name = expression`")
            name, value = stmt.targets[0].id, self.expr(stmt.value)
        elif isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, ast.Name):
                self.fail("assign to exactly one plain name: `name = expression`")
            name = stmt.target.id
            value = self.binop(stmt.op, self.load(name), self.expr(stmt.value))
        else:
            self.fail("every statement must be `name = expression`")
        if name in FUNCTIONS:
            self.fail(f"{name!r} is a function name and cannot be assigned")
        old = self.env.get(name, Node("layer", (name,)) if name in self.layers else None)
        if old is not None and isinstance(old, Node) != isinstance(value, Node):
            self.fail(f"{name!r} was a {_kind(old)} and cannot become a {_kind(value)}")
        self.env[name] = value
        if isinstance(value, Node):
            self.statements.append((name, self.line, value))

    def finish(self, outputs: list[str] | None) -> Program:
        if outputs is None:
            outputs = [n for n in dict.fromkeys(s[0] for s in self.statements) if n in self.layers]
            if not outputs:
                if not self.statements:
                    raise DslError("the program defines no region")
                outputs = [self.statements[-1][0]]
        out: dict[str, Node] = {}
        for name in outputs:
            v = self.env.get(name)
            if not isinstance(v, Node):
                raise DslError(f"output {name!r} is {'a number' if v is not None else 'never assigned'}")
            out[name] = v
        # Post-order from the outputs, children in stored order (commutative operands are already
        # sorted by digest), so the order depends on the DAG and not on how it was spelled.
        order: list[Node] = []
        reach: set[str] = set()
        stack = [(n, False) for n in reversed(out.values())]
        while stack:
            n, expanded = stack.pop()
            if n.digest in reach:
                continue
            if expanded:
                reach.add(n.digest)
                order.append(n)
            else:
                stack.append((n, True))
                stack.extend((c, False) for c in reversed(n.children))
        dead = [f"{name} (line {line})" for name, line, n in self.statements if n.digest not in reach]
        return Program(out, order, dead)

    # ---- expressions -------------------------------------------------------------------------

    def node(self, op: str, args: tuple) -> Node:
        n = Node(op, args)
        if n.digest not in self.interned:
            self.interned[n.digest] = n
        return self.interned[n.digest]

    def load(self, name: str) -> int | Node:
        if name in self.env:
            return self.env[name]
        if name in self.layers:
            return self.node("layer", (name,))
        if name in FUNCTIONS:
            self.fail(f"{name!r} is a function; call it as {name}(...)")
        self.fail(f"undefined name {name!r}")

    def expr(self, e: ast.expr) -> int | Node:
        if isinstance(e, ast.Constant):
            if isinstance(e.value, bool) or not isinstance(e.value, int):
                self.fail(f"only integer literals are allowed, got {e.value!r}")
            return e.value
        if isinstance(e, ast.Name):
            return self.load(e.id)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            v = self.expr(e.operand)
            if isinstance(v, Node):
                self.fail("unary minus applies to numbers, not regions")
            return -v if isinstance(e.op, ast.USub) else v
        if isinstance(e, ast.BinOp):
            return self.binop(e.op, self.expr(e.left), self.expr(e.right))
        if isinstance(e, ast.Call):
            return self.call(e)
        self.fail(f"{type(e).__name__} is not allowed here")

    def binop(self, op: ast.operator, a, b) -> int | Node:
        if isinstance(a, Node) != isinstance(b, Node):
            self.fail("cannot mix a region and a number in one operator")
        if isinstance(a, Node):
            if type(op) not in REGION_OPS:
                hint = "; use | for union" if isinstance(op, ast.Add) else ""
                self.fail(f"operator {_opname(op)} does not apply to regions{hint}")
            return self.region_op(REGION_OPS[type(op)], a, b)
        if isinstance(op, ast.Add):
            return a + b
        if isinstance(op, ast.Sub):
            return a - b
        if isinstance(op, ast.Mult):
            return a * b
        if isinstance(op, ast.Div):
            if b == 0:
                self.fail("division by zero")
            q = Fraction(a, b)
            if q.denominator != 1:
                self.fail(f"{a}/{b} = {float(q):g} is not an integer")
            return int(q)
        self.fail(f"operator {_opname(op)} does not apply to numbers")

    def region_op(self, op: str, a: Node, b: Node) -> Node:
        if op not in COMMUTATIVE:
            return self.node(op, (a, b))
        operands = []
        for x in (a, b):
            operands.extend(x.args if x.op == op else (x,))
        if op in IDEMPOTENT:
            operands = list({x.digest: x for x in operands}.values())
            if len(operands) == 1:
                return operands[0]
        return self.node(op, tuple(sorted(operands, key=lambda x: x.digest)))

    def call(self, e: ast.Call) -> Node:
        if not isinstance(e.func, ast.Name) or e.func.id not in FUNCTIONS:
            name = e.func.id if isinstance(e.func, ast.Name) else ast.unparse(e.func)
            self.fail(f"unknown function {name!r}; the functions are {', '.join(FUNCTIONS)}")
        fn = e.func.id
        if e.keywords:
            self.fail(f"{fn}() takes positional arguments only")
        if fn == "poly":
            form, arity = POLY_SYNTAX[self.poly]
            if len(e.args) != arity:
                self.fail(f"poly is written {form}")
        elif len(e.args) not in FUNCTIONS[fn]:
            self.fail(f"{fn}() takes {' or '.join(map(str, FUNCTIONS[fn]))} arguments, got {len(e.args)}")
        try:
            return getattr(self, f"call_{fn}")(e.args)
        except GeometryError as err:
            self.fail(f"{fn}: {err}")

    def number(self, e: ast.expr, what: str) -> int:
        v = self.expr(e)
        if isinstance(v, Node):
            self.fail(f"{what} must be a number, got a region")
        return v

    def region(self, e: ast.expr, what: str) -> Node:
        v = self.expr(e)
        if not isinstance(v, Node):
            self.fail(f"{what} must be a region, got the number {v}")
        return v

    def pair(self, e: ast.expr, what: str) -> tuple[int, int]:
        if not isinstance(e, (ast.Tuple, ast.List)) or len(e.elts) != 2:
            self.fail(f"{what} must be a pair (x, y)")
        return self.number(e.elts[0], what), self.number(e.elts[1], what)

    def call_rect(self, args) -> Node:
        r = Rect(*(self.number(a, "rect argument") for a in args))
        return self.node("rect", (r.x, r.y, r.w, r.h))

    def call_poly(self, args) -> Node:
        if self.poly == "points":
            if not isinstance(args[0], (ast.Tuple, ast.List)) or \
                    not all(isinstance(c, (ast.Tuple, ast.List)) for c in args[0].elts):
                self.fail(f"poly is written {POLY_SYNTAX['points'][0]}")
            p = Polygon.from_points([self.pair(c, "poly corner") for c in args[0].elts])
            return self.node("poly", (tuple(p.origin), p.deltas))
        origin = self.pair(args[0], "poly origin")
        if not isinstance(args[1], (ast.Tuple, ast.List)):
            self.fail("poly deltas must be a list of (dx, dy) pairs")
        deltas = [self.pair(d, "poly delta") for d in args[1].elts]
        p = Polygon.of(origin, deltas)
        return self.node("poly", (tuple(p.origin), p.deltas))

    def call_size(self, args) -> Node:
        r = self.region(args[0], "size's first argument")
        dx = self.number(args[1], "size amount")
        dy = self.number(args[2], "size amount") if len(args) == 3 else dx
        return r if dx == dy == 0 else self.node("size", (r, dx, dy))

    def call_move(self, args) -> Node:
        r = self.region(args[0], "move's first argument")
        dx, dy = self.number(args[1], "move dx"), self.number(args[2], "move dy")
        return r if dx == dy == 0 else self.node("move", (r, dx, dy))

    def call_scale(self, args) -> Node:
        r = self.region(args[0], "scale's first argument")
        num = self.number(args[1], "scale factor")
        den = self.number(args[2], "scale denominator") if len(args) == 3 else 1
        if num <= 0 or den <= 0:
            self.fail(f"scale factor must be positive, got {num}/{den}")
        f = Fraction(num, den)
        return r if f == 1 else self.node("scale", (r, f.numerator, f.denominator))


# ---- evaluation and text -----------------------------------------------------------------------

def _eval(n: Node, val: dict, engine, layout: Layout | None):
    a = [val[x.digest] if isinstance(x, Node) else x for x in n.args]
    if n.op == "rect":
        return engine.region([Rect(*a)])
    if n.op == "poly":
        return engine.region([Polygon.of(*a)])
    if n.op == "layer":
        if layout is None or a[0] not in layout.layers:
            raise GeometryError(f"no input layer {a[0]!r}")
        return engine.region(layout.layers[a[0]].shapes)
    if n.op in COMMUTATIVE:
        f = {"or": engine.union, "and": engine.intersect, "xor": engine.xor}[n.op]
        acc = a[0]
        for x in a[1:]:
            acc = f(acc, x)
        return acc
    if n.op == "sub":
        return engine.subtract(a[0], a[1])
    return getattr(engine, n.op)(*a)                # size, move, scale


def _ops(n: Node) -> int:
    return len(n.args) - 1 if n.op in COMMUTATIVE else 1


def _variables(n: Node) -> int:
    if n.op == "rect":
        return 5
    if n.op == "poly":
        return 3 + 2 * len(n.args[1])
    if n.op == "layer":
        return 1
    if n.op in COMMUTATIVE:
        return len(n.args) - 1
    if n.op == "sub":
        return 1
    if n.op == "size":
        return 2 if n.args[1] == n.args[2] else 3
    if n.op == "move":
        return 3
    if n.op == "scale":
        return 2 if n.args[2] == 1 else 3
    raise AssertionError(f"no variable count for {n.op}")


def _text(n: Node, ids: dict[str, str]) -> str:
    a = [ids[x.digest] if isinstance(x, Node) else x for x in n.args]
    if n.op == "rect":
        return f"rect({', '.join(map(str, a))})"
    if n.op == "poly":
        return f"poly({a[0]}, {list(a[1])})"
    if n.op == "layer":
        return a[0]
    if n.op in SYMBOL:
        return f" {SYMBOL[n.op]} ".join(a)
    return f"{n.op}({', '.join(map(str, a))})"


def _kind(v) -> str:
    return "region" if isinstance(v, Node) else "number"


def _opname(op: ast.operator) -> str:
    return {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
            ast.Mod: "%", ast.Pow: "**", ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
            ast.LShift: "<<", ast.RShift: ">>", ast.MatMult: "@"}.get(type(op), type(op).__name__)
