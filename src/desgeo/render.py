"""
render.py — layout PNGs: layer colours, an optional grid or labelled ticks, a zoom window, and a
target-vs-current diff.

Rendering rasterises shapes itself (through the raster engine), so it takes shapes, not an
engine's regions: any engine's canonical rects render the same. White background; a pixel covered
by several layers takes the mean of their colours, so overlaps read as a blend.

The origin is a display convention, not geometry: origin="bottom-left" (y up, the EDA convention)
flips the image so larger y is higher; origin="top-left" (y down, the image convention) does not,
so an image row reads directly as a layout y. Coordinates, shapes and every engine are the same
under both; only the picture, the tick labels and the words for up and down change.
pixel_mapping says how a pixel of an image maps back to layout units, for the text that goes
with the image.

    render        Layout -> PIL image, optional layer subset / window / image_px / grid / ticks
    render_diff   target and current shapes -> PIL image: match grey, missing orange, extra blue
    png_bytes     PIL image -> PNG bytes (for a tool result or a file)

Overlays, in layout units, drawn over the shapes: rulers ((x1, y1), (x2, y2), label) as a line
with end bars and the label at its middle, and labels (x, y, text[, rect]) as tagged text
(Set-of-Mark shape ids): always outside the shape, just above the rect on screen (below it at
the image's top edge), in a saturated tag colour no geometry uses, so a tag never reads as a hole
or a feature; nudged off any tag drawn before it. Markups (kind, name, coords) are an agent's own
annotations, drawn in red: a point (x, y) as a dot, a segment (x1, y1, x2, y2) as a line with end
dots, a box (x, y, w, h) as an outline, each with its name beside it. Overlays outside the window
are skipped.

A window (x0, y0, x1, y1) crops in layout units; the output is scaled with nearest-neighbour so
the longer side is image_px. A grid draws thin grey lines every grid units; ticks adds white
margins with labelled ticks every ticks units instead (labels in layout units).
"""
from __future__ import annotations

import io
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from desgeo.layout import Layout
from desgeo.raster import RasterEngine
from desgeo.shapes import GeometryError, Shape

GRID_RGB = (150, 150, 150)
MATCH_RGB = (170, 170, 170)
MISSING_RGB = (230, 120, 0)
EXTRA_RGB = (0, 90, 200)
MARGIN = 44                     # tick margin in pixels, left and bottom
PAD = 14                        # room for the last labels, top and right
TICK_LEN = 6

RULER_RGB = (0, 0, 0)
LABEL_BG = (255, 215, 0)         # saturated yellow: not white, not a layer colour
MARKUP_RGB = (230, 0, 0)
ORIGINS = ("bottom-left", "top-left")

Window = tuple[int, int, int, int]
Ruler = tuple[tuple[int, int], tuple[int, int], str]
Label = tuple  # (x, y, text) or (x, y, text, Rect): the rect the tag should not hide
Markup = tuple[str, str, tuple[int, ...]]  # (kind, name, coords): point | segment | box


def pixel_mapping(width: int, height: int, *, window: Window | None = None, image_px: int = 800,
                  ticks: int | None = None, origin: str = "bottom-left") -> dict:
    """How the pixels of an image rendered with these settings map to layout units: the plot
    area's offset in the image (the tick margins), the scale, and the layout coordinates at the
    plot's top-left pixel. Layout x = x_left + (px - left) / scale; layout y = y_top - (py - top)
    / scale when y is up (bottom-left origin), y_top + (py - top) / scale when y is down."""
    x0, y0, x1, y1 = window if window is not None else (0, 0, width, height)
    scale = image_px / max(x1 - x0, y1 - y0)
    y_up = origin == "bottom-left"
    return {"left": MARGIN if ticks else 0, "top": PAD if ticks else 0, "scale": scale,
            "x_left": x0, "y_top": y1 if y_up else y0, "y_up": y_up}


def render(layout: Layout, *, layers: Sequence[str] | None = None, window: Window | None = None,
           image_px: int = 800, grid: int | None = None, ticks: int | None = None,
           rulers: Sequence[Ruler] = (), labels: Sequence[Label] = (),
           origin: str = "bottom-left", markups: Sequence[Markup] = ()) -> Image.Image:
    eng = RasterEngine(layout.width, layout.height)
    names = list(layers) if layers is not None else list(layout.layers)
    painted = []
    for n in names:
        if n not in layout.layers:
            raise GeometryError(f"no layer {n!r}; layers are {', '.join(layout.layers) or 'none'}")
        painted.append((eng.region(layout.layers[n].shapes).bits, layout.layers[n].colour))
    return _finish(_compose(painted, layout.width, layout.height), layout.width, layout.height,
                   window, image_px, grid, ticks, rulers, labels, origin, markups)


def render_diff(width: int, height: int, target: Iterable[Shape], current: Iterable[Shape], *,
                window: Window | None = None, image_px: int = 800, grid: int | None = None,
                ticks: int | None = None, rulers: Sequence[Ruler] = (),
                labels: Sequence[Label] = (), origin: str = "bottom-left",
                markups: Sequence[Markup] = ()) -> Image.Image:
    eng = RasterEngine(width, height)
    t, c = eng.region(target).bits, eng.region(current).bits
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    img[t & c] = MATCH_RGB
    img[t & ~c] = MISSING_RGB
    img[c & ~t] = EXTRA_RGB
    return _finish(img, width, height, window, image_px, grid, ticks, rulers, labels, origin, markups)


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _compose(painted, width: int, height: int) -> np.ndarray:
    acc = np.zeros((height, width, 3))
    count = np.zeros((height, width), dtype=int)
    for bits, rgb in painted:
        acc[bits] += rgb
        count[bits] += 1
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    on = count > 0
    img[on] = np.round(acc[on] / count[on, None]).astype(np.uint8)
    return img


def _finish(img: np.ndarray, width: int, height: int, window: Window | None, image_px: int,
            grid: int | None, ticks: int | None, rulers: Sequence[Ruler] = (),
            labels: Sequence[Label] = (), origin: str = "bottom-left",
            markups: Sequence[Markup] = ()) -> Image.Image:
    if origin not in ORIGINS:
        raise GeometryError(f"origin must be one of {', '.join(ORIGINS)}, got {origin!r}")
    y_up = origin == "bottom-left"
    x0, y0, x1, y1 = window if window is not None else (0, 0, width, height)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise GeometryError(f"window {window} must lie inside the frame 0..{width} x 0..{height} "
                            f"with x0 < x1 and y0 < y1")
    scale = image_px / max(x1 - x0, y1 - y0)
    pw, ph = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
    crop = img[y0:y1, x0:x1]
    im = Image.fromarray(np.flipud(crop) if y_up else crop).resize((pw, ph), Image.NEAREST)

    def col(x: float) -> int:
        return min(round((x - x0) * scale), pw - 1)

    def row(y: float) -> int:
        return min(ph - round((y - y0) * scale) if y_up else round((y - y0) * scale), ph - 1)

    if grid:
        draw = ImageDraw.Draw(im)
        for v in _steps(x0, x1, grid):
            draw.line([(col(v), 0), (col(v), ph - 1)], fill=GRID_RGB)
        for v in _steps(y0, y1, grid):
            draw.line([(0, row(v)), (pw - 1, row(v))], fill=GRID_RGB)
    if rulers or labels:
        _overlay(im, (x0, y0, x1, y1), col, row, rulers, labels)
    if markups:
        _markups(im, (x0, y0, x1, y1), col, row, markups)
    if not ticks:
        return im
    out = Image.new("RGB", (pw + MARGIN + PAD, ph + MARGIN + PAD), (255, 255, 255))
    out.paste(im, (MARGIN, PAD))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=12)
    draw.rectangle([MARGIN - 1, PAD - 1, MARGIN + pw, PAD + ph], outline=(0, 0, 0))
    for v in _steps(x0, x1, ticks):
        cx = MARGIN + col(v)
        draw.line([(cx, PAD + ph), (cx, PAD + ph + TICK_LEN)], fill=(0, 0, 0))
        draw.text((cx, PAD + ph + TICK_LEN + 1), str(v), fill=(0, 0, 0), font=font, anchor="mt")
    for v in _steps(y0, y1, ticks):
        cy = PAD + row(v)
        draw.line([(MARGIN - TICK_LEN, cy), (MARGIN - 1, cy)], fill=(0, 0, 0))
        draw.text((MARGIN - TICK_LEN - 2, cy), str(v), fill=(0, 0, 0), font=font, anchor="rm")
    return out


def _overlay(im: Image.Image, window: Window, col, row, rulers: Sequence[Ruler],
             labels: Sequence[Label]) -> None:
    x0, y0, x1, y1 = window
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=13)

    def inside(x, y):
        return x0 <= x <= x1 and y0 <= y <= y1

    placed: list[tuple[float, float, float, float]] = []

    def tag(cx, cy, text, avoid: bool = False):
        l, t, r, b = draw.textbbox((cx, cy), text, font=font, anchor="mm")
        box = (l - 2, t - 1, r + 2, b + 1)
        if avoid:
            step = box[3] - box[1] + 1
            for _ in range(8):              # nudge up past the tags already drawn
                if not any(_overlaps(box, q) for q in placed):
                    break
                cy -= step
                box = (box[0], box[1] - step, box[2], box[3] - step)
        placed.append(box)
        draw.rectangle(list(box), fill=LABEL_BG, outline=RULER_RGB)
        draw.text((cx, cy), text, fill=RULER_RGB, font=font, anchor="mm")

    for (ax, ay), (bx, by), text in rulers:
        if not (inside(ax, ay) or inside(bx, by)):
            continue
        pa, pb = (col(ax), row(ay)), (col(bx), row(by))
        draw.line([pa, pb], fill=RULER_RGB, width=2)
        horizontal = abs(pb[0] - pa[0]) >= abs(pb[1] - pa[1])
        for px, py in (pa, pb):
            bar = [(px, py - 5), (px, py + 5)] if horizontal else [(px - 5, py), (px + 5, py)]
            draw.line(bar, fill=RULER_RGB, width=2)
        tag((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, text)
    for x, y, text, *rect in labels:
        if not inside(x, y):
            continue
        cx, cy = col(x), row(y)
        if rect:
            r = rect[0]
            _, t, _, b = draw.textbbox((0, 0), text, font=font, anchor="mm")
            half = (b - t) / 2 + 3
            top, bottom = min(row(r.y), row(r.y1)), max(row(r.y), row(r.y1))
            # outside the shape: above its top edge on screen, below it when that leaves the image
            cy = top - half - 2 if top - 2 * half - 2 >= 0 else bottom + half + 2
        tag(cx, cy, text, avoid=True)


def _markups(im: Image.Image, window: Window, col, row, markups: Sequence[Markup]) -> None:
    x0, y0, x1, y1 = window
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=13)

    def inside(x, y):
        return x0 <= x <= x1 and y0 <= y <= y1

    def dot(px, py, r=4):
        draw.ellipse([px - r, py - r, px + r, py + r], fill=MARKUP_RGB, outline=(255, 255, 255))

    def name_at(px, py, text):
        l, t, r, b = draw.textbbox((px + 7, py - 7), text, font=font, anchor="lb")
        draw.rectangle([l - 2, t - 1, r + 2, b + 1], fill=(255, 255, 255), outline=MARKUP_RGB)
        draw.text((px + 7, py - 7), text, fill=MARKUP_RGB, font=font, anchor="lb")

    for kind, name, c in markups:
        if kind == "point":
            if inside(c[0], c[1]):
                dot(col(c[0]), row(c[1]))
                name_at(col(c[0]), row(c[1]), name)
        elif kind == "segment":
            if inside(c[0], c[1]) or inside(c[2], c[3]):
                a, b = (col(c[0]), row(c[1])), (col(c[2]), row(c[3]))
                draw.line([a, b], fill=MARKUP_RGB, width=2)
                dot(*a, r=3)
                dot(*b, r=3)
                name_at((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, name)
        elif kind == "box":
            bx0, by0, bx1, by1 = c[0], c[1], c[0] + c[2], c[1] + c[3]
            if bx1 >= x0 and bx0 <= x1 and by1 >= y0 and by0 <= y1:
                pa, pb = (col(bx0), row(by0)), (col(bx1), row(by1))
                box = [min(pa[0], pb[0]), min(pa[1], pb[1]), max(pa[0], pb[0]), max(pa[1], pb[1])]
                draw.rectangle(box, outline=MARKUP_RGB, width=2)
                name_at(box[0], box[1], name)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _steps(lo: int, hi: int, step: int) -> range:
    return range(-(-lo // step) * step, hi + 1, step)
