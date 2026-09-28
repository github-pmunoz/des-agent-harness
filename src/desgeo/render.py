"""
render.py — layout PNGs: layer colours, an optional grid or labelled ticks, a zoom window, and a
target-vs-current diff.

Rendering rasterises shapes itself (through the raster engine), so it takes shapes, not an
engine's regions: any engine's canonical rects render the same. White background; a pixel covered
by several layers takes the mean of their colours, so overlaps read as a blend. Masks are
bottom-up and images top-down, so every image is flipped on the way out.

    render        Layout -> PIL image, optional layer subset / window / image_px / grid / ticks
    render_diff   target and current shapes -> PIL image: match grey, missing orange, extra blue
    png_bytes     PIL image -> PNG bytes (for a tool result or a file)

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

Window = tuple[int, int, int, int]


def render(layout: Layout, *, layers: Sequence[str] | None = None, window: Window | None = None,
           image_px: int = 800, grid: int | None = None, ticks: int | None = None) -> Image.Image:
    eng = RasterEngine(layout.width, layout.height)
    names = list(layers) if layers is not None else list(layout.layers)
    painted = []
    for n in names:
        if n not in layout.layers:
            raise GeometryError(f"no layer {n!r}; layers are {', '.join(layout.layers) or 'none'}")
        painted.append((eng.region(layout.layers[n].shapes).bits, layout.layers[n].colour))
    return _finish(_compose(painted, layout.width, layout.height), layout.width, layout.height,
                   window, image_px, grid, ticks)


def render_diff(width: int, height: int, target: Iterable[Shape], current: Iterable[Shape], *,
                window: Window | None = None, image_px: int = 800, grid: int | None = None,
                ticks: int | None = None) -> Image.Image:
    eng = RasterEngine(width, height)
    t, c = eng.region(target).bits, eng.region(current).bits
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    img[t & c] = MATCH_RGB
    img[t & ~c] = MISSING_RGB
    img[c & ~t] = EXTRA_RGB
    return _finish(img, width, height, window, image_px, grid, ticks)


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
            grid: int | None, ticks: int | None) -> Image.Image:
    x0, y0, x1, y1 = window if window is not None else (0, 0, width, height)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise GeometryError(f"window {window} must lie inside the frame 0..{width} x 0..{height} "
                            f"with x0 < x1 and y0 < y1")
    scale = image_px / max(x1 - x0, y1 - y0)
    pw, ph = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
    im = Image.fromarray(np.flipud(img[y0:y1, x0:x1])).resize((pw, ph), Image.NEAREST)

    def col(x: float) -> int:
        return min(round((x - x0) * scale), pw - 1)

    def row(y: float) -> int:
        return min(ph - round((y - y0) * scale), ph - 1)

    if grid:
        draw = ImageDraw.Draw(im)
        for v in _steps(x0, x1, grid):
            draw.line([(col(v), 0), (col(v), ph - 1)], fill=GRID_RGB)
        for v in _steps(y0, y1, grid):
            draw.line([(0, row(v)), (pw - 1, row(v))], fill=GRID_RGB)
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
        draw.line([(MARGIN - TICK_LEN, cy), (MARGIN, cy)], fill=(0, 0, 0))
        draw.text((MARGIN - TICK_LEN - 2, cy), str(v), fill=(0, 0, 0), font=font, anchor="rm")
    return out


def _steps(lo: int, hi: int, step: int) -> range:
    return range(-(-lo // step) * step, hi + 1, step)
