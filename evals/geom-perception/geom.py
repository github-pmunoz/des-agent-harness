"""
geom.py — the raster half of the perception eval: shapes on an integer layout grid -> masks -> PNG.

A stand-in for the raster engine of the planned geometry library. Coordinates are layout units,
origin at the bottom-left, y up. A mask is indexed [y, x], so row 0 is the bottom of the layout
and every PNG is flipped on the way out (and back on the way in).

    rect           (x, y, w, h) -> vertex list
    poly           origin + deltas -> vertex list; the outline closes back to the origin
    polygon_mask   one closed polygon -> bool mask, even-odd over pixel centres (any edge angle)
    union          several polygons -> the union of their masks
    iou            two masks -> intersection over union
    render         [(mask, rgb), ...] -> PNG, resized to image_px, optional grid overlay
    save_mask      mask -> 1-bit PNG, viewable (flipped); load_mask reverses it
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SIZE = 800                      # layout width and height, in units; one unit per pixel at 800 px
GRID_STEP = 100                 # overlay line spacing, in units
GRID_RGB = (150, 150, 150)      # mid grey: visible on white and on black
_CENTRES = np.arange(SIZE) + 0.5


def rect(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def poly(origin: tuple[int, int], deltas: list[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = origin
    vertices = [(x, y)]
    for dx, dy in deltas:
        x, y = x + dx, y + dy
        vertices.append((x, y))
    if vertices[-1] == vertices[0]:
        vertices.pop()
    return vertices


def polygon_mask(vertices) -> np.ndarray:
    """Pixels whose centre lies inside the polygon, by the even-odd rule. Each non-horizontal edge
    toggles every pixel left of it on the rows its half-open y range covers."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    n = len(vertices)
    for i in range(n):
        (x0, y0), (x1, y1) = vertices[i], vertices[(i + 1) % n]
        if y0 == y1:
            continue
        rows = (_CENTRES >= min(y0, y1)) & (_CENTRES < max(y0, y1))
        if not rows.any():
            continue
        x_edge = x0 + (_CENTRES[rows] - y0) * (x1 - x0) / (y1 - y0)
        mask[rows] ^= _CENTRES[None, :] < x_edge[:, None]
    return mask


def union(*polygons) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for p in polygons:
        mask |= polygon_mask(p)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return 1.0 if u == 0 else int(np.count_nonzero(a & b)) / int(u)


def render(layers: list[tuple[np.ndarray, tuple[int, int, int]]], path, grid: bool = False,
           image_px: int = SIZE) -> None:
    """White background; a pixel covered by several layers takes the mean of their colours."""
    acc = np.zeros((SIZE, SIZE, 3))
    count = np.zeros((SIZE, SIZE), dtype=int)
    for mask, rgb in layers:
        acc[mask] += rgb
        count[mask] += 1
    img = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    painted = count > 0
    img[painted] = np.round(acc[painted] / count[painted, None]).astype(np.uint8)
    im = Image.fromarray(np.flipud(img))
    if image_px != SIZE:
        im = im.resize((image_px, image_px), Image.NEAREST)
    if grid:
        draw = ImageDraw.Draw(im)
        scale = image_px / SIZE
        for v in range(0, SIZE + 1, GRID_STEP):
            col = min(round(v * scale), image_px - 1)
            row = min(image_px - round(v * scale), image_px - 1)
            draw.line([(col, 0), (col, image_px - 1)], fill=GRID_RGB)
            draw.line([(0, row), (image_px - 1, row)], fill=GRID_RGB)
    im.save(path)


def save_mask(mask: np.ndarray, path) -> None:
    Image.fromarray(np.flipud(mask)).convert("1").save(path)


def load_mask(path) -> np.ndarray:
    return np.flipud(np.array(Image.open(path).convert("1"), dtype=bool))
