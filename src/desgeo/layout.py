"""
layout.py — a Layout: a frame in database units, a resolution, and named layers of shapes.

The resolution is nanometres per database unit (default 0.1), metadata for reporting physical
sizes; all geometry is integer dbu. Layers keep their insertion order, which is also the render
order, and carry a display colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from desgeo.shapes import GeometryError, Shape

PALETTE = [(220, 40, 40), (40, 70, 220), (30, 150, 60), (200, 140, 0), (140, 60, 180), (0, 150, 160)]


@dataclass
class Layer:
    name: str
    colour: tuple[int, int, int]
    shapes: list[Shape] = field(default_factory=list)


@dataclass
class Layout:
    width: int
    height: int
    resolution: float = 0.1                   # nm per dbu
    layers: dict[str, Layer] = field(default_factory=dict)

    def layer(self, name: str, colour: tuple[int, int, int] | None = None) -> Layer:
        """The named layer, created (with the next palette colour unless given) if missing."""
        if name not in self.layers:
            if not name.isidentifier():
                raise GeometryError(f"layer name {name!r} must be an identifier")
            colour = colour or PALETTE[len(self.layers) % len(PALETTE)]
            self.layers[name] = Layer(name, colour)
        elif colour is not None:
            self.layers[name].colour = colour
        return self.layers[name]

    def add(self, name: str, *shapes: Shape) -> Layer:
        layer = self.layer(name)
        layer.shapes.extend(shapes)
        return layer

    def nm(self, dbu: int) -> float:
        return dbu * self.resolution
