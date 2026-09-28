"""
desgeo — Manhattan layout geometry: shapes, layouts, engines, rendering and the statement DSL.

A pure library with no desh imports; the desh plugin and the geo evals build on it.
"""
from desgeo.dsl import DslError, Node, Program, Result, compile, run
from desgeo.layout import Layer, Layout
from desgeo.metrology import Metrology, Ruler, Snap
from desgeo.raster import Mask, RasterEngine
from desgeo.render import png_bytes, render, render_diff
from desgeo.shapes import GeometryError, Point, Polygon, Rect, Shape

__all__ = ["DslError", "GeometryError", "Layer", "Layout", "Mask", "Metrology", "Node", "Point", "Polygon",
           "Program", "RasterEngine", "Rect", "Result", "Ruler", "Shape", "Snap", "compile", "png_bytes", "render",
           "render_diff", "run"]
