"""Dominio del motor: conceptos del problema, independientes de cualquier librería de visión."""

from panelvault_ai.domain.geometry import Point, Polygon, Rect
from panelvault_ai.domain.panel import PageType, Panel, PanelMap, ReadingDirection

__all__ = ["PageType", "Panel", "PanelMap", "Point", "Polygon", "ReadingDirection", "Rect"]