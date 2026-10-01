"""Resultado del análisis de una página: viñetas, orden de lectura y confianza."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from panelvault_ai.domain.geometry import Polygon, Rect


class ReadingDirection(StrEnum):
    """Sentido de lectura horizontal de la página."""

    LEFT_TO_RIGHT = "ltr"  # cómic occidental
    RIGHT_TO_LEFT = "rtl"  # manga


class PageType(StrEnum):
    """Clasificación general de la composición de la página."""

    GRID = "grid"  # viñetas rectangulares alineadas
    IRREGULAR = "irregular"  # viñetas inclinadas, superpuestas o sin medianil claro
    SPLASH = "splash"  # una sola ilustración ocupa la página
    UNKNOWN = "unknown"


def _check_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"La confianza debe estar entre 0 y 1, se recibió {value}")


@dataclass(frozen=True, slots=True)
class Panel:
    """Una viñeta detectada. Inmutable (Value Object)."""

    shape: Polygon
    order: int
    confidence: float

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError("El orden de lectura no puede ser negativo")
        _check_confidence(self.confidence)

    @property
    def bounds(self) -> Rect:
        return self.shape.bounding_rect


@dataclass(frozen=True, slots=True)
class PanelMap:
    """Mapa de viñetas de una página, en coordenadas de la imagen analizada.

    Invariante: las viñetas están ordenadas y sus órdenes son exactamente 0, 1, ..., n−1.
    """

    page_width: int
    page_height: int
    direction: ReadingDirection
    page_type: PageType
    panels: tuple[Panel, ...]
    confidence: float

    SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("Las dimensiones de la página deben ser positivas")
        _check_confidence(self.confidence)
        object.__setattr__(self, "panels", tuple(self.panels))
        orders = [p.order for p in self.panels]
        if orders != list(range(len(self.panels))):
            raise ValueError(
                f"Los órdenes de lectura deben ser 0..n-1 y estar en secuencia; se recibió {orders}"
            )

    def __len__(self) -> int:
        return len(self.panels)

    def to_normalized_dict(self) -> dict[str, Any]:
        """Representación serializable con coordenadas de 0 a 1, lista para el visor."""
        w, h = self.page_width, self.page_height
        return {
            "schemaVersion": self.SCHEMA_VERSION,
            "direction": self.direction.value,
            "pageType": self.page_type.value,
            "confidence": round(self.confidence, 4),
            "panels": [
                {
                    "order": p.order,
                    "confidence": round(p.confidence, 4),
                    "bbox": _rect_to_list(p.bounds.normalized(w, h)),
                    "polygon": [
                        [round(x, 6), round(y, 6)] for x, y in p.shape.normalized(w, h).points
                    ],
                }
                for p in self.panels
            ],
        }


def _rect_to_list(r: Rect) -> list[float]:
    return [round(r.x, 6), round(r.y, 6), round(r.width, 6), round(r.height, 6)]