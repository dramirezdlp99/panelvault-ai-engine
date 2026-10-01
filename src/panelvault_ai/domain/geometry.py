"""Primitivas geométricas del dominio.

Convención de coordenadas (la misma de las imágenes):
- El origen (0, 0) está en la esquina superior izquierda.
- ``x`` crece hacia la derecha y ``y`` crece hacia abajo.
- Un rectángulo cubre el intervalo semiabierto [x, x + width) × [y, y + height).

Este módulo no depende de OpenCV ni de NumPy: describe *qué* es una región,
no *cómo* se detecta.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """Rectángulo alineado a los ejes. Inmutable (Value Object)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"Un Rect no puede tener dimensiones negativas: "
                f"width={self.width}, height={self.height}"
            )

    # ------------------------------------------------------------------
    # Construcción alternativa
    # ------------------------------------------------------------------
    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> Rect:
        """Crea un Rect a partir de dos esquinas opuestas, en cualquier orden."""
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return cls(left, top, right - left, bottom - top)

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------
    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def is_empty(self) -> bool:
        return self.area == 0

    # ------------------------------------------------------------------
    # Relaciones entre rectángulos
    # ------------------------------------------------------------------
    def intersection(self, other: Rect) -> Rect | None:
        """Región común a ambos rectángulos, o ``None`` si no se solapan.

        Dos rectángulos que solo se tocan en un borde no comparten área,
        por lo que tampoco se consideran solapados.
        """
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def union_bounds(self, other: Rect) -> Rect:
        """Menor rectángulo que contiene a ambos."""
        return Rect.from_corners(
            min(self.x, other.x),
            min(self.y, other.y),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def iou(self, other: Rect) -> float:
        """Intersection over Union: cuánto se solapan dos regiones, de 0 a 1.

        IoU = área(A ∩ B) / área(A ∪ B), donde área(A ∪ B) = área(A) + área(B) − área(A ∩ B).
        Vale 1 si son idénticos y 0 si no comparten área.
        """
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union_area = self.area + other.area - inter.area
        return inter.area / union_area

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def contains(self, other: Rect) -> bool:
        """True si ``other`` queda completamente dentro de este rectángulo."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    # ------------------------------------------------------------------
    # Transformaciones (devuelven un Rect nuevo)
    # ------------------------------------------------------------------
    def scaled(self, sx: float, sy: float | None = None) -> Rect:
        """Escala posición y tamaño. Útil para volver de la imagen reducida a la original."""
        if sy is None:
            sy = sx
        if sx <= 0 or sy <= 0:
            raise ValueError("Los factores de escala deben ser positivos")
        return Rect(self.x * sx, self.y * sy, self.width * sx, self.height * sy)

    def normalized(self, page_width: float, page_height: float) -> Rect:
        """Expresa el rectángulo como fracción (0 a 1) del tamaño de la página.

        Es el formato que consume el visor: sirve para cualquier resolución de la imagen.
        """
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Las dimensiones de la página deben ser positivas")
        return self.scaled(1 / page_width, 1 / page_height)