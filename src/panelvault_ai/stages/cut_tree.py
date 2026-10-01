"""Árbol de cortes que produce el XY-Cut recursivo."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from panelvault_ai.domain import Rect


class CutAxis(StrEnum):
    """Orientación de las líneas de corte de un nodo."""

    HORIZONTAL = "horizontal"  # cortes horizontales: los hijos son filas (arriba → abajo)
    VERTICAL = "vertical"  # cortes verticales: los hijos son columnas (izquierda → derecha)


@dataclass(frozen=True, slots=True)
class CutNode:
    """Región de la página. Una hoja es una viñeta candidata.

    Los hijos se guardan siempre en orden geométrico natural: de arriba a abajo
    para cortes horizontales y de izquierda a derecha para cortes verticales.
    El sentido de lectura (occidental o manga) se aplica después, al recorrerlo.
    """

    rect: Rect
    axis: CutAxis | None = None
    children: tuple[CutNode, ...] = field(default=())

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> Iterator[CutNode]:
        """Hojas en orden geométrico natural (recorrido en profundidad)."""
        if self.is_leaf:
            yield self
            return
        for child in self.children:
            yield from child.leaves()

    @property
    def depth(self) -> int:
        return 0 if self.is_leaf else 1 + max(c.depth for c in self.children)