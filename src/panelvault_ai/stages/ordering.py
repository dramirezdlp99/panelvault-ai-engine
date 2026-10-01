"""Etapa 5: orden de lectura a partir del árbol de cortes."""

from __future__ import annotations

from collections.abc import Iterator

from panelvault_ai.domain import ReadingDirection, Rect
from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.cut_tree import CutAxis, CutNode
from panelvault_ai.stages.keys import CUT_TREE, ORDERED_RECTS


def reading_order(tree: CutNode | None, direction: ReadingDirection) -> tuple[Rect, ...]:
    """Recorre el árbol en profundidad aplicando el sentido de lectura.

    - Las filas (cortes horizontales) se leen siempre de arriba hacia abajo.
    - Las columnas (cortes verticales) se leen de izquierda a derecha en el cómic
      occidental y de derecha a izquierda en el manga.

    Como el árbol agrupa jerárquicamente, una viñeta alta a la izquierda y dos
    apiladas a la derecha se leen correctamente: primero se termina una columna
    entera antes de pasar a la siguiente.
    """
    if tree is None:
        return ()
    return tuple(_walk(tree, direction))


def _walk(node: CutNode, direction: ReadingDirection) -> Iterator[Rect]:
    if node.is_leaf:
        yield node.rect
        return
    children = node.children
    if node.axis is CutAxis.VERTICAL and direction is ReadingDirection.RIGHT_TO_LEFT:
        children = tuple(reversed(children))
    for child in children:
        yield from _walk(child, direction)


@STAGES.register("order")
class ReadingOrderStage(Stage):
    """Convierte el árbol de cortes en una lista de viñetas en orden de lectura."""

    name = "order"
    requires = frozenset({CUT_TREE})
    provides = frozenset({ORDERED_RECTS})

    def __init__(self, direction: str = "ltr") -> None:
        self.direction = ReadingDirection(direction)

    def _process(self, ctx: PageContext) -> None:
        ctx.set(ORDERED_RECTS, reading_order(ctx.get(CUT_TREE), self.direction))