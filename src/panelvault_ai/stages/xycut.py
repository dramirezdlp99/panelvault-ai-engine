"""Etapa 4: XY-Cut recursivo, el núcleo de la segmentación de viñetas."""

from __future__ import annotations

import numpy as np

from panelvault_ai.domain import Rect
from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.cut_tree import CutAxis, CutNode
from panelvault_ai.stages.keys import CONTENT_MASK, CUT_TREE


def find_gaps(empty: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Tramos consecutivos de ``True`` de longitud >= ``min_gap``, como [inicio, fin).

    Se ignoran los tramos que tocan los extremos: esos son márgenes, no medianiles
    entre dos regiones con contenido.
    """
    if empty.size == 0:
        return []
    padded = np.concatenate(([False], empty, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(s), int(e))
        for s, e in zip(starts, ends, strict=True)
        if e - s >= min_gap and s > 0 and e < empty.size
    ]


def split_by_gaps(length: int, gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Segmentos [inicio, fin) que quedan entre los medianiles."""
    segments = []
    start = 0
    for gap_start, gap_end in gaps:
        segments.append((start, gap_start))
        start = gap_end
    segments.append((start, length))
    return segments


@STAGES.register("xycut")
class XYCutStage(Stage):
    """Divide la página recursivamente por sus medianiles y construye un árbol.

    Para cada región:

    1. **Recorta** la región al rectángulo mínimo que contiene contenido
       (descarta márgenes vacíos).
    2. Calcula los **perfiles de proyección**: la fracción de contenido de cada
       fila y de cada columna. Una fila casi vacía es parte de un medianil.
    3. Busca **medianiles**: tramos de filas (o columnas) vacías consecutivas.
    4. Si hay medianiles horizontales, corta en **filas**; si no, prueba con
       columnas. Se prefieren las filas porque en el cómic occidental y en el
       manga una cuadrícula se lee fila por fila, no columna por columna.
    5. Repite sobre cada pedazo. Una región sin medianiles es una **hoja**:
       una viñeta candidata.

    El árbol resultante describe la composición completa de la página, y su
    recorrido en profundidad da directamente el orden de lectura.
    """

    name = "xycut"
    requires = frozenset({CONTENT_MASK})
    provides = frozenset({CUT_TREE})

    def __init__(self, min_gap: int = 4, gap_tolerance: float = 0.01, max_depth: int = 12) -> None:
        if min_gap < 1:
            raise ValueError("min_gap debe ser al menos 1")
        if not 0 <= gap_tolerance < 0.5:
            raise ValueError("gap_tolerance debe estar entre 0 y 0.5")
        self.min_gap = min_gap
        self.gap_tolerance = gap_tolerance
        self.max_depth = max_depth

    def _process(self, ctx: PageContext) -> None:
        mask = ctx.get(CONTENT_MASK) > 0
        height, width = mask.shape
        tree = self._cut(mask, 0, 0, width, height, depth=0)
        ctx.set(CUT_TREE, tree)
        if tree is not None:
            self._debug(ctx, "cortes", self._render(mask, tree))

    def _cut(
        self, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, depth: int
    ) -> CutNode | None:
        region = mask[y1:y2, x1:x2]
        trimmed = self._trim(region)
        if trimmed is None:
            return None
        tx1, ty1, tx2, ty2 = trimmed
        x1, y1, x2, y2 = x1 + tx1, y1 + ty1, x1 + tx2, y1 + ty2
        region = mask[y1:y2, x1:x2]
        rect = Rect(x1, y1, x2 - x1, y2 - y1)

        if depth >= self.max_depth:
            return CutNode(rect)

        rows_empty = region.mean(axis=1) <= self.gap_tolerance
        row_gaps = find_gaps(rows_empty, self.min_gap)
        if row_gaps:
            children = [
                self._cut(mask, x1, y1 + s, x2, y1 + e, depth + 1)
                for s, e in split_by_gaps(y2 - y1, row_gaps)
            ]
            return self._node(rect, CutAxis.HORIZONTAL, children)

        cols_empty = region.mean(axis=0) <= self.gap_tolerance
        col_gaps = find_gaps(cols_empty, self.min_gap)
        if col_gaps:
            children = [
                self._cut(mask, x1 + s, y1, x1 + e, y2, depth + 1)
                for s, e in split_by_gaps(x2 - x1, col_gaps)
            ]
            return self._node(rect, CutAxis.VERTICAL, children)

        return CutNode(rect)

    def _trim(self, region: np.ndarray) -> tuple[int, int, int, int] | None:
        """Rectángulo mínimo con contenido dentro de la región, o None si está vacía."""
        if region.size == 0:
            return None
        rows = np.flatnonzero(region.mean(axis=1) > self.gap_tolerance)
        cols = np.flatnonzero(region.mean(axis=0) > self.gap_tolerance)
        if rows.size == 0 or cols.size == 0:
            return None
        return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1

    @staticmethod
    def _node(rect: Rect, axis: CutAxis, children: list[CutNode | None]) -> CutNode:
        kept = tuple(c for c in children if c is not None)
        if len(kept) == 1:  # un corte que deja un solo pedazo no aporta estructura
            return kept[0]
        return CutNode(rect, axis, kept)

    @staticmethod
    def _render(mask: np.ndarray, tree: CutNode) -> np.ndarray:
        import cv2

        canvas = cv2.cvtColor((mask * 160).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for leaf in tree.leaves():
            r = leaf.rect
            cv2.rectangle(
                canvas,
                (int(r.x), int(r.y)),
                (int(r.right) - 1, int(r.bottom) - 1),
                (0, 0, 255),
                thickness=2,
            )
        return canvas