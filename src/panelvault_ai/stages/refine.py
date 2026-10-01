"""Etapa 4b: dividir viñetas fusionadas por medianiles demasiado delgados para el XY-Cut."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panelvault_ai.domain import Rect
from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.cut_tree import CutAxis, CutNode
from panelvault_ai.stages.keys import CUT_TREE, GUTTER, WORKING_IMAGE, GutterEstimate
from panelvault_ai.stages.xycut import find_gaps


@dataclass(frozen=True, slots=True)
class SplitCandidate:
    """Un posible medianil delgado dentro de una hoja."""

    axis: CutAxis
    start: int  # inicio del medianil, relativo a la hoja
    end: int  # fin (exclusivo)
    score: float


@STAGES.register("refine")
class ThinGutterRefineStage(Stage):
    """Busca, dentro de cada hoja del árbol, medianiles de 1 a pocos píxeles.

    El XY-Cut exige medianiles de ``min_gap`` píxeles para no confundir líneas
    blancas del dibujo con separaciones. Pero algunos cómics separan viñetas con
    un espacio mínimo, y entonces dos viñetas salen fusionadas en una hoja.

    Esta etapa trabaja sobre la imagen en gris (no sobre la máscara) y acepta un
    medianil delgado solo si cumple una firma muy específica de los cómics:

    1. Una franja de filas (o columnas) que atraviesa la hoja de lado a lado y es
       casi toda del color del medianil (``min_light``).
    2. **Flanqueada por tinta a ambos lados**: justo antes y justo después hay
       líneas casi continuas muy oscuras (``min_ink``), que son los marcos de las
       dos viñetas. Una zona clara del dibujo casi nunca está encerrada entre dos
       líneas negras que la recorren completa. La "tinta" es **relativa al color
       del papel** (``ink_contrast``): en papel amarillento de gris 160, un marco
       de gris 60 es tinta aunque esté lejos del negro absoluto.
    3. Lejos de los bordes de la hoja (``min_part``): cada pedazo resultante debe
       conservar una fracción razonable de la hoja.

    Igual que el XY-Cut, se prefieren los cortes en **filas**: solo si no hay
    ningún medianil horizontal creíble se buscan columnas. Si se eligiera por
    puntuación entre ambos ejes, una cuadrícula podría cortarse primero en
    columnas y su orden de lectura saldría por columnas, que es incorrecto.
    Dentro del eje elegido gana la candidata con mejor puntuación; se divide y
    se repite recursivamente en cada mitad. La etapa reemplaza el árbol de cortes por
    una versión refinada; si no encuentra nada, lo deja igual.
    """

    name = "refine"
    requires = frozenset({CUT_TREE, WORKING_IMAGE, GUTTER})
    provides = frozenset({CUT_TREE})

    def __init__(
        self,
        min_light: float = 0.85,
        min_ink: float = 0.5,
        ink_contrast: float = 0.5,
        flank: int = 4,
        min_part: float = 0.15,
        min_size: int = 40,
        max_depth: int = 8,
    ) -> None:
        if not 0 < min_light <= 1 or not 0 < min_ink <= 1:
            raise ValueError("min_light y min_ink deben estar entre 0 y 1")
        if not 0 < ink_contrast < 1:
            raise ValueError("ink_contrast debe estar entre 0 y 1")
        if not 0 < min_part < 0.5:
            raise ValueError("min_part debe estar entre 0 y 0.5")
        self.min_light = min_light
        self.min_ink = min_ink
        self.ink_contrast = ink_contrast
        self.flank = flank
        self.min_part = min_part
        self.min_size = min_size
        self.max_depth = max_depth

    def _process(self, ctx: PageContext) -> None:
        tree = ctx.get(CUT_TREE)
        if tree is None:
            return
        gray = ctx.get(WORKING_IMAGE)
        gutter = ctx.get(GUTTER)
        ctx.set(CUT_TREE, self._refine(tree, gray, gutter))

    def _refine(self, node: CutNode, gray: np.ndarray, gutter: GutterEstimate) -> CutNode:
        if node.is_leaf:
            return self._split_leaf(node.rect, gray, gutter, depth=0)
        children = tuple(self._refine(child, gray, gutter) for child in node.children)
        return CutNode(node.rect, node.axis, children)

    def _split_leaf(
        self, rect: Rect, gray: np.ndarray, gutter: GutterEstimate, depth: int
    ) -> CutNode:
        if depth >= self.max_depth or min(rect.width, rect.height) < 2 * self.min_size:
            return CutNode(rect)
        x1, y1 = int(rect.x), int(rect.y)
        x2, y2 = int(rect.right), int(rect.bottom)
        candidate = self.best_split(gray[y1:y2, x1:x2], gutter)
        if candidate is None:
            return CutNode(rect)

        if candidate.axis is CutAxis.HORIZONTAL:
            first = Rect(x1, y1, x2 - x1, candidate.start)
            second = Rect(x1, y1 + candidate.end, x2 - x1, (y2 - y1) - candidate.end)
        else:
            first = Rect(x1, y1, candidate.start, y2 - y1)
            second = Rect(x1 + candidate.end, y1, (x2 - x1) - candidate.end, y2 - y1)
        children = (
            self._split_leaf(first, gray, gutter, depth + 1),
            self._split_leaf(second, gray, gutter, depth + 1),
        )
        return CutNode(rect, candidate.axis, children)

    def best_split(self, crop: np.ndarray, gutter: GutterEstimate) -> SplitCandidate | None:
        """Mejor medianil delgado dentro de un recorte, o None si no hay ninguno creíble."""
        distance = np.abs(crop.astype(np.int16) - gutter.value)
        light = distance <= gutter.tolerance
        # Contraste máximo posible desde el color del papel: hacia el negro si el
        # papel es claro, hacia el blanco si es oscuro.
        contrast_range = max(gutter.value, 255 - gutter.value)
        ink = distance >= self.ink_contrast * contrast_range

        # Primero filas (perfil por fila, axis=1); solo si no hay, columnas (axis=0).
        for axis, profile_axis in ((CutAxis.HORIZONTAL, 1), (CutAxis.VERTICAL, 0)):
            best = self._best_on_axis(
                axis, light.mean(axis=profile_axis), ink.mean(axis=profile_axis)
            )
            if best is not None:
                return best
        return None

    def _best_on_axis(
        self, axis: CutAxis, light_frac: np.ndarray, ink_frac: np.ndarray
    ) -> SplitCandidate | None:
        """Candidata con mejor puntuación en un eje, aplicando las tres condiciones."""
        length = light_frac.size
        low, high = self.min_part * length, (1 - self.min_part) * length
        best: SplitCandidate | None = None
        for start, end in find_gaps(light_frac >= self.min_light, min_gap=1):
            if start < low or end > high:
                continue
            before = float(ink_frac[max(0, start - self.flank) : start].max())
            after = float(ink_frac[end : end + self.flank].max())
            if before < self.min_ink or after < self.min_ink:
                continue
            score = float(light_frac[start:end].mean()) + (before + after) / 2
            if best is None or score > best.score:
                best = SplitCandidate(axis, start, end, score)
        return best