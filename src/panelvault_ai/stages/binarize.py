"""Etapa 3: separar el contenido de las viñetas del medianil."""

from __future__ import annotations

import cv2
import numpy as np

from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.keys import CONTENT_MASK, GUTTER, WORKING_IMAGE


@STAGES.register("binarize")
class BinarizeStage(Stage):
    """Produce una máscara binaria: 255 donde hay viñeta, 0 donde hay medianil.

    El reto es que el interior de una viñeta suele tener zonas del mismo color
    que el medianil (un cielo blanco, por ejemplo). La solución es topológica:

    1. Se marcan como "candidatos a medianil" los píxeles cuyo color está dentro
       de la tolerancia estimada en la etapa anterior.
    2. Se agrupan en **componentes conexas** (regiones de píxeles vecinos).
    3. Solo es medianil la componente que **toca el borde de la página**. Una zona
       blanca encerrada por el marco de una viñeta no puede alcanzar el borde, así
       que se reclasifica como contenido.
    4. Una apertura morfológica elimina motas de ruido aisladas en el medianil.

    Limitación conocida: si una viñeta no tiene el marco cerrado y su fondo es del
    color del medianil, ese fondo "se fuga" y se confunde con medianil. Las etapas
    de confianza posteriores deben detectar esos casos.
    """

    name = "binarize"
    requires = frozenset({WORKING_IMAGE, GUTTER})
    provides = frozenset({CONTENT_MASK})

    def __init__(self, despeckle_size: int = 3) -> None:
        if despeckle_size < 1:
            raise ValueError("despeckle_size debe ser al menos 1")
        self.despeckle_size = despeckle_size

    def _process(self, ctx: PageContext) -> None:
        gray = ctx.get(WORKING_IMAGE)
        gutter = ctx.get(GUTTER)

        distance = np.abs(gray.astype(np.int16) - gutter.value)
        candidates = (distance <= gutter.tolerance).astype(np.uint8)
        self._debug(ctx, "candidatos_medianil", candidates * 255)

        background = self._border_connected(candidates)
        content = np.where(background, 0, 255).astype(np.uint8)

        if self.despeckle_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (self.despeckle_size, self.despeckle_size)
            )
            content = cv2.morphologyEx(content, cv2.MORPH_OPEN, kernel)

        ctx.set(CONTENT_MASK, content)
        self._debug(ctx, "contenido", content)

    @staticmethod
    def _border_connected(candidates: np.ndarray) -> np.ndarray:
        """Máscara booleana de los candidatos conectados con el borde de la imagen.

        Se usa conectividad 4 (solo arriba, abajo, izquierda y derecha): con
        conectividad 8 el medianil podría "colarse" en diagonal por un marco de un
        píxel de grosor.
        """
        _, labels = cv2.connectedComponents(candidates, connectivity=4)
        border_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        border_labels = border_labels[border_labels != 0]  # 0 = píxeles no candidatos
        return np.isin(labels, border_labels)