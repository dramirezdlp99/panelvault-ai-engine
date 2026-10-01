"""Utilidades compartidas por las pruebas de etapas."""

import numpy as np

from panelvault_ai.domain import Rect
from panelvault_ai.pipeline import PageContext, Pipeline
from panelvault_ai.stages import (
    ORIGINAL_IMAGE,
    BinarizeStage,
    GutterEstimationStage,
    NormalizeStage,
)


def pipeline_basico(target_width: int = 600) -> Pipeline:
    return Pipeline(
        [NormalizeStage(target_width), GutterEstimationStage(), BinarizeStage()],
        initial={ORIGINAL_IMAGE},
    )


def ejecutar(pipeline: Pipeline, imagen: np.ndarray) -> PageContext:
    ctx = PageContext()
    ctx.set(ORIGINAL_IMAGE, imagen)
    return pipeline.run(ctx)


def mascaras_de_verdad(
    shape: tuple[int, int], panels: tuple[Rect, ...], scale: float, holgura: int
) -> tuple[np.ndarray, np.ndarray]:
    """Zonas donde la respuesta es inequívoca, dejando una franja de holgura en los bordes.

    Devuelve (interior_seguro, medianil_seguro): el interior de cada viñeta encogido
    ``holgura`` píxeles, y todo lo que queda a más de ``holgura`` píxeles de cualquier viñeta.
    """
    interior = np.zeros(shape, dtype=bool)
    cerca_de_viñeta = np.zeros(shape, dtype=bool)
    for p in panels:
        r = p.scaled(scale)
        x1, y1, x2, y2 = round(r.x), round(r.y), round(r.right), round(r.bottom)
        interior[y1 + holgura : y2 - holgura, x1 + holgura : x2 - holgura] = True
        cerca_de_viñeta[
            max(0, y1 - holgura) : y2 + holgura, max(0, x1 - holgura) : x2 + holgura
        ] = True
    return interior, ~cerca_de_viñeta