"""Datos que viajan entre las etapas de visión: sus claves tipadas y sus tipos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panelvault_ai.pipeline import ArtifactKey


@dataclass(frozen=True, slots=True)
class GutterEstimate:
    """Color estimado del medianil (el espacio que separa las viñetas).

    ``value`` es la intensidad en escala de grises (0 negro, 255 blanco),
    ``tolerance`` cuánto puede desviarse un píxel para seguir contando como medianil,
    y ``confidence`` qué tan uniforme resultó el margen de la página (0 a 1).
    """

    value: int
    tolerance: int
    confidence: float

    @property
    def is_light(self) -> bool:
        return self.value >= 128


ORIGINAL_IMAGE: ArtifactKey[np.ndarray] = ArtifactKey(
    "original_image", "Página tal como llega: BGR, BGRA o gris, uint8"
)
WORKING_IMAGE: ArtifactKey[np.ndarray] = ArtifactKey(
    "working_image", "Página en gris y reducida al ancho de trabajo"
)
SCALE: ArtifactKey[float] = ArtifactKey(
    "scale", "Factor ancho_trabajo / ancho_original (1.0 si no se redujo)"
)
GUTTER: ArtifactKey[GutterEstimate] = ArtifactKey("gutter", "Color estimado del medianil")
CONTENT_MASK: ArtifactKey[np.ndarray] = ArtifactKey(
    "content_mask", "Máscara binaria: 255 = contenido de viñeta, 0 = medianil"
)