"""Etapa 1: llevar cualquier página a un formato de trabajo uniforme."""

from __future__ import annotations

import cv2
import numpy as np

from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.keys import ORIGINAL_IMAGE, SCALE, WORKING_IMAGE


@STAGES.register("normalize")
class NormalizeStage(Stage):
    """Convierte a escala de grises y reduce la página a un ancho fijo.

    Trabajar siempre al mismo ancho hace que los parámetros de las etapas
    siguientes (grosores, áreas mínimas) signifiquen lo mismo para cualquier
    página, sin importar la resolución del escaneo. Nunca se agranda una imagen:
    inventar píxeles no aporta información.
    """

    name = "normalize"
    requires = frozenset({ORIGINAL_IMAGE})
    provides = frozenset({WORKING_IMAGE, SCALE})

    def __init__(self, target_width: int = 1200) -> None:
        if target_width < 64:
            raise ValueError("El ancho de trabajo debe ser de al menos 64 píxeles")
        self.target_width = target_width

    def _process(self, ctx: PageContext) -> None:
        image = ctx.get(ORIGINAL_IMAGE)
        gray = self._to_gray(image)
        height, width = gray.shape

        if width > self.target_width:
            scale = self.target_width / width
            new_size = (self.target_width, max(1, round(height * scale)))
            # INTER_AREA promedia los píxeles que se fusionan: es el método correcto al reducir.
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        ctx.set(WORKING_IMAGE, gray)
        ctx.set(SCALE, scale)
        self._debug(ctx, "gris", gray)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            raise ValueError(f"Se esperaba una imagen uint8, se recibió {image.dtype}")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Forma de imagen no soportada: {image.shape}")