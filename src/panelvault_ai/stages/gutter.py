"""Etapa 2: estimar el color del medianil a partir del margen de la página."""

from __future__ import annotations

import numpy as np

from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.keys import GUTTER, WORKING_IMAGE, GutterEstimate

# Percentil de las desviaciones que se usa como medida de dispersión.
_SPREAD_PERCENTILE = 90
# Multiplicador sobre esa dispersión: con ruido normal equivale a unas 3 desviaciones estándar.
_SPREAD_MULTIPLIER = 2.0


@STAGES.register("gutter")
class GutterEstimationStage(Stage):
    """Mide el marco exterior de la página para deducir el color del medianil.

    En la gran mayoría de los cómics el margen exterior y los medianiles son del
    mismo color (normalmente blanco, a veces negro). Se toma una franja delgada
    alrededor de todo el borde y se calcula:

    - la **mediana** de sus intensidades: el color del medianil. La mediana, a
      diferencia del promedio, no se deja arrastrar por los pocos píxeles de
      dibujo que puedan tocar el borde;
    - la **dispersión**: el percentil 90 de las distancias a la mediana. Indica
      cuánto varía el color por ruido o compresión, y de ahí sale la tolerancia.
      No se usa la MAD (desviación absoluta mediana) porque falla con blanco
      saturado: si más de la mitad de los píxeles valen exactamente 255, la MAD
      da cero aunque haya ruido. El percentil 90 sigue ignorando hasta un 10 %
      de píxeles de dibujo que toquen el borde;
    - la **confianza**: la fracción de la franja que de verdad cae dentro de la
      tolerancia. Si el dibujo llega hasta el borde (página a sangre), será baja.
    """

    name = "gutter"
    requires = frozenset({WORKING_IMAGE})
    provides = frozenset({GUTTER})

    def __init__(
        self,
        border_fraction: float = 0.015,
        min_tolerance: int = 12,
        max_tolerance: int = 60,
    ) -> None:
        if not 0 < border_fraction < 0.5:
            raise ValueError("border_fraction debe estar entre 0 y 0.5")
        if not 0 < min_tolerance <= max_tolerance:
            raise ValueError("Se requiere 0 < min_tolerance <= max_tolerance")
        self.border_fraction = border_fraction
        self.min_tolerance = min_tolerance
        self.max_tolerance = max_tolerance

    def _process(self, ctx: PageContext) -> None:
        gray = ctx.get(WORKING_IMAGE)
        strip = self._border_strip(gray)

        median = float(np.median(strip))
        deviations = np.abs(strip - median)
        spread = float(np.percentile(deviations, _SPREAD_PERCENTILE))
        tolerance = int(
            np.clip(round(_SPREAD_MULTIPLIER * spread), self.min_tolerance, self.max_tolerance)
        )
        confidence = float(np.mean(deviations <= tolerance))

        ctx.set(GUTTER, GutterEstimate(round(median), tolerance, confidence))

    def _border_strip(self, gray: np.ndarray) -> np.ndarray:
        """Píxeles de las cuatro franjas del borde, como un vector de enteros."""
        height, width = gray.shape
        band = max(2, round(self.border_fraction * min(height, width)))
        parts = (
            gray[:band, :],
            gray[-band:, :],
            gray[band:-band, :band],
            gray[band:-band, -band:],
        )
        return np.concatenate([p.ravel() for p in parts]).astype(np.int16)