"""Etapas concretas del análisis visual.

Importar este paquete registra todas las etapas en ``STAGES`` para poder
construir pipelines por nombre.
"""

from panelvault_ai.stages.binarize import BinarizeStage
from panelvault_ai.stages.gutter import GutterEstimationStage
from panelvault_ai.stages.keys import (
    CONTENT_MASK,
    GUTTER,
    ORIGINAL_IMAGE,
    SCALE,
    WORKING_IMAGE,
    GutterEstimate,
)
from panelvault_ai.stages.normalize import NormalizeStage

__all__ = [
    "CONTENT_MASK",
    "GUTTER",
    "ORIGINAL_IMAGE",
    "SCALE",
    "WORKING_IMAGE",
    "BinarizeStage",
    "GutterEstimate",
    "GutterEstimationStage",
    "NormalizeStage",
]