"""Etapas concretas del análisis visual.

Importar este paquete registra todas las etapas en ``STAGES`` para poder
construir pipelines por nombre.
"""

from panelvault_ai.stages.assemble import AssembleStage
from panelvault_ai.stages.binarize import BinarizeStage
from panelvault_ai.stages.cut_tree import CutAxis, CutNode
from panelvault_ai.stages.gutter import GutterEstimationStage
from panelvault_ai.stages.keys import (
    CONTENT_MASK,
    CUT_TREE,
    GUTTER,
    ORDERED_RECTS,
    ORIGINAL_IMAGE,
    PANEL_MAP,
    SCALE,
    WORKING_IMAGE,
    GutterEstimate,
)
from panelvault_ai.stages.normalize import NormalizeStage
from panelvault_ai.stages.ordering import ReadingOrderStage, reading_order
from panelvault_ai.stages.refine import SplitCandidate, ThinGutterRefineStage
from panelvault_ai.stages.xycut import XYCutStage, find_gaps, split_by_gaps

__all__ = [
    "CONTENT_MASK",
    "CUT_TREE",
    "GUTTER",
    "ORDERED_RECTS",
    "ORIGINAL_IMAGE",
    "PANEL_MAP",
    "SCALE",
    "WORKING_IMAGE",
    "AssembleStage",
    "BinarizeStage",
    "CutAxis",
    "CutNode",
    "GutterEstimate",
    "GutterEstimationStage",
    "NormalizeStage",
    "ReadingOrderStage",
    "SplitCandidate",
    "ThinGutterRefineStage",
    "XYCutStage",
    "find_gaps",
    "reading_order",
    "split_by_gaps",
]