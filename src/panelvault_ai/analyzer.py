"""Fachada del motor: analizar una página con una sola llamada (patrón Facade).

Oculta el pipeline, el registro y las claves del contexto detrás de una clase
simple. Es lo que usarán la línea de comandos, el worker y la API.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

import panelvault_ai.stages  # noqa: F401  (registra las etapas en STAGES)
from panelvault_ai.domain import PanelMap
from panelvault_ai.pipeline import DebugSink, PageContext, Pipeline, build_pipeline
from panelvault_ai.stages import ORIGINAL_IMAGE, PANEL_MAP

_BASE: list[dict[str, Any]] = [
    {"stage": "normalize", "params": {"target_width": 1200}},
    {"stage": "gutter"},
    {"stage": "binarize"},
    {"stage": "xycut"},
]

PRESETS: dict[str, list[dict[str, Any]]] = {
    "western": [
        *_BASE,
        {"stage": "order", "params": {"direction": "ltr"}},
        {"stage": "assemble", "params": {"direction": "ltr"}},
    ],
    "manga": [
        *_BASE,
        {"stage": "order", "params": {"direction": "rtl"}},
        {"stage": "assemble", "params": {"direction": "rtl"}},
    ],
}


class PanelAnalyzer:
    """Analiza páginas de cómic y devuelve su mapa de viñetas."""

    def __init__(self, preset: str = "western", debug: DebugSink | None = None) -> None:
        if preset not in PRESETS:
            raise ValueError(f"Preset desconocido {preset!r}. Disponibles: {sorted(PRESETS)}")
        self.preset = preset
        self.debug = debug
        self.pipeline: Pipeline = build_pipeline(
            copy.deepcopy(PRESETS[preset]), initial={ORIGINAL_IMAGE}
        )

    def analyze(self, image: np.ndarray) -> PanelMap:
        return self.analyze_with_context(image).get(PANEL_MAP)

    def analyze_with_context(self, image: np.ndarray) -> PageContext:
        """Igual que ``analyze`` pero devuelve el contexto completo (tiempos, datos intermedios)."""
        ctx = PageContext(debug=self.debug)
        ctx.set(ORIGINAL_IMAGE, image)
        return self.pipeline.run(ctx)