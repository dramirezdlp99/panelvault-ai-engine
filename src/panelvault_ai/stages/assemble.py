"""Etapa 6: filtrar candidatas y ensamblar el PanelMap final con su confianza."""

from __future__ import annotations

import numpy as np

from panelvault_ai.domain import PageType, Panel, PanelMap, Polygon, ReadingDirection, Rect
from panelvault_ai.pipeline import STAGES, PageContext, Stage
from panelvault_ai.stages.keys import CONTENT_MASK, GUTTER, ORDERED_RECTS, PANEL_MAP


@STAGES.register("assemble")
class AssembleStage(Stage):
    """Descarta candidatas diminutas, mide la confianza y construye el PanelMap.

    - **Filtrado:** una región con menos de ``min_area_fraction`` del área de la
      página no es una viñeta (suele ser un número de página o una firma).
    - **Confianza por viñeta:** fracción de su rectángulo que la máscara marca
      como contenido. Una viñeta rectangular bien detectada ronda 1.0; un valor
      bajo indica una forma irregular o una fusión de varias viñetas.
    - **Confianza de la página:** confianza del medianil × promedio de las viñetas.
    - **Tipo de página:** sin viñetas → desconocido; una sola que cubre casi todo
      el contenido → splash; si no → cuadrícula.

    Esta es una primera versión de la confianza. Las etapas de refinamiento con
    contornos (viñetas inclinadas) la harán más precisa.
    """

    name = "assemble"
    requires = frozenset({ORDERED_RECTS, CONTENT_MASK, GUTTER})
    provides = frozenset({PANEL_MAP})

    def __init__(
        self,
        direction: str = "ltr",
        min_area_fraction: float = 0.01,
        splash_area_fraction: float = 0.6,
    ) -> None:
        if not 0 <= min_area_fraction < 1:
            raise ValueError("min_area_fraction debe estar entre 0 y 1")
        self.direction = ReadingDirection(direction)
        self.min_area_fraction = min_area_fraction
        self.splash_area_fraction = splash_area_fraction

    def _process(self, ctx: PageContext) -> None:
        mask = ctx.get(CONTENT_MASK) > 0
        height, width = mask.shape
        page_area = width * height

        rects = [r for r in ctx.get(ORDERED_RECTS) if r.area >= self.min_area_fraction * page_area]
        panels = tuple(
            Panel(Polygon.from_rect(r), order, self._fill_ratio(mask, r))
            for order, r in enumerate(rects)
        )

        if panels:
            gutter_conf = ctx.get(GUTTER).confidence
            confidence = gutter_conf * float(np.mean([p.confidence for p in panels]))
        else:
            confidence = 0.0

        ctx.set(
            PANEL_MAP,
            PanelMap(
                page_width=width,
                page_height=height,
                direction=self.direction,
                page_type=self._page_type(rects, page_area),
                panels=panels,
                confidence=min(1.0, max(0.0, confidence)),
            ),
        )

    @staticmethod
    def _fill_ratio(mask: np.ndarray, r: Rect) -> float:
        region = mask[int(r.y) : int(r.bottom), int(r.x) : int(r.right)]
        return float(region.mean()) if region.size else 0.0

    def _page_type(self, rects: list[Rect], page_area: int) -> PageType:
        if not rects:
            return PageType.UNKNOWN
        if len(rects) == 1 and rects[0].area >= self.splash_area_fraction * page_area:
            return PageType.SPLASH
        return PageType.GRID