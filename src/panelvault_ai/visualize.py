"""Dibujo del mapa de viñetas sobre la página original."""

from __future__ import annotations

import cv2
import numpy as np

from panelvault_ai.domain import PanelMap

# Paleta categórica (BGR) con buen contraste sobre papel blanco y sobre dibujo.
_COLORS: tuple[tuple[int, int, int], ...] = (
    (235, 99, 37),  # azul
    (36, 160, 70),  # verde
    (40, 110, 230),  # naranja
    (180, 60, 160),  # morado
    (30, 170, 200),  # mostaza
    (150, 120, 20),  # verde azulado
)


def draw_panel_map(image: np.ndarray, panel_map: PanelMap, fill_alpha: float = 0.18) -> np.ndarray:
    """Devuelve una copia de ``image`` con cada viñeta coloreada, enmarcada y numerada.

    Usa las coordenadas normalizadas del mapa, así sirve para la imagen original en
    su resolución completa aunque el análisis se haya hecho sobre una copia reducida.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    canvas = image.copy()
    overlay = image.copy()
    thickness = max(2, round(width / 300))

    polygons = []
    for panel in panel_map.panels:
        points = panel.shape.normalized(panel_map.page_width, panel_map.page_height).points
        pixels = np.array([[x * width, y * height] for x, y in points], dtype=np.int32)
        polygons.append(pixels)
        cv2.fillPoly(overlay, [pixels], _COLORS[panel.order % len(_COLORS)])

    cv2.addWeighted(overlay, fill_alpha, canvas, 1 - fill_alpha, 0, dst=canvas)

    for panel, pixels in zip(panel_map.panels, polygons, strict=True):
        color = _COLORS[panel.order % len(_COLORS)]
        cv2.polylines(canvas, [pixels], isClosed=True, color=color, thickness=thickness)
        _draw_badge(canvas, pixels, f"{panel.order + 1}", color, width)
    return canvas


def _draw_badge(
    canvas: np.ndarray, pixels: np.ndarray, text: str, color: tuple[int, int, int], width: int
) -> None:
    """Círculo con el número de orden en la esquina superior izquierda de la viñeta."""
    radius = max(14, round(width / 45))
    x, y = pixels.min(axis=0) + radius + 4
    center = (int(x), int(y))
    cv2.circle(canvas, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, center, radius, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    scale = radius / 20
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    origin = (center[0] - tw // 2, center[1] + th // 2)
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
    )