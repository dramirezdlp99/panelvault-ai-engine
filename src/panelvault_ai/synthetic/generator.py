"""Generador de páginas de cómic sintéticas con respuesta correcta conocida.

Dibuja viñetas sobre un lienzo en posiciones elegidas por nosotros. Como sabemos
exactamente dónde está cada viñeta y en qué orden se lee, sirve para probar el
pipeline de forma exacta, repetible y sin material con derechos de autor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from panelvault_ai.domain import Rect


@dataclass(frozen=True, slots=True)
class PageStyle:
    """Apariencia de la página sintética."""

    gutter_value: int = 255  # intensidad del medianil (255 = blanco, 0 = negro)
    border_value: int = 0  # color del marco de cada viñeta
    border_thickness: int = 3
    noise_sigma: float = 0.0  # ruido gaussiano, simula escaneos o compresión JPEG
    draw_art: bool = True  # dibuja "ilustraciones" simples dentro de cada viñeta


@dataclass(frozen=True, slots=True)
class SyntheticPage:
    """Imagen generada más su respuesta correcta (viñetas en orden de lectura)."""

    image: np.ndarray  # BGR, uint8
    panels: tuple[Rect, ...]


def grid_layout(
    width: int, height: int, rows: int, cols: int, gutter: int = 20, margin: int = 40
) -> list[Rect]:
    """Cuadrícula regular. Devuelve las viñetas en orden de lectura occidental."""
    cell_w = (width - 2 * margin - (cols - 1) * gutter) / cols
    cell_h = (height - 2 * margin - (rows - 1) * gutter) / rows
    if cell_w <= 0 or cell_h <= 0:
        raise ValueError("La página es demasiado pequeña para esa cuadrícula")
    return [
        Rect(
            round(margin + c * (cell_w + gutter)),
            round(margin + r * (cell_h + gutter)),
            round(cell_w),
            round(cell_h),
        )
        for r in range(rows)
        for c in range(cols)
    ]


def row_layout(
    width: int,
    height: int,
    panels_per_row: Sequence[int],
    seed: int = 0,
    gutter: int = 20,
    margin: int = 40,
) -> list[Rect]:
    """Filas de alturas variables, cada una partida en viñetas de anchos variables.

    Es la composición más común del cómic occidental. El ``seed`` hace que la misma
    llamada produzca siempre la misma página.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(panels_per_row)
    usable_h = height - 2 * margin - (n_rows - 1) * gutter
    row_weights = rng.uniform(0.7, 1.3, n_rows)
    row_heights = np.floor(usable_h * row_weights / row_weights.sum()).astype(int)

    rects: list[Rect] = []
    y = margin
    for row_h, n in zip(row_heights, panels_per_row, strict=True):
        usable_w = width - 2 * margin - (n - 1) * gutter
        col_weights = rng.uniform(0.6, 1.4, n)
        col_widths = np.floor(usable_w * col_weights / col_weights.sum()).astype(int)
        x = margin
        for col_w in col_widths:
            rects.append(Rect(int(x), int(y), int(col_w), int(row_h)))
            x += col_w + gutter
        y += row_h + gutter
    return rects


def pinwheel_layout(width: int, height: int, gutter: int = 20, margin: int = 40) -> list[Rect]:
    """Composición en "molinete": cuatro viñetas giran alrededor de una central.

    Ningún medianil cruza la página de lado a lado, así que un algoritmo de cortes
    rectos como el XY-Cut no puede separarla. Se usa como **límite conocido**.
    Orden devuelto: por fila de su esquina superior izquierda, de izquierda a derecha.
    """
    uw = (width - 2 * margin - 2 * gutter) / 3
    uh = (height - 2 * margin - 2 * gutter) / 3

    def cell(col: float, row: float, cols: float, rows: float) -> Rect:
        x = margin + col * (uw + gutter)
        y = margin + row * (uh + gutter)
        return Rect(
            round(x),
            round(y),
            round(cols * uw + (cols - 1) * gutter),
            round(rows * uh + (rows - 1) * gutter),
        )

    return [
        cell(0, 0, 2, 1),  # arriba, ancha
        cell(2, 0, 1, 2),  # derecha, alta
        cell(0, 1, 1, 2),  # izquierda, alta
        cell(1, 1, 1, 1),  # centro
        cell(1, 2, 2, 1),  # abajo, ancha
    ]


def render_page(
    width: int,
    height: int,
    panels: Sequence[Rect],
    style: PageStyle | None = None,
    seed: int = 0,
) -> SyntheticPage:
    """Dibuja las viñetas indicadas sobre un lienzo del color del medianil."""
    style = style or PageStyle()
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), style.gutter_value, dtype=np.uint8)

    for rect in panels:
        x1, y1 = int(rect.x), int(rect.y)
        x2, y2 = int(rect.right) - 1, int(rect.bottom) - 1
        # Fondo de la viñeta: claro pero distinto del medianil, como el papel de un dibujo.
        fill = 235 if style.gutter_value < 128 else 245
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, thickness=-1)
        if style.draw_art:
            _draw_art(canvas, rect, rng)
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), style.border_value, thickness=style.border_thickness
        )

    if style.noise_sigma > 0:
        noise = rng.normal(0, style.noise_sigma, canvas.shape)
        canvas = np.clip(canvas.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return SyntheticPage(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), tuple(panels))


def _draw_art(canvas: np.ndarray, rect: Rect, rng: np.random.Generator) -> None:
    """Formas al azar que imitan trazos de dibujo.

    Se dibujan sobre una copia recortada de la viñeta, así OpenCV recorta
    automáticamente cualquier trazo que se salga y nunca invade el medianil.
    """
    inset = 6
    x1, y1 = int(rect.x) + inset, int(rect.y) + inset
    x2, y2 = int(rect.right) - inset, int(rect.bottom) - inset
    w, h = x2 - x1, y2 - y1
    if w <= 10 or h <= 10:
        return
    region = canvas[y1:y2, x1:x2].copy()
    for _ in range(int(rng.integers(3, 7))):
        shade = int(rng.integers(20, 200))
        thickness = int(rng.integers(1, 4))
        if rng.random() < 0.5:
            center = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            radius = int(rng.integers(5, max(6, min(w, h) // 3)))
            cv2.circle(region, center, radius, shade, thickness=thickness)
        else:
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(region, p1, p2, shade, thickness=thickness)
    canvas[y1:y2, x1:x2] = region