"""Pruebas del dibujo del mapa de viñetas."""

import numpy as np

from panelvault_ai.analyzer import PanelAnalyzer
from panelvault_ai.synthetic import grid_layout, render_page
from panelvault_ai.visualize import draw_panel_map


def test_dibuja_sobre_una_copia_del_mismo_tamano():
    pagina = render_page(800, 1200, grid_layout(800, 1200, 2, 2))
    mapa = PanelAnalyzer().analyze(pagina.image)
    resultado = draw_panel_map(pagina.image, mapa)
    assert resultado.shape == pagina.image.shape
    assert not np.array_equal(resultado, pagina.image)


def test_no_modifica_la_imagen_original():
    pagina = render_page(800, 1200, grid_layout(800, 1200, 2, 2))
    original = pagina.image.copy()
    draw_panel_map(pagina.image, PanelAnalyzer().analyze(pagina.image))
    assert np.array_equal(pagina.image, original)


def test_usa_coordenadas_normalizadas_en_imagenes_grandes():
    # Analizada a 1200 px pero dibujada sobre la original de 2400 px:
    # el centro de la primera viñeta debe quedar teñido.
    layout = grid_layout(2400, 3600, rows=2, cols=2, gutter=60, margin=120)
    imagen = render_page(2400, 3600, layout).image
    resultado = draw_panel_map(imagen, PanelAnalyzer().analyze(imagen))
    cx, cy = (int(c) for c in layout[0].center)
    assert not np.array_equal(resultado[cy, cx], imagen[cy, cx])


def test_acepta_imagenes_en_gris():
    pagina = render_page(800, 1200, grid_layout(800, 1200, 2, 2))
    gris = pagina.image[:, :, 0]
    assert draw_panel_map(gris, PanelAnalyzer().analyze(pagina.image)).ndim == 3