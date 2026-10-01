"""Pruebas del generador de páginas sintéticas."""

import numpy as np
import pytest

from panelvault_ai.domain import Rect
from panelvault_ai.synthetic import (
    PageStyle,
    grid_layout,
    pinwheel_layout,
    render_page,
    row_layout,
)


def test_grid_layout_produce_filas_por_columnas_en_orden_de_lectura():
    rects = grid_layout(800, 1200, rows=3, cols=2)
    assert len(rects) == 6
    # Orden occidental: la segunda viñeta está a la derecha de la primera, misma fila.
    assert rects[1].x > rects[0].x
    assert rects[1].y == rects[0].y
    # La tercera empieza una fila nueva.
    assert rects[2].y > rects[0].y


def test_grid_layout_rechaza_paginas_demasiado_pequenas():
    with pytest.raises(ValueError):
        grid_layout(100, 100, rows=10, cols=10, gutter=20, margin=40)


def test_row_layout_es_determinista_con_la_misma_semilla():
    assert row_layout(800, 1200, [2, 3, 1], seed=7) == row_layout(800, 1200, [2, 3, 1], seed=7)
    assert row_layout(800, 1200, [2, 3, 1], seed=7) != row_layout(800, 1200, [2, 3, 1], seed=8)


@pytest.mark.parametrize("seed", range(5))
def test_row_layout_no_superpone_viñetas_y_respeta_la_pagina(seed):
    rects = row_layout(800, 1200, [2, 3, 1, 2], seed=seed)
    assert len(rects) == 8
    pagina = Rect(0, 0, 800, 1200)
    for i, a in enumerate(rects):
        assert pagina.contains(a)
        for b in rects[i + 1 :]:
            assert a.intersection(b) is None


def test_render_produce_imagen_bgr_del_tamano_pedido():
    pagina = render_page(800, 1200, grid_layout(800, 1200, 2, 2))
    assert pagina.image.shape == (1200, 800, 3)
    assert pagina.image.dtype == np.uint8
    assert len(pagina.panels) == 4


def test_render_deja_el_medianil_del_color_indicado():
    rects = grid_layout(800, 1200, 2, 2, gutter=30, margin=50)
    for gutter_value in (255, 0):
        imagen = render_page(800, 1200, rects, PageStyle(gutter_value=gutter_value)).image
        # Esquina superior izquierda (margen) y centro de un medianil vertical.
        assert imagen[10, 10, 0] == gutter_value
        assert imagen[300, 400, 0] == gutter_value


def test_el_dibujo_nunca_invade_el_medianil():
    rects = row_layout(800, 1200, [3, 2, 3], seed=11)
    imagen = render_page(800, 1200, rects, PageStyle(border_thickness=1), seed=11).image[:, :, 0]
    dentro = np.zeros(imagen.shape, dtype=bool)
    for r in rects:
        # Se deja 1 píxel extra por el grosor del marco.
        dentro[int(r.y) - 1 : int(r.bottom) + 1, int(r.x) - 1 : int(r.right) + 1] = True
    assert np.all(imagen[~dentro] == 255)


def test_molinete_no_tiene_ningun_medianil_que_cruce_la_pagina():
    rects = pinwheel_layout(800, 1200)
    assert len(rects) == 5
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert a.intersection(b) is None
    # Toda línea vertical u horizontal interior atraviesa al menos una viñeta.
    for x in range(60, 740, 5):
        assert any(r.x <= x < r.right for r in rects)
    for y in range(60, 1140, 5):
        assert any(r.y <= y < r.bottom for r in rects)