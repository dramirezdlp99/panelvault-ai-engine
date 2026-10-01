"""Pruebas del XY-Cut recursivo y de sus funciones auxiliares."""

import numpy as np
import pytest

from panelvault_ai.domain import Rect
from panelvault_ai.pipeline import PageContext
from panelvault_ai.stages import (
    CONTENT_MASK,
    CUT_TREE,
    CutAxis,
    XYCutStage,
    find_gaps,
    split_by_gaps,
)


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def test_find_gaps_encuentra_tramos_internos():
    vacio = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert find_gaps(vacio, min_gap=1) == [(2, 5), (7, 8)]


def test_find_gaps_respeta_la_longitud_minima():
    vacio = np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)
    assert find_gaps(vacio, min_gap=2) == [(1, 4)]


def test_find_gaps_ignora_los_margenes():
    vacio = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert find_gaps(vacio, min_gap=1) == [(4, 5)]


def test_split_by_gaps():
    assert split_by_gaps(10, [(3, 5), (7, 8)]) == [(0, 3), (5, 7), (8, 10)]
    assert split_by_gaps(10, []) == [(0, 10)]


# ----------------------------------------------------------------------
# Etapa completa sobre máscaras dibujadas a mano
# ----------------------------------------------------------------------
def _mascara(alto, ancho, rects):
    m = np.zeros((alto, ancho), dtype=np.uint8)
    for x, y, w, h in rects:
        m[y : y + h, x : x + w] = 255
    return m


def _arbol(mascara, **params):
    ctx = PageContext()
    ctx.set(CONTENT_MASK, mascara)
    XYCutStage(**params).run(ctx)
    return ctx.get(CUT_TREE)


def test_una_sola_region_es_una_hoja_recortada_al_contenido():
    arbol = _arbol(_mascara(100, 100, [(10, 20, 30, 40)]))
    assert arbol.is_leaf
    assert arbol.rect == Rect(10, 20, 30, 40)


def test_pagina_vacia_no_produce_arbol():
    assert _arbol(np.zeros((50, 50), dtype=np.uint8)) is None


def test_prefiere_cortar_en_filas_en_una_cuadricula():
    # Cuadrícula 2x2: deben salir 2 filas, cada una con 2 columnas.
    arbol = _arbol(
        _mascara(100, 100, [(0, 0, 40, 40), (60, 0, 40, 40), (0, 60, 40, 40), (60, 60, 40, 40)])
    )
    assert arbol.axis is CutAxis.HORIZONTAL
    assert len(arbol.children) == 2
    assert all(fila.axis is CutAxis.VERTICAL for fila in arbol.children)
    assert len(list(arbol.leaves())) == 4


def test_columna_alta_a_la_izquierda_y_dos_apiladas_a_la_derecha():
    # Sin un medianil horizontal que cruce toda la página, el primer corte es vertical.
    arbol = _arbol(_mascara(100, 100, [(0, 0, 40, 100), (60, 0, 40, 40), (60, 60, 40, 40)]))
    assert arbol.axis is CutAxis.VERTICAL
    izquierda, derecha = arbol.children
    assert izquierda.is_leaf
    assert derecha.axis is CutAxis.HORIZONTAL
    assert arbol.depth == 2


def test_medianiles_mas_estrechos_que_min_gap_no_cortan():
    mascara = _mascara(100, 100, [(0, 0, 100, 48), (0, 50, 100, 50)])  # medianil de 2 px
    assert _arbol(mascara, min_gap=4).is_leaf
    assert not _arbol(mascara, min_gap=2).is_leaf


def test_tolera_ruido_disperso_dentro_del_medianil():
    mascara = _mascara(100, 100, [(0, 0, 100, 40), (0, 60, 100, 40)])
    mascara[50, 30] = 255  # un píxel de ruido en el medianil (1 % de la fila)
    assert _arbol(mascara, gap_tolerance=0.02).axis is CutAxis.HORIZONTAL


def test_rechaza_parametros_invalidos():
    with pytest.raises(ValueError):
        XYCutStage(min_gap=0)
    with pytest.raises(ValueError):
        XYCutStage(gap_tolerance=0.9)