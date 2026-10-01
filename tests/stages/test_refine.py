"""Pruebas del refinamiento por medianiles delgados.

Se prueba en las dos direcciones: que divida viñetas fusionadas y, igual de
importante, que NO divida una viñeta por una franja clara de su propio dibujo.
"""

import cv2
import numpy as np
import pytest

from panelvault_ai.domain import Rect
from panelvault_ai.pipeline import PageContext
from panelvault_ai.stages import (
    CUT_TREE,
    GUTTER,
    WORKING_IMAGE,
    CutAxis,
    CutNode,
    GutterEstimate,
    ThinGutterRefineStage,
)

MEDIANIL_BLANCO = GutterEstimate(value=255, tolerance=12, confidence=1.0)


def _refinar(gray: np.ndarray) -> CutNode:
    """Ejecuta la etapa sobre una sola hoja que cubre toda la imagen."""
    alto, ancho = gray.shape
    ctx = PageContext()
    ctx.set(WORKING_IMAGE, gray)
    ctx.set(GUTTER, MEDIANIL_BLANCO)
    ctx.set(CUT_TREE, CutNode(Rect(0, 0, ancho, alto)))
    ThinGutterRefineStage().run(ctx)
    return ctx.get(CUT_TREE)


def _viñeta(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, relleno: int = 200) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), relleno, thickness=-1)
    cv2.rectangle(img, (x1, y1), (x2, y2), 0, thickness=3)


def test_divide_dos_viñetas_separadas_por_un_medianil_de_dos_pixeles():
    img = np.full((400, 600), 255, dtype=np.uint8)
    _viñeta(img, 0, 0, 296, 399)
    _viñeta(img, 302, 0, 599, 399)  # entre ambos marcos quedan 2 píxeles blancos
    arbol = _refinar(img)
    assert arbol.axis is CutAxis.VERTICAL
    izquierda, derecha = (hoja.rect for hoja in arbol.leaves())
    assert izquierda.right < derecha.x
    assert abs(izquierda.right - 299) <= 2


def test_prefiere_filas_para_conservar_el_orden_de_lectura():
    img = np.full((600, 600), 255, dtype=np.uint8)
    for x1, x2 in ((0, 296), (302, 599)):
        for y1, y2 in ((0, 296), (302, 599)):
            _viñeta(img, x1, y1, x2, y2)
    arbol = _refinar(img)
    assert arbol.axis is CutAxis.HORIZONTAL
    assert all(fila.axis is CutAxis.VERTICAL for fila in arbol.children)
    assert len(list(arbol.leaves())) == 4


def test_no_divide_por_una_franja_clara_del_dibujo_sin_marcos():
    # Una franja blanca que atraviesa el dibujo, pero rodeada de tonos medios (no de tinta).
    img = np.full((400, 600), 255, dtype=np.uint8)
    _viñeta(img, 0, 0, 599, 399, relleno=128)
    img[3:-3, 298:302] = 255
    assert _refinar(img).is_leaf


def test_no_divide_si_hay_tinta_solo_a_un_lado():
    img = np.full((400, 600), 255, dtype=np.uint8)
    _viñeta(img, 0, 0, 599, 399, relleno=128)
    img[3:-3, 298:302] = 255
    img[3:-3, 295:298] = 0  # línea negra solo a la izquierda de la franja
    assert _refinar(img).is_leaf


def test_ignora_franjas_pegadas_al_borde_de_la_hoja():
    # Un medianil al 5 % del ancho dejaría un pedazo demasiado angosto para ser viñeta.
    img = np.full((400, 600), 255, dtype=np.uint8)
    _viñeta(img, 0, 0, 26, 399)
    _viñeta(img, 32, 0, 599, 399)
    assert _refinar(img).is_leaf


def test_respeta_las_hojas_que_ya_estaban_bien():
    img = np.full((400, 600), 255, dtype=np.uint8)
    _viñeta(img, 0, 0, 599, 399)
    assert _refinar(img).is_leaf


def test_rechaza_parametros_invalidos():
    with pytest.raises(ValueError):
        ThinGutterRefineStage(min_part=0.6)
    with pytest.raises(ValueError):
        ThinGutterRefineStage(min_light=0)