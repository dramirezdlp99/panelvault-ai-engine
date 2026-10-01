"""Pruebas del orden de lectura sobre árboles de cortes."""

from panelvault_ai.domain import ReadingDirection, Rect
from panelvault_ai.stages import CutAxis, CutNode, reading_order

A, B, C, D = (Rect(0, 0, 10, 10), Rect(20, 0, 10, 10), Rect(0, 20, 10, 10), Rect(20, 20, 10, 10))
LTR, RTL = ReadingDirection.LEFT_TO_RIGHT, ReadingDirection.RIGHT_TO_LEFT


def _hoja(r):
    return CutNode(r)


def _cuadricula():
    fila1 = CutNode(Rect(0, 0, 30, 10), CutAxis.VERTICAL, (_hoja(A), _hoja(B)))
    fila2 = CutNode(Rect(0, 20, 30, 10), CutAxis.VERTICAL, (_hoja(C), _hoja(D)))
    return CutNode(Rect(0, 0, 30, 30), CutAxis.HORIZONTAL, (fila1, fila2))


def test_occidental_lee_filas_de_izquierda_a_derecha():
    assert reading_order(_cuadricula(), LTR) == (A, B, C, D)


def test_manga_lee_filas_de_derecha_a_izquierda():
    assert reading_order(_cuadricula(), RTL) == (B, A, D, C)


def test_las_filas_siempre_se_leen_de_arriba_hacia_abajo():
    columna = CutNode(Rect(0, 0, 10, 30), CutAxis.HORIZONTAL, (_hoja(A), _hoja(C)))
    assert reading_order(columna, LTR) == reading_order(columna, RTL) == (A, C)


def test_termina_una_columna_completa_antes_de_pasar_a_la_siguiente():
    alta = Rect(0, 0, 10, 30)
    apiladas = CutNode(Rect(20, 0, 10, 30), CutAxis.HORIZONTAL, (_hoja(B), _hoja(D)))
    arbol = CutNode(Rect(0, 0, 30, 30), CutAxis.VERTICAL, (_hoja(alta), apiladas))
    assert reading_order(arbol, LTR) == (alta, B, D)
    assert reading_order(arbol, RTL) == (B, D, alta)


def test_arbol_vacio():
    assert reading_order(None, LTR) == ()