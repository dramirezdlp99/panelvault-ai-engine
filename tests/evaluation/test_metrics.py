"""Pruebas de las métricas de detección y de orden de lectura."""

import itertools
import random

import pytest

from panelvault_ai.domain import Rect
from panelvault_ai.evaluation import count_inversions, kendall_tau, match_panels, score_page

A, B, C = Rect(0, 0, 10, 10), Rect(20, 0, 10, 10), Rect(0, 20, 10, 10)


# ----------------------------------------------------------------------
# Inversiones y Kendall τ
# ----------------------------------------------------------------------
def _inversiones_ingenuas(valores):
    return sum(
        1 for i, j in itertools.combinations(range(len(valores)), 2) if valores[i] > valores[j]
    )


def test_count_inversions_casos_basicos():
    assert count_inversions([]) == 0
    assert count_inversions([0, 1, 2, 3]) == 0
    assert count_inversions([3, 2, 1, 0]) == 6
    assert count_inversions([1, 0, 3, 2]) == 2


def test_count_inversions_coincide_con_la_version_ingenua():
    # La versión con merge sort es O(n log n); la ingenua es O(n²) pero obviamente correcta.
    rng = random.Random(42)
    for _ in range(300):
        valores = [rng.randrange(20) for _ in range(rng.randrange(0, 30))]
        assert count_inversions(valores) == _inversiones_ingenuas(valores)


def test_kendall_tau_extremos():
    assert kendall_tau([0, 1, 2, 3]) == 1.0
    assert kendall_tau([3, 2, 1, 0]) == -1.0
    assert kendall_tau([0]) == 1.0


def test_kendall_tau_un_intercambio_entre_cuatro():
    # 1 par discordante de 6: τ = 1 − 2·1/6.
    assert kendall_tau([1, 0, 2, 3]) == pytest.approx(1 - 2 / 6)


# ----------------------------------------------------------------------
# Emparejamiento
# ----------------------------------------------------------------------
def test_empareja_cada_viñeta_una_sola_vez():
    detectadas = [A, A]  # la misma viñeta detectada dos veces
    matches = match_panels(detectadas, [A])
    assert len(matches) == 1


def test_prefiere_la_pareja_con_mayor_iou():
    casi_a = Rect(1, 1, 10, 10)
    matches = match_panels([casi_a, A], [A])
    assert matches[0].predicted == 1  # A exacta gana sobre la aproximada


def test_descarta_parejas_bajo_el_umbral():
    desplazada = Rect(8, 0, 10, 10)  # IoU con A ≈ 0.11
    assert match_panels([desplazada], [A]) == []


# ----------------------------------------------------------------------
# Puntaje de página
# ----------------------------------------------------------------------
def test_pagina_perfecta():
    s = score_page([A, B, C], [A, B, C])
    assert (s.precision, s.recall, s.f1, s.order_tau) == (1.0, 1.0, 1.0, 1.0)
    assert s.is_perfect


def test_orden_equivocado_baja_tau_pero_no_la_deteccion():
    s = score_page([B, A, C], [A, B, C])
    assert s.recall == 1.0
    assert s.order_tau < 1.0
    assert not s.is_perfect


def test_viñetas_fusionadas_bajan_el_recall():
    fusion = A.union_bounds(B)
    s = score_page([fusion, C], [A, B, C])
    assert s.recall == pytest.approx(1 / 3)  # solo C coincide
    assert s.precision == pytest.approx(1 / 2)


def test_tau_no_se_define_con_menos_de_dos_parejas():
    assert score_page([C], [A, B, C]).order_tau is None


def test_pagina_sin_viñetas_ni_detecciones_es_perfecta():
    s = score_page([], [])
    assert s.is_perfect