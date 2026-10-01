"""Pruebas del Value Object Rect."""

import pytest

from panelvault_ai.domain import Rect


# ----------------------------------------------------------------------
# Construcción y validación
# ----------------------------------------------------------------------
def test_rechaza_dimensiones_negativas():
    with pytest.raises(ValueError):
        Rect(0, 0, -5, 10)


def test_es_inmutable():
    r = Rect(0, 0, 10, 10)
    with pytest.raises(AttributeError):
        r.x = 5  # type: ignore[misc]


def test_from_corners_acepta_esquinas_en_cualquier_orden():
    assert Rect.from_corners(10, 20, 0, 0) == Rect(0, 0, 10, 20)


def test_propiedades_derivadas():
    r = Rect(10, 20, 30, 40)
    assert r.right == 40
    assert r.bottom == 60
    assert r.area == 1200
    assert r.center == (25, 40)


# ----------------------------------------------------------------------
# Intersección
# ----------------------------------------------------------------------
def test_interseccion_de_rectangulos_solapados():
    a = Rect(0, 0, 10, 10)
    b = Rect(5, 5, 10, 10)
    assert a.intersection(b) == Rect(5, 5, 5, 5)


def test_interseccion_es_conmutativa():
    a = Rect(0, 0, 10, 10)
    b = Rect(3, 4, 20, 2)
    assert a.intersection(b) == b.intersection(a)


def test_rectangulos_separados_no_se_intersectan():
    assert Rect(0, 0, 10, 10).intersection(Rect(20, 20, 5, 5)) is None


def test_rectangulos_que_solo_se_tocan_en_un_borde_no_se_intersectan():
    # Dos viñetas pegadas, sin medianil entre ellas, no comparten área.
    assert Rect(0, 0, 10, 10).intersection(Rect(10, 0, 10, 10)) is None


# ----------------------------------------------------------------------
# IoU
# ----------------------------------------------------------------------
def test_iou_de_rectangulos_identicos_es_uno():
    r = Rect(3, 3, 7, 9)
    assert r.iou(r) == pytest.approx(1.0)


def test_iou_de_rectangulos_separados_es_cero():
    assert Rect(0, 0, 10, 10).iou(Rect(50, 50, 10, 10)) == 0.0


def test_iou_con_solape_parcial():
    # Intersección 5x5 = 25. Unión = 100 + 100 - 25 = 175.
    a = Rect(0, 0, 10, 10)
    b = Rect(5, 5, 10, 10)
    assert a.iou(b) == pytest.approx(25 / 175)


def test_iou_de_rectangulo_contenido_en_otro():
    # El pequeño (área 25) dentro del grande (área 100): IoU = 25 / 100.
    grande = Rect(0, 0, 10, 10)
    pequeno = Rect(2, 2, 5, 5)
    assert grande.iou(pequeno) == pytest.approx(0.25)


# ----------------------------------------------------------------------
# Contención
# ----------------------------------------------------------------------
def test_contains_point_usa_intervalo_semiabierto():
    r = Rect(0, 0, 10, 10)
    assert r.contains_point(0, 0)
    assert r.contains_point(9.9, 9.9)
    assert not r.contains_point(10, 5)


def test_contains_rectangulo():
    r = Rect(0, 0, 10, 10)
    assert r.contains(Rect(2, 2, 5, 5))
    assert r.contains(r)
    assert not r.contains(Rect(8, 8, 5, 5))


def test_union_bounds_envuelve_a_ambos():
    a = Rect(0, 0, 5, 5)
    b = Rect(10, 20, 5, 5)
    assert a.union_bounds(b) == Rect(0, 0, 15, 25)


# ----------------------------------------------------------------------
# Transformaciones
# ----------------------------------------------------------------------
def test_scaled_ida_y_vuelta_recupera_el_original():
    r = Rect(100, 200, 300, 400)
    ida_y_vuelta = r.scaled(0.5).scaled(2)
    assert ida_y_vuelta == r


def test_scaled_rechaza_factores_no_positivos():
    with pytest.raises(ValueError):
        Rect(0, 0, 10, 10).scaled(0)


def test_normalized_expresa_fracciones_de_la_pagina():
    # Viñeta que ocupa la mitad derecha de una página de 1000x1500.
    vineta = Rect(500, 0, 500, 1500)
    n = vineta.normalized(1000, 1500)
    assert n.x == pytest.approx(0.5)
    assert n.y == pytest.approx(0.0)
    assert n.width == pytest.approx(0.5)
    assert n.height == pytest.approx(1.0)