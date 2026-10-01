"""Pruebas del Value Object Polygon: shoelace, centroide, convexidad y Sutherland–Hodgman."""

import pytest

from panelvault_ai.domain import Polygon, Rect

CUADRADO = Polygon(((0, 0), (10, 0), (10, 10), (0, 10)))


def test_rechaza_menos_de_tres_vertices():
    with pytest.raises(ValueError):
        Polygon(((0, 0), (1, 1)))


def test_area_por_shoelace():
    assert CUADRADO.area == pytest.approx(100)


def test_area_no_depende_del_sentido_de_los_vertices():
    invertido = Polygon(tuple(reversed(CUADRADO.points)))
    assert invertido.area == pytest.approx(100)


def test_area_de_triangulo():
    triangulo = Polygon(((0, 0), (4, 0), (0, 3)))
    assert triangulo.area == pytest.approx(6)


def test_centroide_de_cuadrado():
    assert CUADRADO.centroid == pytest.approx((5, 5))


def test_centroide_es_centro_de_masa_y_no_promedio_de_vertices():
    # Un vértice extra en mitad de un lado no mueve el centro de masa,
    # pero sí movería el promedio simple de vértices.
    con_vertice_extra = Polygon(((0, 0), (5, 0), (10, 0), (10, 10), (0, 10)))
    assert con_vertice_extra.centroid == pytest.approx((5, 5))


def test_from_rect_conserva_area_y_limites():
    r = Rect(2, 3, 8, 4)
    p = Polygon.from_rect(r)
    assert p.area == pytest.approx(r.area)
    assert p.bounding_rect == r


def test_bounding_rect_de_poligono_inclinado():
    rombo = Polygon(((5, 0), (10, 5), (5, 10), (0, 5)))
    assert rombo.bounding_rect == Rect(0, 0, 10, 10)


def test_convexidad():
    assert CUADRADO.is_convex
    en_forma_de_l = Polygon(((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)))
    assert not en_forma_de_l.is_convex


def test_clip_de_cuadrados_solapados():
    otro = Polygon(((5, 5), (15, 5), (15, 15), (5, 15)))
    inter = CUADRADO.clip(otro)
    assert inter is not None
    assert inter.area == pytest.approx(25)
    assert inter.bounding_rect == Rect(5, 5, 5, 5)


def test_clip_funciona_con_cualquier_sentido_del_recortador():
    otro = Polygon(((5, 5), (5, 15), (15, 15), (15, 5)))  # sentido opuesto
    inter = CUADRADO.clip(otro)
    assert inter is not None
    assert inter.area == pytest.approx(25)


def test_clip_sin_solape_devuelve_none():
    lejano = Polygon(((50, 50), (60, 50), (60, 60), (50, 60)))
    assert CUADRADO.clip(lejano) is None


def test_clip_de_viñeta_inclinada():
    # Rombo inscrito en el cuadrado: queda completamente dentro.
    rombo = Polygon(((5, 0), (10, 5), (5, 10), (0, 5)))
    inter = rombo.clip(CUADRADO)
    assert inter is not None
    assert inter.area == pytest.approx(rombo.area)


def test_clip_rechaza_recortador_no_convexo():
    en_forma_de_l = Polygon(((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)))
    with pytest.raises(ValueError):
        CUADRADO.clip(en_forma_de_l)


def test_iou_de_poligonos_coincide_con_iou_de_rectangulos():
    a, b = Rect(0, 0, 10, 10), Rect(5, 5, 10, 10)
    assert Polygon.from_rect(a).iou(Polygon.from_rect(b)) == pytest.approx(a.iou(b))


def test_iou_de_poligono_consigo_mismo_es_uno():
    assert CUADRADO.iou(CUADRADO) == pytest.approx(1.0)


def test_normalized():
    n = CUADRADO.normalized(20, 40)
    coordenadas = [c for punto in n.points for c in punto]
    assert coordenadas == pytest.approx([0, 0, 0.5, 0, 0.5, 0.25, 0, 0.25])