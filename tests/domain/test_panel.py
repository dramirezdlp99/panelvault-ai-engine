"""Pruebas de Panel y PanelMap."""

import pytest

from panelvault_ai.domain import PageType, Panel, PanelMap, Polygon, ReadingDirection, Rect


def _panel(rect: Rect, order: int, confidence: float = 0.9) -> Panel:
    return Panel(Polygon.from_rect(rect), order, confidence)


def _mapa(panels: list[Panel]) -> PanelMap:
    return PanelMap(
        page_width=1000,
        page_height=1500,
        direction=ReadingDirection.LEFT_TO_RIGHT,
        page_type=PageType.GRID,
        panels=tuple(panels),
        confidence=0.85,
    )


def test_panel_rechaza_confianza_fuera_de_rango():
    with pytest.raises(ValueError):
        _panel(Rect(0, 0, 10, 10), 0, confidence=1.5)


def test_panel_rechaza_orden_negativo():
    with pytest.raises(ValueError):
        _panel(Rect(0, 0, 10, 10), -1)


def test_panel_bounds():
    assert _panel(Rect(1, 2, 3, 4), 0).bounds == Rect(1, 2, 3, 4)


def test_panelmap_acepta_ordenes_consecutivos():
    mapa = _mapa([_panel(Rect(0, 0, 500, 750), 0), _panel(Rect(500, 0, 500, 750), 1)])
    assert len(mapa) == 2


def test_panelmap_rechaza_ordenes_con_huecos():
    with pytest.raises(ValueError):
        _mapa([_panel(Rect(0, 0, 10, 10), 0), _panel(Rect(20, 0, 10, 10), 2)])


def test_panelmap_rechaza_ordenes_desordenados():
    with pytest.raises(ValueError):
        _mapa([_panel(Rect(0, 0, 10, 10), 1), _panel(Rect(20, 0, 10, 10), 0)])


def test_panelmap_vacio_es_valido():
    # Una página sin viñetas detectadas (por ejemplo, una portada) es un resultado legítimo.
    assert len(_mapa([])) == 0


def test_to_normalized_dict():
    mapa = _mapa([_panel(Rect(500, 0, 500, 750), 0)])
    data = mapa.to_normalized_dict()
    assert data["schemaVersion"] == 1
    assert data["direction"] == "ltr"
    assert data["pageType"] == "grid"
    assert data["panels"][0]["order"] == 0
    assert data["panels"][0]["bbox"] == [0.5, 0.0, 0.5, 0.5]
    assert data["panels"][0]["polygon"][2] == [1.0, 0.5]