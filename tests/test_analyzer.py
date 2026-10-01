"""Pruebas de extremo a extremo: imagen de entrada → PanelMap, sobre páginas sintéticas."""

import cv2
import numpy as np
import pytest

from panelvault_ai.analyzer import PanelAnalyzer
from panelvault_ai.domain import PageType, Rect
from panelvault_ai.synthetic import PageStyle, grid_layout, render_page, row_layout

UMBRAL_IOU = 0.9


def _verificar_orden(mapa, esperadas: list[Rect]):
    """Cada viñeta detectada debe coincidir (IoU alto) con la esperada en la misma posición."""
    assert len(mapa) == len(esperadas)
    for panel, esperada in zip(mapa.panels, esperadas, strict=True):
        assert panel.bounds.iou(esperada) >= UMBRAL_IOU, (panel.order, panel.bounds, esperada)


ESTILOS = {
    "medianil_blanco": PageStyle(),
    "medianil_negro": PageStyle(gutter_value=0, border_value=255),
    "con_ruido": PageStyle(noise_sigma=8),
}


@pytest.mark.parametrize("estilo", ESTILOS.values(), ids=ESTILOS.keys())
@pytest.mark.parametrize("seed", range(6))
def test_detecta_todas_las_viñetas_en_orden_occidental(estilo, seed):
    layout = row_layout(800, 1200, [2, 3, 1, 2], seed=seed)
    pagina = render_page(800, 1200, layout, estilo, seed)
    mapa = PanelAnalyzer("western").analyze(pagina.image)
    _verificar_orden(mapa, list(pagina.panels))
    assert mapa.page_type is PageType.GRID
    assert mapa.confidence > 0.9


def test_manga_invierte_el_orden_dentro_de_cada_fila():
    layout = grid_layout(800, 1200, rows=2, cols=3)
    pagina = render_page(800, 1200, layout)
    mapa = PanelAnalyzer("manga").analyze(pagina.image)
    esperado = [layout[2], layout[1], layout[0], layout[5], layout[4], layout[3]]
    _verificar_orden(mapa, esperado)
    assert mapa.direction.value == "rtl"


def test_columna_alta_con_viñetas_apiladas():
    alta = Rect(40, 40, 340, 1120)
    arriba = Rect(400, 40, 360, 550)
    abajo = Rect(400, 610, 360, 550)
    pagina = render_page(800, 1200, [alta, arriba, abajo])
    _verificar_orden(PanelAnalyzer("western").analyze(pagina.image), [alta, arriba, abajo])
    _verificar_orden(PanelAnalyzer("manga").analyze(pagina.image), [arriba, abajo, alta])


def test_las_coordenadas_se_reportan_en_la_imagen_de_trabajo_y_normalizadas():
    # Página grande (2400 px): se reduce a 1200, pero lo normalizado no depende del tamaño.
    layout = grid_layout(2400, 3600, rows=2, cols=2, gutter=60, margin=120)
    mapa = PanelAnalyzer().analyze(render_page(2400, 3600, layout).image)
    assert (mapa.page_width, mapa.page_height) == (1200, 1800)
    primera = mapa.to_normalized_dict()["panels"][0]["bbox"]
    esperada = layout[0].normalized(2400, 3600)
    assert primera == pytest.approx(
        [esperada.x, esperada.y, esperada.width, esperada.height], abs=0.01
    )


def test_ignora_el_numero_de_pagina():
    layout = grid_layout(800, 1200, rows=2, cols=2)
    imagen = render_page(800, 1200, layout).image
    cv2.putText(imagen, "12", (390, 1190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    assert len(PanelAnalyzer().analyze(imagen)) == 4


def test_pagina_splash():
    pagina = render_page(800, 1200, [Rect(40, 40, 720, 1120)])
    mapa = PanelAnalyzer().analyze(pagina.image)
    assert len(mapa) == 1
    assert mapa.page_type is PageType.SPLASH


def test_pagina_en_blanco():
    mapa = PanelAnalyzer().analyze(np.full((1200, 800, 3), 255, dtype=np.uint8))
    assert len(mapa) == 0
    assert mapa.page_type is PageType.UNKNOWN
    assert mapa.confidence == 0.0


def test_preset_desconocido():
    with pytest.raises(ValueError, match="western"):
        PanelAnalyzer("inexistente")


def test_el_contexto_expone_los_tiempos_de_cada_etapa():
    ctx = PanelAnalyzer().analyze_with_context(
        render_page(800, 1200, grid_layout(800, 1200, 2, 2)).image
    )
    nombres = [t.stage_name for t in ctx.timings]
    assert nombres == ["normalize", "gutter", "binarize", "xycut", "order", "assemble"]