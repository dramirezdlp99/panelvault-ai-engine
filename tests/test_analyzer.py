"""Pruebas de extremo a extremo: imagen de entrada → PanelMap, sobre páginas sintéticas y reales."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from panelvault_ai.analyzer import PanelAnalyzer
from panelvault_ai.domain import PageType, Rect
from panelvault_ai.imageio import read_image
from panelvault_ai.synthetic import PageStyle, grid_layout, render_page, row_layout

UMBRAL_IOU = 0.9
FIXTURES = Path(__file__).parent / "fixtures"


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


@pytest.mark.parametrize("medianil", [6, 7])
@pytest.mark.parametrize("seed", range(3))
def test_separa_viñetas_con_medianiles_demasiado_delgados_para_el_xy_cut(medianil, seed):
    # Con medianiles de 6-7 px, el espacio blanco visible entre marcos es de 2-3 px:
    # menos que el min_gap del XY-Cut. Lo resuelve la etapa de refinamiento.
    layout = row_layout(800, 1200, [2, 3, 1, 2], seed=seed, gutter=medianil)
    pagina = render_page(800, 1200, layout, PageStyle(noise_sigma=4), seed)
    _verificar_orden(PanelAnalyzer("western").analyze(pagina.image), list(pagina.panels))


def test_papel_amarillento_con_un_trazo_que_cruza_el_medianil():
    # Reproduce el fallo encontrado en una página real: papel de gris 160 y la cola de
    # un globo que cruza el medianil entre dos viñetas. Ese trazo deja un pedazo de
    # medianil encerrado, la binarización lo marca como contenido y el XY-Cut no corta.
    # La etapa de refinamiento debe separarlas.
    imagen = np.full((900, 700), 160, dtype=np.uint8)
    viñetas = [Rect(30, 30, 301, 391), Rect(342, 30, 329, 391), Rect(30, 440, 641, 431)]
    for r in viñetas:
        esquinas = ((int(r.x), int(r.y)), (int(r.right) - 1, int(r.bottom) - 1))
        cv2.rectangle(imagen, *esquinas, 205, thickness=-1)
        cv2.rectangle(imagen, *esquinas, 60, thickness=3)
    cv2.line(imagen, (300, 80), (370, 80), 40, thickness=3)
    ruido = np.random.default_rng(0).normal(0, 5, imagen.shape)
    imagen = np.clip(imagen + ruido, 0, 255).astype(np.uint8)
    _verificar_orden(PanelAnalyzer().analyze(imagen), viñetas)


def test_pagina_real_de_dominio_publico():
    # Página de un cómic de dominio público (Wikimedia Commons). Respuesta verificada a
    # mano: 7 viñetas, la 1 y la 2 separadas por un medianil parcialmente cruzado.
    mapa = PanelAnalyzer().analyze(read_image(FIXTURES / "real1.jpg"))
    esperadas = [
        (0.028, 0.017, 0.306, 0.314),
        (0.347, 0.017, 0.268, 0.314),
        (0.627, 0.019, 0.348, 0.313),
        (0.027, 0.372, 0.472, 0.287),
        (0.512, 0.343, 0.462, 0.316),
        (0.011, 0.670, 0.516, 0.318),
        (0.538, 0.670, 0.435, 0.318),
    ]
    obtenidas = [p["bbox"] for p in mapa.to_normalized_dict()["panels"]]
    assert len(obtenidas) == len(esperadas)
    for obtenida, esperada in zip(obtenidas, esperadas, strict=True):
        assert obtenida == pytest.approx(list(esperada), abs=0.02)


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
    assert nombres == ["normalize", "gutter", "binarize", "xycut", "refine", "order", "assemble"]