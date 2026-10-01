"""Pruebas de BinarizeStage sobre páginas sintéticas con respuesta conocida."""

import cv2
import numpy as np
import pytest
from paginas import ejecutar, mascaras_de_verdad, pipeline_basico

from panelvault_ai.stages import CONTENT_MASK, SCALE
from panelvault_ai.synthetic import PageStyle, grid_layout, render_page, row_layout

ESTILOS = {
    "medianil_blanco": PageStyle(),
    "medianil_negro": PageStyle(gutter_value=0, border_value=255),
    "con_ruido": PageStyle(noise_sigma=6),
}


@pytest.mark.parametrize("estilo", ESTILOS.values(), ids=ESTILOS.keys())
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_separa_viñetas_del_medianil(estilo, seed):
    pagina = render_page(800, 1200, row_layout(800, 1200, [2, 3, 1], seed=seed), estilo, seed)
    ctx = ejecutar(pipeline_basico(), pagina.image)
    mascara = ctx.get(CONTENT_MASK)

    interior, medianil = mascaras_de_verdad(mascara.shape, pagina.panels, ctx.get(SCALE), 3)
    assert np.all(mascara[interior] == 255), "hay huecos dentro de alguna viñeta"
    assert np.all(mascara[medianil] == 0), "hay contenido falso en el medianil"


def test_zonas_del_color_del_medianil_encerradas_por_el_marco_son_contenido():
    # Página blanca con una viñeta de interior también blanco: solo el marco la delimita.
    imagen = np.full((600, 400), 255, dtype=np.uint8)
    cv2.rectangle(imagen, (50, 50), (349, 549), 0, thickness=3)
    ctx = ejecutar(pipeline_basico(), imagen)
    mascara = ctx.get(CONTENT_MASK)
    assert mascara[300, 200] == 255  # centro de la viñeta
    assert mascara[10, 10] == 0  # margen


def test_elimina_motas_aisladas_en_el_medianil():
    imagen = render_page(800, 1200, grid_layout(800, 1200, 2, 2, gutter=40)).image
    imagen[600:602, 395:397] = 0  # mota de 2x2 en mitad del medianil horizontal
    ctx = ejecutar(pipeline_basico(target_width=800), imagen)
    assert ctx.get(CONTENT_MASK)[600, 396] == 0


def test_funciona_desde_un_pipeline_construido_por_nombre():
    import panelvault_ai.stages  # noqa: F401  (registra las etapas)
    from panelvault_ai.pipeline import build_pipeline
    from panelvault_ai.stages import ORIGINAL_IMAGE

    pipeline = build_pipeline(
        [
            {"stage": "normalize", "params": {"target_width": 600}},
            {"stage": "gutter"},
            {"stage": "binarize"},
        ],
        initial={ORIGINAL_IMAGE},
    )
    pagina = render_page(800, 1200, grid_layout(800, 1200, 2, 2))
    ctx = ejecutar(pipeline, pagina.image)
    assert ctx.get(CONTENT_MASK).shape == (900, 600)