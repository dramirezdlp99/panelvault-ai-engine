"""Pruebas de GutterEstimationStage."""

import numpy as np
import pytest

from panelvault_ai.pipeline import PageContext
from panelvault_ai.stages import GUTTER, WORKING_IMAGE, GutterEstimationStage
from panelvault_ai.synthetic import PageStyle, grid_layout, render_page


def _estimar(gray):
    ctx = PageContext()
    ctx.set(WORKING_IMAGE, gray)
    GutterEstimationStage().run(ctx)
    return ctx.get(GUTTER)


def _pagina(style):
    return render_page(800, 1200, grid_layout(800, 1200, 3, 2), style).image[:, :, 0]


def test_detecta_medianil_blanco():
    g = _estimar(_pagina(PageStyle()))
    assert g.value == 255
    assert g.is_light
    assert g.confidence == pytest.approx(1.0)


def test_detecta_medianil_negro():
    g = _estimar(_pagina(PageStyle(gutter_value=0, border_value=255)))
    assert g.value == 0
    assert not g.is_light


def test_la_tolerancia_crece_con_el_ruido():
    limpia = _estimar(_pagina(PageStyle()))
    ruidosa = _estimar(_pagina(PageStyle(noise_sigma=12)))
    assert ruidosa.tolerance > limpia.tolerance
    assert ruidosa.confidence > 0.95


def test_baja_confianza_cuando_el_dibujo_llega_al_borde():
    # Página "a sangre": la mitad del borde es dibujo oscuro, no medianil.
    gray = np.full((1200, 800), 255, dtype=np.uint8)
    gray[:, :400] = 30
    assert _estimar(gray).confidence < 0.7