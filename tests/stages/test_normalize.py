"""Pruebas de NormalizeStage."""

import numpy as np
import pytest

from panelvault_ai.pipeline import PageContext, StageExecutionError
from panelvault_ai.stages import ORIGINAL_IMAGE, SCALE, WORKING_IMAGE, NormalizeStage


def _correr(imagen, target_width=1200):
    ctx = PageContext()
    ctx.set(ORIGINAL_IMAGE, imagen)
    NormalizeStage(target_width).run(ctx)
    return ctx


def test_reduce_al_ancho_de_trabajo_conservando_la_proporcion():
    ctx = _correr(np.zeros((3000, 2000, 3), dtype=np.uint8))
    assert ctx.get(WORKING_IMAGE).shape == (1800, 1200)
    assert ctx.get(SCALE) == pytest.approx(0.6)


def test_nunca_agranda_imagenes_pequenas():
    ctx = _correr(np.zeros((900, 600, 3), dtype=np.uint8))
    assert ctx.get(WORKING_IMAGE).shape == (900, 600)
    assert ctx.get(SCALE) == 1.0


@pytest.mark.parametrize("canales", [None, 3, 4])
def test_acepta_gris_bgr_y_bgra(canales):
    forma = (100, 80) if canales is None else (100, 80, canales)
    ctx = _correr(np.full(forma, 200, dtype=np.uint8))
    resultado = ctx.get(WORKING_IMAGE)
    assert resultado.ndim == 2
    assert resultado[0, 0] == 200


def test_rechaza_imagenes_que_no_son_uint8():
    with pytest.raises(StageExecutionError, match="uint8"):
        _correr(np.zeros((100, 100), dtype=np.float32))


def test_rechaza_ancho_de_trabajo_absurdo():
    with pytest.raises(ValueError):
        NormalizeStage(target_width=10)