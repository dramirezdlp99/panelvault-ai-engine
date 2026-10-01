"""Pruebas del destino de depuración en archivos."""

import numpy as np

from panelvault_ai.pipeline import FileDebugSink


def test_escribe_png_numerados_en_orden(tmp_path):
    sink = FileDebugSink(tmp_path / "salida")
    imagen = np.full((8, 8), 255, dtype=np.uint8)
    sink.emit("binarize", "mascara", imagen)
    sink.emit("xy cut", "cortes/v1", imagen)
    archivos = sorted(p.name for p in (tmp_path / "salida").iterdir())
    assert archivos == ["01_binarize_mascara.png", "02_xy_cut_cortes_v1.png"]