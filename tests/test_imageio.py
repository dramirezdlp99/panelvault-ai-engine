"""Pruebas de lectura y escritura de imágenes, incluidas rutas con tildes y ñ."""

import numpy as np
import pytest

from panelvault_ai.imageio import read_image, write_image


def test_ida_y_vuelta_con_ruta_con_tildes_y_enie(tmp_path):
    carpeta = tmp_path / "Cómics de Año"
    carpeta.mkdir()
    imagen = np.zeros((20, 30, 3), dtype=np.uint8)
    imagen[5:10, 5:10] = (0, 128, 255)
    ruta = carpeta / "página 1.png"
    write_image(ruta, imagen)
    assert np.array_equal(read_image(ruta), imagen)


def test_lee_imagenes_en_gris_como_bgr(tmp_path):
    ruta = tmp_path / "gris.png"
    write_image(ruta, np.full((10, 10), 200, dtype=np.uint8))
    assert read_image(ruta).shape == (10, 10, 3)


def test_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "no_existe.png")


def test_archivo_que_no_es_imagen(tmp_path):
    ruta = tmp_path / "texto.png"
    ruta.write_text("esto no es una imagen")
    with pytest.raises(ValueError, match="no es una imagen"):
        read_image(ruta)