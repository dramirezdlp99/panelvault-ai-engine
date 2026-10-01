"""Pruebas de humo: verifican que el paquete y sus dependencias se instalaron bien."""

import cv2
import numpy as np

import panelvault_ai


def test_paquete_expone_version_instalada():
    assert panelvault_ai.__version__ != "0.0.0"


def test_opencv_opera_sobre_matrices_numpy():
    imagen = np.zeros((10, 10), dtype=np.uint8)
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    assert binaria.shape == (10, 10)