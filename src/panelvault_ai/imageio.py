"""Lectura y escritura de imágenes segura para rutas de Windows.

``cv2.imread`` y ``cv2.imwrite`` fallan en Windows con rutas que contienen
caracteres fuera de ASCII (tildes, ñ). Leer los bytes con NumPy y decodificarlos
en memoria evita el problema en cualquier sistema operativo.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Lee una imagen como BGR uint8. Lanza ``ValueError`` si no es una imagen válida."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path.name} no es una imagen que OpenCV pueda leer")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Escribe una imagen; el formato se deduce de la extensión (.png, .jpg, ...)."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"No se pudo codificar la imagen {path}")
    encoded.tofile(path)