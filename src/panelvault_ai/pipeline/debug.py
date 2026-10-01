"""Destinos de las imágenes de depuración que emiten las etapas (Observer + Null Object)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class DebugSink(ABC):
    """Recibe imágenes intermedias del pipeline. Las etapas no saben qué se hace con ellas."""

    @abstractmethod
    def emit(self, stage_name: str, label: str, image: np.ndarray) -> None: ...


class NullDebugSink(DebugSink):
    """No hace nada. Es el destino por defecto en producción: costo cero."""

    def emit(self, stage_name: str, label: str, image: np.ndarray) -> None:
        return None


class MemoryDebugSink(DebugSink):
    """Guarda las imágenes en memoria. Útil en pruebas y en notebooks."""

    def __init__(self) -> None:
        self.images: list[tuple[str, str, np.ndarray]] = []

    def emit(self, stage_name: str, label: str, image: np.ndarray) -> None:
        self.images.append((stage_name, label, image.copy()))


class FileDebugSink(DebugSink):
    """Escribe cada imagen como PNG numerado en una carpeta, en el orden en que se emiten."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def emit(self, stage_name: str, label: str, image: np.ndarray) -> None:
        from panelvault_ai.imageio import write_image  # import diferido: evita ciclos

        self._counter += 1
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{stage_name}_{label}")
        write_image(self.directory / f"{self._counter:02d}_{safe}.png", image)