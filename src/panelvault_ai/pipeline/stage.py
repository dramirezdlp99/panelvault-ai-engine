"""Clase base de las etapas del pipeline (patrón Template Method)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from panelvault_ai.pipeline.context import ArtifactKey, PageContext, StageTiming
from panelvault_ai.pipeline.errors import PanelVaultError, StageExecutionError


class Stage(ABC):
    """Un paso del análisis de una página.

    Las subclases declaran qué datos necesitan (``requires``) y cuáles producen
    (``provides``), e implementan solo ``_process``. El método ``run`` es fijo para
    todas: valida entradas, mide el tiempo, envuelve errores y verifica que la etapa
    haya cumplido su contrato de salida.
    """

    name: ClassVar[str] = ""
    requires: ClassVar[frozenset[ArtifactKey[Any]]] = frozenset()
    provides: ClassVar[frozenset[ArtifactKey[Any]]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    def run(self, ctx: PageContext) -> None:
        missing = [k.name for k in self.requires if not ctx.has(k)]
        if missing:
            raise StageExecutionError(self.name, f"faltan datos de entrada: {sorted(missing)}")

        start = time.perf_counter()
        try:
            self._process(ctx)
        except PanelVaultError:
            raise
        except Exception as exc:  # cualquier fallo inesperado queda atribuido a esta etapa
            raise StageExecutionError(self.name, f"{type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - start

        not_produced = [k.name for k in self.provides if not ctx.has(k)]
        if not_produced:
            raise StageExecutionError(
                self.name, f"no produjo los datos que declara: {sorted(not_produced)}"
            )
        ctx.timings.append(StageTiming(self.name, elapsed))

    @abstractmethod
    def _process(self, ctx: PageContext) -> None:
        """Algoritmo propio de la etapa. Lee de ``ctx`` lo que requiere y escribe lo que provee."""

    def _debug(self, ctx: PageContext, label: str, image: np.ndarray) -> None:
        """Emite una imagen intermedia hacia el destino de depuración configurado."""
        ctx.debug.emit(self.name, label, image)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"