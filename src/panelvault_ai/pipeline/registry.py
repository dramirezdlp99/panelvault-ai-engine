"""Registro de componentes por nombre (patrones Registry + Factory).

Permite armar pipelines desde una configuración y agregar etapas o estrategias
nuevas sin modificar el núcleo: basta con decorar la clase con ``@registry.register``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from panelvault_ai.pipeline.context import ArtifactKey
from panelvault_ai.pipeline.errors import PipelineConfigurationError, UnknownComponentError
from panelvault_ai.pipeline.pipeline import Pipeline
from panelvault_ai.pipeline.stage import Stage


class Registry[T]:
    """Asocia nombres con clases de un mismo tipo base."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._components: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._components and self._components[name] is not cls:
                raise PipelineConfigurationError(
                    f"Ya existe un componente de tipo {self.kind} llamado {name!r}"
                )
            self._components[name] = cls
            return cls

        return decorator

    def create(self, name: str, **params: Any) -> T:
        try:
            cls = self._components[name]
        except KeyError:
            raise UnknownComponentError(
                f"No hay ningún componente de tipo {self.kind} llamado {name!r}. "
                f"Disponibles: {self.names()}"
            ) from None
        try:
            return cls(**params)
        except TypeError as exc:
            raise PipelineConfigurationError(
                f"Parámetros inválidos para {self.kind} {name!r}: {exc}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._components)

    def __contains__(self, name: object) -> bool:
        return name in self._components


STAGES: Registry[Stage] = Registry("etapa")


def build_pipeline(
    spec: Iterable[Mapping[str, Any]],
    initial: Iterable[ArtifactKey[Any]] = (),
    registry: Registry[Stage] = STAGES,
) -> Pipeline:
    """Construye un pipeline a partir de una especificación declarativa.

    Ejemplo de ``spec``::

        [{"stage": "normalize", "params": {"target_width": 1200}},
         {"stage": "binarize"}]
    """
    stages: list[Stage] = []
    for position, entry in enumerate(spec, start=1):
        if "stage" not in entry:
            raise PipelineConfigurationError(f"La entrada {position} no indica 'stage'")
        stages.append(registry.create(entry["stage"], **dict(entry.get("params", {}))))
    return Pipeline(stages, initial=initial)