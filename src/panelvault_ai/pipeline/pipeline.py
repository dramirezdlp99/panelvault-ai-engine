"""Cadena ordenada de etapas, validada al construirse."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from panelvault_ai.pipeline.context import ArtifactKey, PageContext
from panelvault_ai.pipeline.errors import PipelineConfigurationError
from panelvault_ai.pipeline.stage import Stage


class Pipeline:
    """Ejecuta etapas en orden sobre un PageContext.

    Al construirse simula el flujo de datos: recorre las etapas acumulando lo que
    cada una produce y falla de inmediato si alguna necesita algo que ninguna
    etapa anterior (ni la entrada inicial) aporta.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        initial: Iterable[ArtifactKey[Any]] = (),
    ) -> None:
        if not stages:
            raise PipelineConfigurationError("Un pipeline necesita al menos una etapa")
        self._stages = tuple(stages)
        self._initial = frozenset(initial)
        self._validate()

    def _validate(self) -> None:
        names = [s.name for s in self._stages]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise PipelineConfigurationError(f"Etapas repetidas en el pipeline: {duplicates}")

        available = set(self._initial)
        for position, stage in enumerate(self._stages, start=1):
            missing = sorted(k.name for k in stage.requires - available)
            if missing:
                raise PipelineConfigurationError(
                    f"La etapa {position} ({stage.name}) necesita {missing}, "
                    f"pero ninguna etapa anterior lo produce"
                )
            available |= stage.provides

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    @property
    def provides(self) -> frozenset[ArtifactKey[Any]]:
        """Todo lo que existirá en el contexto al terminar."""
        result = set(self._initial)
        for stage in self._stages:
            result |= stage.provides
        return frozenset(result)

    def run(self, ctx: PageContext) -> PageContext:
        missing = sorted(k.name for k in self._initial if not ctx.has(k))
        if missing:
            raise PipelineConfigurationError(f"El contexto inicial no contiene {missing}")
        for stage in self._stages:
            stage.run(ctx)
        return ctx

    def __repr__(self) -> str:
        return f"Pipeline({' -> '.join(s.name for s in self._stages)})"