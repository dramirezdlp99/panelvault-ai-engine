"""Contexto compartido entre etapas (patrón Blackboard con claves tipadas).

Cada dato que viaja por el pipeline se identifica con un ``ArtifactKey``. La clave
lleva el tipo del valor, así que el resto del código sabe qué obtiene sin conversiones
sueltas, y las etapas pueden declarar exactamente qué claves leen y escriben.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from panelvault_ai.pipeline.debug import DebugSink, NullDebugSink
from panelvault_ai.pipeline.errors import PanelVaultError


@dataclass(frozen=True, slots=True)
class ArtifactKey[T]:
    """Identificador tipado de un dato del contexto.

    Dos claves son iguales si tienen el mismo nombre.
    """

    name: str
    description: str = field(default="", compare=False)

    def __repr__(self) -> str:
        return f"ArtifactKey({self.name!r})"


class MissingArtifactError(PanelVaultError):
    """Se pidió un dato que ninguna etapa anterior produjo."""


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_name: str
    seconds: float


class PageContext:
    """Estado de una página mientras atraviesa el pipeline."""

    def __init__(self, debug: DebugSink | None = None) -> None:
        self._artifacts: dict[ArtifactKey[Any], Any] = {}
        self.debug: DebugSink = debug or NullDebugSink()
        self.timings: list[StageTiming] = []

    def set[T](self, key: ArtifactKey[T], value: T) -> None:
        self._artifacts[key] = value

    def get[T](self, key: ArtifactKey[T]) -> T:
        try:
            return cast(T, self._artifacts[key])
        except KeyError:
            raise MissingArtifactError(f"El contexto no contiene {key.name!r}") from None

    def has(self, key: ArtifactKey[Any]) -> bool:
        return key in self._artifacts

    @property
    def available(self) -> frozenset[ArtifactKey[Any]]:
        return frozenset(self._artifacts)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)