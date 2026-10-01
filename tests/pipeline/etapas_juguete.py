"""Etapas de juguete para probar el núcleo del pipeline sin depender de visión por computador."""

import numpy as np

from panelvault_ai.pipeline import ArtifactKey, PageContext, Stage

NUMERO: ArtifactKey[int] = ArtifactKey("numero")
DOBLE: ArtifactKey[int] = ArtifactKey("doble")
TEXTO: ArtifactKey[str] = ArtifactKey("texto")


class Duplicar(Stage):
    name = "duplicar"
    requires = frozenset({NUMERO})
    provides = frozenset({DOBLE})

    def _process(self, ctx: PageContext) -> None:
        ctx.set(DOBLE, ctx.get(NUMERO) * 2)
        self._debug(ctx, "lienzo", np.zeros((4, 4), dtype=np.uint8))


class Describir(Stage):
    name = "describir"
    requires = frozenset({DOBLE})
    provides = frozenset({TEXTO})

    def _process(self, ctx: PageContext) -> None:
        ctx.set(TEXTO, f"el doble es {ctx.get(DOBLE)}")


class Incumplida(Stage):
    """Declara que produce TEXTO pero no lo hace."""

    name = "incumplida"
    requires = frozenset({NUMERO})
    provides = frozenset({TEXTO})

    def _process(self, ctx: PageContext) -> None:
        pass


class Explosiva(Stage):
    name = "explosiva"
    requires = frozenset({NUMERO})

    def _process(self, ctx: PageContext) -> None:
        raise ZeroDivisionError("división entre cero")