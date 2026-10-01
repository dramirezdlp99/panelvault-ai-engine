"""Pruebas del contexto compartido entre etapas."""

import pytest
from etapas_juguete import DOBLE, NUMERO

from panelvault_ai.pipeline import ArtifactKey, MissingArtifactError, NullDebugSink, PageContext


def test_guarda_y_recupera_datos_por_clave():
    ctx = PageContext()
    ctx.set(NUMERO, 21)
    assert ctx.get(NUMERO) == 21
    assert ctx.has(NUMERO)
    assert not ctx.has(DOBLE)


def test_pedir_un_dato_inexistente_falla_con_mensaje_claro():
    with pytest.raises(MissingArtifactError, match="doble"):
        PageContext().get(DOBLE)


def test_claves_con_el_mismo_nombre_son_iguales():
    assert ArtifactKey("numero", "otra descripción") == NUMERO


def test_por_defecto_la_depuracion_no_hace_nada():
    assert isinstance(PageContext().debug, NullDebugSink)