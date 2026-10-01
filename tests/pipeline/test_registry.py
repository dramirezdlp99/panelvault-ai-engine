"""Pruebas del Registry y de la construcción de pipelines desde configuración."""

import pytest
from etapas_juguete import NUMERO, TEXTO, Describir, Duplicar

from panelvault_ai.pipeline import (
    PageContext,
    PipelineConfigurationError,
    Registry,
    Stage,
    UnknownComponentError,
    build_pipeline,
)


@pytest.fixture
def registro() -> Registry[Stage]:
    r: Registry[Stage] = Registry("etapa")
    r.register("duplicar")(Duplicar)
    r.register("describir")(Describir)
    return r


def test_crea_componentes_por_nombre(registro):
    assert isinstance(registro.create("duplicar"), Duplicar)
    assert registro.names() == ["describir", "duplicar"]
    assert "duplicar" in registro


def test_nombre_desconocido_lista_los_disponibles(registro):
    with pytest.raises(UnknownComponentError, match="describir"):
        registro.create("inexistente")


def test_no_permite_dos_clases_con_el_mismo_nombre(registro):
    class Otra(Stage):
        def _process(self, ctx):
            pass

    with pytest.raises(PipelineConfigurationError):
        registro.register("duplicar")(Otra)


def test_parametros_invalidos_dan_error_de_configuracion(registro):
    with pytest.raises(PipelineConfigurationError, match="Parámetros inválidos"):
        registro.create("duplicar", parametro_que_no_existe=1)


def test_build_pipeline_desde_especificacion(registro):
    spec = [{"stage": "duplicar"}, {"stage": "describir", "params": {}}]
    pipeline = build_pipeline(spec, initial={NUMERO}, registry=registro)
    ctx = PageContext()
    ctx.set(NUMERO, 10)
    pipeline.run(ctx)
    assert ctx.get(TEXTO) == "el doble es 20"


def test_build_pipeline_exige_el_campo_stage(registro):
    with pytest.raises(PipelineConfigurationError, match="'stage'"):
        build_pipeline([{"params": {}}], registry=registro)