"""Pruebas de la validación y ejecución del Pipeline."""

import pytest
from etapas_juguete import NUMERO, TEXTO, Describir, Duplicar

from panelvault_ai.pipeline import PageContext, Pipeline, PipelineConfigurationError


def test_ejecuta_las_etapas_en_orden():
    pipeline = Pipeline([Duplicar(), Describir()], initial={NUMERO})
    ctx = PageContext()
    ctx.set(NUMERO, 4)
    pipeline.run(ctx)
    assert ctx.get(TEXTO) == "el doble es 8"
    assert [t.stage_name for t in ctx.timings] == ["duplicar", "describir"]


def test_detecta_el_orden_incorrecto_al_construirse():
    # Describir necesita DOBLE, que solo produce Duplicar: el error salta antes de procesar nada.
    with pytest.raises(PipelineConfigurationError, match=r"La etapa 1 \(describir\)"):
        Pipeline([Describir(), Duplicar()], initial={NUMERO})


def test_detecta_entradas_iniciales_faltantes_al_construirse():
    with pytest.raises(PipelineConfigurationError, match="numero"):
        Pipeline([Duplicar()])


def test_rechaza_etapas_repetidas():
    with pytest.raises(PipelineConfigurationError, match="repetidas"):
        Pipeline([Duplicar(), Duplicar()], initial={NUMERO})


def test_rechaza_pipeline_vacio():
    with pytest.raises(PipelineConfigurationError):
        Pipeline([])


def test_run_verifica_que_el_contexto_traiga_la_entrada_inicial():
    pipeline = Pipeline([Duplicar()], initial={NUMERO})
    with pytest.raises(PipelineConfigurationError, match="contexto inicial"):
        pipeline.run(PageContext())


def test_provides_resume_todo_lo_que_existira_al_final():
    pipeline = Pipeline([Duplicar(), Describir()], initial={NUMERO})
    assert {k.name for k in pipeline.provides} == {"numero", "doble", "texto"}