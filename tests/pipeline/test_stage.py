"""Pruebas del Template Method de Stage."""

import pytest
from etapas_juguete import DOBLE, NUMERO, Duplicar, Explosiva, Incumplida

from panelvault_ai.pipeline import MemoryDebugSink, PageContext, StageExecutionError


def test_ejecuta_el_algoritmo_y_registra_el_tiempo():
    ctx = PageContext()
    ctx.set(NUMERO, 5)
    Duplicar().run(ctx)
    assert ctx.get(DOBLE) == 10
    assert [t.stage_name for t in ctx.timings] == ["duplicar"]
    assert ctx.timings[0].seconds >= 0


def test_falla_si_faltan_datos_de_entrada():
    with pytest.raises(StageExecutionError, match="faltan datos de entrada"):
        Duplicar().run(PageContext())


def test_falla_si_no_produce_lo_que_declara():
    ctx = PageContext()
    ctx.set(NUMERO, 1)
    with pytest.raises(StageExecutionError, match="no produjo"):
        Incumplida().run(ctx)


def test_los_errores_inesperados_quedan_atribuidos_a_la_etapa():
    ctx = PageContext()
    ctx.set(NUMERO, 1)
    with pytest.raises(StageExecutionError) as info:
        Explosiva().run(ctx)
    assert info.value.stage_name == "explosiva"
    assert isinstance(info.value.__cause__, ZeroDivisionError)


def test_las_imagenes_de_depuracion_llegan_al_destino_configurado():
    sink = MemoryDebugSink()
    ctx = PageContext(debug=sink)
    ctx.set(NUMERO, 1)
    Duplicar().run(ctx)
    assert [(stage, label) for stage, label, _ in sink.images] == [("duplicar", "lienzo")]


def test_el_nombre_por_defecto_es_el_de_la_clase():
    from panelvault_ai.pipeline import Stage

    class SinNombre(Stage):
        def _process(self, ctx):
            pass

    assert SinNombre.name == "SinNombre"