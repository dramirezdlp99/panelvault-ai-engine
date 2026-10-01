"""Pruebas del banco de evaluación."""

from pathlib import Path

from panelvault_ai.evaluation import (
    annotated_cases,
    format_report,
    run_benchmark,
    summarize,
    synthetic_cases,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
CATEGORIAS_RESUELTAS = {
    "filas (blanco)",
    "filas (negro)",
    "filas (ruido)",
    "medianil delgado",
    "manga",
}


def test_genera_todas_las_categorias():
    categorias = {c.category for c in synthetic_cases(seeds=1)}
    assert CATEGORIAS_RESUELTAS < categorias
    assert {"límite: molinete", "límite: pegadas"} < categorias


def test_las_categorias_resueltas_son_perfectas():
    casos = [c for c in synthetic_cases(seeds=3) if c.category in CATEGORIAS_RESUELTAS]
    resultados = run_benchmark(casos)
    assert all(r.score.is_perfect for r in resultados), [
        r.case_name for r in resultados if not r.score.is_perfect
    ]


def test_carga_paginas_reales_anotadas():
    casos = list(annotated_cases(FIXTURES))
    assert [c.name for c in casos] == ["real1"]
    assert len(casos[0].truth) == 7
    assert run_benchmark(casos)[0].score.is_perfect


def test_resumen_incluye_total_y_reporte_legible():
    resultados = run_benchmark(list(synthetic_cases(seeds=1)))
    resumen = summarize(resultados)
    assert resumen[-1].category == "TOTAL"
    assert resumen[-1].pages == len(resultados)
    reporte = format_report(resumen)
    assert "Precisión" in reporte
    assert "TOTAL" in reporte