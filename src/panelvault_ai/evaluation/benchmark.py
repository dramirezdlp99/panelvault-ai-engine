"""Banco de pruebas: genera páginas variadas, las analiza y resume las métricas.

Incluye dos fuentes de páginas con respuesta conocida:

- **Sintéticas**, generadas con semillas: muchas, variadas y reproducibles.
- **Reales anotadas**: imágenes con un archivo ``.json`` al lado que lista sus
  viñetas en orden de lectura, con coordenadas normalizadas (0 a 1).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from panelvault_ai.analyzer import PanelAnalyzer
from panelvault_ai.domain import Rect
from panelvault_ai.evaluation.metrics import PageScore, score_page
from panelvault_ai.imageio import read_image
from panelvault_ai.synthetic import (
    PageStyle,
    grid_layout,
    pinwheel_layout,
    render_page,
    row_layout,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Una página con su respuesta correcta, en coordenadas normalizadas (0 a 1)."""

    name: str
    category: str
    preset: str
    image: np.ndarray
    truth: tuple[Rect, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_name: str
    category: str
    score: PageScore


def _normalize_all(rects: Sequence[Rect], width: int, height: int) -> tuple[Rect, ...]:
    return tuple(r.normalized(width, height) for r in rects)


def _manga_order(rects: Sequence[Rect], rows: int, cols: int) -> list[Rect]:
    """Reordena una cuadrícula (dada en orden occidental) al orden de lectura manga."""
    return [rects[r * cols + c] for r in range(rows) for c in reversed(range(cols))]


def synthetic_cases(seeds: int = 10) -> Iterator[BenchmarkCase]:
    """Páginas sintéticas, ``seeds`` variantes por categoría.

    Incluye a propósito una categoría de **límites conocidos**: composiciones que el
    motor actual no resuelve (molinete, viñetas pegadas sin medianil). Un banco de
    pruebas que solo contiene casos fáciles no mide nada.
    """
    w, h = 800, 1200
    styles = {
        "blanco": PageStyle(),
        "negro": PageStyle(gutter_value=0, border_value=255),
        "ruido": PageStyle(noise_sigma=10),
    }
    for seed in range(seeds):
        for style_name, style in styles.items():
            layout = row_layout(w, h, [2, 3, 1, 2], seed=seed)
            page = render_page(w, h, layout, style, seed)
            yield BenchmarkCase(
                f"filas-{style_name}-{seed}",
                f"filas ({style_name})",
                "western",
                page.image,
                _normalize_all(page.panels, w, h),
            )

        pinwheel = pinwheel_layout(w, h)
        page = render_page(w, h, pinwheel, seed=seed)
        yield BenchmarkCase(
            f"molinete-{seed}",
            "límite: molinete",
            "western",
            page.image,
            _normalize_all(pinwheel, w, h),
        )

        touching = row_layout(w, h, [2, 2], seed=seed, gutter=0)
        page = render_page(w, h, touching, seed=seed)
        yield BenchmarkCase(
            f"pegadas-{seed}",
            "límite: pegadas",
            "western",
            page.image,
            _normalize_all(touching, w, h),
        )

        thin = row_layout(w, h, [3, 2, 3], seed=seed, gutter=6 + seed % 2)
        page = render_page(w, h, thin, PageStyle(noise_sigma=4), seed)
        yield BenchmarkCase(
            f"delgados-{seed}",
            "medianil delgado",
            "western",
            page.image,
            _normalize_all(thin, w, h),
        )

        rows, cols = 2 + seed % 2, 2 + (seed // 2) % 2
        grid = grid_layout(w, h, rows, cols)
        page = render_page(w, h, grid, seed=seed)
        yield BenchmarkCase(
            f"manga-{seed}",
            "manga",
            "manga",
            page.image,
            _normalize_all(_manga_order(grid, rows, cols), w, h),
        )


def annotated_cases(directory: str | Path) -> Iterator[BenchmarkCase]:
    """Páginas reales: cada imagen ``X.jpg``/``X.png`` con su anotación ``X.json``.

    Formato de la anotación::

        {"preset": "western", "panels": [[x, y, ancho, alto], ...]}

    con las viñetas en orden de lectura y coordenadas normalizadas.
    """
    for annotation in sorted(Path(directory).glob("*.json")):
        image_path = next(
            (p for p in annotation.parent.glob(annotation.stem + ".*") if p.suffix != ".json"),
            None,
        )
        if image_path is None:
            continue
        data = json.loads(annotation.read_text(encoding="utf-8"))
        yield BenchmarkCase(
            annotation.stem,
            "real",
            data.get("preset", "western"),
            read_image(image_path),
            tuple(Rect(*box) for box in data["panels"]),
        )


def run_benchmark(
    cases: Sequence[BenchmarkCase] | Iterator[BenchmarkCase],
) -> list[CaseResult]:
    analyzers: dict[str, PanelAnalyzer] = {}
    results = []
    for case in cases:
        analyzer = analyzers.setdefault(case.preset, PanelAnalyzer(case.preset))
        panel_map = analyzer.analyze(case.image)
        predicted = [
            p.bounds.normalized(panel_map.page_width, panel_map.page_height)
            for p in panel_map.panels
        ]
        results.append(CaseResult(case.name, case.category, score_page(predicted, case.truth)))
    return results


@dataclass(frozen=True, slots=True)
class Summary:
    category: str
    pages: int
    perfect: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    order_tau: float | None  # promedio sobre las páginas donde se pudo medir


def summarize(results: Sequence[CaseResult]) -> list[Summary]:
    """Resumen por categoría y total. Precisión y recall se calculan sobre el conjunto
    (micro-promedio): todas las viñetas cuentan igual, sin importar la página."""
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        groups.setdefault(result.category, []).append(result)
    summaries = [_summary(name, group) for name, group in groups.items()]
    if len(groups) > 1:
        summaries.append(_summary("TOTAL", list(results)))
    return summaries


def _summary(category: str, group: Sequence[CaseResult]) -> Summary:
    tp = sum(r.score.true_positives for r in group)
    predicted = sum(r.score.predicted_count for r in group)
    truth = sum(r.score.truth_count for r in group)
    precision = tp / predicted if predicted else 1.0
    recall = tp / truth if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = [m.iou for r in group for m in r.score.matches]
    taus = [r.score.order_tau for r in group if r.score.order_tau is not None]
    return Summary(
        category=category,
        pages=len(group),
        perfect=sum(r.score.is_perfect for r in group),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=sum(matched) / len(matched) if matched else 0.0,
        order_tau=sum(taus) / len(taus) if taus else None,
    )


def format_report(summaries: Sequence[Summary]) -> str:
    header = (
        f"{'Categoría':<20} {'Páginas':>7} {'Perfectas':>9} {'Precisión':>9} "
        f"{'Recall':>7} {'F1':>6} {'IoU':>6} {'τ orden':>8}"
    )
    lines = [header, "-" * len(header)]
    for s in summaries:
        tau = f"{s.order_tau:>8.3f}" if s.order_tau is not None else f"{'—':>8}"
        lines.append(
            f"{s.category:<20} {s.pages:>7} {s.perfect:>9} {s.precision:>9.3f} "
            f"{s.recall:>7.3f} {s.f1:>6.3f} {s.mean_iou:>6.3f} {tau}"
        )
    return "\n".join(lines)