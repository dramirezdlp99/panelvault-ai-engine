"""Evaluación cuantitativa del motor: métricas de detección y de orden de lectura."""

from panelvault_ai.evaluation.benchmark import (
    BenchmarkCase,
    CaseResult,
    Summary,
    annotated_cases,
    format_report,
    run_benchmark,
    summarize,
    synthetic_cases,
)
from panelvault_ai.evaluation.metrics import (
    Match,
    PageScore,
    count_inversions,
    kendall_tau,
    match_panels,
    score_page,
)

__all__ = [
    "BenchmarkCase",
    "CaseResult",
    "Match",
    "PageScore",
    "Summary",
    "annotated_cases",
    "count_inversions",
    "format_report",
    "kendall_tau",
    "match_panels",
    "run_benchmark",
    "score_page",
    "summarize",
    "synthetic_cases",
]