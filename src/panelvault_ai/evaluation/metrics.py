"""Métricas de evaluación: qué tan bien coinciden las viñetas detectadas con las reales.

Dos preguntas distintas, dos familias de métricas:

- **Detección** (¿encontró las viñetas?): se emparejan detecciones con viñetas
  reales por IoU y se calculan precisión, exhaustividad (recall) y F1.
- **Orden de lectura** (¿las puso en el orden correcto?): sobre las parejas
  encontradas se calcula la correlación de Kendall τ entre el orden detectado
  y el real.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from panelvault_ai.domain import Rect

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Match:
    predicted: int  # índice en la lista detectada (su posición en el orden detectado)
    truth: int  # índice en la lista real (su posición en el orden real)
    iou: float


def match_panels(
    predicted: Sequence[Rect],
    truth: Sequence[Rect],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Match]:
    """Empareja detecciones con viñetas reales, cada una como máximo una vez.

    Algoritmo voraz por IoU descendente (el estándar en evaluación de detectores):
    se calculan todos los pares con IoU >= umbral, se ordenan de mayor a menor y se
    acepta cada par si ninguno de sus dos miembros fue emparejado antes.
    """
    pairs = [
        (p.iou(t), i, j)
        for i, p in enumerate(predicted)
        for j, t in enumerate(truth)
        if p.iou(t) >= iou_threshold
    ]
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for iou, i, j in pairs:
        if i not in used_pred and j not in used_truth:
            used_pred.add(i)
            used_truth.add(j)
            matches.append(Match(i, j, iou))
    return sorted(matches, key=lambda m: m.predicted)


def count_inversions(values: Sequence[int]) -> int:
    """Pares (i, j) con i < j y values[i] > values[j], en O(n log n) con merge sort.

    Al mezclar dos mitades ordenadas, cada vez que se toma un elemento de la mitad
    derecha, todos los que quedan en la izquierda son mayores que él: cada uno
    forma una inversión. Se suman sin compararlos uno por uno.
    """

    def sort_and_count(items: list[int]) -> tuple[list[int], int]:
        if len(items) <= 1:
            return items, 0
        middle = len(items) // 2
        left, inv_left = sort_and_count(items[:middle])
        right, inv_right = sort_and_count(items[middle:])
        merged: list[int] = []
        inversions = inv_left + inv_right
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions

    return sort_and_count(list(values))[1]


def kendall_tau(values: Sequence[int]) -> float:
    """Kendall τ entre el orden de ``values`` y su orden ascendente, de −1 a 1.

    1 = orden idéntico, −1 = completamente invertido. Con menos de dos elementos
    no hay pares que comparar y se devuelve 1.
    """
    n = len(values)
    if n < 2:
        return 1.0
    total_pairs = n * (n - 1) // 2
    return 1 - 2 * count_inversions(values) / total_pairs


@dataclass(frozen=True, slots=True)
class PageScore:
    """Resultado de comparar una página detectada con su respuesta correcta."""

    predicted_count: int
    truth_count: int
    matches: tuple[Match, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def precision(self) -> float:
        """De lo que detectó, ¿qué fracción eran viñetas reales?"""
        return self.true_positives / self.predicted_count if self.predicted_count else 1.0

    @property
    def recall(self) -> float:
        """De las viñetas reales, ¿qué fracción encontró?"""
        return self.true_positives / self.truth_count if self.truth_count else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def mean_iou(self) -> float:
        return sum(m.iou for m in self.matches) / len(self.matches) if self.matches else 0.0

    @property
    def order_tau(self) -> float | None:
        """Kendall τ del orden de lectura sobre las viñetas emparejadas.

        Es ``None`` si hay menos de dos parejas: sin pares no hay orden que medir, y
        reportar 1.0 haría pasar por perfecta una página donde casi nada se detectó.
        """
        if len(self.matches) < 2:
            return None
        return kendall_tau([m.truth for m in self.matches])

    @property
    def is_perfect(self) -> bool:
        complete = self.predicted_count == self.truth_count == self.true_positives
        return complete and self.order_tau in (None, 1.0)


def score_page(
    predicted: Sequence[Rect],
    truth: Sequence[Rect],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> PageScore:
    """Ambas listas deben venir en orden de lectura."""
    return PageScore(
        len(predicted), len(truth), tuple(match_panels(predicted, truth, iou_threshold))
    )