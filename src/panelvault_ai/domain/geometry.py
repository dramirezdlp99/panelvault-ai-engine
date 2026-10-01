"""Primitivas geométricas del dominio.

Convención de coordenadas (la misma de las imágenes):
- El origen (0, 0) está en la esquina superior izquierda.
- ``x`` crece hacia la derecha y ``y`` crece hacia abajo.
- Un rectángulo cubre el intervalo semiabierto [x, x + width) × [y, y + height).

Este módulo no depende de OpenCV ni de NumPy: describe *qué* es una región,
no *cómo* se detecta.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    """Rectángulo alineado a los ejes. Inmutable (Value Object)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"Un Rect no puede tener dimensiones negativas: "
                f"width={self.width}, height={self.height}"
            )

    # ------------------------------------------------------------------
    # Construcción alternativa
    # ------------------------------------------------------------------
    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> Rect:
        """Crea un Rect a partir de dos esquinas opuestas, en cualquier orden."""
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return cls(left, top, right - left, bottom - top)

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------
    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def is_empty(self) -> bool:
        return self.area == 0

    # ------------------------------------------------------------------
    # Relaciones entre rectángulos
    # ------------------------------------------------------------------
    def intersection(self, other: Rect) -> Rect | None:
        """Región común a ambos rectángulos, o ``None`` si no se solapan.

        Dos rectángulos que solo se tocan en un borde no comparten área,
        por lo que tampoco se consideran solapados.
        """
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def union_bounds(self, other: Rect) -> Rect:
        """Menor rectángulo que contiene a ambos."""
        return Rect.from_corners(
            min(self.x, other.x),
            min(self.y, other.y),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def iou(self, other: Rect) -> float:
        """Intersection over Union: cuánto se solapan dos regiones, de 0 a 1.

        IoU = área(A ∩ B) / área(A ∪ B), donde área(A ∪ B) = área(A) + área(B) − área(A ∩ B).
        Vale 1 si son idénticos y 0 si no comparten área.
        """
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union_area = self.area + other.area - inter.area
        return inter.area / union_area

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def contains(self, other: Rect) -> bool:
        """True si ``other`` queda completamente dentro de este rectángulo."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    # ------------------------------------------------------------------
    # Transformaciones (devuelven un Rect nuevo)
    # ------------------------------------------------------------------
    def scaled(self, sx: float, sy: float | None = None) -> Rect:
        """Escala posición y tamaño. Útil para volver de la imagen reducida a la original."""
        if sy is None:
            sy = sx
        if sx <= 0 or sy <= 0:
            raise ValueError("Los factores de escala deben ser positivos")
        return Rect(self.x * sx, self.y * sy, self.width * sx, self.height * sy)

    def normalized(self, page_width: float, page_height: float) -> Rect:
        """Expresa el rectángulo como fracción (0 a 1) del tamaño de la página.

        Es el formato que consume el visor: sirve para cualquier resolución de la imagen.
        """
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Las dimensiones de la página deben ser positivas")
        return self.scaled(1 / page_width, 1 / page_height)


Point = tuple[float, float]


def _signed_area(points: tuple[Point, ...]) -> float:
    """Área con signo por la fórmula del cordón de zapato (shoelace).

    Suma de los productos cruzados de cada vértice con el siguiente, dividida entre 2.
    El signo indica la orientación del recorrido de los vértices.
    """
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2


def _cross(o: Point, a: Point, b: Point) -> float:
    """Producto cruzado (a − o) × (b − o): indica de qué lado de o→a queda b."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _line_intersection(p: Point, q: Point, a: Point, b: Point) -> Point:
    """Punto donde el segmento p→q cruza la recta infinita que pasa por a y b."""
    dx_pq, dy_pq = q[0] - p[0], q[1] - p[1]
    dx_ab, dy_ab = b[0] - a[0], b[1] - a[1]
    denom = dx_pq * dy_ab - dy_pq * dx_ab
    t = ((a[0] - p[0]) * dy_ab - (a[1] - p[1]) * dx_ab) / denom
    return (p[0] + t * dx_pq, p[1] + t * dy_pq)


@dataclass(frozen=True, slots=True)
class Polygon:
    """Polígono simple (sus lados no se cruzan). Inmutable (Value Object).

    Representa viñetas inclinadas o irregulares que un Rect no describe bien.
    """

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        pts = tuple((float(x), float(y)) for x, y in self.points)
        if len(pts) < 3:
            raise ValueError("Un Polygon necesita al menos 3 vértices")
        object.__setattr__(self, "points", pts)

    @classmethod
    def from_rect(cls, rect: Rect) -> Polygon:
        return cls(
            (
                (rect.x, rect.y),
                (rect.right, rect.y),
                (rect.right, rect.bottom),
                (rect.x, rect.bottom),
            )
        )

    @property
    def area(self) -> float:
        return abs(_signed_area(self.points))

    @property
    def centroid(self) -> Point:
        """Centro de masa del polígono (no el promedio de sus vértices)."""
        a = _signed_area(self.points)
        if a == 0:  # polígono degenerado: se usa el promedio como respaldo
            n = len(self.points)
            return (sum(p[0] for p in self.points) / n, sum(p[1] for p in self.points) / n)
        cx = cy = 0.0
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            f = x1 * y2 - x2 * y1
            cx += (x1 + x2) * f
            cy += (y1 + y2) * f
        return (cx / (6 * a), cy / (6 * a))

    @property
    def bounding_rect(self) -> Rect:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Rect.from_corners(min(xs), min(ys), max(xs), max(ys))

    @property
    def is_convex(self) -> bool:
        """True si todos los giros entre lados consecutivos tienen el mismo sentido."""
        n = len(self.points)
        sign = 0
        for i in range(n):
            c = _cross(self.points[i], self.points[(i + 1) % n], self.points[(i + 2) % n])
            if c != 0:
                s = 1 if c > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
        return True

    def clip(self, clipper: Polygon) -> Polygon | None:
        """Recorte de Sutherland–Hodgman: la parte de este polígono dentro de ``clipper``.

        ``clipper`` debe ser convexo. Se recorre cada lado del recortador y se
        descartan los vértices que quedan del lado de afuera, agregando los puntos
        donde los lados del polígono cruzan ese borde.
        """
        if not clipper.is_convex:
            raise ValueError("El polígono recortador debe ser convexo")
        orientation = 1 if _signed_area(clipper.points) > 0 else -1
        output = list(self.points)
        m = len(clipper.points)
        for i in range(m):
            a, b = clipper.points[i], clipper.points[(i + 1) % m]
            candidates, output = output, []
            if not candidates:
                break
            for j, current in enumerate(candidates):
                previous = candidates[j - 1]
                current_in = _cross(a, b, current) * orientation >= 0
                previous_in = _cross(a, b, previous) * orientation >= 0
                if current_in:
                    if not previous_in:
                        output.append(_line_intersection(previous, current, a, b))
                    output.append(current)
                elif previous_in:
                    output.append(_line_intersection(previous, current, a, b))
        if len(output) < 3 or _signed_area(tuple(output)) == 0:
            return None
        return Polygon(tuple(output))

    def iou(self, other: Polygon) -> float:
        """IoU entre polígonos. ``other`` debe ser convexo (se usa como recortador)."""
        inter = self.clip(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        return inter_area / (self.area + other.area - inter_area)

    def scaled(self, sx: float, sy: float | None = None) -> Polygon:
        if sy is None:
            sy = sx
        if sx <= 0 or sy <= 0:
            raise ValueError("Los factores de escala deben ser positivos")
        return Polygon(tuple((x * sx, y * sy) for x, y in self.points))

    def normalized(self, page_width: float, page_height: float) -> Polygon:
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Las dimensiones de la página deben ser positivas")
        return self.scaled(1 / page_width, 1 / page_height)