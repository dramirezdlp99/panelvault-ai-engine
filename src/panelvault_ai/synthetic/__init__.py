"""Páginas de cómic sintéticas con respuesta correcta conocida, para pruebas y evaluación."""

from panelvault_ai.synthetic.generator import (
    PageStyle,
    SyntheticPage,
    grid_layout,
    pinwheel_layout,
    render_page,
    row_layout,
)

__all__ = [
    "PageStyle",
    "SyntheticPage",
    "grid_layout",
    "pinwheel_layout",
    "render_page",
    "row_layout",
]