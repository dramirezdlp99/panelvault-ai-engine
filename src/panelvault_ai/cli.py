"""Línea de comandos del motor: analizar páginas reales o una página de demostración.

Uso:
    panelvault-ai analyze RUTA [RUTA ...] [--preset manga] [--out CARPETA] [--debug]
    panelvault-ai demo [--out CARPETA]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from panelvault_ai import __version__
from panelvault_ai.analyzer import PRESETS, PanelAnalyzer
from panelvault_ai.domain import PanelMap
from panelvault_ai.imageio import read_image, write_image
from panelvault_ai.pipeline import FileDebugSink, PanelVaultError
from panelvault_ai.stages import PANEL_MAP
from panelvault_ai.synthetic import PageStyle, render_page, row_layout
from panelvault_ai.visualize import draw_panel_map

DEFAULT_OUT = Path("debug_output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panelvault-ai", description="Detección de viñetas y orden de lectura en cómics."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analiza una o varias páginas.")
    analyze.add_argument("images", nargs="+", type=Path, help="Rutas de las imágenes.")
    analyze.add_argument("--preset", choices=sorted(PRESETS), default="western")
    analyze.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Carpeta de salida.")
    analyze.add_argument(
        "--debug", action="store_true", help="Guarda las imágenes intermedias de cada etapa."
    )

    demo = sub.add_parser("demo", help="Genera una página sintética y la analiza.")
    demo.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Carpeta de salida.")
    demo.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "demo":
        page = render_page(
            900, 1350, row_layout(900, 1350, [2, 3, 1, 2], seed=args.seed), PageStyle(), args.seed
        )
        source = args.out / "demo_pagina.png"
        write_image(source, page.image)
        print(f"Página de demostración creada en {source}")
        return _analyze_all([source], "western", args.out, debug=False)

    return _analyze_all(args.images, args.preset, args.out, args.debug)


def _analyze_all(paths: Sequence[Path], preset: str, out: Path, debug: bool) -> int:
    failures = 0
    for path in paths:
        try:
            _analyze_one(path, preset, out, debug)
        except (OSError, ValueError, PanelVaultError) as exc:
            failures += 1
            print(f"ERROR en {path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def _analyze_one(path: Path, preset: str, out: Path, debug: bool) -> None:
    image = read_image(path)
    sink = FileDebugSink(out / f"{path.stem}_etapas") if debug else None
    ctx = PanelAnalyzer(preset, debug=sink).analyze_with_context(image)
    panel_map = ctx.get(PANEL_MAP)

    overlay_path = out / f"{path.stem}_vinetas.png"
    json_path = out / f"{path.stem}_mapa.json"
    write_image(overlay_path, draw_panel_map(image, panel_map))
    json_path.write_text(
        json.dumps(panel_map.to_normalized_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(_summary(path, image, panel_map, [(t.stage_name, t.seconds) for t in ctx.timings]))
    print(f"  Imagen con viñetas: {overlay_path}")
    print(f"  Mapa JSON:          {json_path}")
    if sink is not None:
        print(f"  Etapas intermedias: {sink.directory}")


def _summary(
    path: Path, image: np.ndarray, panel_map: PanelMap, timings: list[tuple[str, float]]
) -> str:
    height, width = image.shape[:2]
    lines = [
        "",
        f"{path.name}  ({width}x{height} px)",
        f"  Viñetas: {len(panel_map)}   Tipo: {panel_map.page_type.value}   "
        f"Lectura: {panel_map.direction.value}   Confianza: {panel_map.confidence:.2f}",
    ]
    for panel in panel_map.panels:
        b = panel.bounds.normalized(panel_map.page_width, panel_map.page_height)
        lines.append(
            f"    #{panel.order + 1:<2} x={b.x:.3f} y={b.y:.3f} "
            f"w={b.width:.3f} h={b.height:.3f}  confianza={panel.confidence:.2f}"
        )
    total = sum(s for _, s in timings)
    detail = ", ".join(f"{name} {s * 1000:.0f}" for name, s in timings)
    lines.append(f"  Tiempo: {total * 1000:.0f} ms  ({detail})")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())