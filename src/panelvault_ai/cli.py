"""Línea de comandos del motor: analizar páginas reales o una página de demostración.

Uso:
    panelvault-ai analyze RUTA [RUTA ...] [--preset manga] [--out CARPETA] [--debug]
    panelvault-ai demo [--out CARPETA]
    panelvault-ai evaluate [--seeds N] [--annotations CARPETA]
    panelvault-ai request RUTA [--url URL] [--preset manga]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from panelvault_ai import __version__
from panelvault_ai.analyzer import PRESETS, PanelAnalyzer
from panelvault_ai.api.security import signed_headers
from panelvault_ai.domain import PanelMap
from panelvault_ai.evaluation import (
    annotated_cases,
    format_report,
    run_benchmark,
    summarize,
    synthetic_cases,
)
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

    evaluate = sub.add_parser("evaluate", help="Mide precisión, recall y orden de lectura.")
    evaluate.add_argument(
        "--seeds", type=int, default=10, help="Variantes sintéticas por categoría."
    )
    evaluate.add_argument(
        "--annotations", type=Path, help="Carpeta con páginas reales anotadas (imagen + .json)."
    )

    request = sub.add_parser(
        "request", help="Envía una imagen firmada a la API del motor, como lo hará el backend."
    )
    request.add_argument("image", type=Path, help="Ruta de la imagen.")
    request.add_argument("--url", default="http://127.0.0.1:8001", help="URL base de la API.")
    request.add_argument("--preset", choices=sorted(PRESETS), default="western")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "evaluate":
        return _evaluate(args.seeds, args.annotations)
    if args.command == "request":
        return _request(args.image, args.url, args.preset)

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


def _evaluate(seeds: int, annotations: Path | None) -> int:
    cases = list(synthetic_cases(seeds))
    if annotations is not None:
        cases.extend(annotated_cases(annotations))
    print(f"Evaluando {len(cases)} páginas...\n")
    results = run_benchmark(cases)
    print(format_report(summarize(results)))
    failed = [r for r in results if not r.score.is_perfect]
    if failed:
        print(f"\nPáginas no perfectas: {len(failed)}")
        for r in failed:
            s = r.score
            print(f"  {r.case_name:<16} detectadas={s.predicted_count} reales={s.truth_count}")
    return 0


def _request(image: Path, base_url: str, preset: str) -> int:
    """Firma y envía una imagen a la API, exactamente como lo hará el backend."""
    secret = os.environ.get("PANELVAULT_ENGINE_SECRET", "")
    if not secret:
        print("ERROR: define PANELVAULT_ENGINE_SECRET con el mismo valor del servidor.")
        return 1
    body = image.read_bytes()
    path = "/v1/analyze"
    headers = signed_headers(secret, "POST", path, body)
    headers["Content-Type"] = "application/octet-stream"
    url = f"{base_url.rstrip('/')}{path}?preset={preset}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"ERROR {exc.code}: {exc.read().decode(errors='replace')}")
        return 1
    except urllib.error.URLError as exc:
        print(f"ERROR: no se pudo conectar con {base_url} ({exc.reason}). ¿Está corriendo?")
        return 1
    panels = data["panelMap"]["panels"]
    print(f"{image.name}: {len(panels)} viñetas en {data['elapsedMs']} ms")
    print(json.dumps(data["panelMap"], indent=2, ensure_ascii=False))
    return 0


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