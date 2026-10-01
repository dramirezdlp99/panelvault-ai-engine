"""API HTTP interna del motor (FastAPI).

Solo la consume el backend; nunca el navegador. Ejecutar en local con::

    uvicorn panelvault_ai.api.app:create_app --factory --port 8001

``--factory`` hace que uvicorn llame a ``create_app()`` al arrancar: la
configuración se lee entonces del entorno y no al importar el módulo.

Este módulo no usa ``from __future__ import annotations`` a propósito: FastAPI lee
las anotaciones de tipo en tiempo de ejecución para saber de dónde sale cada
parámetro, y con anotaciones diferidas no podría resolver la dependencia local
``signed_body``.
"""

import time
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from panelvault_ai import __version__
from panelvault_ai.analyzer import PRESETS, PanelAnalyzer
from panelvault_ai.api.security import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    SignatureError,
    verify,
)
from panelvault_ai.api.settings import EngineSettings
from panelvault_ai.stages import PANEL_MAP

ANALYZE_PATH = "/v1/analyze"


def create_app(settings: EngineSettings | None = None) -> FastAPI:
    settings = settings or EngineSettings.from_env()
    analyzers = {name: PanelAnalyzer(name) for name in PRESETS}

    app = FastAPI(
        title="PanelVault AI Engine",
        version=__version__,
        description="Segmentación de viñetas y orden de lectura. Uso interno del backend.",
    )

    async def signed_body(request: Request) -> bytes:
        """Dependencia: lee el cuerpo con límite de tamaño y verifica la firma HMAC."""
        declared = request.headers.get("content-length")
        if declared is not None and int(declared) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Imagen demasiado grande")
        body = await request.body()
        if len(body) > settings.max_upload_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Imagen demasiado grande")
        try:
            verify(
                settings.secret,
                request.headers.get(TIMESTAMP_HEADER),
                request.headers.get(SIGNATURE_HEADER),
                request.method,
                request.url.path,
                body,
                settings.signature_ttl_seconds,
            )
        except SignatureError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from None
        return body

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Sin autenticación a propósito: lo usan el keep-alive y el ping de despertar."""
        return {"status": "ok", "version": __version__, "presets": sorted(PRESETS)}

    @app.post(ANALYZE_PATH)
    def analyze(
        body: Annotated[bytes, Depends(signed_body)],
        preset: Annotated[str, Query()] = "western",
    ) -> dict[str, Any]:
        """Recibe los bytes de una imagen (JPEG, PNG o WebP) y devuelve su mapa de viñetas.

        Es una función normal (no ``async``): FastAPI la ejecuta en un hilo aparte,
        así el análisis, que usa CPU, no bloquea las demás peticiones del servidor.
        """
        if preset not in analyzers:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"Preset desconocido. Disponibles: {sorted(analyzers)}",
            )
        image = _decode(body, settings.max_pixels)
        started = time.perf_counter()
        ctx = analyzers[preset].analyze_with_context(image)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "engineVersion": __version__,
            "preset": preset,
            "imageWidth": int(image.shape[1]),
            "imageHeight": int(image.shape[0]),
            "elapsedMs": round(elapsed_ms, 1),
            "stages": {t.stage_name: round(t.seconds * 1000, 1) for t in ctx.timings},
            "panelMap": ctx.get(PANEL_MAP).to_normalized_dict(),
        }

    return app


def _decode(body: bytes, max_pixels: int) -> np.ndarray:
    if not body:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "El cuerpo está vacío")
    data = np.frombuffer(body, dtype=np.uint8)
    # Antes de decodificar a tamaño completo se lee una versión reducida 8 veces por lado
    # para conocer las dimensiones: una imagen de 30 000 x 30 000 píxeles cabe en pocos
    # KB comprimida pero ocuparía GB en memoria. En JPEG esta lectura reducida es barata;
    # en PNG OpenCV igual decodifica completo, así que el límite de bytes del cuerpo
    # sigue siendo la primera barrera.
    header = cv2.imdecode(data, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if header is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "El cuerpo no es una imagen válida"
        )
    if header.shape[0] * header.shape[1] * 64 > max_pixels:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Imagen con demasiados píxeles")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "El cuerpo no es una imagen válida"
        )
    return image