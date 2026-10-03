"""Configuración del motor, leída siempre de variables de entorno.

Nada sensible vive en el código: el secreto compartido con el backend llega por
``PANELVAULT_ENGINE_SECRET``. Si falta, el servidor se niega a arrancar en lugar
de quedar expuesto sin autenticación.

La documentación interactiva (``/docs``) está apagada por defecto: en producción no
hay motivo para publicar la forma de la API. En local se enciende con
``PANELVAULT_ENABLE_DOCS=true``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

MIN_SECRET_LENGTH = 32
_TRUE_VALUES = frozenset({"1", "true", "yes", "si", "sí", "on"})


class ConfigurationError(RuntimeError):
    """La configuración del entorno es inválida o incompleta."""


@dataclass(frozen=True, slots=True)
class EngineSettings:
    secret: str
    max_upload_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_pixels: int = 40_000_000  # protege contra imágenes gigantes ("bombas de descompresión")
    signature_ttl_seconds: int = 300  # antigüedad máxima de una petición firmada
    docs_enabled: bool = False  # /docs, /redoc y /openapi.json; solo para desarrollo

    def __post_init__(self) -> None:
        if len(self.secret) < MIN_SECRET_LENGTH:
            raise ConfigurationError(
                f"PANELVAULT_ENGINE_SECRET debe tener al menos {MIN_SECRET_LENGTH} caracteres"
            )
        if self.max_upload_bytes <= 0 or self.max_pixels <= 0 or self.signature_ttl_seconds <= 0:
            raise ConfigurationError("Los límites del motor deben ser positivos")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> EngineSettings:
        env = os.environ if env is None else env
        secret = env.get("PANELVAULT_ENGINE_SECRET", "")
        if not secret:
            raise ConfigurationError(
                "Falta la variable de entorno PANELVAULT_ENGINE_SECRET (secreto compartido "
                "con el backend). Revisa .env.example."
            )
        try:
            return cls(
                secret=secret,
                max_upload_bytes=int(env.get("PANELVAULT_MAX_UPLOAD_BYTES", 15 * 1024 * 1024)),
                max_pixels=int(env.get("PANELVAULT_MAX_PIXELS", 40_000_000)),
                signature_ttl_seconds=int(env.get("PANELVAULT_SIGNATURE_TTL", 300)),
                docs_enabled=_parse_bool(env.get("PANELVAULT_ENABLE_DOCS", "")),
            )
        except ValueError as exc:
            raise ConfigurationError(f"Valor numérico inválido en la configuración: {exc}") from exc


def _parse_bool(value: str) -> bool:
    """Interpreta valores típicos de variables de entorno ("true", "1", "sí"...)."""
    return value.strip().lower() in _TRUE_VALUES
