"""Firma HMAC de peticiones entre el backend y el motor.

El motor no se expone al navegador: solo el backend puede llamarlo. Para probarlo,
cada petición lleva dos cabeceras:

- ``X-PanelVault-Timestamp``: segundos Unix del momento en que se firmó.
- ``X-PanelVault-Signature``: HMAC-SHA256, en hexadecimal, del texto canónico
  ``"{timestamp}\\n{MÉTODO}\\n{ruta}\\n{sha256 del cuerpo}"`` con el secreto compartido.

Así se garantiza:

- **Autenticidad**: sin el secreto no se puede producir una firma válida.
- **Integridad**: cambiar un solo byte del cuerpo o de la ruta invalida la firma.
- **Protección contra repetición**: una petición capturada deja de servir pasados
  ``ttl`` segundos, porque el timestamp forma parte de lo firmado.
"""

from __future__ import annotations

import hashlib
import hmac
import time

TIMESTAMP_HEADER = "X-PanelVault-Timestamp"
SIGNATURE_HEADER = "X-PanelVault-Signature"


class SignatureError(Exception):
    """La petición no trae una firma válida."""


def canonical_message(timestamp: str, method: str, path: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return f"{timestamp}\n{method.upper()}\n{path}\n{body_hash}".encode()


def sign(secret: str, timestamp: str, method: str, path: str, body: bytes) -> str:
    """Firma que debe calcular el cliente (el backend) para una petición."""
    message = canonical_message(timestamp, method, path, body)
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def signed_headers(
    secret: str, method: str, path: str, body: bytes, now: float | None = None
) -> dict[str, str]:
    """Cabeceras listas para enviar. Útil en pruebas y como referencia para el backend."""
    timestamp = str(int(time.time() if now is None else now))
    return {
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: sign(secret, timestamp, method, path, body),
    }


def verify(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    method: str,
    path: str,
    body: bytes,
    ttl_seconds: int,
    now: float | None = None,
) -> None:
    """Lanza ``SignatureError`` si la firma falta, expiró o no coincide."""
    if not timestamp or not signature:
        raise SignatureError("Faltan las cabeceras de firma")
    try:
        sent_at = int(timestamp)
    except ValueError:
        raise SignatureError("Timestamp inválido") from None
    current = time.time() if now is None else now
    if abs(current - sent_at) > ttl_seconds:
        raise SignatureError("La firma expiró o el reloj del cliente está desfasado")
    expected = sign(secret, timestamp, method, path, body)
    # compare_digest tarda lo mismo sin importar dónde difieran las cadenas: así no se
    # puede adivinar la firma carácter por carácter midiendo tiempos de respuesta.
    if not hmac.compare_digest(expected, signature.lower()):
        raise SignatureError("Firma inválida")