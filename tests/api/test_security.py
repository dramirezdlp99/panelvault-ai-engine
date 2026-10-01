"""Pruebas de la firma HMAC entre backend y motor."""

import pytest

from panelvault_ai.api import SIGNATURE_HEADER, TIMESTAMP_HEADER, SignatureError, sign, verify
from panelvault_ai.api.security import signed_headers

SECRETO = "s" * 32
CUERPO = b"bytes de una imagen"
AHORA = 1_800_000_000


def _verificar(timestamp, firma, metodo="POST", ruta="/v1/analyze", cuerpo=CUERPO, ahora=AHORA):
    verify(SECRETO, timestamp, firma, metodo, ruta, cuerpo, ttl_seconds=300, now=ahora)


def test_una_firma_correcta_se_acepta():
    _verificar(str(AHORA), sign(SECRETO, str(AHORA), "POST", "/v1/analyze", CUERPO))


def test_signed_headers_produce_cabeceras_validas():
    cabeceras = signed_headers(SECRETO, "POST", "/v1/analyze", CUERPO, now=AHORA)
    _verificar(cabeceras[TIMESTAMP_HEADER], cabeceras[SIGNATURE_HEADER])


def test_la_firma_es_determinista_y_hexadecimal():
    a = sign(SECRETO, "1", "POST", "/x", b"a")
    assert a == sign(SECRETO, "1", "POST", "/x", b"a")
    assert len(a) == 64
    int(a, 16)  # no lanza: es hexadecimal


@pytest.mark.parametrize(
    "cambio",
    [
        {"cuerpo": CUERPO + b"!"},
        {"ruta": "/v1/otra"},
        {"metodo": "PUT"},
    ],
    ids=["cuerpo_alterado", "ruta_alterada", "metodo_alterado"],
)
def test_cualquier_alteracion_invalida_la_firma(cambio):
    firma = sign(SECRETO, str(AHORA), "POST", "/v1/analyze", CUERPO)
    with pytest.raises(SignatureError, match="inválida"):
        _verificar(str(AHORA), firma, **cambio)


def test_otro_secreto_no_sirve():
    firma = sign("x" * 32, str(AHORA), "POST", "/v1/analyze", CUERPO)
    with pytest.raises(SignatureError):
        _verificar(str(AHORA), firma)


def test_una_peticion_vieja_se_rechaza_aunque_la_firma_sea_correcta():
    viejo = str(AHORA - 301)
    firma = sign(SECRETO, viejo, "POST", "/v1/analyze", CUERPO)
    with pytest.raises(SignatureError, match="expiró"):
        _verificar(viejo, firma)


def test_faltan_cabeceras():
    with pytest.raises(SignatureError, match="Faltan"):
        _verificar(None, None)


def test_timestamp_no_numerico():
    with pytest.raises(SignatureError, match="Timestamp"):
        _verificar("ayer", "abc")