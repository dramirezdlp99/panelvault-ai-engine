"""Pruebas de la API HTTP del motor con un cliente de pruebas (sin servidor real)."""

import cv2
import pytest
from fastapi.testclient import TestClient

from panelvault_ai.api import ANALYZE_PATH, EngineSettings, create_app, signed_headers
from panelvault_ai.synthetic import grid_layout, render_page

SECRETO = "secreto-de-pruebas-con-mas-de-32-caracteres"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    ajustes = EngineSettings(secret=SECRETO, max_upload_bytes=2_000_000, max_pixels=5_000_000)
    return TestClient(create_app(ajustes))


def _jpeg(cols: int = 2) -> bytes:
    imagen = render_page(800, 1200, grid_layout(800, 1200, 2, cols)).image
    ok, codificada = cv2.imencode(".jpg", imagen)
    assert ok
    return codificada.tobytes()


def _post(cliente, cuerpo: bytes, query: str = "", firmar: bool = True):
    cabeceras = signed_headers(SECRETO, "POST", ANALYZE_PATH, cuerpo) if firmar else {}
    cabeceras["Content-Type"] = "application/octet-stream"
    return cliente.post(ANALYZE_PATH + query, content=cuerpo, headers=cabeceras)


def test_health_no_requiere_firma(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"
    assert "manga" in respuesta.json()["presets"]


def test_analiza_una_pagina_firmada(cliente):
    respuesta = _post(cliente, _jpeg())
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert len(datos["panelMap"]["panels"]) == 4
    assert datos["panelMap"]["direction"] == "ltr"
    assert (datos["imageWidth"], datos["imageHeight"]) == (800, 1200)
    assert "xycut" in datos["stages"]


def test_preset_manga_por_query(cliente):
    respuesta = _post(cliente, _jpeg(cols=3), query="?preset=manga")
    assert respuesta.status_code == 200
    assert respuesta.json()["panelMap"]["direction"] == "rtl"


def test_sin_firma_responde_401(cliente):
    assert _post(cliente, _jpeg(), firmar=False).status_code == 401


def test_firma_de_otro_cuerpo_responde_401(cliente):
    cabeceras = signed_headers(SECRETO, "POST", ANALYZE_PATH, b"otro cuerpo")
    respuesta = cliente.post(ANALYZE_PATH, content=_jpeg(), headers=cabeceras)
    assert respuesta.status_code == 401


def test_preset_desconocido_responde_422(cliente):
    assert _post(cliente, _jpeg(), query="?preset=webtoon").status_code == 422


def test_cuerpo_que_no_es_imagen_responde_422(cliente):
    assert _post(cliente, b"esto no es una imagen").status_code == 422


def test_cuerpo_vacio_responde_422(cliente):
    assert _post(cliente, b"").status_code == 422


def test_imagen_demasiado_pesada_responde_413(cliente):
    assert _post(cliente, b"\0" * 2_000_001).status_code == 413


def test_imagen_con_demasiados_pixeles_responde_422():
    pequeno = EngineSettings(secret=SECRETO, max_pixels=100_000)
    respuesta = _post(TestClient(create_app(pequeno)), _jpeg())  # 800x1200 = 960 000 px
    assert respuesta.status_code == 422


def test_sin_documentacion_publica_por_defecto(cliente):
    for ruta in ("/docs", "/redoc", "/openapi.json"):
        assert cliente.get(ruta).status_code == 404


def test_la_documentacion_se_puede_encender_para_desarrollo():
    ajustes = EngineSettings(secret=SECRETO, docs_enabled=True)
    cliente_con_docs = TestClient(create_app(ajustes))
    assert cliente_con_docs.get("/docs").status_code == 200
    assert cliente_con_docs.get("/openapi.json").json()["info"]["title"] == "PanelVault AI Engine"
