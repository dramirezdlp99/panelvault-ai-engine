"""Pruebas de la configuración del motor desde variables de entorno."""

import pytest

from panelvault_ai.api import ConfigurationError, EngineSettings


def test_lee_la_configuracion_del_entorno():
    ajustes = EngineSettings.from_env(
        {"PANELVAULT_ENGINE_SECRET": "s" * 40, "PANELVAULT_MAX_UPLOAD_BYTES": "1000"}
    )
    assert ajustes.secret == "s" * 40
    assert ajustes.max_upload_bytes == 1000
    assert ajustes.signature_ttl_seconds == 300


def test_sin_secreto_se_niega_a_arrancar():
    with pytest.raises(ConfigurationError, match="PANELVAULT_ENGINE_SECRET"):
        EngineSettings.from_env({})


def test_rechaza_secretos_cortos():
    with pytest.raises(ConfigurationError, match="32"):
        EngineSettings.from_env({"PANELVAULT_ENGINE_SECRET": "corto"})


def test_rechaza_numeros_invalidos():
    with pytest.raises(ConfigurationError, match="numérico"):
        EngineSettings.from_env(
            {"PANELVAULT_ENGINE_SECRET": "s" * 32, "PANELVAULT_MAX_PIXELS": "muchos"}
        )


def test_la_documentacion_esta_apagada_por_defecto():
    assert EngineSettings.from_env({"PANELVAULT_ENGINE_SECRET": "s" * 32}).docs_enabled is False


@pytest.mark.parametrize("valor", ["true", "TRUE", "1", "sí", " yes "])
def test_la_documentacion_se_enciende_con_valores_verdaderos(valor):
    ajustes = EngineSettings.from_env(
        {"PANELVAULT_ENGINE_SECRET": "s" * 32, "PANELVAULT_ENABLE_DOCS": valor}
    )
    assert ajustes.docs_enabled is True


@pytest.mark.parametrize("valor", ["false", "0", "no", ""])
def test_la_documentacion_sigue_apagada_con_otros_valores(valor):
    ajustes = EngineSettings.from_env(
        {"PANELVAULT_ENGINE_SECRET": "s" * 32, "PANELVAULT_ENABLE_DOCS": valor}
    )
    assert ajustes.docs_enabled is False
