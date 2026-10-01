"""Pruebas de la línea de comandos."""

import json

from panelvault_ai.cli import main


def test_demo_genera_pagina_imagen_y_json(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    for nombre in ("demo_pagina.png", "demo_pagina_vinetas.png", "demo_pagina_mapa.json"):
        assert (tmp_path / nombre).is_file()
    mapa = json.loads((tmp_path / "demo_pagina_mapa.json").read_text(encoding="utf-8"))
    assert len(mapa["panels"]) == 8
    assert "Viñetas: 8" in capsys.readouterr().out


def test_analyze_con_debug_guarda_las_etapas(tmp_path):
    main(["demo", "--out", str(tmp_path)])
    pagina = tmp_path / "demo_pagina.png"
    assert (
        main(["analyze", str(pagina), "--preset", "manga", "--debug", "--out", str(tmp_path)]) == 0
    )
    etapas = sorted(p.name for p in (tmp_path / "demo_pagina_etapas").iterdir())
    assert etapas[0].startswith("01_normalize")
    mapa = json.loads((tmp_path / "demo_pagina_mapa.json").read_text(encoding="utf-8"))
    assert mapa["direction"] == "rtl"


def test_un_archivo_invalido_no_detiene_a_los_demas(tmp_path, capsys):
    main(["demo", "--out", str(tmp_path)])
    codigo = main(
        [
            "analyze",
            str(tmp_path / "no_existe.png"),
            str(tmp_path / "demo_pagina.png"),
            "--out",
            str(tmp_path),
        ]
    )
    salida = capsys.readouterr()
    assert codigo == 1
    assert "ERROR" in salida.err
    assert "Viñetas: 8" in salida.out


def test_evaluate_imprime_el_reporte(capsys):
    assert main(["evaluate", "--seeds", "1"]) == 0
    salida = capsys.readouterr().out
    assert "TOTAL" in salida
    assert "límite: molinete" in salida


def test_request_sin_secreto_avisa_y_falla(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PANELVAULT_ENGINE_SECRET", raising=False)
    main(["demo", "--out", str(tmp_path)])
    assert main(["request", str(tmp_path / "demo_pagina.png")]) == 1
    assert "PANELVAULT_ENGINE_SECRET" in capsys.readouterr().out


def test_request_sin_servidor_explica_el_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PANELVAULT_ENGINE_SECRET", "s" * 40)
    main(["demo", "--out", str(tmp_path)])
    codigo = main(["request", str(tmp_path / "demo_pagina.png"), "--url", "http://127.0.0.1:9"])
    assert codigo == 1
    assert "no se pudo conectar" in capsys.readouterr().out