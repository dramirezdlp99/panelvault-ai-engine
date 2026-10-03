# PanelVault AI Engine

Motor de visión por computador de **PanelVault**: recibe una página de cómic y devuelve
el **mapa de viñetas** (rectángulos normalizados) en **orden de lectura**, tanto para
cómic occidental (izquierda → derecha) como para manga (derecha → izquierda).

Es 100 % gratuito y local: OpenCV + NumPy, sin APIs de pago ni modelos en la nube.

## Cómo encaja en PanelVault

```
Frontend (Next.js) ──► Backend (Spring Boot) ──HMAC──► AI Engine (este repo)
                         │  cola de trabajos
                         └─ PostgreSQL (caché por SHA-256 de la imagen)
```

- Solo el **backend** llama a este motor. Cada petición va firmada con HMAC-SHA256
  usando un secreto compartido (`PANELVAULT_ENGINE_SECRET`); sin firma válida → `401`.
- El backend guarda el resultado por hash de la imagen, así una misma página nunca se
  analiza dos veces.
- `/health` no requiere firma: el backend lo usa para "despertar" el servicio en planes
  gratuitos que se duermen por inactividad.

## Arquitectura interna

El análisis es un **pipeline de etapas** (patrón *pipeline* + registro de etapas):

| Etapa | Qué hace |
|-------|----------|
| `normalize` | Lleva cualquier página a un formato de trabajo uniforme |
| `gutter` | Estima el color del medianil (espacio entre viñetas) desde el margen |
| `binarize` | Separa el contenido de las viñetas del medianil |
| `xycut` (+ `cut_tree`) | XY-Cut recursivo: corta la página en un árbol de cortes |
| `refine` | Divide viñetas fusionadas por medianiles demasiado delgados |
| `ordering` | Orden de lectura a partir del árbol (preset `western` o `manga`) |
| `assemble` | Filtra candidatas y ensambla el `PanelMap` final con su confianza |

Carpetas de `src/panelvault_ai`:

- `domain/`: modelos inmutables (`Panel`, geometría).
- `pipeline/`: contexto, etapas, registro y modo depuración.
- `stages/`: cada etapa del análisis.
- `api/`: aplicación FastAPI, verificación HMAC y configuración.
- `synthetic/` y `evaluation/`: generador de páginas sintéticas y métricas
  (precisión, *recall*, orden de lectura) para medir el motor sin datos con derechos.

## Requisitos

- Python 3.12 o superior.

## Instalación local (Windows, CMD)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Pruebas y calidad

```bat
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

## Uso por línea de comandos

```bat
panelvault-ai demo
panelvault-ai analyze pagina.jpg --preset manga --debug
panelvault-ai evaluate --seeds 50
```

## Levantar la API en local

```bat
set PANELVAULT_ENGINE_SECRET=el-mismo-valor-que-usa-el-backend
set PANELVAULT_ENABLE_DOCS=true
uvicorn panelvault_ai.api.app:create_app --factory --port 8001
```

- Salud: <http://127.0.0.1:8001/health>
- Documentación (solo si `PANELVAULT_ENABLE_DOCS=true`): <http://127.0.0.1:8001/docs>
- Probar una petición firmada igual que el backend:

```bat
panelvault-ai request pagina.jpg --url http://127.0.0.1:8001 --preset western
```

## Variables de entorno

| Variable | Obligatoria | Por defecto | Descripción |
|----------|-------------|-------------|-------------|
| `PANELVAULT_ENGINE_SECRET` | Sí | — | Secreto HMAC compartido con el backend (mín. 32 caracteres). |
| `PANELVAULT_ENABLE_DOCS` | No | `false` | Activa `/docs`, `/redoc` y `/openapi.json`. |
| `PANELVAULT_MAX_UPLOAD_BYTES` | No | `15728640` | Tamaño máximo de imagen (15 MiB). |
| `PANELVAULT_MAX_PIXELS` | No | `40000000` | Máximo de píxeles por imagen (evita "bombas" de descompresión). |
| `PANELVAULT_SIGNATURE_TTL` | No | `300` | Segundos de validez de una firma (anti-repetición). |
| `PORT` | No | `8001` | Puerto en Docker (Render lo inyecta solo). |

Ver `.env.example`. Ningún secreto real se sube a Git.

## API

### `GET /health`

```json
{"status": "ok", "version": "0.1.0", "presets": ["manga", "western"]}
```

### `POST /v1/analyze?preset=western|manga`

- Cuerpo: bytes crudos de la imagen (JPEG, PNG o WebP), `Content-Type: application/octet-stream`.
- Cabeceras `X-PanelVault-Timestamp` y `X-PanelVault-Signature`: HMAC-SHA256 de
  `timestamp\nMETHOD\nruta\nsha256(cuerpo)` (la ruta sin la query).
- Respuestas: `200` mapa de viñetas, `401` firma inválida o caducada,
  `413` imagen demasiado grande, `422` imagen o preset no válidos.

## Seguridad

- Firma HMAC con comparación en tiempo constante y ventana de validez.
- Límite de tamaño revisado antes de procesar la imagen y límite de píxeles al decodificar.
- Documentación interactiva apagada por defecto.
- El servidor se niega a arrancar si falta el secreto o es demasiado corto.
- La imagen Docker corre con un usuario sin privilegios.

## Docker / despliegue

```bat
docker build -t panelvault-ai .
docker run --rm -p 8001:8001 -e PANELVAULT_ENGINE_SECRET=... panelvault-ai
```

En Render (plan gratuito): servicio web desde este repositorio con el `Dockerfile`,
definiendo `PANELVAULT_ENGINE_SECRET` en *Environment*. El backend apunta a la URL pública
del servicio con su variable `PANELVAULT_ENGINE_URL`.
