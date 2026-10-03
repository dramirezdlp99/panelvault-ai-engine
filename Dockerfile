# Imagen del motor de IA de PanelVault.
# Ligera (python slim + OpenCV headless), sin usuario root y con el puerto que
# indique la plataforma (Render inyecta PORT; en local se usa 8001).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Primero solo lo necesario para instalar: si no cambian, Docker reutiliza la capa.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin panelvault
USER panelvault

EXPOSE 8001

# El secreto PANELVAULT_ENGINE_SECRET llega como variable de entorno de la plataforma,
# nunca dentro de la imagen.
CMD ["sh", "-c", "uvicorn panelvault_ai.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8001}"]
