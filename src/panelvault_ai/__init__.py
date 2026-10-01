"""PanelVault AI Engine: segmentación de viñetas y orden de lectura de cómics."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("panelvault-ai")
except PackageNotFoundError:  # el paquete no está instalado
    __version__ = "0.0.0"