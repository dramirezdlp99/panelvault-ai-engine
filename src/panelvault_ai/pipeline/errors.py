"""Jerarquía de errores del motor.

Permite distinguir fallos de configuración de fallos de ejecución.
"""


class PanelVaultError(Exception):
    """Raíz de todos los errores propios del motor."""


class PipelineConfigurationError(PanelVaultError):
    """El pipeline está mal armado (se detecta antes de procesar cualquier imagen)."""


class UnknownComponentError(PipelineConfigurationError):
    """Se pidió al registro un componente que no existe."""


class StageExecutionError(PanelVaultError):
    """Una etapa falló mientras procesaba una página."""

    def __init__(self, stage_name: str, message: str) -> None:
        super().__init__(f"[{stage_name}] {message}")
        self.stage_name = stage_name