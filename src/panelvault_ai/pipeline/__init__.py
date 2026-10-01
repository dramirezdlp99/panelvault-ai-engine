"""Núcleo del pipeline: etapas, contexto, validación, registro y depuración."""

from panelvault_ai.pipeline.context import (
    ArtifactKey,
    MissingArtifactError,
    PageContext,
    StageTiming,
)
from panelvault_ai.pipeline.debug import DebugSink, FileDebugSink, MemoryDebugSink, NullDebugSink
from panelvault_ai.pipeline.errors import (
    PanelVaultError,
    PipelineConfigurationError,
    StageExecutionError,
    UnknownComponentError,
)
from panelvault_ai.pipeline.pipeline import Pipeline
from panelvault_ai.pipeline.registry import STAGES, Registry, build_pipeline
from panelvault_ai.pipeline.stage import Stage

__all__ = [
    "STAGES",
    "ArtifactKey",
    "DebugSink",
    "FileDebugSink",
    "MemoryDebugSink",
    "MissingArtifactError",
    "NullDebugSink",
    "PageContext",
    "PanelVaultError",
    "Pipeline",
    "PipelineConfigurationError",
    "Registry",
    "Stage",
    "StageExecutionError",
    "StageTiming",
    "UnknownComponentError",
    "build_pipeline",
]