"""Shared pipeline stage names for configuration and CLI validation."""

from typing import Literal, get_args

PipelineStage = Literal["prepare", "encode", "generate", "evaluate", "dose-evaluate", "all"]

PIPELINE_STAGES: tuple[str, ...] = get_args(PipelineStage)
