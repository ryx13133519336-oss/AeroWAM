"""Shared multi-task, multi-scene episode and model contracts.

The core package is deliberately independent from Gazebo, PX4 and any one
task.  Scene adapters produce public observations plus separate supervision
and audit branches; model adapters consume only :mod:`model_io` inputs.
"""

from .audit_schema import AuditSummary, Provenance, validate_audit
from .episode_schema import EpisodeRecord, FutureObservationTarget, MessageRecord, PublicObservation, TaskSupervision
from .model_io import ModelInput, ModelOutput, WorldModelInput, WorldModelOutput
from .scene_schema import SceneSpec, validate_scene
from .task_schema import TaskSpec, validate_task

__all__ = [
    "AuditSummary", "EpisodeRecord", "FutureObservationTarget", "MessageRecord",
    "ModelInput", "ModelOutput", "Provenance", "PublicObservation",
    "TaskSpec", "TaskSupervision", "SceneSpec", "WorldModelInput",
    "WorldModelOutput", "validate_audit", "validate_scene", "validate_task",
]
