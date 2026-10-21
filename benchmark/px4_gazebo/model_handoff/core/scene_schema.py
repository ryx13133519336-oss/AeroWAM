"""Scene declarations.  Geometry and private truth stay in scene adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import assert_json_safe, identifier, unique_strings

SCENE_SCHEMA = "aerowam-scene/1"


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    scene_family_id: str
    scene_version: str = "1"
    adapter_id: str = ""
    public_channels: tuple[str, ...] = ("dual_rgb", "local_state", "messages")
    safety_contract: str = ""
    supports_tasks: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value, name in ((self.scene_id, "scene_id"), (self.scene_family_id, "scene_family_id"), (self.scene_version, "scene_version")):
            identifier(value, name)
        if self.adapter_id:
            identifier(self.adapter_id, "adapter_id")
        unique_strings(self.public_channels, "public_channels")
        if self.safety_contract:
            identifier(self.safety_contract, "safety_contract")
        for task in self.supports_tasks:
            identifier(task, "supports_tasks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA, "scene_id": self.scene_id,
            "scene_family_id": self.scene_family_id, "scene_version": self.scene_version,
            "adapter_id": self.adapter_id, "public_channels": list(self.public_channels),
            "safety_contract": self.safety_contract, "supports_tasks": list(self.supports_tasks),
        }


def validate_scene(value: Mapping[str, Any]) -> SceneSpec:
    if not isinstance(value, Mapping) or value.get("schema") != SCENE_SCHEMA:
        raise ValueError("unsupported scene schema")
    expected = {"schema", "scene_id", "scene_family_id", "scene_version", "adapter_id", "public_channels", "safety_contract", "supports_tasks"}
    if set(value) != expected:
        raise ValueError(f"scene fields must be exactly {sorted(expected)}")
    assert_json_safe(value)
    return SceneSpec(
        scene_id=value["scene_id"], scene_family_id=value["scene_family_id"],
        scene_version=value["scene_version"], adapter_id=value["adapter_id"],
        public_channels=tuple(value["public_channels"]), safety_contract=value["safety_contract"],
        supports_tasks=tuple(value["supports_tasks"]),
    )


SUBT_URBAN_STAIR = SceneSpec(
    scene_id="subt_urban_stair_v1", scene_family_id="subt_urban_2_story",
    adapter_id="subt_handoff_scene_adapter_v1", safety_contract="subt_no_flight_proxy_v1",
    supports_tasks=("handoff",),
)
