"""Independent audit and provenance declarations.

These flags describe evidence boundaries.  They never turn a kinematic proxy
or teacher trace into a model score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import assert_json_safe, identifier, unique_strings

AUDIT_SCHEMA = "aerowam-audit/1"
PROVENANCE_SCHEMA = "aerowam-provenance/1"


@dataclass(frozen=True)
class Provenance:
    source_id: str
    source_kind: str
    files_sha256: Mapping[str, str] = field(default_factory=dict)
    parent_episode_id: str = ""
    capture_tool: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        identifier(self.source_id, "source_id")
        identifier(self.source_kind, "source_kind")
        if self.parent_episode_id:
            identifier(self.parent_episode_id, "parent_episode_id")
        if self.capture_tool:
            identifier(self.capture_tool, "capture_tool")
        unique_strings(self.notes, "notes", min_items=0) if self.notes else None
        import re
        for path, digest in self.files_sha256.items():
            if not isinstance(path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("files_sha256 must map paths to SHA256 strings")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PROVENANCE_SCHEMA, "source_id": self.source_id, "source_kind": self.source_kind,
                "files_sha256": dict(self.files_sha256), "parent_episode_id": self.parent_episode_id,
                "capture_tool": self.capture_tool, "notes": list(self.notes)}


@dataclass(frozen=True)
class AuditSummary:
    passed: bool
    independent_audit: bool
    model_score_eligible: bool = False
    flight_admitted: bool = False
    privileged_teacher: bool = False
    kinematic_proxy: bool = False
    action_conditioned_world_model: bool = False
    checks: Mapping[str, bool] = field(default_factory=dict)
    limitations: tuple[str, ...] = field(default_factory=tuple)
    audit_id: str = ""

    def __post_init__(self) -> None:
        if self.audit_id:
            identifier(self.audit_id, "audit_id")
        if not self.passed and self.model_score_eligible:
            raise ValueError("failed audit cannot be score eligible")
        if (not self.independent_audit or self.privileged_teacher or self.kinematic_proxy) and self.model_score_eligible:
            raise ValueError("teacher/proxy or non-independent evidence cannot be model score eligible")
        if self.flight_admitted and (not self.independent_audit or self.privileged_teacher or self.kinematic_proxy):
            raise ValueError("flight admission requires independent non-proxy evidence")
        if any(type(value) is not bool for value in self.checks.values()):
            raise ValueError("audit checks must be booleans")
        assert_json_safe(self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": AUDIT_SCHEMA, "passed": self.passed, "independent_audit": self.independent_audit,
                "model_score_eligible": self.model_score_eligible, "flight_admitted": self.flight_admitted,
                "privileged_teacher": self.privileged_teacher, "kinematic_proxy": self.kinematic_proxy,
                "action_conditioned_world_model": self.action_conditioned_world_model,
                "checks": dict(self.checks), "limitations": list(self.limitations), "audit_id": self.audit_id}


def validate_audit(value: Mapping[str, Any]) -> AuditSummary:
    if not isinstance(value, Mapping) or value.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported audit schema")
    expected = {"schema", "passed", "independent_audit", "model_score_eligible", "flight_admitted", "privileged_teacher", "kinematic_proxy", "action_conditioned_world_model", "checks", "limitations", "audit_id"}
    if set(value) != expected:
        raise ValueError(f"audit fields must be exactly {sorted(expected)}")
    for key in ("passed", "independent_audit", "model_score_eligible", "flight_admitted", "privileged_teacher", "kinematic_proxy", "action_conditioned_world_model"):
        if type(value[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not isinstance(value["checks"], Mapping) or not isinstance(value["limitations"], list):
        raise ValueError("invalid audit checks or limitations")
    assert_json_safe(value)
    return AuditSummary(
        passed=value["passed"], independent_audit=value["independent_audit"],
        model_score_eligible=value["model_score_eligible"], flight_admitted=value["flight_admitted"],
        privileged_teacher=value["privileged_teacher"], kinematic_proxy=value["kinematic_proxy"],
        action_conditioned_world_model=value["action_conditioned_world_model"], checks=dict(value["checks"]),
        limitations=tuple(value["limitations"]), audit_id=value["audit_id"],
    )
