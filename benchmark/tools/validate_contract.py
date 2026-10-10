"""Dependency-free smoke validation for the initial AeroWAM-Duo contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec"
FIXTURES = ROOT / "fixtures" / "mvp_tasks.json"


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    for name in ("task_schema.json", "episode_schema.json", "metrics_schema.json"):
        data = load(SPEC / name)
        require(data.get("$schema"), f"missing JSON schema declaration in {name}")

    fixture = load(FIXTURES)
    require(fixture["schema_version"] == "0.1", "fixture schema version mismatch")
    tasks = fixture["task_templates"]
    require(len(tasks) == 6, "MVP must contain exactly six task templates")
    require(fixture["seed_policy"] == [0, 1, 2], "MVP seed policy must be [0, 1, 2]")
    ids = set()
    families = set()
    for task in tasks:
        task_id = task["task_id"]
        require(task_id not in ids, f"duplicate task id: {task_id}")
        ids.add(task_id)
        families.add(task["family"])
        require(len(task["agents"]) == 2, f"{task_id} must have two agents")
        prohibited = set(task["observation_contract"]["prohibited_fields"])
        require("target_pose" in prohibited, f"{task_id} exposes target pose")
        require("teammate_pose" in prohibited, f"{task_id} exposes teammate pose")
        require(task["split"]["name"] in {"source_train", "target_zero_shot", "target_few_shot"}, f"bad split in {task_id}")
        require(task["split"]["allowed_for_training"] == (task["split"]["name"] == "source_train"), f"training flag mismatch in {task_id}")
        require(set(task["hidden_events"]) <= {"coverage", "pre_position", "handoff", "verify", "collision", "timeout"}, f"bad event in {task_id}")
    require(families == {"joint_inspection", "observation_handoff", "search_verify"}, "MVP must cover all three task families")
    print(f"validated {len(tasks)} task templates and {len(tasks) * len(fixture['seed_policy'])} seed fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
