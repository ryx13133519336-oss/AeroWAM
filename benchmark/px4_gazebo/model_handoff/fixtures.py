"""Deterministic appearance/occluder fixtures for the model benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from ..occlusion_geometry import point_aabb_distance


COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("sphere", "cube", "cylinder")
STRIPES = ("black", "white")
ALL_APPEARANCES = tuple((c, s, p) for c in COLORS for s in SHAPES for p in STRIPES)
VALIDATION_APPEARANCES = (
    ("red", "cylinder", "white"),
    ("green", "sphere", "black"),
    ("blue", "cube", "black"),
    ("yellow", "sphere", "white"),
)
TEST_APPEARANCES = (
    ("red", "cube", "white"),
    ("green", "cylinder", "black"),
    ("blue", "sphere", "white"),
    ("yellow", "cube", "black"),
)
TRAIN_APPEARANCES = tuple(
    appearance for appearance in ALL_APPEARANCES
    if appearance not in {*VALIDATION_APPEARANCES, *TEST_APPEARANCES}
)

WALL_BLOCKING_CENTER = (2.0, -0.2, 2.0)
WALL_BLOCKING_HALF_EXTENTS = (0.25, 0.9, 2.0)
WALL_BLOCKING_STAGING = (4.0, -3.1, 2.0)


@dataclass(frozen=True)
class Appearance:
    color: str
    shape: str
    stripe: str

    @classmethod
    def from_tuple(cls, value: tuple[str, str, str]) -> "Appearance":
        return cls(*value)

    def phrase(self) -> str:
        return f"{self.color} {self.shape} with one wide {self.stripe} stripe"


@dataclass(frozen=True)
class OccluderLayout:
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class HandoffFixture:
    fixture_id: str
    split: str
    seed: int
    target: Appearance
    distractors: tuple[Appearance, Appearance]
    instruction: str
    target_start: tuple[float, float, float]
    target_end: tuple[float, float, float]
    occluder: OccluderLayout
    perturbation: bool
    staging_position: tuple[float, float, float] | None = None
    scene_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["distractors"] = [asdict(item) for item in self.distractors]
        # Preserve the bytes/digest of the already frozen v3 fixtures.
        for key in ("staging_position", "scene_profile"):
            if result[key] is None:
                result.pop(key)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HandoffFixture":
        return cls(
            fixture_id=str(value["fixture_id"]), split=str(value["split"]),
            seed=int(value["seed"]), target=Appearance(**value["target"]),
            distractors=tuple(Appearance(**item) for item in value["distractors"]),  # type: ignore[arg-type]
            instruction=str(value["instruction"]),
            target_start=tuple(float(x) for x in value["target_start"]),  # type: ignore[arg-type]
            target_end=tuple(float(x) for x in value["target_end"]),  # type: ignore[arg-type]
            occluder=OccluderLayout(
                center=tuple(float(x) for x in value["occluder"]["center"]),  # type: ignore[arg-type]
                half_extents=tuple(float(x) for x in value["occluder"]["half_extents"]),  # type: ignore[arg-type]
            ),
            perturbation=bool(value.get("perturbation", False)),
            staging_position=(
                tuple(float(x) for x in value["staging_position"])
                if value.get("staging_position") is not None else None
            ),
            scene_profile=value.get("scene_profile"),
        )


def _different_shape(shape: str, index: int) -> str:
    candidates = [item for item in SHAPES if item != shape]
    return candidates[index % len(candidates)]


def _different_color(color: str, index: int) -> str:
    candidates = [item for item in COLORS if item != color]
    return candidates[index % len(candidates)]


def _fixture(split: str, index: int, appearance_value: tuple[str, str, str], *, ood_occluder: bool) -> HandoffFixture:
    rng = random.Random(f"aerowam:{split}:{index}")
    target = Appearance.from_tuple(appearance_value)
    same_color = Appearance(target.color, _different_shape(target.shape, index), target.stripe)
    same_shape = Appearance(_different_color(target.color, index), target.shape, target.stripe)
    if ood_occluder:
        center = (rng.choice((1.35, 3.05)), -0.72 + rng.uniform(-0.06, 0.06), 2.0)
        half_extents = (rng.choice((0.68, 0.82)), rng.uniform(0.39, 0.46), 2.0)
    else:
        center = (2.2 + rng.uniform(-0.25, 0.25), -0.72 + rng.uniform(-0.04, 0.04), 2.0)
        half_extents = (0.6 + rng.uniform(-0.08, 0.08), 0.45 + rng.uniform(-0.03, 0.03), 2.0)
    fixture_id = f"{split}_{index:03d}"
    return HandoffFixture(
        fixture_id=fixture_id, split=split, seed=index, target=target,
        distractors=(same_color, same_shape),
        instruction=(
            f"Track the {target.phrase()}. UAV0 starts tracking; UAV1 must pre-position, "
            "take over through the wall occlusion, and confirm the same target."
        ),
        target_start=(-2.0, -1.6, 2.0), target_end=(8.0, -1.6, 2.0),
        occluder=OccluderLayout(center=center, half_extents=half_extents),
        perturbation=(index % 20) < 3,
    )


def generate_fixtures(profile: str = "safe_occlusion_v3") -> list[HandoffFixture]:
    if profile not in {"safe_occlusion_v3", "wall_blocking_v1"}:
        raise ValueError(f"unknown scene profile: {profile}")
    result: list[HandoffFixture] = []
    for index in range(240):
        result.append(_fixture("train", index, TRAIN_APPEARANCES[index % 16], ood_occluder=False))
    for index in range(40):
        result.append(_fixture("validation", index, VALIDATION_APPEARANCES[index % 4], ood_occluder=False))
    for index in range(40):
        result.append(_fixture("test_appearance", index, TEST_APPEARANCES[index % 4], ood_occluder=False))
    for index in range(40):
        result.append(_fixture("test_occluder", index, TRAIN_APPEARANCES[index % 16], ood_occluder=True))
    for index in range(40):
        result.append(_fixture("test_combined", index, TEST_APPEARANCES[index % 4], ood_occluder=True))
    if profile == "wall_blocking_v1":
        nominal = OccluderLayout(WALL_BLOCKING_CENTER, WALL_BLOCKING_HALF_EXTENTS)
        # Keep training, validation and appearance-only testing exactly on
        # the user-verified task. OOD layouts are reserved for later oracle
        # validation; they keep the same transverse-wall task family.
        unseen = (
            OccluderLayout((1.90, -0.15, 2.0), (0.18, 0.80, 2.0)),
            OccluderLayout((2.10, -0.27, 2.0), (0.32, 0.95, 2.0)),
        )
        result = [replace(
            row,
            occluder=(unseen[row.seed % 2] if row.split in {"test_occluder", "test_combined"} else nominal),
            staging_position=WALL_BLOCKING_STAGING,
            scene_profile=profile,
        ) for row in result]
    validate_fixtures(result)
    return result


def validate_fixtures(fixtures: Iterable[HandoffFixture]) -> None:
    rows = list(fixtures)
    expected = {"train": 240, "validation": 40, "test_appearance": 40, "test_occluder": 40, "test_combined": 40}
    counts = {split: sum(row.split == split for row in rows) for split in expected}
    if counts != expected or len(rows) != 400:
        raise ValueError(f"fixture counts do not match the frozen design: {counts}")
    ids = [row.fixture_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("fixture ids are not unique")
    train = {(row.target.color, row.target.shape, row.target.stripe) for row in rows if row.split == "train"}
    validation = {(row.target.color, row.target.shape, row.target.stripe) for row in rows if row.split == "validation"}
    test = {(row.target.color, row.target.shape, row.target.stripe) for row in rows if row.split in {"test_appearance", "test_combined"}}
    if train & validation or train & test or validation & test:
        raise ValueError("appearance combinations leak across train/validation/test")
    for axis in (COLORS, SHAPES, STRIPES):
        axis_index = (COLORS, SHAPES, STRIPES).index(axis)
        if {value[axis_index] for value in train} != set(axis):
            raise ValueError("each appearance factor must occur in training")
    nominal_spawns = ((0.0, 0.0, 2.0), (3.0, 0.35, 2.0))
    for row in rows:
        # The widest target body is 0.32 m in radius.  The wall must block
        # sight lines without intersecting any moving target appearance.
        if row.occluder.center[1] - row.occluder.half_extents[1] - row.target_start[1] <= 0.34:
            raise ValueError(
                f"fixture {row.fixture_id} occluder intersects the target lane"
            )
        for spawn in nominal_spawns:
            clearance = point_aabb_distance(
                spawn, row.occluder.center, row.occluder.half_extents
            )
            if clearance < 0.35:
                raise ValueError(
                    f"fixture {row.fixture_id} occluder overlaps a nominal UAV spawn"
                )


def manifest_digest(fixtures: Iterable[HandoffFixture]) -> str:
    payload = "".join(
        json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in fixtures
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(path: Path, profile: str = "safe_occlusion_v3") -> str:
    fixtures = generate_fixtures(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for fixture in fixtures:
            handle.write(json.dumps(fixture.to_dict(), sort_keys=True) + "\n")
    digest = manifest_digest(fixtures)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def read_manifest(path: Path) -> list[HandoffFixture]:
    with path.open("r", encoding="utf-8") as handle:
        fixtures = [HandoffFixture.from_dict(json.loads(line)) for line in handle if line.strip()]
    validate_fixtures(fixtures)
    return fixtures


def load_fixture(path: Path, fixture_id: str) -> HandoffFixture:
    matches = [item for item in read_manifest(path) if item.fixture_id == fixture_id]
    if len(matches) != 1:
        raise ValueError(f"fixture {fixture_id!r} not found exactly once in {path}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen 240/40/120 Handoff manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("safe_occlusion_v3", "wall_blocking_v1"), default="safe_occlusion_v3")
    args = parser.parse_args()
    digest = write_manifest(args.output, args.profile)
    print(f"wrote 400 fixtures to {args.output}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
