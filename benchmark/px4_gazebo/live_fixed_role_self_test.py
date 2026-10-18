"""Non-flight tests for the live FixedRole batch and evaluator tools."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from .live_fixed_role_batch import (
    ensure_new_output,
    inspect_seeds,
    next_pending_seed,
    runner_command,
    seed_output_path,
    unresolved_statuses,
    validate_episode_result,
)
from .live_fixed_role_evaluate import load_live_records, summarize_records
from .live_sim_manager import (
    ProcessInfo,
    Px4GazeboConfig,
    Px4GazeboSession,
    has_dds_agent,
    has_image_bridge,
    gazebo_gui_window_ids,
    gazebo_gui_environment,
    px4_environment,
    rendered_gui_frame,
    stale_simulator_pids,
)


def _record(
    *, episode_id: str, seed: int, success: bool,
    termination_reason: str, preposition_error_m: float,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "task_id": "live_dual_control_observation_diagnostic",
        "seed": seed,
        "outcome": {
            "terminated": True,
            "termination_reason": termination_reason,
            "success": success,
            "metrics": {
                "control_success": 1.0 if success else 0.0,
                "pre_position_success": 1.0 if success else 0.0,
                "collision_rate": 0.0,
                "preposition_error_m": preposition_error_m,
                "minimum_separation_m": 3.0 if success else 2.5,
                "flight_time_s": 20.0 if success else 5.0,
                "camera_frames_uav_0": 400 if success else 50,
                "camera_frames_uav_1": 402 if success else 51,
                "communication_bytes": 128 if success else 64,
            },
        },
    }


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    base_env = {
        "KEEP": "yes",
        "PX4_SYS_AUTOSTART": "4001",
        "PX4_GZ_STANDALONE": "1",
        "PX4_GZ_WORLD": "old",
        "PX4_GZ_MODEL_POSE": "9,9",
    }
    env0 = px4_environment(
        index=0, world="lawn", spawn_offset_ned=(0.0, 3.0, 0.0), base=base_env,
    )
    assert env0["PX4_UXRCE_DDS_NS"] == "px4_0"
    assert env0["PX4_SIM_MODEL"] == "gz_x500_mono_cam"
    assert "PX4_SYS_AUTOSTART" not in env0
    assert env0["PX4_GZ_WORLD"] == "lawn"
    assert "PX4_GZ_MODEL_POSE" not in env0
    assert "PX4_GZ_STANDALONE" not in env0
    assert env0["KEEP"] == "yes"

    env1 = px4_environment(
        index=1, world="lawn", spawn_offset_ned=(0.0, 3.0, 0.0), base=base_env,
    )
    assert env1["PX4_UXRCE_DDS_NS"] == "px4_1"
    assert env1["PX4_SIM_MODEL"] == "gz_x500_mono_cam"
    assert "PX4_SYS_AUTOSTART" not in env1
    # NED (north=0, east=3) must become Gazebo ENU (x=3, y=0).
    assert env1["PX4_GZ_MODEL_POSE"] == "3,0"
    assert "PX4_GZ_WORLD" not in env1
    assert "PX4_GZ_STANDALONE" not in env1

    processes = [
        ProcessInfo(10, 1, 10, 10, ("./build/px4_sitl_default/bin/px4", "-i", "0")),
        ProcessInfo(11, 1, 11, 11, ("./build/px4_sitl_default/bin/px4", "-i", "1")),
        ProcessInfo(12, 10, 10, 10, (
            "gz", "sim", "-r", "/home/ryx/PX4-Autopilot/Tools/simulation/gz/worlds/lawn.sdf",
        )),
        ProcessInfo(13, 10, 10, 10, ("gz", "sim", "-g")),
        ProcessInfo(14, 1, 14, 14, ("MicroXRCEAgent", "udp4", "-p", "8888")),
        ProcessInfo(15, 1, 15, 15, (
            "ros2", "run", "ros_gz_image", "image_bridge",
            "/world/lawn/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
            "/world/lawn/model/x500_mono_cam_1/link/camera_link/sensor/camera/image",
        )),
        ProcessInfo(16, 1, 16, 16, ("QGroundControl.exe",)),
        ProcessInfo(17, 1, 17, 17, ("python3", "other_job.py")),
    ]
    assert stale_simulator_pids(processes, world="lawn") == {10, 11, 12, 13}
    # Ruby rewrites the real server's argv into a single process title.
    ruby_processes = [
        ProcessInfo(20, 1, 20, 20, (
            "gz sim --verbose=1 -r -s /home/ryx/PX4-Autopilot/Tools/simulation/gz/worlds/lawn.sdf   ",
        )),
        ProcessInfo(21, 1, 20, 20, ("gz sim -g",)),
        ProcessInfo(22, 1, 22, 22, (
            "gz sim -r -s /home/ryx/PX4-Autopilot/Tools/simulation/gz/worlds/other.sdf",
        )),
        ProcessInfo(23, 1, 23, 23, ("python3", "-c", "gz sim -r lawn.sdf")),
        ProcessInfo(24, 1, 24, 24, ("gz sim 'invalid",)),
    ]
    assert stale_simulator_pids(ruby_processes, world="lawn") == {20, 21}
    with tempfile.TemporaryDirectory() as directory:
        session = Px4GazeboSession(Px4GazeboConfig())
        session._log_dir = Path(directory)
        session._processes = [("px4_0", SimpleNamespace(poll=lambda: None))]
        log_path = Path(directory) / "px4_0.log"
        log_path.write_text("INFO [px4] startup script started\n", encoding="utf-8")
        session._check_processes()
        log_path.write_text(
            "ERROR [px4] Startup script returned with return value: 2\n", encoding="utf-8",
        )
        try:
            session._check_processes()
        except RuntimeError as exc:
            assert "startup script failed with code 2" in str(exc)
            assert str(log_path) in str(exc)
        else:
            raise AssertionError("PX4 startup failure must be detected while its shell is alive")
    assert has_dds_agent(processes)
    assert has_image_bridge(processes, world="lawn")
    assert gazebo_gui_window_ids(
        '  0x600012 "Gazebo Sim": ("gz-sim-gui" "Gazebo GUI")  1200x1000+38+59\n'
    ) == ["0x600012"]

    graphics_base = {
        "WSL_DISTRO_NAME": "Ubuntu-22.04", "KEEP": "yes",
        "QT_QUICK_BACKEND": "software", "GALLIUM_DRIVER": "d3d12",
    }
    graphics_snapshot = dict(graphics_base)
    default_env, renderer = gazebo_gui_environment(graphics_base)
    assert renderer == "hardware" and default_env["MESA_D3D12_ADAPTER_NAME"] == "NVIDIA"
    assert "GALLIUM_DRIVER" not in default_env
    assert default_env["QSG_RENDER_LOOP"] == "basic"
    gui_env, renderer = gazebo_gui_environment(graphics_base, "software")
    assert renderer == "software" and gui_env["GALLIUM_DRIVER"] == "llvmpipe"
    assert gui_env["QT_QUICK_BACKEND"] == "opengl"
    assert gui_env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert graphics_base == graphics_snapshot  # Sensor/model process env is untouched.
    hardware_env, renderer = gazebo_gui_environment(gui_env, "hardware")
    assert renderer == "hardware" and "GALLIUM_DRIVER" not in hardware_env
    assert "LIBGL_ALWAYS_SOFTWARE" not in hardware_env
    assert gazebo_gui_environment({"WSL_DISTRO_NAME": "Ubuntu", "MESA_D3D12_ADAPTER_NAME": "AMD"})[0]["MESA_D3D12_ADAPTER_NAME"] == "AMD"
    assert gazebo_gui_environment({}, "auto")[1] == "hardware"

    from PIL import Image
    with tempfile.TemporaryDirectory() as directory:
        frames = Path(directory)
        (frames / "partial.png").write_bytes(b"\\x89PNG")
        assert rendered_gui_frame(frames) is None
        with Image.new("RGB", (128, 128), "black") as frame:
            frame.save(frames / "blank.png")
            assert rendered_gui_frame(frames) is None
            frame.putpixel((64, 64), (255, 255, 255))
            frame.save(frames / "scene.png")
        assert rendered_gui_frame(frames) == frames / "scene.png"

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert seed_output_path(root, 2).name == "seed_002.jsonl"

        success = _record(
            episode_id="fixed-role-seed-000", seed=0, success=True,
            termination_reason="success", preposition_error_m=0.04,
        )
        failed = _record(
            episode_id="fixed-role-seed-001", seed=1, success=False,
            termination_reason="startup stream timeout", preposition_error_m=99.0,
        )
        _write(seed_output_path(root, 0), [success])
        _write(seed_output_path(root, 1), [failed])

        statuses = inspect_seeds(root, (0, 1, 2))
        assert [status.state for status in statuses] == ["success", "failed", "pending"]
        assert next_pending_seed(statuses) == 2
        assert [status.seed for status in unresolved_statuses(statuses)] == [1]

        try:
            validate_episode_result(root, 2, 1)
        except RuntimeError as exc:
            assert "exited with code 1" in str(exc)
        else:
            raise AssertionError("nonzero runner exit must stop automatic mode")

        try:
            validate_episode_result(root, 1, 0)
        except RuntimeError as exc:
            assert "output state is failed" in str(exc)
        else:
            raise AssertionError("failed formal output must stop automatic mode")

        validated = validate_episode_result(root, 0, 0)
        assert validated.state == "success"

        try:
            ensure_new_output(seed_output_path(root, 0))
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing seed output must never be overwritten")

        command = runner_command(
            seed=2,
            output=seed_output_path(root, 2),
            world="lawn",
            altitude=2.0,
            preposition_distance=1.5,
            duration=10.0,
            spawn_offset_ned=(0.0, 3.0, 0.0),
        )
        assert command[command.index("--seed") + 1] == "2"
        assert command[command.index("--world") + 1] == "lawn"

        combined = root / "combined.jsonl"
        _write(combined, [success, failed, success])
        records, metadata = load_live_records([combined])
        assert len(records) == 2
        assert metadata["duplicate_records"] == 1

        summary = summarize_records(records, metadata)
        groups = summary["groups"]
        assert groups["all"]["count"] == 2
        assert groups["all"]["success_rate"] == 0.5
        assert groups["successful"]["count"] == 1
        assert groups["failed"]["count"] == 1
        error_stats = groups["successful"]["metrics"]["preposition_error_m"]
        assert error_stats["count"] == 1
        assert error_stats["mean"] == 0.04
        assert groups["termination_reason_counts"] == {
            "startup stream timeout": 1,
            "success": 1,
        }

    print("live FixedRole batch/evaluator non-flight self-test: PASS")


if __name__ == "__main__":
    main()
