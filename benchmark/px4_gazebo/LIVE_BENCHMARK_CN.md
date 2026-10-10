# PX4-Gazebo 最小 Benchmark：双机实时诊断

当前实现是“实时双机控制/观测诊断”，不是完整的视觉 handoff benchmark。它先验证同一条安全闭环是否稳定：FixedRole policy → ROS 2/PX4 Offboard → 双机状态与双相机 → episode JSONL → schema 校验。暂时不会生成虚假的 handoff、verify 或目标识别指标。

## 1. 启动顺序

以下命令在 WSL Ubuntu 22.04 中执行。项目路径固定为 `/mnt/d/AeroWAM`，不要使用桌面上的旧 C 盘副本。

### 终端 A：Micro XRCE-DDS Agent

```bash
source /opt/ros/humble/setup.bash
MicroXRCEAgent udp4 -p 8888
```

保持这个终端运行。

### 终端 B/C：PX4 SITL 双机和 Gazebo

按照目前已经验证成功的 PX4-Gazebo 双机启动方式启动两个 `x500_mono_cam`。启动后应看到：

```text
/px4_0/fmu/out/vehicle_odometry
/px4_0/fmu/out/vehicle_status_v4
/px4_1/fmu/out/vehicle_odometry
/px4_1/fmu/out/vehicle_status_v4
```

相机话题应为：

```text
/world/lawn/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
/world/lawn/model/x500_mono_cam_1/link/camera_link/sensor/camera/image
```

### 终端 D：ROS 2 环境

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd /mnt/d/AeroWAM
```

## 2. 飞行前诊断

先确认两个状态和里程计都能收到：

```bash
ros2 topic list -t | grep -E '^/px4_[01]/fmu/out/(vehicle_status_v4|vehicle_odometry)$'

timeout 10 ros2 topic echo --once --qos-reliability best_effort \
  /px4_0/fmu/out/vehicle_status_v4 px4_msgs/msg/VehicleStatus

timeout 10 ros2 topic echo --once --qos-reliability best_effort \
  /px4_1/fmu/out/vehicle_status_v4 px4_msgs/msg/VehicleStatus
```

两个状态消息都应包含 `pre_flight_checks_pass: true`。相机频率检查：

```bash
timeout 10 ros2 topic hz \
  /world/lawn/model/x500_mono_cam_0/link/camera_link/sensor/camera/image
timeout 10 ros2 topic hz \
  /world/lawn/model/x500_mono_cam_1/link/camera_link/sensor/camera/image
```

两路均有持续频率（通常约 17–20 Hz）后再运行实时 runner。若没有频率，先修复 Gazebo/ROS bridge，不要解锁。

## 3. 无飞行自测

这一步不会连接 ROS，也不会启动飞机：

```bash
cd /mnt/d/AeroWAM
python3 -m compileall benchmark/px4_gazebo
python3 -m benchmark.px4_gazebo.live_benchmark_self_test
python3 -m benchmark.px4_gazebo.live_fixed_role_self_test
```

看到 `live benchmark non-flight self-test: PASS` 才继续。

## 4. 出生点偏移必须确认

`--uav1-spawn-offset-ned N E D` 是第二架飞机相对第一架飞机的 Gazebo 出生点偏移，单位为米，坐标是 NED（北、东、下）。两架 PX4 的 `vehicle_odometry` 都是各自本地原点，不能直接相减，所以这个参数必须与实际启动参数一致。

例如第二机确实在第一机东侧 3 m、同高度出生时，才使用：

```text
--uav1-spawn-offset-ned 0 3 0
```

如果启动脚本使用了其他位置，就替换成实际值。不要为了让程序启动而猜这个参数；错误偏移会使最小距离和预定位指标失真。

## 5. 首次实时诊断运行

QGroundControl 只用于监控地图、姿态、解锁状态和故障；运行期间不要用 QGC 手动切换模式或抢控制权。

确认所有仿真终端在线后，在终端 D 执行：

```bash
cd /mnt/d/AeroWAM
python3 -m benchmark.px4_gazebo.live_benchmark_runner \
  --world lawn \
  --altitude 2 \
  --preposition-distance 1.5 \
  --duration 10 \
  --uav1-spawn-offset-ned 0 3 0 \
  --output /mnt/d/AeroWAM-data/episodes/live_fixed_role_diagnostic.jsonl
```

其中 `0 3 0` 必须按上一节替换。程序会先等待遥测和双相机，预发布 20 Hz setpoint，再以 10 Hz 运行 FixedRole policy，最后自动降落并锁定。任何异常或 `Ctrl+C` 都会进入安全降落流程。

建议第一次只使用 `--duration 5` 或 `10` 秒。确认日志和轨迹正常后，再逐步增加时长；不要一开始就做长时间运行。

## 6. 查看和复验 episode

```bash
tail -n 1 /mnt/d/AeroWAM-data/episodes/live_fixed_role_diagnostic.jsonl | python3 -m json.tool
```

日志的 `outcome.metrics` 当前包括：控制成功、预定位成功、最小机间距离、预定位误差、通信字节数、飞行时间和两路相机帧数。当前 `handoff_success`、目标身份错误等字段不会被伪造填入。

如果需要单独复验 schema：

```bash
python3 - <<'PY'
import json
from pathlib import Path
import jsonschema

log = Path('/mnt/d/AeroWAM-data/episodes/live_fixed_role_diagnostic.jsonl')
schema = Path('/mnt/d/AeroWAM/benchmark/spec/episode_schema.json')
record = json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
jsonschema.validate(record, json.loads(schema.read_text(encoding='utf-8')))
print('episode schema: PASS')
PY
```

## 7. Live FixedRole 当前验收边界

完成上述步骤后，最小骨架就具备：双机独立 Offboard、双机起飞/悬停/降落、双相机观测、速度/加速度/地理围栏/最小间距检查、异常安全降落、单 episode JSONL 和离线 schema 复验。

这部分仍是 Live FixedRole 基线，不应与后续静态遮挡 handoff 结果混在同一个正式结果目录中。

## 8. 正式 FixedRole 五个 seed

正式结果使用独立目录：

```text
/mnt/d/AeroWAM-data/episodes/fixed_role/
```

先查看 0–4 五个 seed 的状态：

```bash
cd /mnt/d/AeroWAM
python3 -m benchmark.px4_gazebo.live_fixed_role_batch status
```

查看下一次应执行的单个飞行命令，但不启动飞行：

```bash
python3 -m benchmark.px4_gazebo.live_fixed_role_batch command
```

保留原来的手动模式。只有在 PX4 双机、Gazebo `lawn` world、QGroundControl 和两路相机都确认正常后，才手动执行一次：

```bash
python3 -m benchmark.px4_gazebo.live_fixed_role_batch run-next
```

`run-next` 每次只选择第一个 pending seed，也不会覆盖已有 seed 文件。手动模式每轮完成后必须：

1. 确认两架飞机已经落地并锁定。
2. 人工重启或重置 PX4 与 Gazebo。
3. 重新检查 ROS 2 状态和两路相机。
4. 再次运行 `status` 和 `command`，确认后才执行下一 seed。

### 自动复位模式

运行前只保留一个 DDS Agent：

```bash
MicroXRCEAgent udp4 -p 8888
```

首次只测试一个 pending seed：

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd /mnt/d/AeroWAM

python3 -m benchmark.px4_gazebo.live_fixed_role_batch run-auto-next
```

`run-auto-next` 会精确关闭旧 PX4 SITL/Gazebo，重新启动两架 `gz_x500_mono_cam`，等待两路状态、里程计和相机健康，运行一个 seed，然后完整关闭本轮 PX4/Gazebo。它不会关闭 QGroundControl，也不会关闭预先运行的 DDS Agent。每轮日志写到：

```text
/mnt/d/AeroWAM-data/px4_logs/fixed_role_auto/seed_XXX/
```

确认单轮自动复位稳定后，才运行全部剩余 pending seed：

```bash
python3 -m benchmark.px4_gazebo.live_fixed_role_batch run-auto
```

`run-auto` 在每个 seed 之间执行完整 teardown 和全新启动。任何启动失败、健康检查失败、飞行进程非零退出、`failed` 或 `invalid` 输出都会立即停止，不会继续下一 seed，也不会覆盖已有结果。自动模式会关闭当前 PX4/Gazebo，因此同时进行其他 Gazebo 工作时不要使用。

如果飞行前就因 PX4 健康检查、遥测或相机问题终止，调度器会把该 seed 标记为 `failed`，并阻止跳到后续 seed。先重启并修复 PX4/Gazebo；确认这是基础设施启动失败而不是有效实验后，将失败记录移出正式目录，例如：

```bash
mkdir -p /mnt/d/AeroWAM-data/episodes/aborted_startup
mv /mnt/d/AeroWAM-data/episodes/fixed_role/seed_000.jsonl \
  /mnt/d/AeroWAM-data/episodes/aborted_startup/seed_000_preflight.jsonl
```

再次运行 `status` 时该 seed 应恢复为 `pending`。不要删除失败记录，也不要将 `aborted_startup` 放在 `fixed_role` 目录内部，否则正式汇总会递归读入它。

五个 seed 完成后汇总：

```bash
python3 -m benchmark.px4_gazebo.live_fixed_role_evaluate \
  /mnt/d/AeroWAM-data/episodes/fixed_role \
  --json-output /mnt/d/AeroWAM-data/episodes/fixed_role_summary.json \
  --markdown-output /mnt/d/AeroWAM-data/episodes/fixed_role_summary.md
```

旧文件 `/mnt/d/AeroWAM-data/episodes/live_fixed_role_diagnostic.jsonl` 只用于调试验证，其中可能包含早期失败记录，不属于正式五个 seed 的结果目录。

## 9. Static Occlusion Handoff v1 非飞行验收

Static Occlusion Handoff v1 先在确定性运动学 fixture 中冻结几何、通信协议、事件和 evaluator 语义，暂不启动 PX4/Gazebo 飞行。记录使用 `schema_version=0.2`，由 `benchmark/spec/episode_schema_v0_2.json` 校验；原 Live FixedRole 日志继续使用 `episode_schema.json` 的 0.1 契约。

执行完整非飞行验收：

```bash
cd /mnt/d/AeroWAM
python3 -m compileall benchmark/px4_gazebo
python3 -m benchmark.px4_gazebo.handoff_v1_self_test
```

自测必须同时覆盖：FixedRole 超时、Oracle 成功、规则策略成功、同 seed 完全确定性、通信延迟、强制丢包、碰撞终止、超时终止、0.2 schema 校验，以及从 episode JSON 重放得到的指标与 `outcome.metrics` 一致。

也可以单独运行三种策略：

```bash
python3 -m benchmark.px4_gazebo.benchmark_runner --policy fixed  --seeds 0 1 2 --drop-probability 0
python3 -m benchmark.px4_gazebo.benchmark_runner --policy oracle --seeds 0 1 2 --drop-probability 0
python3 -m benchmark.px4_gazebo.benchmark_runner --policy rule   --seeds 0 1 2 --drop-probability 0
```

`fixed` 是无 handoff 的失败对照；`oracle` 使用精确遮挡几何作为上界；`rule` 使用任务时序、预定位状态和已送达消息。正式比较时三者必须使用相同 seeds、带宽、延迟和丢包率。评估器只读取 JSON 记录，重算双机路径、逐机/联合可见率、最长中断、恢复/交接延迟、消息尝试/送达/丢弃、通信字节数、最小间距和碰撞率。

只有此自测稳定通过后，才进入 PX4/Gazebo 静态遮挡场景适配；在此之前不要把 fixture 成绩作为真实飞行 benchmark 结果。
