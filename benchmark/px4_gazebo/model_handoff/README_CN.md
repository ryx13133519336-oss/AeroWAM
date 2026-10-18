# PX4/Gazebo 双机视觉 Handoff 模型实验

本目录实现统一的 `DualVLN-Hier`、`JointBC` 和 `DualWAM-MPC` 对比管线。学习策略只接收语言、两路 256x256 JPEG、两机自身状态、历史动作和协议消息；场景真值只写入独立的 `evaluator/` 目录。

## 当前采集基准：已验收的横向墙体

2026-10-09，`handoff_wall_blocking_retry1/seed_000.jsonl` 的规则飞行已由用户验收。
接下来的数据使用 `manifests/handoff_wall_blocking_v1.jsonl`，保留旧 v3 清单和结果。
训练、验证和外观测试固定使用同一目标路线、墙中心 `(2.0, -0.2, 2.0)`、
半尺寸 `(0.25, 0.9, 2.0)`、补位点 `(4.0, -3.1, 2.0)`。
清单显式记录补位点，避免正式采集重新由墙体厚度推导出过近的横穿位置。
清单 SHA-256 为 `667309f16248102da5010b6b3fc70e3982569a9d08e4cf7bb81fe9237690666f`。
两个遮挡 OOD 布局仅保留用于后续测试，尚未通过 Gazebo oracle 飞行验收，当前不采集它们。

先采集两个不加扰动的训练 episode。保持一个 `MicroXRCEAgent udp4 -p 8888`
运行，不需要预先打开 Gazebo。窗口可见，但正式采集不显示圆盘和路线提示：

```bash
cd /mnt/d/AeroWAM
source ~/venvs/aerowam-model/bin/activate
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash

python -m benchmark.px4_gazebo.model_handoff.collect \
  --mode collect \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_wall_blocking_v1.jsonl \
  --splits train --limit 2 --perturb-probability 0 \
  --show-gui --gui-renderer hardware \
  --session-startup-timeout 150 \
  --auto-log-dir /mnt/d/AeroWAM-data/px4_logs/oracle_wall_blocking_smoke_v1 \
  --output-dir /mnt/d/AeroWAM-data/oracle_results_wall_blocking_smoke_v1 \
  --dataset-dir /mnt/d/AeroWAM-data/model_handoff_dataset_wall_blocking_smoke_v1

python -m benchmark.px4_gazebo.model_handoff.dataset \
  /mnt/d/AeroWAM-data/model_handoff_dataset_wall_blocking_smoke_v1/*
```

验收应看到 `train_000/001: dataset validated (...)`、`oracle success` 和
`oracle collect complete: 2 fixtures`。采集在归档成功结果前验证 JPEG 解码、
256x256 RGB、图像校验和、100 ms 同步、动作标签、真值独立存储及样本对齐。
场景更新失败归为 `simulator_error`，自动复位重试；oracle 碰撞或任务失败则停止。
这些离线检查不替代真实飞行及相机画面验收。

两个 episode 通过后再按 240 train / 40 validation 采集，恢复约 15% 扰动，
然后缓存同一 OpenCLIP 编码器的特征并先训练 JointBC。后续所有采集、训练和
评测使用新清单和独立目录；下文旧 v3 示例路径仅供旧实验追溯。

## Handoff v3 核心任务

当前正式任务为 `handoff_safe_occlusion_v3`。墙体位于 UAV0 观察航线与目标直线路径之间，不与目标实体相交；目标经过墙后时，墙体遮挡 UAV0 的视线，UAV1 必须绕墙端补位并接管。它把角色交接拆成可观察、可验收的四个阶段：

1. UAV0 先到蓝色获取点，然后沿蓝色观察走廊跟随目标，在墙前建立稳定跟踪。
2. UAV1 先飞到橙色墙端安全点，并连续稳定 5 个策略周期；随后才允许横移到绿色 staging 点。
3. 目标进入墙体遮挡前，UAV0 发出 Handoff request；UAV1 只有在 staging 稳定且看到正确目标后才接管。
4. UAV1 沿黄色段持续跟踪并确认身份；UAV0 移到青色近侧支援点，不再遮挡主跟踪航线。

灰色带黄色警示边的实体是有真实碰撞几何的遮挡墙。彩色小柱和细线只在 `--visual-handoff` 人工演示中出现，表示飞行航点在地面的投影，不是降落坪；飞机应从其上方经过。演示还包含同色异形和同形异色两个干扰物。正式模型运行会隐藏所有彩色航点和路线，防止视觉提示泄漏。

控制器对墙体使用 `0.5 m` 膨胀安全区，并同时检查当前机位和相邻采样点之间的扫掠线段。UAV1 未在墙端安全点稳定前不能转向 staging。默认规则/oracle 的最大双机距离在全部 400 个冻结 fixture 的离线动态检查中限制为 `6.1 m`。

### v3 可视化飞行验收

先确保另一个终端中恰好运行一个 `MicroXRCEAgent udp4 -p 8888`。运行一次不飞行的 GUI 预检；自动复位单独启动 Gazebo GUI，将日志保存为 `gazebo_gui.log`，等待窗口映射、非空首帧保存为 `gui_first_frame.png`，并确认 GUI 持续存活后才允许起飞：

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
python3 -m benchmark.px4_gazebo.live_handoff_batch check-gui \
  --world lawn --session-startup-timeout 150 \
  --auto-log-dir /mnt/d/AeroWAM-data/px4_logs/handoff_v3_gui_check
```

窗口应显示约 30 秒，然后自动关闭；也可通过 `--gui-hold-seconds 60` 延长观察时间。默认 `--gui-renderer auto` 使用硬件 OpenGL；WSLg 下为 GUI 选择 NVIDIA D3D12 adapter（保留用户显式设置），使用 GLX 和 Qt `basic` 串行渲染循环。相机仿真和模型推理继续使用各自原有环境。`--gui-renderer software` 仅作为 CPU OpenGL 诊断选项，可能增加负载，不推荐用于飞行。Qt 的场景图仍使用 OpenGL，不能设成 `QT_QUICK_BACKEND=software`。这些启动兼容设置不替代 Windows 驱动更新，也不保证消除所有驱动故障。

预检通过后，在 WSL 运行以下命令，它会为 seed 000 自动启动并关闭 PX4/Gazebo，使用新目录保留已完成的 v3 结果。终端显示 `Gazebo GUI scene rendered and window ready` 后切换到 Gazebo 窗口；落地后自动复位会关闭窗口，这是正常行为。

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
python3 -m benchmark.px4_gazebo.live_handoff_batch run-auto-next \
  --seeds 0 --visual-handoff --policy rule \
  --bandwidth 256 --latency-ms 0 --drop-probability 0 \
  --world lawn --altitude 2 --duration 24 \
  --uav1-spawn-offset-ned 0 3 0 --session-startup-timeout 150 \
  --auto-log-dir /mnt/d/AeroWAM-data/px4_logs/handoff_v3_gui_fix \
  --output-dir /mnt/d/AeroWAM-data/episodes/handoff_v3_gui_fix
```

Gazebo 中应看到目标沿墙的远侧经过，而不是穿墙；UAV0 的视线被墙挡住时，UAV1 已绕墙端到远侧并接管。日志应有 `occlusion_start`、`handoff`、`verify`，且无 `collision`。检查记录：

若 WSLg 的 Gazebo GUI 在启动时以 `-11` 退出，可在后续批量命令中同时传入 `--visual-handoff --no-gui`：它保留相同的演示轨迹、场景标记和真实相机流，只跳过 GUI 窗口。没有产生 episode 文件的 seed 仍为 pending，可在原输出目录继续运行；无 GUI 的结果不能充当人工画面验收。不要把 GUI 驱动崩溃计作策略失败。

```bash
python3 - <<'PY'
import json
from benchmark.px4_gazebo.evaluator import replay_metric_differences
from benchmark.px4_gazebo.occlusion_geometry import segment_intersects_aabb

path = '/mnt/d/AeroWAM-data/episodes/handoff_v3_gui_fix/seed_000.jsonl'
with open(path, encoding='utf-8') as handle:
    record = json.loads(next(line for line in handle if line.strip()))
task = record['task_spec']
blocked = sum(
    not step['truth']['target_visibility']['uav_0']
    and segment_intersects_aabb(
        step['truth']['agent_positions']['uav_0'], step['truth']['target_pose'],
        task['occluder_center'], task['occluder_half_extents'],
    )
    for step in record['steps']
)
events = {event['event_type'] for step in record['steps'] for event in step['events']}
print('wall_blocked_uav0_steps:', blocked)
print('events:', sorted(events))
print('success:', record['outcome']['success'])
print('metrics:', record['outcome']['metrics'])
print('replay_differences:', replay_metric_differences(record))
PY
```

验收条件为 `wall_blocked_uav0_steps > 0`、成功交接和身份确认、`collision_rate = 0`、`longest_interruption_steps <= 10`、`replay_differences = {}`。仅通过离线自检不等于已通过真实相机画面验收。

## 固定约束

- 模型策略 2 Hz，PX4 setpoint 20 Hz。
- body-FLU waypoint 上限为 `dx/dy=1 m`、`dz=0.5 m`、`dyaw=15 deg`。
- 通信固定为 `256 B/s, 0 ms, 0 drop`。
- 单次推理超时悬停，连续三次超时安全降落。
- 正式模式关闭 acquisition/clearance/staging/handoff/support/route 标记。
- `simulator_error` 不进入模型分数，自动重试并保存在 `.attempts/`。

## 外部 VLN/WAM 接口

外部模型不需要导入 PX4 或 evaluator。它只需在 `HOST:PORT` 提供与
`policy_client.py` 兼容的 TCP 服务：每个请求和响应均为“4 字节大端长度 + JSON”。
请求协议版本为 `aerowam-model-handoff/1`，包含语言、同步双路 JPEG、两机自身
状态、最近 4 个动作和最近 16 条协议消息，不包含目标坐标、遮挡几何或评测真值。
响应格式为：

```json
{
  "protocol": "aerowam-model-handoff/1",
  "waypoints": {
    "uav_0": [0.0, 0.0, 0.0, 0.0],
    "uav_1": [0.0, 0.0, 0.0, 0.0]
  },
  "intent": "none"
}
```

`waypoints` 是两机 body-FLU 的 `[dx, dy, dz, dyaw]`；`intent` 只能是
`none`、`handoff` 或 `confirm`。在 `runs-config` 为每个模型填写不同的
`endpoint`，提前启动服务，再给批处理增加 `--external-policy-servers` 即可。
所有内置和外部模型使用同一批 fixture 与同一 evaluator。报告中的
`task_success` 均值就是模型成功率，同时给出 95% 置信区间、碰撞率、视觉覆盖、
平均/最大双机距离、路径长度和推理延迟。

## 0. 环境与自测

在 RTX 5060 Laptop 上先安装与驱动匹配的 CUDA PyTorch，再安装其余依赖：

```bash
cd /mnt/d/AeroWAM
python3 -m pip install -r benchmark/px4_gazebo/model_handoff/requirements.txt
python3 -m benchmark.px4_gazebo.model_handoff.self_test
python3 -m benchmark.px4_gazebo.live_handoff_self_test
```

## 1. 冻结 fixture manifest

仓库已经包含冻结清单。重新生成时必须得到文档记录的 SHA-256：

```bash
python3 -m benchmark.px4_gazebo.model_handoff.fixtures \
  --output benchmark/px4_gazebo/model_handoff/manifests/handoff_v3.jsonl
```

预期 SHA-256：`1486d421895e8ca531e081545794d190881d147d551340f4ae0b43cebf31a2fe`。旧 `handoff_v1.jsonl` 不用于 v3 训练或评测。

清单包含 240 个 train、40 个 validation，以及各 40 个 `test_appearance`、`test_occluder`、`test_combined`。

## 2. Oracle 采集与测试 fixture 验证

以下命令自行启动、停止 PX4/Gazebo，不需要预先打开 Gazebo。运行前只需启动 Micro XRCE-DDS Agent，并 source ROS：

```bash
source /opt/ros/humble/setup.bash
source ~/px4_ros2_ws/install/setup.bash
cd /mnt/d/AeroWAM

python3 -m benchmark.px4_gazebo.model_handoff.collect \
  --mode collect \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_v3.jsonl \
  --splits train validation \
  --output-dir /mnt/d/AeroWAM-data/oracle_results \
  --dataset-dir /mnt/d/AeroWAM-data/model_handoff_dataset

python3 -m benchmark.px4_gazebo.model_handoff.collect \
  --mode validate \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_v3.jsonl \
  --splits test_appearance test_occluder test_combined \
  --output-dir /mnt/d/AeroWAM-data/oracle_test_validation
```

先用 `--limit 2` 做飞行 smoke。只有全部 120 个测试 fixture 的 oracle 结果为 success，才能开始正式模型比较。

## 3. 冻结 OpenCLIP 特征

### 确认监督与旧数据修复

专家控制为 10 Hz，常规记录为 2 Hz，交接请求会额外记录。
确认事件可能出现在非记录时刻；采集器现在保留待确认标志，在确认已经发生且
UAV1 持续可见的后续记录行标注 `confirm`。它是专家监督标签，不表示原始 oracle
发送过确认消息。正式模型评测仍要求模型显式输出确认，验收规则保持原样。

对全部缺失确认标签的旧采集，可使用 `model_handoff.repair_dataset`，传入
`--source` 原训练集、`--results` 正式成功结果、`--manifest` 原冻结清单和
`--output` 新目录。工具先检查完整覆盖、成功结果、指标回放及图像完整性，
再使用完整日志真值与采样行的精确对应关系，在验证事件之后且满足持续可见的
第一条已有图像上补标。图像、时间戳、航点、真值和飞行结果均保留；动作历史
由此前记录的专家航点重建。每条补标依据保存在 `evaluator/supervision_repair.json`，
汇总在 `supervision_repair_report.json`。输出必须为新目录，原数据保留。

补标后必须重新生成特征，不能沿用旧缓存。标签校验不能代替图像身份识别检查
或真实闭环评测；先完成小规模训练和 validation 闭环 smoke 再做长训练。
新采集验收要求同时存在 `handoff`、`confirm` 和一致的历史航点，可通过
`model_handoff.dataset --require-expert-supervision` 单独检查。旧结果的
`already verified` 只表示断点跳过成功飞行，不会自动修复旧训练目录。

```bash
python3 -m benchmark.px4_gazebo.model_handoff.features \
  /mnt/d/AeroWAM-data/model_handoff_dataset \
  --splits train \
  --output /mnt/d/AeroWAM-data/features/handoff_train_vitb32.pt

python3 -m benchmark.px4_gazebo.model_handoff.features \
  /mnt/d/AeroWAM-data/model_handoff_dataset \
  --splits validation \
  --output /mnt/d/AeroWAM-data/features/handoff_validation_vitb32.pt
```

## 4. 三方法、三训练 seed

先用 `JointBC` 和小步数检查闭环，再运行完整训练。三个方法都受 20M 可训练参数和 50k step 上限约束：

```bash
mkdir -p /mnt/d/AeroWAM-data/checkpoints
for method in joint_bc dual_vln_hier dual_wam_mpc; do
  for seed in 0 1 2; do
    python3 -m benchmark.px4_gazebo.model_handoff.train \
      --method "$method" --seed "$seed" --steps 50000 \
      --train-cache /mnt/d/AeroWAM-data/features/handoff_train_vitb32.pt \
      --validation-cache /mnt/d/AeroWAM-data/features/handoff_validation_vitb32.pt \
      --output "/mnt/d/AeroWAM-data/checkpoints/${method}_seed_${seed}.pt"
  done
done
```

## 5. 10-episode smoke 与 120-fixture 正式评测

批处理默认按需启动一个策略服务器，因此不需要手动开九个终端。每个 fixture 内的方法顺序会确定性随机化：

批次应在固定代码和检查点下独立运行，不应运行中修改验收逻辑。观测采集的
缺帧、相机同步等异常标记为 `simulator_error`，安全降落后自动重试；重试耗尽
会停止批次。模型推理异常、连续推理超时和碰撞仍作为模型失败保存，不为了
获得成功结果而重跑。相机同步 100 ms 和默认推理 400 ms 门限不变。
旧版已经误标为 `safety_abort` 的相机同步故障不会被断点续跑自动修复；须保留
原始记录并将其移出正式结果目录后，才会补跑该 fixture。

想观察学习模型，在 `model_handoff.batch run` 命令后加 `--show-gui`；oracle 的 `model_handoff.collect` 也支持该选项。它只打开观察窗口，保持正式运行的人工标记隐藏。CPU OpenGL 会增加显示开销，因此 GUI 适合检查 smoke episode；统一推理延迟比较仍用默认无 GUI 模式。

```bash
python3 -m benchmark.px4_gazebo.model_handoff.batch run \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_v3.jsonl \
  --runs-config benchmark/px4_gazebo/model_handoff/runs.example.json \
  --splits test_appearance --limit 10 \
  --output-dir /mnt/d/AeroWAM-data/model_handoff_smoke

python3 -m benchmark.px4_gazebo.model_handoff.batch run \
  --manifest benchmark/px4_gazebo/model_handoff/manifests/handoff_v3.jsonl \
  --runs-config benchmark/px4_gazebo/model_handoff/runs.example.json \
  --splits test_appearance test_occluder test_combined \
  --output-dir /mnt/d/AeroWAM-data/model_handoff_test
```

## 6. 配对分层 bootstrap 汇总

```bash
python3 -m benchmark.px4_gazebo.model_handoff.evaluate \
  /mnt/d/AeroWAM-data/model_handoff_test \
  --json-output /mnt/d/AeroWAM-data/model_handoff_summary.json \
  --markdown-output /mnt/d/AeroWAM-data/model_handoff_summary.md \
  --csv-output /mnt/d/AeroWAM-data/model_handoff_summary.csv \
  --plot-output /mnt/d/AeroWAM-data/model_handoff_task_success_ci.png
```

主排名使用三个 OOD split 的 `task_success`。当置信区间重叠时，再比较视觉中断、碰撞、平均/最大双机距离和路径长度。若还需报告 ID-to-OOD gap，额外在 validation fixture 上运行同一批 checkpoint；报告会明确将它标为 validation reference，而不会混入 120 个正式 OOD 测试 episode。

## 7. WAM 修正版与预测能力边界

2026-10-13 的 legacy_v1 validation 结果为 0/40。新增 `--wam-revision expert_prior_v2`
用于诊断动作先验基线，旧 checkpoint 仍默认使用 legacy_v1。当前专家缓存没有危险或
联合不可见样本，新版本禁用未经验证的奖励规划，实际为 imitation_fallback。
它不能证明自主 MPC 或组合迁移能力。训练/规划修正、独立结果与后续数据要求见
[WAM_V2_CN.md](WAM_V2_CN.md)。外部适配和部分可观测、通信/队友变化、组合迁移仍在
[TODO_CN.md](TODO_CN.md) 中独立推进。

## 8. 决策导向 benchmark 草案

用户最终指定两张图为准：组合式协作泛化为主线，候选协作决策评测为核心。
先集中式双机观测、暂缓通信实现，完整版本仍包括跨任务的延迟通信组合。
预测层以历史、指令和候选动作序列为输入，决策层以真实分支的候选效果评价
ranking、Utility Regret、风险校准和闭环成功。设计与前置实现门槛见
[DECISION_BENCHMARK_CN.md](DECISION_BENCHMARK_CN.md)。旧交接版本继续保留，
草案还不是可运行的采集/评测命令。

2026-10-14 已加入可选发布控制 trace（dataset/2）、外部模型 factory 服务入口与
独立候选预测/选择 draft 接口；非飞行测试通过。接口语义、检查命令和单场景 trace
smoke 见 [INTERFACE_CN.md](INTERFACE_CN.md)。单场景仿真采集与严格数据验证已通过，
结果和重复帧修复见 [TRACE_SMOKE_CN.md](TRACE_SMOKE_CN.md)。新增 controls/2 时钟证据
单场景采集通过结构验证，离线 draft 质量检查发现图像偏旧/复用与 host 时钟配对异常；
controls/3 的双帧前进/新鲜度门控和时钟读取区间也已通过首次单场景结构验证，
图像年龄约150 ms且无复用；仍发现约104 ms发布间隔及RUN内host时钟偏移不一致。
不作为精确转移监督就绪证据。尚未完成时钟对齐与
多任务候选闭环，不需要因此重跑旧专家数据或启动新一轮长训练。

新 trace pilot 可启用 `--require-trace-receive-quality`：默认20秒无飞行host时钟检查，
异常不启动仿真；成功任务的质量报告保存在 evaluator，质量拒绝保留attempt且不晋升
正式训练目录，单独报告trace_quality_error。默认旧采集不启用该门槛。
独立 `clock_probe` 已在无仿真时复现约1.8~1.9秒UNIX偏移变化，另一20秒窗口通过；
异常间歇出现，仍需运行中质量验收。命令及边界见 [INTERFACE_CN.md](INTERFACE_CN.md)。
