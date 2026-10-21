# SubT 下一批终端运行命令

开发集 8/8 已完成并独立复核，第一版探索训练和重算审计已完成。下一步补齐方向留出集并评估冻结 checkpoint；模型接口检查与代理执行另行运行，不与采集或 warehouse Gazebo 同时运行。所有新输出保留在数据盘，训练权重和图片不提交 GitHub。

环境提示：`ryx@RYX:...$` 是 WSL Bash，运行 bash 代码块；`PS D:\AeroWAM>` 是 Windows PowerShell，运行 powershell 代码块。在 WSL 输入 `exit` 返回 PowerShell。

## 1. 开发集 8 格（已完成，以下仅用于恢复）

在 WSL 运行。通信延迟固定 0.5 秒，normal/delayed 是队友代理运动差异。已审计通过的 episode 会重新核验后跳过；失败目录保留并新增 attempt。

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_batch \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --design /mnt/d/AeroWAM-data/subt_stair_fixedcomm_design_20261021_v3.json \
  --output-root /mnt/d/AeroWAM-data/subt_stair_matrix_capture_20261021_v1 \
  --split development_training_only \
  --existing-episodes /mnt/d/AeroWAM-data/subt_stair_blue_swapped_motion_only_v1
```

完成判据：batch_summary.json 中 8 个条件全部为 collected_and_audited 或 skipped_passed；失败条件不计覆盖。当前代码的 already_passed 只表示本次开始前已通过的数量，不等于最终覆盖。遇到失败可以重新执行同一命令，无须删除目录。

这一步产生特权教师事件监督和下一观察序列，没有实际飞控动作，不提供动作条件世界模型监督。

## 2. 单独采集方向留出集

步骤 1 退出并释放 Gazebo 后运行。下行只作评估，不能加入训练或用来挑 checkpoint；它仍属于同一场景族，不等于独立场景。

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_batch \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --design /mnt/d/AeroWAM-data/subt_stair_fixedcomm_design_20261021_v3.json \
  --output-root /mnt/d/AeroWAM-data/subt_stair_direction_test_20261021_v1 \
  --split held_out_route_test \
  --existing-episodes /mnt/d/AeroWAM-data/subt_stair_descent_test_red_swapped_normal_comm05_20261021_v1 \
                      /mnt/d/AeroWAM-data/subt_stair_descent_test_blue_swapped_normal_comm05_20261021_v1
```

完成判据同上。8 格、每格一次只是条件覆盖，不是统计重复或组合迁移结论。

2026-10-21 首轮下行结果为 4/8：红球四格通过，蓝方块四格因遮挡边缘 RGB 颜色支持不足拒绝（第 101 或 117 帧、17–19 个目标掩码像素，WSL 支持数为 0、要求至少 2）。新采集统一使用显式 JPEG 4:4:4、quality=100，避免默认色度子采样损失；不改颜色阈值、教师事件或路线，不使用目标掩码修图。编码版本写入 public metadata，独立审计检查实际采样格式。旧失败数据不重编码或补写通过，原始目录保留。

补采仍使用上面的同一条 WSL 批量命令：程序会重新核验并跳过红球四格，仅为蓝方块缺失条件新建 `attempt002`。代码已经在共享工作区，无须 `git pull`。这是待真实采集验证的修复，离线测试通过不保证新蓝方块 episode 已准入。旧通过数据、训练权重和测试划分保持不动；本轮新旧 JPEG 编码混合，是后续解释离线结果的额外限制，不能据此作严格的外观因素比较。正式对比需要统一编码的数据版本。

先在步骤 2 命令末尾加 `--limit 1`，只补一条验证；该条 collected_and_audited 后，去掉 `--limit 1` 补其余缺失项。如果仍 audit_failed，先读取新目录 batch_audit_error.txt，不继续重复采集。

## 3. 真实 Gazebo 代理动作接口 smoke（已通过一次）

hold smoke 已在 `D:\AeroWAM-data\subt_proxy_hold_20261021_190821` 通过：11 帧、11 次动作服务请求，独立审计通过，清理错误为空。hold reference 只返回零动作且不报告事件，预期不会完成 handoff；本步骤用途是验证实时传感、core 协议、动作服务和清理，不是模型成绩。不能与步骤 1/2 并发。若后续模型输出异常、缺帧、代理碰撞或进程退出，保留目录用于诊断。

```bash
cd /mnt/d/AeroWAM
proxy_run_dir="/mnt/d/AeroWAM-data/subt_proxy_hold_$(date +%Y%m%d_%H%M%S)"
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_proxy \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --model-manifest benchmark/px4_gazebo/model_handoff/manifests/subt_core_hold_reference.json \
  --adapter benchmark.px4_gazebo.model_handoff.subt_handoff_model_io:hold \
  --seconds 5 --wall-timeout-s 300 --output "$proxy_run_dir"
```

命令成功退出后，在同一终端独立审计：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_proxy_audit \
  --run-dir "$proxy_run_dir" --output "${proxy_run_dir}.audit.json"
```

检查 proxy_summary.json 的 status=complete、owned_processes_stopped=true、cleanup_errors=[]，以及审计 passed=true。程序将模型动作变成相机代理位姿请求；service_accepted 只证明服务接受，不能证明曝光时刻精确执行。PX4 未启动，flight_admitted/formal_score 均为 false。只有独立审计通过，才能称这条代理接口 smoke 通过。

使用已适配模型时改 adapter 和 manifest，或把 adapter 换为 --policy-endpoint HOST:PORT。服务端必须接受 core 协议；旧单 intent 的 ModelJointAction 需要显式适配，不能固定 uav_0/uav_1 为 source/receiver。

## 3a. 冻结事件头的跨 Windows/WSL 协议 smoke

训练 checkpoint 使用 Windows scikit-learn，Gazebo 使用 WSL Python；不要在 WSL 直接加载该 joblib。先开一个 **Windows PowerShell** 窗口并保持服务运行：

```powershell
Set-Location D:\AeroWAM
python -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_checkpoint_server `
  --checkpoint D:\AeroWAM-data\subt_fixedcomm_shared_event_20261021_161727 `
  --host 0.0.0.0 --port 8765
```

再开一个 **WSL Bash** 窗口，确认没有其他 Gazebo 任务后运行：

```bash
cd /mnt/d/AeroWAM
proxy_run_dir="/mnt/d/AeroWAM-data/subt_proxy_checkpoint_$(date +%Y%m%d_%H%M%S)"
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_proxy \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --model-manifest benchmark/px4_gazebo/model_handoff/manifests/subt_fixedcomm_event_head_diagnostic.json \
  --policy-endpoint 127.0.0.1:8765 \
  --seconds 5 --wall-timeout-s 300 --output "$proxy_run_dir"
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_proxy_audit \
  --run-dir "$proxy_run_dir" --output "${proxy_run_dir}.audit.json"
```

2026-10-21 已完成该 smoke：`D:\AeroWAM-data\subt_proxy_checkpoint_20261021_192240`，`status=complete`、`model_action_services_accepted=11`、`samples=11`、`owned_processes_stopped=true`、`cleanup_errors=[]`，独立审计 `passed=true`（`frames_verified=11`、`model_actions_traced=11`）。`proxy_event_success=false` 是预期结果：适配器只输出 hold 动作并对无框事件 abstain。该测试证明冻结 checkpoint 已进入实时请求链；不能证明 handoff 成功、世界模型有效或 PX4 飞行。后续更换 checkpoint、模型或任务时仍需重新跑同一审计；若 WSL 无法连接 `127.0.0.1:8765`，停止代理并把连接错误发来，不要改成任意公网地址。

## 3b. SubT PX4 只读控制链检查

启动检查已经证明 SubT 资源、双机相机和 PX4 遥测可用，但 `flight_ready=false`。在接入真实动作前，先运行下面的显式控制链检查。它只验证六个逐机控制 writer 的匹配，并发送每机一条 `command=400` 的未解锁 disarm 请求；不发送 offboard、轨迹、解锁或起飞命令。

```bash
cd /mnt/d/AeroWAM
source /opt/ros/humble/setup.bash
source /home/ryx/px4_ros2_ws/install/setup.bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.subt_urban startup \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --px4-root /home/ryx/PX4-Autopilot \
  --agent /mnt/d/AeroWAM-data/tools/microxrce_agent_v242_humble/MicroXRCEAgent \
  --ros-domain-id 83 --timeout-s 120 --control-link-check \
  --output-dir "/mnt/d/AeroWAM-data/subt_startup_control_$(date +%Y%m%d_%H%M%S)"
```

准入前应同时看到 `startup_passed=true`、`control_link_verified=true`、两条新鲜 `result=0` disarm ACK、`motion_commands_sent=0` 和 `owned_processes_stopped=true`。失败目录必须保留，不能放宽超时或把启动检查写成飞行成绩。即使通过，`flight_admitted=false` 仍保持不变。

## 4. 开发集 8 格全部通过后训练探索性基线

以下在 Windows PowerShell 运行，使用已确认安装 scikit-learn 的 Windows Python。--train-root 会重新独立核验数据、为每格选择一条通过的 episode；不足 8 格就报错，不训练。测试使用独立目录的红球下行 episode，不加入训练，也不作为 checkpoint 选择依据。

```powershell
Set-Location D:\AeroWAM
$subtFitDir = 'D:\AeroWAM-data\subt_fixedcomm_shared_event_' + (Get-Date -Format yyyyMMdd_HHmmss)
python -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_learn train `
  --train-root D:\AeroWAM-data\subt_stair_matrix_capture_20261021_v1 `
  --extra-train-episodes D:\AeroWAM-data\subt_stair_blue_swapped_motion_only_v1 `
  --test-episode D:\AeroWAM-data\subt_stair_descent_test_red_swapped_normal_comm05_20261021_v1 `
  --output-dir $subtFitDir
```

成功后同一终端独立重算：

```powershell
python -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_learn audit --run-dir $subtFitDir
```

本轮修正了共享事件头的 self/peer 特征排列，交换两机身份时同步交换 RGB、位姿、消息计数和近因特征。输出仍只是集中式事件分类与下一观察 Ridge 基线；世界模型没有动作条件，不能把离线事件 F1 当作闭环任务成功率，不能写入正式榜单。旧历史 checkpoint 仍按旧排列审计，不静默改变其解释。

## 5. 下行 8 格完成后评估冻结模型，不重新训练

当前冻结 checkpoint：`D:\AeroWAM-data\subt_fixedcomm_shared_event_20261021_161727`。单条红球下行诊断 macro F1=0.39493，found/confirm 均未命中；55 次 request 预测只有 1 次正确（精确率 1.82%）。RGB 下一观察 RMSE=0.03681，而沿用当前画面的 persistence 为 0.01797。该基线尚未证明迁移能力，不根据这条测试结果调参。补充诊断及独立重算保存在数据盘 `subt_frozen_single_diagnostic_20261021_v1.json` 和 `.audit.json`。

以下在 Windows PowerShell 执行；测试矩阵不足 8 格会直接拒绝，不会拿部分条件充当完整评估。评估读取固定权重，不拟合、不选阈值，输出混淆矩阵、逐事件 precision/recall/F1、逐条件结果以及 persistence 比较。录制消息来自教师，仍是离线诊断。

```powershell
Set-Location D:\AeroWAM
$subtEvalDir = 'D:\AeroWAM-data\subt_frozen_descent_eval_' + (Get-Date -Format yyyyMMdd_HHmmss)
python -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_frozen_eval evaluate `
  --run-dir D:\AeroWAM-data\subt_fixedcomm_shared_event_20261021_161727 `
  --test-root D:\AeroWAM-data\subt_stair_direction_test_20261021_v1 `
  --extra-test-episodes D:\AeroWAM-data\subt_stair_descent_test_red_swapped_normal_comm05_20261021_v1 D:\AeroWAM-data\subt_stair_descent_test_blue_swapped_normal_comm05_20261021_v1 `
  --output-dir $subtEvalDir
if ($LASTEXITCODE -eq 0) {
  python -B -m benchmark.px4_gazebo.model_handoff.subt_handoff_frozen_eval audit --output-dir $subtEvalDir
}
```

采集端曾出现蓝方块 RGB 颜色审计失败；同一 JPEG 在 Windows Pillow 12/libjpeg 9 和 WSL libjpeg 6.2 的解码会改变阈值边缘的颜色支持计数。保留 `batch_audit_error.txt` 的失败目录不计入覆盖，不能换平台补写 passed 来绕过。新报错包含帧号、agent、支持像素、阈值和 JPEG 版本，便于定位；审计阈值保持不变。当前 8 格留出评估已经在真实审计数据上完成，不能把代理或离线指标写成闭环任务成功率。

## 后续顺序

1. 开发集补齐并重新核验覆盖后导出训练集，禁止纳入旧 2 秒通信或 legacy 不明条件数据。
2. 修正并训练身份可交换的共享事件策略；现有轻量模型只是探索诊断，不能直接作为飞行策略。
3. 新代理入口审计通过后接入训练 checkpoint，记录真实模型事件；增加独立队友策略行为，而非把教师时间表当模型策略。
4. 生成并审计真实动作对应的下一观察数据，再训练动作条件世界模型；未执行动作不能配对固定回放帧作动力学成绩。
5. SubT 双机 PX4 只读检查、机体包络与控制链准入通过后接真实飞控。
6. 冻结模型，在下行、独立 warehouse 场景和新任务上跑对照与重复，合格 evaluation reports 才进入正式榜单。

2026-10-21 短时验证：Windows/WSL 各 16 项新离线测试通过；真实蓝色已审计 episode 的 4 步 hold 回放和独立请求重建通过。该 hold 回放不是学习模型成绩，也没有执行飞控。
