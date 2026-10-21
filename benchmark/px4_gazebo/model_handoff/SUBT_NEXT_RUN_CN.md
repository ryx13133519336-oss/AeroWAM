# SubT 下一批终端运行命令

先续采专家矩阵；模型接口检查与代理执行另行运行，不与采集或 warehouse Gazebo 同时运行。所有新输出保留在数据盘，训练权重和图片不提交 GitHub。

## 1. 继续开发集 8 格

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

## 3. 可选：真实 Gazebo 代理动作接口 smoke

此新入口尚待真实 Gazebo 验收。hold reference 只返回零动作且不报告事件，预期不会完成 handoff；用途是验证实时传感、core 协议、动作服务和清理。先只跑 5 秒仿真时间；不能与步骤 1/2 并发。若模型输出异常、缺帧、代理碰撞或进程退出，保留目录用于诊断。

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

## 后续顺序

1. 开发集补齐并重新核验覆盖后导出训练集，禁止纳入旧 2 秒通信或 legacy 不明条件数据。
2. 修正并训练身份可交换的共享事件策略；现有轻量模型只是探索诊断，不能直接作为飞行策略。
3. 新代理入口审计通过后接入训练 checkpoint，记录真实模型事件；增加独立队友策略行为，而非把教师时间表当模型策略。
4. 生成并审计真实动作对应的下一观察数据，再训练动作条件世界模型；未执行动作不能配对固定回放帧作动力学成绩。
5. SubT 双机 PX4 只读检查、机体包络与控制链准入通过后接真实飞控。
6. 冻结模型，在下行、独立 warehouse 场景和新任务上跑对照与重复，合格 evaluation reports 才进入正式榜单。

2026-10-21 短时验证：Windows/WSL 各 16 项新离线测试通过；真实蓝色已审计 episode 的 4 步 hold 回放和独立请求重建通过。该 hold 回放不是学习模型成绩，也没有执行飞控。
