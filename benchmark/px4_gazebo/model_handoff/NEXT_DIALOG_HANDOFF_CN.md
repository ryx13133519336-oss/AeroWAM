# AeroWAM 下一对话接管上下文

更新时间：2026-10-21

最新实测：开发集已独立复核 8/8、1,888 双机样本；用户已完成探索训练和独立 audit，checkpoint 为 `D:\AeroWAM-data\subt_fixedcomm_shared_event_20261021_161727`，保持冻结。单条下行 macro F1=0.39493，found/confirm 未命中；55 次 request 只有 1 次标签匹配。RGB RMSE=0.03681，差于 persistence=0.01797；补充诊断与单独重算位于数据盘 `subt_frozen_single_diagnostic_20261021_v1.json` 及 `.audit.json`。不根据留出结果调参。新增 `subt_handoff_frozen_eval.py` 待下行 8 格审计齐全后只读评估固定权重，再单独 audit；具体命令见 `SUBT_NEXT_RUN_CN.md`。

下行采集中蓝方块已有颜色审计失败：同一 JPEG 在 WSL libjpeg 6.2 与 Windows Pillow 12/libjpeg 9 的支持像素可能不同。有 `batch_audit_error.txt` 的目录保持排除，禁止换平台覆盖审计。现有阈值未放宽，错误信息已增加帧号、agent、支持计数和 JPEG 版本。失败目录保留，尚未完成采集修复或全矩阵实测。新的训练/回放/批量和冻结评估入口拒绝已知失败标记。

当前用户约束：暂不研究通信影响，所有新采集固定消息延迟 0.5 s。当前开发计划为目标外观 × 身份置换 × 队友轨迹的 8 格；旧 16 格计划及 2 s 数据保留作历史证据，不用于本阶段条件覆盖。长时采集由用户在 WSL 执行，助手不自动启动或持续等待。

## 研究主线（不可偏移）

> 在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

所有后续设计都必须围绕这条主线。两架无人机没有永久的“搜索机/确认机”角色；角色只能由当前可见性、通信和任务状态临时形成。不得把 `uav_0` 固定解释为搜索者，也不得把颜色绑定到角色。

## 仓库和运行位置

- 实际仓库：`D:\AeroWAM`，WSL 路径：`/mnt/d/AeroWAM`
- 数据根目录：`D:\AeroWAM-data`，WSL 路径：`/mnt/d/AeroWAM-data`
- 不要使用空的桌面副本 `C:\Users\R\Desktop\AeroWAM`
- 固定场景资源：`/mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2`
- 相机/实例模型：`/mnt/d/AeroWAM-data/annotation_models_v1`
- Git remote：`git@github.com:ryx13133519336-oss/AeroWAM.git`

WSL 中运行 Gazebo/PX4 前，先确认没有其他任务占用；不要全局 kill 其他用户进程。当前工作区有大量与本任务无关的已修改/未跟踪文件，必须保留，提交时只 stage 本任务文件。

## 已完成的 Git 提交

当前本线最新提交：

- `130a522` `Add stair handoff GUI route preview`
- `1694d29` `Keep Gazebo target preview alive during GUI stalls`
- `6acda55` `Decouple stair teammate motion and communication conditions`
- `4bdcd87` `Add stair-crossing handoff demonstration pipeline`

以上提交已经推送到 GitHub `main`。更早的本线提交还包括 SubT Urban 场景和可移动工业标靶接入。不要重写历史或回滚其他工作。

## 当前已有的楼梯 handoff 管线

主要文件：

- `benchmark/px4_gazebo/model_handoff/subt_handoff_route.py`：SubT Urban 两层折返楼梯路线、网格净空、目标扫掠检查
- `benchmark/px4_gazebo/model_handoff/subt_handoff_expert.py`：特权运动学教师、可见性阶段、`found -> handoff_request -> handoff_accept -> confirm`
- `benchmark/px4_gazebo/model_handoff/subt_handoff_collect.py`：Gazebo 运动学代理采集，不启动 PX4
- `benchmark/px4_gazebo/model_handoff/subt_handoff_inspect.py`：独立图像哈希、实例掩码、时间戳、因果回放审计
- `benchmark/px4_gazebo/model_handoff/subt_handoff_data.py`：训练导出，公开输入和监督严格分离
- `benchmark/px4_gazebo/model_handoff/subt_handoff_batch.py`：可恢复串行条件采集与逐 episode 独立审计
- `benchmark/px4_gazebo/model_handoff/subt_handoff_learn.py`：公开观测事件基线与下一观察预测诊断；不提供飞控或正式模型成绩
- `benchmark/px4_gazebo/model_handoff/subt_handoff_design.py`：固定种子 2×2×2 条件清单，通信固定 0.5 s
- `benchmark/px4_gazebo/model_handoff/subt_handoff_preview.py`：楼梯 handoff GUI 预览，目标沿楼梯上升
- `benchmark/px4_gazebo/model_handoff/SUBT_URBAN_CN.md`：完整命令和证据边界

当前楼梯路线约 34.027 m，目标中心垂直跨度约 10 m。相机代理不是无人机实体，目标载体不是物理轮式动力学。

## 重要：两个预览入口不同

旧入口只展示一楼工业标靶水平往返，不是楼梯 handoff：

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_urban preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --case revealed --move-targets --preview-seconds 30 --target-speed 0.2 \
  --output-dir "/mnt/d/AeroWAM-data/subt_gui_preview_$(date +%Y%m%d_%H%M%S)"
```

楼梯 handoff 必须使用新入口：

```bash
cd /mnt/d/AeroWAM
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.subt_handoff_preview \
  --package /mnt/d/AeroWAM-data/external_scenes/subt_urban_pinned_v2 \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --target blue:cube --swapped --speed 0.4 --preview-seconds 100 \
  --output-dir "/mnt/d/AeroWAM-data/subt_stair_gui_$(date +%Y%m%d_%H%M%S)"
```

结束后的 `preview_summary.json` 应有 `reached_upper_floor: true`。该预览仍然是 Gazebo 运动学演示，不是 PX4 飞行或模型成绩。

## 已验证数据

已独立审计通过的三条可用于开发导出的 episode：

1. `D:\AeroWAM-data\subt_stair_expert_red_v1`：旧版红球，legacy，无条件元数据和位姿请求日志
2. `D:\AeroWAM-data\subt_stair_blue_swapped_motion_only_v1`：蓝方块，`swapped=true`，队友轨迹 delayed，通信 0.5 s
3. `D:\AeroWAM-data\subt_stair_blue_swapped_comm_only_v1`：蓝方块，`swapped=true`，队友轨迹 normal，通信 2 s

两条新蓝色 episode 各有 236 个双视角样本、472 张 RGB、约 10 m 跨层跨度；独立审计重放事件成功，位姿请求分别核验约 2358/2361 条。红色 legacy 数据没有伪造补写条件。

当前开发导出：

`D:\AeroWAM-data\subt_stair_handoff_training_v2`

- 3 个 episode
- 708 个联合样本
- 1416 张 RGB
- 705 个下一观察预测对
- `development_training_only`
- `model_score_eligible=false`
- `flight_admitted=false`
- `condition_coverage.complete_2x2x2x2=false`

旧的 `subt_stair_handoff_training_v1` 保留，不覆盖。一个旧蓝色基线在 Windows 复核通过，但 WSL 独立复核拒绝，因此不能未经重新采集就纳入新的 WSL 导出。

## 当前证据能说明什么

- SubT Urban 两层建筑、楼梯、遮挡和本地纹理资源可以加载。
- 目标沿折返楼梯路线的运动学规划通过固定网格扫掠检查。
- 两机身份可交换，且队友轨迹变化和通信延迟现在可以独立设置。
- Gazebo RGB/实例图、位姿反馈和因果 handoff 事件可以被独立审计。
- GUI 楼梯预览已经实际到达二楼。

## 当前证据不能说明什么

以下都还没有完成，不能在论文、报告或对话中宣称已经完成：

- 已完成一次轻量探索性基线：上行两条现代 episode 训练，反向下行红球 episode 留出评估；独立重算通过，但不是正式模型成绩。
- 没有独立新几何测试路线或新场景，当前只有一个 scene family。
- 已加入同一 scene family 的下行方向留出路线 `stair_descent_v1`；一条红球身份置换 episode 已独立审计通过，不能把它称作独立场景。
- 当前上行 8 格固定通信开发矩阵只有 1 格经独立审计通过：蓝方块、身份置换、delayed、0.5 s。normal/2 s 不计本阶段覆盖；红色 legacy 不补写条件。
- 没有真正分散式双无人机策略；教师相机朝向使用隐藏目标位姿。已新增 `subt_handoff_model_eval.py`，复用 core 模型协议对独立审计 episode 做公开输入回放；保存完整请求/响应，由 `subt_handoff_model_audit.py` 单独重建审计。回放的 `success=null`，不生成正式榜单成绩。
- 没有动作条件世界模型；目前没有“飞控动作 -> 下一状态”的真实执行记录。
- 楼梯 handoff 还没有 PX4 飞行放行，所有数据明确 `flight_admitted=false`。新增私有 `subt_handoff_evaluator.py` 状态机已测试逐机角色置换、消息因果和遮挡持续性；仍等待真实执行器提供相机/PX4 遥测、动作追踪和私有物理遮挡证据。
- 目标是 `set_pose_vector` 运动学代理，不是带接触/轮驱动/碰撞故障的物理目标。
- 没有系统验证丢包、乱序、延迟抖动、传感器噪声、失帧、通信中断。
- 每个条件没有独立重复，不能做显著性检验或稳定性结论。

## 推荐的下一步顺序

### 第一阶段：补齐可解释的数据条件

使用 `subt_handoff_design.py` 生成的固定种子清单，补齐目标外观、身份置换、队友轨迹的 8 格交叉条件，通信延迟固定 0.5 s。优先重新采集红色目标的现代条件，因为旧红色是 legacy；每条采集都必须通过 `subt_handoff_inspect`，不能只看 `summary.json`。

采集器现支持显式 `development_training_only` / `held_out_route_test` 分区，并核对上行 `stair_ascent_v1` 与下行 `stair_descent_v1`。`subt_handoff_batch.py` 会串行跳过已通过 episode、保留失败目录并在每条完成后运行独立审计；留出路线不能进入训练导出。

### 第二阶段：建立真正的学习评测

先保持开发集和测试集严格按 episode/路线/场景分组，不能随机切连续帧。当前 `subt_handoff_learn.py` 只提供集中式公开观测事件分类和 8×8 下一观察 Ridge 诊断，已报告事件准确率、宏 F1、稀有事件召回、下一帧 RGB RMSE 与公开代理位姿误差；它没有训练分散式策略、动作条件世界模型或真实 PX4 控制。正式评测仍需更多独立 episode，至少报告 handoff 成功率、确认延迟、原观察者失视后的恢复率、下一帧 RGB/状态预测误差和队友运动变化敏感性；本阶段不评通信影响。

### 第三阶段：加入独立路线/场景留出

当前已加入下行方向留出作为同场景族任务测试，但它不是独立场景。仍需新增不同几何结构或不同场景路线，禁止将同一场景的相邻帧当独立泛化证据。只有完成真正留出测试后，才能讨论“组合迁移到新任务”。

### 第四阶段：再接 PX4 闭环

先完成无模型的 PX4/ Gazebo 双机只读启动和机体净空检查，再将策略动作映射到飞控控制器。不能把 `set_pose_vector` 请求当作真实飞行动作，也不能为通过预览而关闭预检或修改 WSL 校时链路。

## 运行和提交规则

- 每次使用全新输出目录，避免覆盖已有证据。
- 运行前检查本次命令是否会启动 PX4；GUI/教师预览不应启动 PX4。
- 不并发运行同一 Gazebo/PX4 任务。
- 先运行回归测试，再提交代码。
- 只 stage 本次任务文件；不要提交数据盘的大图、日志、模型和其他对话生成资料。
- 不要修改系统时钟、WSL 校时链路或放宽飞控安全门槛来绕过问题。
- 经过验证的代码才提交和推送；当前 GitHub `main` 已包含本文件对应代码的 `130a522`。

## 接管对话的第一条建议消息

可以把下面这段直接发送给下一个对话：

> 请接管 `D:\AeroWAM` 中的 AeroWAM 工作。研究主线是“在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务”。不要固定两架无人机角色，也不要把特权教师、运动学代理或 Gazebo 预览说成模型成绩或 PX4 飞行。先阅读 `benchmark/px4_gazebo/model_handoff/NEXT_DIALOG_HANDOFF_CN.md` 和 `SUBT_URBAN_CN.md`，检查当前 Git 提交 `130a522`。下一优先级是补齐楼梯 handoff 的条件矩阵和独立测试场景，然后训练/评测策略与世界模型；所有新证据必须通过独立审计，提交时只 stage 相关文件，不要碰工作区其他未相关修改。

## 当前研究推进边界

8 格只补齐部分可观测跨层 handoff 的基础条件，normal/delayed 仅代表两种预设队友运动。需要在训练前固定新任务和场景留出，补充失视后恢复及未见队友行为，再比较无协作/无预测基线、仅协作技能、仅预测模块、协作技能与预测组合的适当对照。当前事件分类与下一观察 Ridge 尚未形成组合决策闭环，不能证明组合迁移。下行方向反转只作初步方向留出。现阶段不继续用 2 s episode 训练当前固定通信实验；历史轻量基线保留但不作为本阶段结果。
