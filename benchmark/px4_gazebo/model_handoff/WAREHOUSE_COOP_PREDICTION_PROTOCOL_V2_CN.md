# Warehouse 协作与预测协议 v2（设计复核稿）

这份协议替代“单一路线、单次交接”的 v1 作为后续实现依据。它的目的不是让模型记住仓库中某个固定路线，而是分别测量：

1. 在遮挡下根据历史观测预测目标未来状态；
2. 根据队友的实际行为选择何时请求、确认或等待；
3. 把这两种能力迁移到没有见过的布局、运动模式和队友策略。

用户已确认沿此设计继续实现。下文是目标设计，实际完成范围见末尾，不能把设计要求当成已验证结果。

## 为什么 v1 证据不够

- 固定的南到北轨迹和固定速度可以被时间模板替代，不需要世界模型；
- 固定的南侧可见、中央双盲、北侧可见顺序无法区分真正的因果判断和阶段标签匹配；
- 队友没有延迟、拒绝、误报、提前确认等行为变化，不能测协作策略；
- 只有一次 handoff 成功/失败，没有未来状态预测的概率输出和评分；
- 一个障碍布局不能证明对新场景、新运动或新队友的组合迁移。

## 任务族

每个 episode 仍使用 AWS warehouse，但由私有 episode 配置采样以下潜变量。配置不进入公开模型输入。

### 场景族

- `layout_family`：至少 6 个布局，包含主隔断、错列货架端、C 形舱、L 形舱和两种开口方向；
- `obstacle_scale`：遮挡体尺寸在审计安全范围内随机变化；
- `view_pair`：南北、东西和斜向互补观察位；
- `target_lane`：至少 3 条不相同但碰撞安全的目标通道。

训练、验证、测试必须按布局族拆分，测试布局不能只是在训练布局上平移一个常数。

### 目标运动族

- `constant_velocity`：速度从有限集合采样；
- `accelerate_brake`：分段恒加速度，拐点和终点隐藏；
- `pause_reverse`：随机停顿、短时反向后继续；
- `branch`：在遮挡前从两个安全出口中选择一个；
- `distractor`：非目标物块移动，目标保持或改变速度。

每个 episode 的轨迹只在私有配置和真实 mover 中存在。公开输入只包含历史图像、历史报告和公开时间戳。

### 队友行为族

队友不是固定教师脚本，至少包含：

- `reliable_relay`：按证据请求和确认；
- `delayed_relay`：观测和消息有可审计延迟；
- `conservative_relay`：看见目标但在置信度不足时等待；
- `noisy_relay`：偶发错误报告，但不会泄露目标真值；
- `missing_relay`：短时不响应，随后恢复。

队友策略只用于生成私有行为轨迹。正式模型不能读取队友类型标签。

## 每个 episode 的公开任务流程

任务流程不公开轨迹和角色：

1. `observe`：各机建立自己的 grounded track；
2. `predict`：在目标可能进入遮挡前，提交未来 `0.5 s/1.0 s/2.0 s` 的位置和可见性分布；
3. `request_or_wait`：根据自己的预测和队友消息决定请求、等待或继续观察；
4. `relay`：接收方在获得独立证据后报告目标状态和置信度；
5. `verify`：源机检查接力是否与新证据一致；
6. `return`：完成返航和降落。

不要求每个 episode 都发生交接。若目标没有进入可靠遮挡区，正确行为可以是继续观察或拒绝交接。这样可以防止模型把“每次都必须 handoff”当作捷径。

## 可计算的预测输出

扩展 warehouse adapter 的公开动作协议，但不修改 core schema：

- 位置预测：每个 horizon 输出二维位置均值和协方差；
- 可见性预测：目标在每个 horizon 对本机、队友和联合视图的可见概率；
- 事件预测：进入遮挡、离开遮挡、队友可确认的概率；
- 协作动作：`request`、`wait`、`confirm`、`reject`，每次带 grounded evidence 引用。

私有审计用真实 mover pose、真实遮挡几何和实际队友消息重放评分。模型不得从未来真值或教师标签构造预测。

## 指标与榜单

不能只报一次 handoff success。每个模型至少报告：

- `prediction_nll`：位置分布的负对数似然；
- `visibility_brier`：可见/遮挡事件的 Brier 分数；
- `calibration_error`：置信度校准误差；
- `handoff_success`：满足物理遮挡、双方 grounded evidence 和安全飞行的比例；
- `unsafe_handoff_rate`：没有接收方证据却确认的比例；
- `communication_efficiency`：达到可靠确认所需的消息数和延迟；
- `joint_flight_validity`：无碰撞、无越界、odometry 新鲜、正常降落的比例。

正式总分使用固定权重前先公开分项成绩；任何单项失败不能被其他项平均掩盖。一次 GUI 预览、教师数据重放或 flight-check 只能作为环境/执行证据，不能进入模型榜单。

## 迁移验证矩阵

至少做以下四个对照，每项使用同一私有测试集和独立 episode 种子：

| 条件 | 训练布局 | 训练运动 | 训练队友 | 用途 |
|---|---|---|---|---|
| `seen` | 见过 | 见过 | 见过 | 检查基础闭环 |
| `new_layout` | 未见过 | 见过 | 见过 | 检查几何迁移 |
| `new_motion` | 见过 | 未见过 | 见过 | 检查世界模型预测迁移 |
| `new_teammate` | 见过 | 见过 | 未见过 | 检查协作策略迁移 |
| `composed` | 未见过 | 未见过 | 未见过 | 检查组合迁移 |

对照模型至少包含：未适配 SmolVLM、只训练报告/动作 adapter、加入预测头的模型、加入队友行为训练的模型。只有当 `composed` 相对未适配基线在预测和安全协作两项都改善，才能声称“学到的能力组合迁移”。

## 数据和证据边界

公开输入保留 RGB、公开时间戳、自己的历史报告、队友公开消息和动作接口。目标真值、障碍物尺寸/位姿、真实遮挡射线、实例标签、Gazebo/PX4 真值、队友潜变量、教师策略和拟合中间结果只进入私有审计或训练数据的受控部分。

训练集、验证集、测试集按 `layout_family × target_motion_family × teammate_family` 做组合拆分，并固定 episode seed 清单。发布模型时同时发布 adapter、协议版本、训练组合清单、测试种子哈希和分项成绩。

## 实施顺序

1. 先在不启动 PX4 的离线 mover/可见性重放中验证潜变量和指标；
2. 再接入多布局 warehouse wrapper，生成每个 layout 的独立 clearance/contact 证据；
3. 先做静态和动态 scene self-test，再做无模型 bounded flight-check；
4. 最后接入 SmolVLM，分开运行 seen/new/composed 测试集；
5. 只有分项指标和迁移矩阵完整后，才生成可上传的榜单 manifest。

建议把当前已改的 v1 代码暂存为实验分支，等 v2 任务族确认后再收敛到正式运行入口。

## 当前实现状态（2026-10-22）

仓库已提供第一版可复现的离线任务族和公开输入 baseline：

- `warehouse_v2_task.py`：任务 ID `warehouse_coop_prediction`，场景 ID `aws_warehouse_handoff_v2`，五个固定迁移 split，以及公开输入、私有 truth 和评分器；
- `warehouse_v2_baseline.py`：只读取 `ModelInput` 的 constant-velocity / 保守交接 baseline；
- `warehouse_v2_scene_adapter.py`：把已验证的真实 Warehouse `ModelObservation` 公共包转换为 v2 `ModelInput`，保持角色未分配并拒绝私有真值；
- `warehouse_v2_scene_adapter.py`：函数 `to_model_input` / `from_wire_to_model_input` 转为 `ModelInput`，拒绝未知上下文、未来曝光和同帧报告；跨调用的 episode 和乱序检查由运行时负责；
- `warehouse_v2_task_self_test.py`、`warehouse_v2_baseline_self_test.py`：任务边界、core roundtrip、评分器和报告回归测试；
- `warehouse_v2_scene_adapter_self_test.py`：验证 live observation 的公开边界和同步约束；
- `warehouse_v2_dataset_20261021_v1`：每个 split 两个 episode 的可查看数据样例；
- `warehouse_v2_baseline_20261021_v1`：五个 split 的 NLL、visibility Brier、校准误差、危险交接率和协作成功率诊断报告。

task_version 为 `2`，数据集 manifest 为 `aerowam-warehouse-v2-dataset/3`，评分 schema 为 `aerowam-warehouse-v2-score/2`；公开 `episodes.jsonl` 只有 episode index，公开模型输入在 `model_inputs.jsonl`，私有真值在 `private_truth/`。合成输入的 `rgb=null`，没有图像资源。

当前默认导出为五个固定条件组合，尚未完整随机采样6/5/5族；合成布局只定义可见区间，尚未生成6套真实 Gazebo 布局。中央 baseline 读取双机视图，不能用它验证通信限制、真实队友变化或训练迁移收益。完整二维协方差、事件预测和真实 v2 任务动作执行仍待接入。

运行入口：

```powershell
$env:PYTHONPATH='D:\AeroWAM'
python -m benchmark.px4_gazebo.model_handoff.warehouse_v2_task `
  --output 'D:\AeroWAM-data\warehouse_v2_dataset_20261022_v2' --episodes-per-split 2
python -m benchmark.px4_gazebo.model_handoff.warehouse_v2_baseline `
  --output 'D:\AeroWAM-data\warehouse_v2_baseline_20261022_v2' --episodes-per-split 2
```

以上结果仍是 synthetic kinematics，`model_score_eligible=false`、`flight_admitted=false`。接入 Gazebo/PX4 和 SmolVLM 之前，不能把这份报告当作正式榜单成绩。

## 当前真实接入边界

`warehouse_v2_scene_adapter.py` 是真实采集的第一层接口，不是场景运行器。它接收现有 Warehouse 运行时产生的 `aerowam-model-handoff/1` 公共包，转换为相同 Core `ModelInput`；输入只包括双 RGB、各机自己的状态、已送达消息和动作历史。`role_assignment` 保持为空，角色由模型输出和私有 evaluator 从实际观测归因。

它不会把 `warehouse_v2_dataset_20261021_v1` 中的 synthetic RGB 路径当成图像，也不会读取 `private_truth`、目标位姿、遮挡几何、教师标签或队友潜变量。要进行真实 v2 评测，还必须由 Gazebo/PX4 scene adapter 产生带真实 JPEG、实际时间戳和独立场景审计的公开包，再将该公开包送入 SmolVLM；在此之前所有输出仍是诊断，不能进入正式榜单。
