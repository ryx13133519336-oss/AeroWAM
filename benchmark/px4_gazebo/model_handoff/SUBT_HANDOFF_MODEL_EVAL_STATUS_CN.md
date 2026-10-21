# SubT 楼梯 handoff：模型闭环与榜单缺口

## 当前已经具备

- 楼梯路线、目标颜色/形状、身份置换、队友运动变化和固定通信延迟的开发矩阵；
- RGB、公开本机状态、已送达消息、事件监督和未来观察监督的分离格式；
- 独立数据审计，能够拒绝旧目录、颜色支持错误、时序错误和私有字段泄漏；
- SubT 训练导出到统一 core episode 的 adapter；
- 外部模型的旧 handoff transport、adapter manifest 和无飞行接口自测；
- 新增的 evaluation report 和 leaderboard 聚合规则。

这些内容还不能证明模型已经在楼梯场景执行 handoff，也不能产生正式成功率。

## 还缺什么

### 1. SubT 模型闭环 runner

当前 `subt_handoff_collect.py` 是特权专家/运动学代理采集器：它沿规划路线移动目标和相机代理，不接收模型动作。因此它只能产生训练/接口诊断数据，不能作为模型成绩。

需要新增一个独立的 SubT evaluation runner：

- 启动固定版本 SubT Urban 楼梯 world；
- 两架真实 PX4/Gazebo x500 由模型输出控制，或明确标记为 `diagnostic_proxy`；
- 每个决策点只构造公开 `ModelInput`，不把目标真值、路线、掩码、私有可见性或 evaluator 结果传给模型；
- 接收模型动作和事件，记录原始请求、原始响应、发布控制、接收遥测和时间戳；
- 私有 evaluator 独立判断 `found -> handoff_request -> handoff_accept -> confirm`、安全停止、超时和基础设施错误；
- 每个 episode 结束后生成 `aerowam-subt-handoff-evaluation/1` report。

代理 runner 可以先用于接口 smoke，但必须将 `kinematic_proxy=true`、`flight_admitted=false`、`model_score_eligible=false` 写入报告，不能进入正式榜单。

### 2. 可复现的模型身份

每个模型必须有 adapter manifest、代码/配置/权重 SHA256、训练数据 ID、训练 seed、输入权限、运行设备和超时设置。没有身份绑定的“某个模型输出”不能写入榜单。

### 3. 真正的队友行为变化

当前 `normal/delayed` 主要是响应时间/运动行为代理，不是独立队友策略。正式主线需要冻结至少两个可复现队友 policy：每个 policy 有自己的 manifest、动作日志和行为版本，且不把队友私有策略状态传给被测模型。

### 4. 正式测试矩阵和重复

当前楼梯下行路线是同一场景族的方向留出，不是独立场景。正式榜单前还需要：

- 固定训练、开发、同场景方向留出、独立场景留出和新任务留出；
- 每个条件多个独立 episode/repeat，按 episode 计分，不按帧计分；
- 新场景、新任务和测试队友行为不参与训练、阈值选择和 checkpoint 选择；
- 固定通信延迟为 0.5 秒，通信影响暂不作为实验因素。

warehouse 可作为第一套独立场景族，但必须先完成统一 adapter 和独立审计。

### 5. 榜单输入和排名规则

`subt_handoff_leaderboard.py` 只聚合合格 evaluation report：

- `closed_loop=true`；
- `public_input_only=true`；
- `independent_audit=true`；
- `flight_admitted=true`；
- `action_trace_verified=true`；
- `privileged_teacher=false`、`kinematic_proxy=false`；
- `status=complete` 且 `success` 是布尔值；
- 固定通信延迟为 0.5 秒；
- 同一 method/condition/repeat 不得重复。

教师、代理、规则策略、仿真错误和安全停止会保留在审计输出中，但不进入正式成功率分母，也不排名。

## 推荐执行顺序

1. 先完成 SubT evaluation runner 的无飞行 adapter smoke，确认模型请求/响应、超时、控制记录和私有 evaluator 对齐。
2. 用一个明确的外部模型和一个规则 reference 做小规模双条件闭环，验证失败分类和清理。
3. 通过 PX4/Gazebo 真实 x500 闭环准入，再开始正式重复采集。
4. 在楼梯开发矩阵上训练/调参，只在 `development_training_only` 使用专家数据。
5. 在 `stair_descent_v1` 和独立 warehouse 场景上运行冻结 checkpoint，生成 evaluation reports。
6. 运行 leaderboard 聚合器，发布成功率、有效 episode 数、覆盖条件和基础设施失败数。
7. 再加入 search/inspection 等任务，使用新的 task namespace 和 evaluator，不修改旧 handoff 字段语义。

## 现在用户还需要准备的内容

- 选定第一批要接入的开源模型及其 checkpoint/权重来源；
- 给每个模型指定 adapter manifest 和推理超时；
- 准备能在当前 WSL/PX4/Gazebo 环境运行的 SubT x500 world 与安全清理路径；
- 与 warehouse 对话冻结队友行为 manifest、独立场景 adapter 和评测起点；
- 确认正式榜单只接受真实 PX4/Gazebo 闭环，还是先发布单独的 `diagnostic_proxy` 表。

## 检查命令

```bash
python -m benchmark.px4_gazebo.model_handoff.core.core_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_self_test
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_leaderboard_self_test
```

生成榜单时输入的是独立 evaluation reports，而不是训练数据、专家标签或 Gazebo 预览目录：

```bash
python -m benchmark.px4_gazebo.model_handoff.subt_handoff_leaderboard \
  --reports /path/to/evaluation_report_*.json \
  --output /path/to/subt_handoff_leaderboard.json
```
