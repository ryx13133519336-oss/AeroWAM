# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 结构，版本号暂按日期和阶段记录。

## [未发布] - 2026-10-15

### 新增

- 增加 `benchmark/px4_gazebo/task_registry.py`，按任务族注册和创建 `TaskPlugin`。
- 增加 `benchmark/px4_gazebo/generic_runner.py`，提供与具体任务族无关的 mock 运行入口。
- 增加 `benchmark/px4_gazebo/tasks/search_handoff/`，包含参数化搜索、交接、复核任务族、manifest 和 smoke test。

### 变更

- 将之前顶层的单文件 `search_verify` 示例调整为任务族目录结构，避免每个任务变体复制 runner、通信和策略接口。
- 长程任务沿用现有 `TaskPlugin`、`CommunicationChannel`、`PolicyContext` 和 episode 记录边界。
- 任务 observation 继续禁止目标位置、目标身份、队友真值位置、全局地图、未来轨迹和 truth 字段。

### 验证

- `python -m benchmark.px4_gazebo.contract_smoke`
- `python -m benchmark.px4_gazebo.model_handoff.self_test`
- `python -m benchmark.px4_gazebo.tasks.search_handoff.self_test`

### 已知限制

- 新任务当前是纯 Python mock，尚未接入 Gazebo/PX4 SITL。
- 通用 runner 当前使用零动作 smoke policy；正式策略比较仍使用现有 Handoff runner。
- episode record 的逐步轨迹记录和公共 evaluator 注册需要下一阶段补齐。
