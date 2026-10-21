# Warehouse v2 变更记录（2026-10-21）

本次变更完成了第一个符合“部分可观测 + 队友行为变化 + 预测/协作迁移”主线的 warehouse 任务族离线闭环。

## 版本与接口

- core schema：未修改。`benchmark/px4_gazebo/model_handoff/core/` 不需要迁移。
- task：`warehouse_coop_prediction`，`task_version=1`。
- scene：`aws_warehouse_handoff_v2`，`scene_version=2`。
- 公开输入仍使用 core `ModelInput`；私有 future supervision 和 truth 只给可信 evaluator/训练分支。
- 新增公开 baseline 报告接口：`aerowam-warehouse-v2-baseline-report/1`。

## 已修改/新增文件

- `warehouse_v2_task.py`：任务生成、五 split、评分器；修复危险请求判定，使“源机不可见且接收机可见”是安全请求条件；补齐 6 个布局、5 个运动族和 5 个队友行为族。
- `warehouse_v2_task.py`：数据集导出 schema 升到 `aerowam-warehouse-v2-dataset/2`；公开 `episodes.jsonl` 现在只有 episode index，完整 future supervision 不再随公开索引发布。
- `warehouse_v2_scene_adapter.py`：新增 `WarehouseV2SceneAdapter`，将现有 `ModelObservation` 公开输入映射到 core `PublicObservation` 和 `ModelInput`；严格检查 episode、时间戳、消息预算和 truth-free 字段。
- `warehouse_v2_scene_adapter_self_test.py`：新增 adapter 边界回归测试。
- `warehouse_v2_task_self_test.py`：修复公开边界断言并增加潜变量族回归测试。
- `warehouse_v2_baseline.py`：只消费公开 `ModelInput` 的 constant-velocity / 保守交接 baseline 和五 split 汇总。
- `warehouse_v2_baseline_self_test.py`：baseline 和报告产物自测。
- `WAREHOUSE_COOP_PREDICTION_PROTOCOL_V2_CN.md`：记录当前实现和运行入口。

## 队友同步

请重新下载/同步包含上述文件的仓库版本，并重新生成 v2 数据集（旧 `warehouse_v2_dataset_20261021_v1` 的 `episodes.jsonl` 仍是旧格式，建议删除后重建）。若只想运行现有 core adapter，不需要任何 core schema 迁移；若要运行 warehouse v2，必须同步 `warehouse_v2_task.py`、`warehouse_v2_baseline.py`、`warehouse_v2_scene_adapter.py` 及对应 self-test 和协议/变更记录。

## 证据边界

`D:\AeroWAM-data\warehouse_v2_dataset_20261021_v1` 和 `warehouse_v2_baseline_20261021_v1` 是可查看的离线 synthetic 诊断，不能作为 PX4 飞行证据或正式模型榜单成绩。Gazebo/PX4 scene adapter、独立遮挡审计和 SmolVLM 接入仍是下一阶段。
