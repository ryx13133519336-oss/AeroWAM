# Warehouse v2 变更记录（更新于 2026-10-22）

目前提供任务与评分接口、合成轨迹诊断和真实公开观测的转换函数。尚未完成 SmolVLM 的真实仓库预测/协作闭环，也没有正式榜单成绩或学习迁移结论。

## 同学需要同步的接口变化

- core schema 未修改。
- task `warehouse_coop_prediction` 的 `task_version` 升到 `2`，evaluator ID 为 `warehouse_coop_prediction_evaluator_v2`。
- 离线数据集 manifest 升到 `aerowam-warehouse-v2-dataset/3`。`episodes.jsonl` 只有公开索引；`model_inputs.jsonl` 只含公开输入；`private_truth/` 只给可信 evaluator。合成观测 `rgb=null`，明确没有图像，不能送给 VLM 作为真实 RGB。
- 评分输出升到 `aerowam-warehouse-v2-score/2`。预测必须覆盖全部预期样本和三个 horizon；缺失样本时主要预测指标为 null、交接不能成功。重复/越界样本、非有限数和非法概率被拒绝；无请求的确认也计为错误。评分不再依赖是否导出训练监督。
- live adapter 是函数 API：`to_model_input(observation, task_context=...)`、`from_wire_to_model_input(payload, task_context=...)` 和 `decode_rgb(item)`，没有 `WarehouseV2SceneAdapter` 类。adapter ID 升到 `warehouse_v2_live_scene_adapter_v2`。上下文只允许固定 horizons、0..8 消息预算和未分配角色；拒绝未来曝光、同帧报告、未知潜变量字段。函数本身不做跨调用的 episode/单调时序审计，这仍由运行时负责。

## 已修复的诊断问题

加减速轨迹不再在分段边界瞬移，反向分支能够离开盲区；队友延迟使用实际模拟送达队列并保留报告曝光时间；CV baseline 在遮挡期按已经流逝的时间外推。校准误差使用预测类别的置信度。关闭监督时不生成私有未来监督，但可信评分仍可用单独 truth 重放。

## 真实仓库执行器变化

仓库保留 AWS 背景，采用八个物理遮挡体、南侧可见—双机盲区—北侧可见的移动目标路线。保守的递归 box 证明覆盖整个目标和相机不确定区域；采集要求所有遮挡体的实际根位姿与审计几何一致，缺失、重复、移位或旋转均拒绝。像素证据与完整遮挡证明冲突时拒绝。仓库 handoff evaluator v2 允许来源机先跟踪/请求、接收机在来源机被物理遮挡后重现目标并接受，再保持至少一秒有效报告；旧的共享可见接受仍支持。

所有旧场景哈希的 clearance/preflight/contact/flight 准入均需重新生成。不得用旧的 v29/v30 归档授予新布局飞行准入。物理飞行诊断的往返方向移向出生安全区内部，原安全门不变。

## 实现范围和证据边界

代码声明了6个布局族、5个运动族和5个队友族，但默认导出只使用五个固定组合；目前布局只是合成可见区间，没有六套经真实 clearance 的 Gazebo 布局。中央 baseline 可以直接读取双机公开视图，不是通信受限的分布式协作实验；队友消息也尚未构成可控的真实队友执行策略。不能据此声称训练或组合迁移有效。

当前数据盘 `warehouse_v2_dataset_20261022_v2`、`warehouse_v2_baseline_20261022_v2` 为合成诊断，`model_score_eligible=false`、`flight_admitted=false`。旧数据和报告保持原样，比较时必须核对 task/dataset/score 版本。

## 同步和检查

同学同步本次发布分支的 v2 Python 文件及本文、协议后重新生成数据；core 不迁移。运行 warehouse v2 三组 self-test 和 `python -m benchmark.px4_gazebo.model_handoff.core.core_self_test`。完整真实执行器另需匹配的仓库场景文件、安全门、WSL 依赖和新准入证据，单独下载 v2 接口不能视为已具备飞行能力。
