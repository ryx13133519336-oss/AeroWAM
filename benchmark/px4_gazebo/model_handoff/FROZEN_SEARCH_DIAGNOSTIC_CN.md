# 冻结模型搜索闭环诊断（2026-10-19，运行前约束）

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

本轮只补齐已有模型的真实搜索行为诊断，不训练，不调评分，不改变角色自由要求。
采用 experimental-design 的整次任务为单位、事先固定运行顺序和保留全部失败原则。
当前仅一个蓝色目标、静态开放场景、nominal队友条件，不能估计组合迁移或队友适应。

## 运行前固定

1. `dual_vln_hier_confirm_fix_v1_seed_0.pt`，SHA256
   `e6cc49b672f2c87453147d4931c21bfd3b8deb1ecf0ff1ede25698a409ada8e5`，一次。
2. `dual_wam_mpc_confirm_fix_v1_seed_0.pt`，SHA256
   `53119eae2f847b1b7b48036b608584a89fb511129c22abdefd9a50dd093780e6`，一次。

顺序由Python `random.Random(20261019).shuffle`对上述两个方法的初始列表生成；
这是两轮探索性诊断的顺序预定，不是充分平衡的随机区组试验。
两次均 `policy_seed=0`（torch与随机候选规划），与训练seed、场景seed分开记录。
原JointBC是前一天的历史诊断，未固定规划seed，不当作本日同期随机对照。
同种子不保证跨平台位级一致，也不消除仿真启动/物理执行的随机性。

每次重建独立Gazebo/PX4，固定原文“找到蓝色方块，并让另一架无人机确认”，
私有正确目标blue，`checkpoint_search --role-mode unassigned`，无机号角色指定。
动作只来自检查点；事件来自同一公开辅助报告器，私有验收不回传策略。
RUN 90秒墙钟、启动150秒、推理400ms、时钟10ms、局部位移2.5m门槛不变。
每轮使用已经验证的临时WSL隔离/恢复入口，不改Windows服务或持久配置。

任何任务失败如实计入，不能以失败为由自动重跑；若发生启动/时钟/恢复失败，
停止余下队列，先核对环境。两轮都不晋升训练数据，不把连续观测帧当独立重复。
不以其中一轮偶然成功宣称模型优越，不从该诊断推断正式任务成功率。

## 预先指定的记录项

整轮category/reason，found/confirm独立验收，实际决策数，正常降落及进程清理，
时钟全程质量/双层恢复，检查点前后哈希，成功推理耗时，以及失败请求和实际位移。
若搜索未成，区分未观察、未接近、未稳定、未协作确认；不只报告末尾降落日志。
候选动作条件预测准确率没有独立真值，本轮仍不报告该指标。

## 运行入口

```powershell
D:/anaconda3/python.exe -X utf8 -B -m benchmark.px4_gazebo.model_handoff.wsl_clock_session `
  --mode pilot --method dual_vln_hier --policy-seed 0 `
  --output-dir D:/AeroWAM-data/frozen_dualvln_blue_20261019_v1 --execute
```

第二轮方法改为`dual_wam_mpc`，输出目录为
`D:/AeroWAM-data/frozen_dualwam_blue_20261019_v1`。每个目录只能创建一次。

以上为运行前协议。实际结果如下，保留未执行和失败，不补跑替换。

## 本轮实际结果

| 方法 | 执行结果 | 发现/确认 | 后续处理 |
|---|---|---|---|
| DualVLN-Hier | 真实起飞，14次决策，model_failure / search_local_travel_budget_exceeded | 0 / 0 | 正常降落、进程清理、校时恢复通过 |
| DualWAM-MPC | 环境前置拒绝，模型监督器未启动，未飞行 | 未评估 | 校时恢复通过，按协议停止，不重试 |

DualVLN的RUN约8.442秒墙钟 / 3.432秒仿真，最大推理43.680ms，权重未变。
含启动至清理的120.620秒会话通过原10ms门槛，2406个时钟样本；原始目录
`D:/AeroWAM-data/frozen_dualvln_blue_20261019_v1`。
最终拒绝观测保存在`pilot/condition_00/policy/failed_request.json`，
UAV0/UAV1距起点0.269/2.508m；拒绝请求未下发。两机最大相对转角约0.304/4.903°。
公开颜色框有一次有效蓝色框，但没有一次满足居中报告要求；私有已交付帧标注
没有有效蓝色框。二者不是相同可见性指标，不能相互替代。图像显示末段蓝物体
已出现于右侧，不能将失败写成“物体从未可见”。末次拒绝输入没有对应私有标注。

DualWAM目录`D:/AeroWAM-data/frozen_dualwam_blue_20261019_v1`。
Ubuntu包装器在读回offset=1908us、frequency=0、tick=10000、status=0时拒绝，
原始错误为`AssertionError:`，未开始20秒预检，也没有实际模型或任务成绩。
仅靠该字段不能推断已超过10ms质量门槛；本次没有测量该门槛。上游Linux6.18
的`ntp_update_offset`在STA_PLL未开启时直接返回，说明不能无条件假设
ADJ_OFFSET=0请求一定把读数清零；当前未改变状态位、未追加系统修正。
源码只读核对地址：
`https://kernel.googlesource.com/pub/scm/linux/kernel/git/torvalds/linux/+/refs/tags/v6.18/kernel/time/ntp.c`。
WSL具体内核是否另有差异、残留修正如何在既定权限范围内消除，仍待单独验证。

后续代码仅把此空白断言改为含实际读数的RuntimeError，拒绝条件原样保留；
入口报告新增clock_quality_evaluated/task_evaluated，未执行时pass/success为null。
旧会话文件不改写；这是报告修复，尚未再次执行提权会话验证。

独立分析：`D:/AeroWAM-data/frozen_search_diagnostic_audit_20261019_v1/summary.json`
及`dualvln_contact_sheet.png`。两轮均根层恢复确认、Ubuntu服务active，检查无仿真残留。
本轮不是完成了三方法闭环比较，不能把DualWAM的环境拒绝计为模型任务失败。

模型选择/种子/报告接线在Windows、WSL各104项回归通过；DualWAM相同种子
三帧离线重放两次响应完全一致（`dualwam_seeded_precheck_20261019_v1/v2`），
这验证种子实现，不是两个任务重复或预测准确率。本轮没有训练和门限变更。
