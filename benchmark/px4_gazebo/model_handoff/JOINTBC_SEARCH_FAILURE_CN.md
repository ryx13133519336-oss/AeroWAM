# JointBC 角色自由搜索：首轮真实闭环失败审计（2026-10-18）

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

本次已有真实模型起飞、决策和落地；搜索未完成。下面是单轮蓝色目标诊断，
不是正式成功率、动态分工或组合迁移成绩。不用规则转向替代模型动作，不扩大预算，
不把这次失败加入训练集。旧 handoff 检查点对新 search 任务属于分布外应用。

## 身份和原始结果

- 原始目录：`D:/AeroWAM-data/wsl_clock_repaired_jointbc_20261018_v2`。
- 权重：`joint_bc_confirm_fix_v1_seed_0.pt`；SHA256
  `d65ba56323d2a1e426e2c810bbaccf5e33f9a0a7c36201b1d37b87ab197f93af`。
- 输入录像：`pilot/condition_00/policy/requests.jsonl`；SHA256
  `5c318852dac883342e0ecd5e72e46d94492aa8c5b71622dd0dfc0149facf3966`。
- 指令原文：“找到蓝色方块，并让另一架无人机确认”；`role_mode=unassigned`，
  nominal 队友响应，两机均可成为发现者，另一架须独立确认。
- 辅助报告器为公开颜色/移动证据规则，既不生成动作，也不代表学习到的报告能力。
- 原始分类保留：`model_failure / policy_inference_error`，错误消息为
  `RGB search local travel budget exceeded`。不回写旧 summary 伪装新分类。
- 219.17 秒受监督会话通过原有 10ms 时钟门；RUN 约 13.06 秒墙钟 / 5.116 秒仿真。
  22 次成功推理最大耗时 84.92ms，未触发 400ms 超时。
- found=0，confirm=0，全部学习意图为 none；正常降落、进程回收均确认。

## 录像和像素核对

| 公开观测指标（22 帧） | UAV0 | UAV1 |
|---|---:|---:|
| 末帧距首帧水平位移 | 0.669m | 2.479m |
| 相对首帧最大绝对转角 | 6.327° | 5.591° |
| 公开报告器有效蓝色框帧数 | 0 | 0 |
| 独立私有标注有效蓝色框帧数 | 0 | 0 |

末帧 UAV1 **有部分蓝色物体露在右边缘**：公开颜色支持 641 像素，范围
`[248,66,256,156]`，触及 256px 画面边界。不能把“没有合格目标框”写成
“没有蓝色像素”或“物体从未进入画面”。私有可见框同样未合格；私有标注
只用于事后核对，从未送入模型。

两机未完成足够转向和完整观察，其中 UAV1 持续平移并接近原有 2.5m 局部预算。
旧 runner 未保存最终被拒绝的请求，因此**不能精确确认触发时哪架越界及越界数值**；
UAV1 的末帧 2.479m 是已保存观测，不能冒充最终失败帧。
动作标签由当前机体 FLU 速度乘 0.5s 生成，推理结果也以同一约定转回速度；
目前未发现坐标符号错误。该检查不等于证明执行时序或曝光遥测已精确对齐。

独立审计产物：`D:/AeroWAM-data/jointbc_search_failure_audit_20261018_v1/`
中的 `audit.json`、`contact_sheet.png`、`replay_comparison.json`。原始证据未覆盖。

## 本次修复与验证

保留 2.5m 上限和原有 found/confirm 门槛。固定/角色自由辅助报告器共用相同
距离检查，超过上限抛出明确的 `SearchTravelBudgetExceeded`，携带双方实际位移、
上限及越界机号。服务器/客户端保留此错误；后续飞行评分仍为 model_failure，
具体原因为 `search_local_travel_budget_exceeded`，不再与神经推理异常混淆。

runner 将失败观测单独保存为 `policy/failed_request.json`，明确 response=null、
action_applied=false；`requests.jsonl` 仍只存已经交付的响应。失败输入不获得
动作、报告或任务进度信用。超时和普通异常不会因相似字符串被误标为距离耗尽。

Windows 与 WSL 各 98 项相关回归通过，含真实 TCP 错误往返、2.5m 边界、双方越界、
角色互换和普通错误区分。冻结权重对原始 22 帧再次推理，22/22 动作逐项完全
一致，最大差值 0；重放最大耗时 30.50ms，无训练、无新飞行。
本次错误记录路径已通过针对性回归，尚未在新一轮真实越界飞行中触发验证。

WSL 重放命令（输出目录必须另取新名字）：

```bash
cd /mnt/d/AeroWAM
/home/ryx/venvs/aerowam-model/bin/python3 -B -u \
  -m benchmark.px4_gazebo.model_handoff.checkpoint_search_probe \
  --checkpoint /mnt/d/AeroWAM-data/checkpoints/joint_bc_confirm_fix_v1_seed_0.pt \
  --requests /mnt/d/AeroWAM-data/wsl_clock_repaired_jointbc_20261018_v2/pilot/condition_00/policy/requests.jsonl \
  --reporter approach --role-mode unassigned \
  --output-dir /mnt/d/AeroWAM-data/jointbc_search_failure_replay_NEW
```

重放使用已经录制的观测/动作历史，不产生反事实轨迹，不能计为第二次闭环成功或失败。

## 后续边界

先保留该负例作为旧 JointBC 的诊断基线，再对已有冻结模型做同权限接入和预算核对，
选择有明确比较意义的下一轮闭环。不会重复运行同一模型期待偶然成功，也不根据此
单例认定某一架必须担任特定角色。后续须分别获得协作技能、候选动作条件预测、
队友行为变化和新任务组合留出的证据；当前静态蓝色搜索不能替代这些结果。
