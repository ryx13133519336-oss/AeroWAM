# 不预分配角色的模型接入边界（2026-10-18）

## 不变的主线

在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

任务不将 UAV0/UAV1 永久指定为发现者、确认者、高空规划者或低空执行者；
也不将高度直接等同于角色。协作中可以形成分工、转移职责；是否选择固定分工
是参评方法的策略属性，而不是 benchmark 给出的答案或评分特权。
本次改动只实现角色自由的离线任务/验收入口，不声称动态分工已学会。

## 已实现的任务与验收入口

新任务使用 `role_free_task(...)`，公开 schema 为
`aerowam-language-task/2-draft`，`role_mode=unassigned`，无预分配 `roles`。
人工指令原文保留，目标由独立私有标注绑定；不解析指令来产生答案。
模板不指定机号；人工指令仍需审查，工具不会自动识别其中隐含的固定分工。
配对目标调用 `paired_target_tasks(..., role_mode="unassigned")`。

搜索中，任一架的有效 found 可以建立此次发现证据；另一架必须独立 confirm。
交接中，任一架完成连续跟踪后都可 request，另一架 accept 后须重新累计跟踪并 confirm。
普通 observe 不锁定角色。巡检不指定哪架负责哪一视角，但仍要求两架均贡献、
覆盖互补视角并分别确认。单机完成两份证据不能伪装双机协作。
评分记录 `evidence_roles` 只是此次发现/交接的事后归属，不是永久分工，也不回传策略。

角色自由不等于忽略事件职责：同一次交接中的请求方不能接受自己的交接，
发现者不能自己完成“另一架确认”。当前状态机支持一次发现/交接的验收，
不是任意多轮角色轮换或完整协作调度器。

旧 `LanguageTask` 默认值、`templated_task` 和旧 search_pilot 保留 /1 固定角色兼容，
仅供已有诊断复现，不能将旧结果重新标成角色自由 benchmark。
旧 MotionSearchGate 与规则搜索策略仍含固定机号；SearchEpisode 构造时拒绝
给 /2 任务挂载旧运动评分门，避免静默按旧规则评分。尚未迁移实际飞行 runner。

## AeroDuo 来源与适配审计

本地代码：`benchmark/third_party/aeroduo`；只读检查，未改动第三方源代码。
本地 commit：`e19fd01e8c6b33c39b4ce0b24f23aadd7ff8698a`。
官方代码：[buaa-colalab/AeroDuo](https://github.com/buaa-colalab/AeroDuo)。
论文标识：[2508.15232](https://arxiv.org/abs/2508.15232)，由官方 README 链接；
前一轮 arXiv API 查询超时、重试 429，本次不将其写为已成功核对论文全文。

代码事实和对本项目的含义：

| 位置 | 代码事实 | 接入边界 |
|---|---|---|
| README | 原方法采用 dual-altitude 分工 | 可保留为方法自身的固定角色局限，不写入公共任务要求 |
| pilot_llm/api_for_airsim.py: CallPilot | 需要 BEV RGB、深度、点云和 high/low UAV locations | 当前前视 RGB + 各机独立局部状态不能无损提供这些输入 |
| pilot_llm/models/pilot_llm.py | 自定义空间 mask head，推理返回概率图 logits | 不是现成双机动作接口；不能以通用 Qwen 文本输出来冒充官方 PilotLLM |
| api_for_airsim.py: postprocess_llm_output | 深度/坐标图参与占据掩码、概率图到航点投影；默认还使用 A* | 任何替换均属适配方法的一部分，必须披露且无真值 |
| full_eval.py: prepare_llm_inputs | 把 target_positions 传给 CallPilot，但该版 CallPilot 未使用此参数 | 不把仅有形参误报成神经网络读取真值；本项目适配接口仍不得含该字段 |
| full_eval.py: 主循环 | drone2_waypoint = traj_status.target_positions，随后交给 makeActions | 不能直接复用此整机闭环路径作为本项目无目标真值基线 |

上述检查针对固定版本的执行路径，不对原论文整个实验设置作超出证据的推断。
交换 UAV 编号只能检查评分和接口偏置；不能靠一次编号交换，把固定高低空方法
变成已学会动态角色切换的模型。不强求所有方法的输出严格置换等变。

当前结论：AeroDuo 是可保留的固定分工外部参考候选，**尚不具备直接接入资格**。
不为迁就它新增只有它能读取的深度、真值地图、目标坐标或全局状态。
若将来设立 RGB-D/BEV 扩展轨，必须给该轨全部方法同等权限、另列结果，
不能与当前 RGB 权限轨混排。若只复用空间预测组件，则明确命名为组件适配，
不宣称复现完整 AeroDuo，也不冒称候选动作条件世界模型。

## 权重预检（只获取元数据，未下载/运行）

2026-10-18 查询 Hugging Face 官方 API 的 `?blobs=true`：

- 基座 `Qwen/Qwen2-VL-2B-Instruct`，revision
  `895c3a49bc3fa70a340399125c650a463535e71c`；模型卡 license=apache-2.0。
  两个 safetensors 分片合计 4,418,050,768 字节。
- 任务权重 `salome1023/AeroDuo-PilotLLM`，revision
  `1614a43e584e975bd2a7832d5c467f0e4846b11c`；模型卡未声明 license，不能由代码
  Apache-2.0 自动推定任务权重许可证。
  adapter_model.safetensors 为 18,516,456 字节，但官方 CallPilot 还加载
  `pytorch_model/mp_rank_00_model_states.pt`，为 8,881,300,727 字节。
- 因此按官方 loader 不能只下载约 18 MB LoRA；上述主要文件合计约 13.32 GB，
  尚不包括 tokenizer/配置、运行依赖、缓存与内存。optimizer states 不用于推理。
- 该 revision 的 adapter_config 为 LoRA r=8、alpha=8，未声明 modules_to_save；
  不能据此假设自定义 mask head 只靠 LoRA 就能恢复。
- 已检查的本地 HF hub 缓存仅见 CLIP，未发现该基座与任务权重。

可复查端点：
`https://huggingface.co/api/models/Qwen/Qwen2-VL-2B-Instruct?blobs=true`、
`https://huggingface.co/api/models/salome1023/AeroDuo-PilotLLM?blobs=true`、
任务仓库该 revision 的 `adapter_config.json`。
这是文件元数据审计，不是权重完整性、运行兼容性或模型能力验证。

## 下一步门槛

先完成同一 RGB 权限下首个真实模型的适配可行性选择；不强行把 AeroDuo
列为必须先跑通的主模型。公开输入、原生输出、动作/报告转换及各模块来源要列清。
通过权限与角色边界后，才准备权重、隔离依赖环境，记录实际模型输出、时延和失败。
新权重下载、依赖环境和飞行有额外资源/权限要求时再请求，不在现有环境里静默替换。

首个模型接入不等于主线完成。后续仍须有候选动作条件未来预测与独立评分，
以及预先冻结的任务/可观测性/队友行为组合留出；不以静态目标成功替代迁移结果。
不将命令减速/延后代理条件冒充独立队友策略，不将动作变化冒充预测准确。
正式测试不用于调提示、角色分配规则或评分阈值；未来组合按原有设计规范分组隔离。

## 本次验证边界

`role_free_self_test.py` 新增 21 项合成回归：两方向发现/交接、巡检视角互换、
同机自确认拒绝、遮挡打断 dwell、错误目标、旧版本兼容、配对目标及协调器集成。
总相关回归为 189 项；另运行 interface_self_test 的真实本地回环接口自检。
不调用真实外部模型，不启动 Gazebo/PX4，不训练，不报告外部模型成功率。
这些检查不能证明真实像素识别、移动安全、队友策略适应或组合迁移。
