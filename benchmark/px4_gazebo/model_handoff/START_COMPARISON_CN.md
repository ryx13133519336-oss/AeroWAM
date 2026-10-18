# 启动接收状态离线比较：不认证候选同起点

主线仍是：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和
世界模型预测能力能否组合迁移到新任务。任务不预先指定两机永久角色。
本工具是获得真实候选后果之前的证据检查，不是世界模型成绩或迁移实验。

## 当前功能边界

`start_comparison` 仅读取已保存的 `disarmed-startup-evidence/1-draft` 或 `/2-draft`。
显式选择左右各一个index；不最近邻搜帧、不挑选最接近的一对、不拟合坐标变换、
不重放、不查询当前时钟/服务、不执行模型或仿真。普通搜索/飞行日志不是此格式。

输入两机按原编号对应，不为提高一致性而交换编号；同时交换两份记录的编号时，
诊断结果对称。这里的编号对应不是任务分工，不引入scout/confirmer固定角色。

逐机报告：

- 原始NED坐标的三维位置/速度L2差、角速度分量L2差。
- wxyz四元数的最短旋转测地角，q与-q等价；计算角度时归一化，但另检查范数误差，
  不借归一化修复不合格状态。零四元数、非有限数值和未知NED frame拒绝。
- pose/velocity frame、reset_counter、quality、arming/nav/failsafe/preflight逐项相等性。
  reset不同不能被“位置很接近”掩盖；quality相等也不代表其质量优良。

`all_numeric_checks_passed` 只表示这些原始数组差值在声明范围内；
`all_discrete_checks_passed` 单独报告，不把离散模式相同解释为完整内部状态相同。
不相减不同会话的PX4/host/sim时间戳，也不从相同场景哈希推断共用原点。
各机PX4局部原点和朝向跨会话尚未核实，故位置/姿态/速度差不是经认证的物理误差。
角速度只比较所保存的PX4分量，不声称已校验跨会话机体坐标参照。

以下均不可由此工具填成true：`start_matched`、`physical_state_comparable`、
`common_coordinate_reference_verified`、`formal_benchmark_result`。
输出恒为 `ineligible_start_evidence`，不产生candidate outcomes、utility或评分输入。
即使比较同一帧自身、数值完全相同且时钟通过，仍缺少实际分支和必要状态证据。

## 规格与证据校验

规格schema为 `aerowam-startup-comparison-spec/1-draft`，仅支持
`purpose=exploratory_saved_disarmed_receive_state_diagnostic`。必须显式提供5个非负有限
容差：`position_l2_m`、`velocity_l2_m_s`、`attitude_geodesic_rad`（不超过pi）、
`angular_velocity_l2_rad_s`、`quaternion_norm_error`（小于1）。闭区间包含边界；无隐藏默认值。
输出绑定规格及原始manifest/samples/session-clock SHA256，并记录所选index。
保存后选择的容差不等于采集前冻结的正式起点容差；不能据此作确认性检验。

检查整份capture所有帧的8个artifact字节数/哈希，而非仅检查所选帧；拒绝路径逃逸、
重复JSON key、计数/行号不符、图像间隔不足250仿真毫秒、odometry sample stamp不前进。
目录内解析到外部的链接同样拒绝。输入只读，输出独占创建，不覆盖源文件或旧报告。
哈希检查只证明与提供的索引字节一致，不是来源签名；本比较器不反序列化protobuf/CDR，
也不独立证明JSON状态等于CDR、接收年龄合格或状态在曝光时刻对应。运行时应使用
停止写入的保存目录；不提供对并发修改者的安全隔离。

从原始host读数重算采样窗口时钟质量，保持10ms整段/相邻变化和读取不确定性门槛。
`/1-draft` 的 `complete=true` 只覆盖局部采样，整体时钟质量标为缺失，绝不自动升级。
`/2-draft` 另校验连续clock artifact哈希，重算既有 `session_quality`（含250ms最大间隔、
boot/clocksource、monitor error），检查与保存报告一致、覆盖capture时间范围及boot身份。
即使修改保存的quality flag或重新绑定了原始文件哈希，也不能用假通过摘要跳过重算。
时钟质量不合格仍保留失败，不校正/detrend、不取最佳子区间、不放宽门槛。
缺少被声明的原始artifact等非法/损坏输入非零报错，不输出伪通过结果。

## 2026-10-18离线复核

新增32项合成/CLI回归；连同时钟、启动、候选评分、组合留出及适配器回归，最终
Windows/WSL各210项通过。期间发现的旧Windows busy响应中断另用3项确定性测试
修复，失败记录与修复边界见ADAPTER_RUNTIME_CN.md，不把此前失败计为通过。
不创建真实分支、不收集新样本、不训练、不起飞，也没有暂停/恢复同步服务。

私有 `AeroWAM-data/start_comparison_offline_20261018_v1/` 固定使用全部3组的第0帧：
r1与r2首帧；连续门控记录首帧与自身；旧局部完成记录首帧与自身。后两组是明确的
拒绝逻辑对照，不是两个独立会话，更不是候选重复。容差仅为探索性示例：
位置0.1m、速度0.1m/s、姿态0.05rad、角速度0.1rad/s、四元数范数误差0.0001。
不在看到差值后调整阈值，不以这些阈值制定正式研究结论。

| 已保存输入 | 数值检查 | 额外发现 | 候选起点资格 |
| --- | --- | --- | --- |
| r1/r2首帧 | 未全部通过 | UAV1原始位置数组差约0.100564m；UAV0 reset counter不同；两轮时钟均失败 | 拒绝 |
| `/2`连续门控首帧与自身 | 全通过 | 原始连续记录重算RAW偏移幅度下界16.717833ms，仍大于10ms | 拒绝 |
| `/1`旧局部完成首帧与自身 | 全通过 | 原manifest complete=true不代表整段质量，本工具无内置绑定的连续记录 | 拒绝 |

后一份旧记录另有此前外部连续诊断失败，详见 STARTUP_EVIDENCE_CN.md；本工具不按目录
名字猜测并自动关联那份外部报告，因此本输出标为不可核实，而不是整体通过。
每个capture的64个artifact均检查；Windows复核前后共265个源文件SHA256一致
（4份capture各manifest/samples/64文件，加`/2`连续clock文件），没有反写原始记录。
两端报告不包含本机绝对路径，便于比较内容一致性。
3组Windows/WSL报告分别逐字节一致，SHA256为：

- r1/r2：`d689adef10a52582b1fc7cd1d032d55aec73018d49bb5855c9c40c81a1252a20`。
- 连续门控自身对照：`56252ab681eabf414a7a8a71de7049a8f286b6430e90aa2d538c341eac91b293`。
- 旧局部完成自身对照：`96fd605da5de105b151906812d83d1e77bdfb8e0ff88fd330c210afea8db08ea`。

## 复现与下一关

```bash
python3 -B -m unittest benchmark.px4_gazebo.model_handoff.start_comparison_self_test
python3 -B -m benchmark.px4_gazebo.model_handoff.start_comparison \
  --left /absolute/private/left/startup_evidence --left-index 0 \
  --right /absolute/private/right/startup_evidence --right-index 0 \
  --spec /absolute/private/comparison_spec.json \
  --output /absolute/private/new_comparison.json
```

正常写出拒绝诊断退出2；非法输入/IO错误非零退出，已有输出不覆盖。本CLI不存在“匹配
成功退出0”路径，不能拿进程正常生成报告或纯数值通过作为评分器的start_matched。
现有candidate_scoring仍信任其外部匹配字段，本工具没有自动接入或改变它的语义。

真实候选比较仍需：跨会话逐机坐标参照、曝光状态对齐、同一决策及前缀重放身份、
物理/控制器/估计器内部状态、RNG、队列，以及目标/任务/队友状态。没有完整恢复时，
应独立设计新会话同前缀重放及可比性检查，不能把set_pose当恢复。
正式采集前冻结容差/重试规则，按决策起点与环境seed配对候选、记录随机执行顺序及
session/day/hardware区组；帧数不是独立重复。符合实验设计技能的配对/重复原则，
但这里尚未生成正式执行日程、独立重复或因果/迁移估计。
这些条件满足后再获取真实后果、接预测评分，最终检验角色自由与新任务组合迁移。
