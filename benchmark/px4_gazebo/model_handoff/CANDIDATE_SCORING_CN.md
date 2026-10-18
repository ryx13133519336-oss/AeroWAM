# 候选条件预测的独立离线评分器

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

`candidate_scoring.py` 实现既有 DECISION_BENCHMARK_CN.md 草案中的一部分决策指标，
不生成预测、执行动作、生成私有答案或重新定义正式实验。模型预测与私有分支结果分文件。
当前没有真实配对分支成绩；下文验证全部明确标注为合成 fixture，不是模型性能。

## 输入与信息隔离

1. `request.json`：原候选协议 `DecisionRequest`，含公开历史、原文指令、动作候选、窗口。
2. `spec.json`：显式提供任务、独立 utility 定义/版本、计划 repeat IDs、utility 平局 epsilon、
   top-k 和预测碰撞风险筛选阈值（可为null）。本工具不指定正式重复数、权重或阈值，
   不从预测或私有结果反推/调节这些设置。正式定义需在数据采集/预测前独立冻结。
3. `forecast.json`：harness 用 `forecast_record(...)` 包装 `CandidatePrediction`，
   绑定 request/spec SHA256、方法和声明的预测来源，逐候选覆盖且不重复。不读私有结果。
4. `evaluator/outcomes.json`：同一 request/spec 下每个 candidate × repeat 的私有结果，
   含 paired_start_id、外部起点匹配结果、状态、窗口末逐机可见性、窗口内事件、
   独立 evaluator 算出的 utility 和唯一 evidence_ref。不允许复用一条引用代替多个分支。

无预分配角色，无 source/receiver 机号。当前评分入口只接 `intent=none` 的双机动作候选，
拒绝旧 intent→固定发送方向映射。显式角色自由交接技能候选/执行器仍需后续接线。
私有标签只进评分器，不回填到 forecast 或策略请求；一个普通动作标签不构成预测。

SHA256绑定检查数据一致性，**不能证明预测在答案之前生成**，也不是签名或访问隔离。
`source_kind`、起点匹配结论和 evidence_ref 均来自可信 harness/evaluator；本评分器不读取
引用证据文件、不重放仿真、不认证原点/速度/估计器/控制器状态或模型身份。
真实运行仍须独立验证匹配证据、utility实现/版本、时间窗口和写入顺序。

## 完整性与失败处理

- 必须覆盖预先声明的全部候选和重复；同一次重复的 paired_start_id 一致，
  所有外部起点匹配标志为true。缺失、不匹配、utility不可用或基础设施故障，
  整个决策返回 `ineligible_evidence`、metrics=null，不只评分成功/存活候选。
- `simulator_error` / `evaluator_error` 的标签与utility必须为null，不能填0冒充能力失败。
  失败仍留在原始分支文件中；此工具不删数据、不自动重试或填补。
- `safety_stop` 可保留外部冻结utility定义给出的负收益，因此不从排序中删除。
  提前终止不能编造窗口末可见性或未观察到的负事件：末帧标签须为null，
  窗口事件只允许确实观察到的true或未知null；不能把安全终止本身认定为物理碰撞。
- 任一分支窗口删失时，整个决策的概率评分暂不提供（`horizon_censored`）。
  不将缺失标签填false，也不只取提前观察到的正例做有偏的概率评分。
  完整窗口标签必须齐全。后续如要用生存/删失方法需另行设计，不由此工具推断。
- 非交接任务的handoff标签为null、预测头为0且不评分；inspection的confirmation同样
  不适用。inspection成功须由其独立视角验收完成，不能用确认概率冒充巡检成功。
- 风险筛选后所有候选被拒绝则明确弃权；selected/regret/best-hit为null，
  不擅自执行hover、不虚构弃权收益，也不把弃权算作零regret。

## 已实现指标及范围

- 先对同一候选的计划重复utility取均值，再算 utility MAE/RMSE、epsilon近似最优集合、
  选中是否在集合中、`max实际均值－选中实际均值` regret。
  regret参照全部候选的实际utility，不是风险可行oracle；不能据此否定合理的保守选择。
  epsilon影响最优集合/可区分对，不把原始utility差直接归零。
- 同决策点内可区分候选对的排序准确率；实际差≤epsilon不评分该对，预测精确平局计0.5。
  没有可区分对时为null，不填100%。top-k按预测utility排序、稳定平局按请求顺序，
  报告是否覆盖至少一个实际近似最优候选；不是最优集合的集合召回比例。
  top-k指标不应用风险过滤，实际选择则使用原 `select(...)` 的可选风险过滤。
- Brier分别评估两机的窗口末可见性，及适用的窗口内handoff/confirmation/collision。
  对每条完整分支计算 `(预测概率－bool结果)^2` 再平均，**不是**对平均标签的平方误差。
  同时报告预测均值、观察频率和分支观测数；不把相机/帧/分支数当独立样本量。

这里只输出单决策点离线度量，不汇总跨场景独立统计、不做显著性/置信区间、
Spearman、log loss或可靠性分箱，也不提供闭环成功、安全认证或组合迁移结论。
所有候选输出相同的退化基线可被如实评分；格式通过不证明使用了候选条件。
需要对真实模型做候选/历史干预和实际后果验证，不能由评分器强制制造差异。

## 可复现的合成验收

```bash
python3 -B -m unittest benchmark.px4_gazebo.model_handoff.candidate_scoring_self_test
python3 -B -m benchmark.px4_gazebo.model_handoff.candidate_scoring_self_test \
  --write-synthetic-demo /absolute/new/demo_directory
python3 -B -m benchmark.px4_gazebo.model_handoff.candidate_scoring \
  --request /absolute/new/demo_directory/public/request.json \
  --spec /absolute/new/demo_directory/public/spec.json \
  --forecast /absolute/new/demo_directory/public/forecast.json \
  --private-outcomes /absolute/new/demo_directory/evaluator/outcomes.json \
  --output /absolute/new/demo_directory/score.json
```

输入只读；输出独占创建，已有文件不覆盖。成功评分退出0，不足证据保存状态后退出2，
契约/标签非法则报错，不生成伪成绩。JSON重复key拒绝。

2026-10-18验证：新增23项手算/边界/CLI测试，两端相关236项本地回归通过，
另原interface_self_test通过。回归总数含未随本工具发布的Qwen/CLIP本地诊断；
新评分器与23项测试不依赖那些模块或模型库。
剔除这些未发布模块后，独立的200项基础+评分回归在Windows/WSL也均通过。
`AeroWAM-data/candidate_scoring_synthetic_20261018_v1` 保存显式合成输入和两端独立CLI结果：
正常排序regret=0，故意反排regret=1；camera Brier各0.0625、confirmation 0.04、collision 0.025；
删除一个候选重复后整组metrics=null、退出2。不是Qwen或其他实际模型的成绩。

下一关是获得并验证真实候选分支起点匹配与后果，再接真实预测器评分。
当前单次搜索日志、静态遮挡图像和VLM动作菜单输出都不能替代这组分支。
部分可观测、独立队友策略变化、组合留出及角色自由闭环仍未由本工具验证。
