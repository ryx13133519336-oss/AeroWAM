# WAM v2 修正与适用范围

旧基线保留：BC 40/40，VLN 35/40，WAM legacy_v1 0/40。
本次只修正统计一致性，并增加独立的 expert_prior_v2，不覆盖旧权重或结果。

## 身份指标

live metrics 的 identity_error_rate 分母改为 identity_error 加成功的
identity_confirm 事件数，与序列化事件回放一致。之前使用 verify 状态，
导致 validation_012 在确认身份但未完成交接时记录 1.0，回放得到 0.5。
task_success、碰撞规则、安全距离和推理时限都不改变。
repair_metrics 创建独立、完整的结果副本，记录源文件与副本 SHA256 和唯一修改项。
原始实验文件保持不变；副本是指标修正，不是重新跑出的实验。

## expert_prior_v2

- 新 checkpoint 的 model_config 显式包含 wam_revision；旧 checkpoint 默认 legacy_v1。
- 使用四帧视觉、自身状态和动作历史；输入不包含 evaluator 真值。
- 转移训练条件改为记录的专家建议动作，终端样本不伪造下一帧转移；辅助监督对齐下一帧。
- 增加固定的下一帧视觉特征与状态监督，避免仅使用可变化的潜变量目标。
- 规划可在学到的专家动作附近搜索，单轴归一化偏差上限 0.15，候选包含专家动作。
- 当前缓存 collision 全为 0、联合 visibility 全为 1。这些预测头没有证据支持动作排序。
  v2 明确关闭 visibility/collision/handoff 规划奖励，实际使用 imitation_fallback，
  输出学到的动作，不执行无有效目标的随机搜索。checkpoint metadata 记录这一模式。
- 参数预算、训练 50k 上限、相机 100ms 同步和策略 400ms 时限沿用原设置。

v2 是具有辅助转移预测的专家动作先验基线，不是验证过的自主世界模型规划器。
当前记录的专家动作在执行前仍可能经过扰动、限幅，且采样间隔包含中间控制；
所以新转移预测仍只是观测数据上的代理目标，不能声称是精确动作后果模型。
真正的自主规划与组合迁移需实际执行动作序列、持续时间、不同安全/可见性样本、
独立的预测检验及新任务/队友/通信条件。一次标签修正无法补足这些信息。

## 重新训练（新输出）

```bash
python3 -B -u -m benchmark.px4_gazebo.model_handoff.train \
  --method dual_wam_mpc --wam-revision expert_prior_v2 \
  --seed 0 --steps 50000 --batch-size 32 --device cuda \
  --train-cache /mnt/d/AeroWAM-data/features/handoff_train_confirm_fix_v1.pt \
  --validation-cache /mnt/d/AeroWAM-data/features/handoff_validation_confirm_fix_v1.pt \
  --output /mnt/d/AeroWAM-data/checkpoints/dual_wam_mpc_expert_prior_v2_seed_0.pt
```

训练前禁止覆盖已有 checkpoint。接入检查先做小规模 validation smoke，不因模型失败重复挑选成功。
配置文件 runs.wam_expert_prior_v2.json 指向独立 checkpoint；验证结果也必须使用新目录。
接口注册与外部模型适配仍按 TODO_CN.md 的下一阶段推进。
