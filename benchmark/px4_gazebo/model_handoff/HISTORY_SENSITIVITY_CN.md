# 冻结 JointBC 的历史图像敏感性诊断

主线：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。
本步骤承接 [SEARCH_HISTORY_CN.md](SEARCH_HISTORY_CN.md) 的真实渲染配对，检查已有模型
的计算是否受历史图像影响。它不替代真实决策/后果预测，不是左右问答正确率或闭环成功率。

## 为什么不直接给旧模型算左右准确率

现有 JointBC 只有双机四维动作、intent 和 phase 输出，没有 left/right 分类或语言答案头。
不能把某个动作符号、intent 或规则颜色识别器临时映射成正确答案后，冒称原模型会回答。
本诊断不添加/训练分类头，不更换权重，不把规则动作当作模型动作，不运行飞行。

## 运行前固定的对照

只读本地已安装的 `joint_bc_confirm_fix_v1_seed_0.pt`，复用既有 OpenCLIP 编码器及模型。
不下载外部模型，HF_HUB_OFFLINE=1，不读 Gazebo 世界、实例图、私有答案或 inspection。
直接从每个 episode 的四个问题 × 两模式共16个公开 JSON 读取 RGB 和问题。
严格检查字段、双相机、256×256 RGB PNG、历史/当前非图像字段相同、配对当前输入相同。
公开 JSON 的 schema 不直接冒充 ModelObservation；这里只把公开像素/文本编码送入模型计算。

三种输入条件：

| 条件 | 历史槽 | 当前槽 |
| --- | --- | --- |
| current_only | 重复当前 hidden 图像 | 当前 hidden |
| with_history | 本 episode 最后一组 seen 图像 | 相同当前 hidden |
| other_history | 另一个 episode 最后一组 seen 图像 | 相同当前 hidden |

历史槽按已有服务首次样本左侧补齐语义构造：h=4 时为 [seen, seen, seen, hidden]；
无历史则为 [hidden, hidden, hidden, hidden]。这是有序快照消融，不是假造四帧连续飞行。
对每个问题文本固定；两机 state 与 previous_action 全部设零，明确是未测量/未执行的占位，
不用不同姿态或上次模型动作掩盖图像历史差异。这些状态及问答指令对旧交接模型存在分布变化。

四问题 × 两 episode × 三模式 × 两次数值重复，共48次纯前向计算，seed=20261018随机
顺序在初始化模型前写入 design.json。不使用跨调用的 policy.history；同一模型 eval()、
inference_mode()、确定性算法；直接观察 encode() 的 latent 和原 heads 输出。
不调用 MPC，不加 RGB 辅助报告器，不提交动作到任何仿真/控制器。

每条输入重复两次作为数值稳定性负对照，不能作为独立实验重复。固定 atol=1e-6、
rtol=1e-5；相同输入重复及两个 current_only 布局输出必须在容差内一致，互换历史
必须复现另一布局的 with_history 输出。失败保留所有已完成记录，不自动重试。
差异指标为 max_abs 与 L2；超出数值容差只表示敏感，不表示语义正确或有用记忆。
同时记录原始图像/预处理张量/当前视觉特征指纹、历史特征差异、模型配置、实际加载
编码器 state_dict 指纹、权重及公开输入运行前后 SHA256。重复像素的特征缓存不引入私有信息。

这里仍只有一个场景组。四个问题和两次数值重复不能当作独立场景、多seed或模型能力样本量。
延迟不采用原飞行400ms门槛：不测试TCP、在线采样或闭环预算，不据此声称首次超时已修复。

## 复现

使用已有模型 venv；输出必须是新目录。HF_HOME 指向既有缓存，不下载/安装：

```bash
export HF_HOME=/mnt/d/AeroWAM-data/cache/huggingface
export HF_HUB_OFFLINE=1
/home/ryx/venvs/aerowam-model/bin/python3 -B -u \
  -m benchmark.px4_gazebo.model_handoff.history_sensitivity \
  --public-dir /mnt/d/AeroWAM-data/search_history_pair_20261018_v1 \
  --checkpoint /mnt/d/AeroWAM-data/checkpoints/joint_bc_confirm_fix_v1_seed_0.pt \
  --output-dir /mnt/d/AeroWAM-data/history_sensitivity_jointbc_new --device cuda

/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.history_sensitivity \
  --output-dir /mnt/d/AeroWAM-data/history_sensitivity_jointbc_new --inspect-only
```

离线检查只用 numpy/Pillow，不加载模型，核对48条响应的完整顺序和差异/数值控制。
产物留在本地，不提交权重、输入图像或完整 latent。后续要单独接入有声明答案/预测
输出的模型及独立验收；不要从敏感性推导判断正确、队友适应或组合迁移成功。

## 2026-10-18 首次 JointBC 结果

本地目录 `D:/AeroWAM-data/history_sensitivity_jointbc_20261018_v1`，一次加载、48次前向
完成，没有重跑推理或训练。checkpoint SHA256 为
`d65ba56323d2a1e426e2c810bbaccf5e33f9a0a7c36201b1d37b87ab197f93af`，
实际 OpenCLIP state_dict SHA256 为
`23d600b99b3ccb9624c82150e8e48fd95c78da21be8d8a918a411fadf303718d`。
PyTorch 2.7.0+cu128、CUDA；权重及16个公开输入运行前后未变。

负对照通过：重复相同输入稳定，两个 episode 的 current_only 输出相同；other_history
复现对侧 with_history。四个问题中，过去布局差异均经过实际编码器保留，双机历史视觉
特征 L2 差异为0.384144。每个问题的真实/替换历史比较在两个 episode 对称，结果如下：

| 问题 | 替换过去图像后的动作最大绝对差 | latent L2 差 |
| --- | --- | --- |
| uav_0 红 | 0.032256 | 3.197760 |
| uav_0 蓝 | 0.030999 | 3.149060 |
| uav_1 红 | 0.032557 | 3.208474 |
| uav_1 蓝 | 0.031259 | 3.157019 |

动作差是8维归一化模型动作中最大的分量，不是实测米数、角度或实际飞行变化。
with_history 对 current_only 的动作最大分量差为0.179593–0.227484；intent/phase logits
也变化，但没有把这些变化映射为左右答案或成功率。只能说该 checkpoint 的输出在
这组受控离线输入上对过去图像敏感，不是记忆正确/决策更好或迁移成功。

原始输出在 WSL 和 Windows 均离线复查通过。首次 Windows 重算因 BLAS 浮点范数
最后几位不同而被字典精确比较误拒绝；检查器对已算出的浮点差异指标采用1e-12
绝对/相对重算容差，数值控制原定的1e-6/1e-5门槛、真假标志、响应顺序与原始
模型结果均未改变，也没有重跑模型挑选结果。新增回归拒绝1e-8指标变化和标志改变。
Windows/WSL各156项相关回归通过（含13项新增敏感性/公开边界/完整性测试）。

后续重点是匹配可独立评分的语义判断或候选后果输出，而不是再扩大量重复前向。
旧动作模型缺少左右答案头，此处不报告准确率；也没有解决旧400ms首次请求问题。
