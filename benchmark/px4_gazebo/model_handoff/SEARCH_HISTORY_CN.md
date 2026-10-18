# 当前相同、历史不同：信息必要性诊断

主线仍是：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。
本步骤只验证其中一个前提：任务能否确实要求历史，而不是仅看当前画面就能回答。
它是独立的 RGB 历史判断协议，不是飞行任务、现有 ModelObservation 接口或世界模型预测评测。

## 采集前固定的设计

一个场景组，两个布局，各采集一次。布局 normal 保持红球 ENU(4,1,2.4)、蓝块
ENU(4,-1,2.4)；swapped 只互换两个目标的位置。身份、形状、纹理、实例标签、墙、
光照、相机内参及 seen → hidden → reacquired 的视点脚本均不变。
复用同世界时序采集器，每个布局有独立服务器/随机 partition；失败保留，不自动重试。
默认 seed=20261018，以 Python Random(seed).shuffle 固定两布局顺序；启动任何服务器前
写入 design.json。场景 XML 去掉两个目标 pose 后必须完全一致。

每个布局在每个阶段接受所有满足时序门槛的采样帧，不按答案或像素相似度挑选。
两个布局的全部 hidden 帧，分别按相机，在原生 RGB 与 256×256 RGB 上逐像素一致，
才允许通过；一像素不同也失败，不使用近似阈值，不替换或合成遮挡图片。
沿用实例图零目标像素、墙可见、稳定期、同戳双相机及位姿回读检查。

四个配对问题为两路相机 × 红/蓝目标：在该相机上次同时看见两个物体的画面中，
指定目标位于另一物体的左侧还是右侧？标准答案来自私有实例像素中心的左右顺序。
此外每一帧 seen 的 RGB 主色阈值参考必须独立得出相同顺序，两物体中心至少相距4像素。
参考算法只是展示信息可读，不是学习模型或记忆能力结果；它也不承担一般视觉识别评测。

公开导出分为 current_only 和 with_history。两者都有相同问题和双相机最后一帧 hidden
RGB，后者仅增加最后一帧 seen 的双相机 RGB。图片是256×256、无叠框的 PNG base64，
JSON 不携带布局、答案、场景路径、时戳、位置或原始消息元数据。对应问题的两布局
current_only JSON 必须字节一致，历史 JSON 必须不同，私有答案必须相反。
真实采样时间保留在私有原始记录中；公开只表示有序快照，没有伪造等间隔时间或动作。
后续调用模型时只能传 JSON，必须隔离跨 episode 状态，不能把私有文件路径或运行顺序
作为提示。这里只构造导出，不调用模型。

若通过，同一确定性无历史函数对每个配对问题至多答对一半。这是输入不可区分的逻辑
上限，不是实际模型得分，也不是统计检验结果。两布局、四问题和各阶段多帧都属于
同一个场景组；不能算作四个独立场景或拆到训练/测试两边。

## 复现

在有 Gazebo 8 Python bindings 的 WSL 运行，输出目录必须不存在：

```bash
/usr/bin/python3 -B -u -m benchmark.px4_gazebo.model_handoff.search_history_probe \
  --models /mnt/d/AeroWAM-data/annotation_models_v1 \
  --output-dir /mnt/d/AeroWAM-data/search_history_pair_new \
  --seed 20261018 --timeout-s 90

/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.search_history_probe \
  --output-dir /mnt/d/AeroWAM-data/search_history_pair_new --inspect-only
```

产物包括采集前设计、每布局的世界/原始消息/采集与时序检查、私有 inspection.json、
history_pair.png 以及不含私有答案的公开请求 JSON。整个目录是私有诊断数据，留在本地。
公开导出也不能与私有检查、世界、文件路径一起作为模型输入。

不启动 PX4，不训练，不进行模型推理；memory_use_verified、world_model_prediction_verified、
teammate_adaptation_verified、flight_verified、compositional_transfer_verified 均保持 false。
后续还需独立模型历史消融、更多场景组、真实决策效用、队友行为变化和候选动作后果预测，
才能逐步回答主线问题。

## 2026-10-18 首次真实配对

`D:/AeroWAM-data/search_history_pair_20261018_v1` 首次采集通过；默认 seed 的顺序为
swapped、normal。每布局3阶段、每阶段5组双机同戳采样，共30组联合观测、60对
RGB/实例图。两个服务器各启动一次，正常停止，cleanup_errors 为空，事后检查无残留。

| 私有布局 | seen 采样时间/s | hidden 采样时间/s | reacquired 采样时间/s |
| --- | --- | --- | --- |
| swapped | 1.13–2.18 | 3.37–4.43 | 5.58–6.64 |
| normal | 1.13–2.18 | 3.34–4.39 | 5.55–6.61 |

每相机共10张 hidden RGB，在原生1280×960及256×256上分别全部逐像素相同。
所有 hidden 帧两目标的实例像素均为0，墙可见。所有 seen 的 RGB 参考与实例左右顺序
一致。两个相机的红目标答案均为 right → left，蓝目标均为 left → right；四个
current_only 配对请求分别字节相同，仅加入历史后可区分。以下为256图像 SHA256：

- uav_0 hidden：`fbedc914df2906231d23a88e9d8ad031a2aef682ce37fc93be79287d536626ba`
- uav_1 hidden：`9238c84904eafe9cb23116914e934725283594019a59763fb4f8c1964a117148`

`history_pair.png` 已目视核对；原始消息独立离线复查通过。旧
`search_occlusion_sequence_20261018_v2` 在扩展布局参数后仍可离线验收。
Windows 与 WSL 各143项相关回归通过，包括12项新增的配对干预、错误位姿、单像素
差异、当前元数据捷径、历史混入私有答案、相同历史/答案和不完整相机输入等拒绝测试。

结论仅为这一场景组的历史信息必要性对照成立；尚未调用学习模型或证明实际记忆、
动作价值、队友适应、世界模型预测和组合迁移。reacquired 只用于私有时序验收，
不进入本轮历史问题的公开输入，避免引入未来观测。
