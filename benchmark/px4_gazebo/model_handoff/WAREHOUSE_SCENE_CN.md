# 外部仓库场景：静态接入验收

主线不变：在部分可观测、队友行为变化的环境中，评测双无人机协作技能和世界模型预测能力能否组合迁移到新任务。

## 本阶段结果

2026-10-19 在现有 WSL / Gazebo Sim 8.15.0 完成真实静态渲染。不是上游截图、模型回放，也不是飞行成绩。

- 导入约 14 × 21 米仓库，25 个背景实例、25 个原始碰撞网格；包含纹理货架、箱堆、墙体、地面及杂物。
- 两组视点，每组两个 1280 × 960 RGB/私有实例分割相机，每个相机获得 8 对不同且完全匹配的时间戳。
- 两组之间只改变相机位姿，背景、目标、灯光和传感器资源相同。不是连续运动或记忆测试。
- 前方低视点中，红球两相机均为 0 像素，蓝块为 0 / 1 像素；进入货架间视点后两者均有有效标注框。
- 不启动 PX4，不解锁，不固定无人机角色，不向策略暴露目标坐标或实例标签。简单场景默认行为不变。
- Windows Python 与 WSL Python 的 33 项导入/标注回归测试均通过，最终检查程序复核保存的原始消息后仍通过静态验收。

以下是实际采集后按现有规则缩放到 256 × 256 的私有目标像素数，各自 8 帧一致：

| 静态视点 | 相机 | 红球 | 蓝块 |
| --- | --- | ---: | ---: |
| rack_front | uav_0 | 0 | 0 |
| rack_front | uav_1 | 0 | 1 |
| rack_back | uav_0 | 708 | 434 |
| rack_back | uav_1 | 271 | 668 |

静态验收要求每个目标至少在一个前方相机的所有帧中完全隐藏，并在至少一个后方相机的所有帧中获得有效框。RGB 主色只用于标签对齐的健全性检查，不替代私有分割标注。背景使用私有标签 100。

本机成功数据：`/mnt/d/AeroWAM-data/external_scenes/warehouse_static_check_20261019_v5/`，含原始 protobuf、世界文件、逐帧哈希、报告及 `warehouse_views.png`。失败尝试 v1–v4 保留，不当作成功数据。大体积资产和原始数据不入 Git。

成功世界 SHA256：

```text
rack_front 91aac467258c114b1b97f143b65bb94cbf6a7b3891daddbf93bee28dcc89586f
rack_back  c45db7db2ab6a93cdb7b91486313acb87830db49ed06852a0999091088f288b6
```

## 资源来源与转换

来源：[AWS RoboMaker Small Warehouse World](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world)，固定提交 `3c23a698bf0b4e366ddf8b084af507c519bd3483`。

下载地址：`https://codeload.github.com/aws-robotics/aws-robomaker-small-warehouse-world/zip/3c23a698bf0b4e366ddf8b084af507c519bd3483`。

归档 SHA256：`00cd9528271002b30396e2957da9fb969584c9b18d5b8ad9a8b97fde2c14f214`。保留上游 `LICENSE`（Amazon 的宽松许可文本），按该文件履行署名和许可保留要求；不是原创实景扫描或照片级重建。

使用 `no_roof_small_warehouse.world`。只提取该世界引用的模型及必要 SDF、配置、DAE、PNG、许可，共 76 个文件。校验固定文件哈希表以及 COLLADA 纹理引用；不运行上游 launch、插件或脚本，不在线解析 Fuel。未使用桌子模型的旧 Windows 绝对纹理路径不会进入包。

转换时展开模型 include，保留实例位姿、碰撞网格、摩擦、DAE 内部单位/变换与相对纹理布局；资源 URI 改为本地绝对路径。仅移除 Classic 的 visual layer 元数据及静态链接不用的惯性块。原地面惯性不满足物理三角不等式，Harmonic 会拒绝整个世界，因此不原样沿用；不借此删除碰撞。

两个诊断目标统一缩小到原尺寸的 0.6 倍，主体、条纹、碰撞一起缩放以放入货架间隔。这是明确的静态诊断布置，不是角色分配、默认飞行高度或策略导航路线。低视点不代表已验证的 UAV 可飞位姿。Gazebo 提示 COLLADA 顶点 COLOR 不受支持；当前纹理已实际渲染，仍需后续场景外观与接触检查。

## 复现

已有包可复用；首次准备使用一个不存在的新目录。输出目录也必须不存在。运行于能加载 `gz.transport13`、`gz.msgs10`、NumPy、Pillow 的 Linux Python：

```bash
/usr/bin/python3 -B -m benchmark.px4_gazebo.model_handoff.warehouse_probe \
  --archive /path/to/pinned-source.zip \
  --package /path/to/new-warehouse-package \
  --models /path/to/annotation_models_v1 \
  --output-dir /path/to/new-static-capture --timeout-s 60
```

再次采集省略 `--archive`；只复核已有原始消息增加 `--inspect-only`。命令使用独立 `GZ_PARTITION`，只清理自己启动的 Gazebo 进程组并记录清理结果，不修改时钟或校时服务。

## 下一步边界

静态验收通过不等于碰撞引擎接触已验证、航路安全、模型成功率或组合迁移有效。当前代码没有把仓库暗接到原 `search_pilot`，也没有解除旧场景的 2.5 米运动保护、10 毫秒校时门或 400 毫秒在线推理预算。

下一步先验证货架间适合飞行的高度、净空、碰撞接触和独立安全边界，再显式接入不指定角色的 search。之后在同一场景族接入移动目标 handoff 与 inspection，并变化起点、遮挡及队友行为。目标选择、分工应由任务和观测产生，而不是写死 UAV0 搜索 / UAV1 确认。正式训练/测试场景划分和模型成绩另行记录，不能用这四个固定视点冒充泛化评测。
