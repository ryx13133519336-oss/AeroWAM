# AeroWAM 的 C/D 盘存储方案

## 已采用的目录布局

代码继续保留在：

```text
C:\Users\R\Desktop\AeroWAM
```

体积较大的可再生内容放在：

```text
D:\AeroWAM-data\
  datasets\
  checkpoints\
  px4_logs\
  episodes\
  gazebo_assets\
  cache\
  conda_envs\
  conda_pkgs\
```

WSL 中同一目录为 `/mnt/d/AeroWAM-data`。

## 每次启动时加载路径

在 WSL、项目根目录执行：

```bash
source benchmark/px4_gazebo/env/aerowam_storage.sh
python3 -m benchmark.px4_gazebo.paths --create
```

在 Windows PowerShell 中执行：

```powershell
. .\benchmark\px4_gazebo\env\aerowam_storage.ps1
python -m benchmark.px4_gazebo.paths --create
```

`run_fixture` 不指定 `--output` 时，会自动将 episode 写入
`D:\AeroWAM-data\episodes`（WSL 中为 `/mnt/d/AeroWAM-data/episodes`）。

## 哪些内容适合放 D 盘

- 数据集、示范轨迹和视频帧
- AeroDuo、UA-NWM、DINO 等模型权重
- Hugging Face、Torch 和 pip 下载缓存
- Gazebo Fuel 下载的模型和自定义世界资源
- PX4 ULog、rosbag2、episode JSONL 和评测结果

## 哪些内容不要直接放 `/mnt/d` 执行

PX4 源码/build、ROS 2 workspace、Micro XRCE-DDS Agent 源码以及 Linux Conda
环境包含大量小文件、Linux 权限和符号链接。直接在 `/mnt/d` 编译或运行通常更慢，
也更容易遇到权限或链接问题。应继续把它们放在 WSL 的 Linux 文件系统中。

如果 WSL 虚拟磁盘本身占满 C 盘，最佳方案是把整个 Ubuntu 发行版迁移到 D 盘，
这样 Linux 内仍使用 ext4，但物理空间来自 D 盘。先运行 `wsl --version`；新版 WSL
支持直接迁移。迁移前关闭 Gazebo、PX4、ROS 2 和所有 WSL 终端，在管理员 PowerShell
中确认发行版名称及状态：

```powershell
wsl --list --verbose
wsl --shutdown
New-Item -ItemType Directory -Force D:\WSL\Ubuntu-22.04
wsl --manage Ubuntu-22.04 --move D:\WSL\Ubuntu-22.04
```

必须把示例中的 `Ubuntu-22.04` 换成 `wsl --list --verbose` 显示的准确名称。若当前
WSL 不支持 `--manage ... --move`，可以使用导出/导入作为备用方案，但不要在验证备份
前注销旧发行版：

```powershell
wsl --export Ubuntu-22.04 D:\WSL\Ubuntu-22.04\ubuntu.tar
wsl --import Ubuntu-22.04 D:\WSL\Ubuntu-22.04 D:\WSL\Ubuntu-22.04\ubuntu.tar --version 2
```

确认新发行版可以启动、原有用户和 PX4 文件都在之后，再按需处理旧发行版；不要为了
释放空间而在迁移验证前执行 `wsl --unregister`。

## Conda 说明

`conda_envs` 和 `conda_pkgs` 目录已预留，但默认没有把 Linux Conda 环境放到
`/mnt/d`。优先迁移整个 WSL 发行版；只有磁盘紧急且能接受速度、权限兼容风险时，
再单独修改 Conda 的 `envs_dirs` 和 `pkgs_dirs`。
