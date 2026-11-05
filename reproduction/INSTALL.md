# 新机器安装与资产下载指南

适用交付分支：`stage1-go2-simulation-release`。下面命令在新机器的 Ubuntu 22.04 x86_64 **Bash** 终端执行，不需要访问原服务器。运行结果和验收标准见[根目录 README](../README.md)。

## 1. 硬件、系统与驱动

使用 NVIDIA RTX GPU，至少 14 GiB 空闲显存，建议 24 GiB 显卡；至少 56 GiB 空闲主机内存。下载原始场景约 51 GB，连同解压、环境和资产副本建议预留至少 200 GB 磁盘。

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates bzip2 build-essential pkg-config \
  ffmpeg p7zip-full libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 libgl1 libgomp1
nvidia-smi
stat -fc %T /sys/fs/cgroup
systemd-run --user --scope --property=MemoryMax=24G --property=MemorySwapMax=0 true
```

`nvidia-smi` 必须能正常列出 GPU。原验证机器使用 **550.107.02** 驱动；新机器按实际显卡型号从 [NVIDIA 官方驱动入口](https://www.nvidia.com/Download/index.aspx)选择兼容驱动，安装并重启后再次检查。不同硬件和驱动尚未实测。

cgroup 检查应输出 `cgroup2fs`，`systemd-run` 命令应正常结束。如果提示无法连接用户总线，先检查 `systemctl --user status`，在正常用户登录会话中配置好用户服务后再运行任务。

## 2. 安装 Conda 和 Python 3.10

已有 Conda 可跳过下载。以下使用 Miniconda 官方 Linux x86_64 安装器；交互安装时阅读许可并选择安装位置，参考 [Conda 官方安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)。

```bash
curl -fL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -o /tmp/Miniconda3-Linux-x86_64.sh
bash /tmp/Miniconda3-Linux-x86_64.sh
# 假设安装到 $HOME/miniconda3；自定义位置时修改下面路径。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n go2-simulation python=3.10 -y
conda activate go2-simulation
python --version
```

后续每次新开终端，先执行 `source "$HOME/miniconda3/etc/profile.d/conda.sh"` 和 `conda activate go2-simulation`。如 Conda 提示软件源条款，按软件源提示处理后重试。

## 3. 克隆交付分支并安装依赖

```bash
git clone --branch stage1-go2-simulation-release --single-branch \
  https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'isaacsim==4.2.0.2' \
  'isaacsim-extscache-physics==4.2.0.2' 'isaacsim-extscache-kit==4.2.0.2' \
  'isaacsim-extscache-kit-sdk==4.2.0.2' --extra-index-url https://pypi.nvidia.com
python -m pip install -r reproduction/requirements.txt
python -m pip install --no-deps -e .
export ISAAC_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD"
```

核心版本为 Python **3.10**、Isaac Sim **4.2.0.2**、PyTorch **2.5.1**；其他版本见 [requirements.txt](requirements.txt)。Isaac Sim 使用 [NVIDIA 官方 Python 安装方式](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)，但版本必须使用上方命令，不照抄最新版文档中的版本号。首次运行按提示阅读并接受 NVIDIA Isaac Sim / Omniverse 许可。

此交付无需另外部署 Qwen 服务、训练 policy，也无需另外安装完整 IsaacLab 源码。GPU 驱动与 PyTorch wheel 自带 CUDA 运行时不是同一个组件，不要根据 `nvidia-smi` 的 CUDA 字样自行替换交付的 torch 版本。

安装后检查：

```bash
python -m pip check
python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
python -c 'from importlib.metadata import version; print("Isaac Sim:", version("isaacsim")); assert version("isaacsim") == "4.2.0.2"'
python -c 'from pxr import Sdf, UsdUtils; import cv2, scipy; print("Asset preparation imports OK")'
python delivery/verify_delivery.py
```

`pip check` 应无冲突，CUDA 检查应为 `True`，交付校验应为 `failures: 0`。如依赖冲突或导入失败，先保存完整报错并解决，不跳过检查或随意升级核心依赖。

## 4. 资产来源与下载

| 内容 | 来源及位置 | 是否单独下载 |
| --- | --- | --- |
| Go2 policy、部署契约 | 分支内 `delivery/model/` | 否，随 Git 克隆 |
| Go2 机器人 USD | `delivery/assets-source/go2_training_flattened.usdc` | 否，随 Git 克隆 |
| 箱体场景覆盖层 | `delivery/assets-source/after_scene.usda` | 否，随 Git 克隆 |
| T01–T10 固定任务输入 | `delivery/inputs/T01` 至 `T10` | 否，随 Git 克隆 |
| 场景几何、模型、纹理 | [GRScenes 数据集](https://huggingface.co/datasets/InternRobotics/GRScenes) | 是，约 51 GB 三个分卷 |
| OmniPBR 材质、仿真扩展 | Isaac Sim 及 extscache 安装包 | 随第 3 步安装 |

数据集固定 revision 为 `4760b9031881c985e3582075cb2d8dbde1530a68`。下载脚本从该 revision 读取以下文件，并用 `7z` 解压：

```text
scenes/GRScenes-100/home_scenes.z01
scenes/GRScenes-100/home_scenes.z02
scenes/GRScenes-100/home_scenes.zip
```

执行：

```bash
# 仍在仓库根目录，且已激活 go2-simulation 环境。
python reproduction/download_dataset.py --download-dir "$HOME/go2-data-download"
```

下载中断可重新执行同一命令，复用 Hugging Face 本地缓存。不要改动三个分卷的名称、版本或相对位置。如果上游提示 401/403，先在数据集页面确认访问要求，再通过 `huggingface-cli login` 配置接收方自己的凭据。

也可在数据集固定 revision 的页面手动下载上述三个文件，保持同一目录、同名，再用 `7z x home_scenes.zip -o/path/to/extracted` 解压。不要混用不同 revision 的分卷。

GRScenes 使用 **CC BY-NC-SA 4.0**，使用和分享需遵守非商业、署名、相同方式共享条件。机器人资产保留其上游许可约束。

## 5. 定位解压根目录并准备可迁移资产

解压层级以实际压缩包为准。以下命令识别同时包含 `models/`、`Materials/`、`scenes/` 的目录；仅找到一个时才继续：

```bash
export GO2_EXTRACTED="$HOME/go2-data-download/extracted"
export GO2_DATASET_HOME="$(python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['GO2_EXTRACTED']).resolve()
candidates = [root] + [p for p in root.rglob('home_scenes') if p.is_dir()]
roots = [p for p in candidates if all((p / n).is_dir() for n in ('models', 'Materials', 'scenes'))]
if len(roots) != 1:
    raise SystemExit(f'Expected one dataset root under {root}; found {len(roots)}')
print(roots[0])
PY
)"
test -n "$GO2_DATASET_HOME" && \
python reproduction/prepare_assets.py \
  --dataset-home "$GO2_DATASET_HOME" --output "$HOME/go2-simulation-assets"
cat "$HOME/go2-simulation-assets/asset-preparation.json"
```

准备程序逐一核对 **1680** 个原始依赖文件的 SHA256，复制所需文件与已交付机器人/覆盖层，修复绝对引用并检查依赖闭包。原始数据不会修改，`--output` 必须为尚不存在的目录。

成功报告中 `source_dataset_hashes_verified`、`all_dependencies_inside_output` 均为 `true`；原验证得到 **1107** 个 USD 层、**6252** 个资产依赖。准备后的结构：

```text
$HOME/go2-simulation-assets/
├── after_scene.usda
├── go2_training_flattened.usdc
├── asset-preparation.json
└── home_scenes/
    ├── models/
    ├── Materials/
    └── scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd/
```

SHA256 不一致时，不要关闭校验，应检查 revision、分卷完整性和原始文件是否被修改。准备中途失败时保留日志，改用新的 `--output` 路径重试。

## 6. 首次运行与全部任务

```bash
python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
  --output "$HOME/go2-runs/T01-first" --task T01 --gpu 0
python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
  --output "$HOME/go2-runs/T10-first" --task T10 --gpu 0
```

`--gpu` 使用 `nvidia-smi` 的设备编号，不设置 `CUDA_VISIBLE_DEVICES`。启动器串行等待资源条件连续满足 60 秒；日志中的等待不属于安装失败。需要五路录像时加 `--record-video`。

全部十项运行命令、结果检查和回归测试见[README 第 3–5 节](../README.md#3-检查安装与代码回归)。每次使用新的输出目录。

## 7. 验证范围

本指南给出新机器所需的安装与下载流程。已执行的是原 Linux GPU 主机上的独立代码目录、迁移资产验证，T01/T10 成功，回归 1008 项通过。全新 Conda 安装、约 51 GB 公开数据完整下载及接收方新硬件尚未端到端实测；提供流程不等同于这些步骤已经通过验证。详细记录见 [validation.json](validation.json)。
