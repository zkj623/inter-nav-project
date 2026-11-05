# stage1-go2-simulation-release 在线建图与推箱导航交付

本分支交付同一 MV7 场景的 T01–T10 固定任务：从空地图利用真实 LiDAR 建图，完成普通导航或侧推箱体、重新扫描、穿越并在目标处停稳。任务完成即通过；不设置耗时门槛。路径曲折度全部 >1.3，初始杂乱度 38.195603% >20%。

## 交付内容与代码范围

- `grutopia/demo/go2_online_interactive_navigation.py`、`go2_side_clearance_navigation.py`：任务入口。
- `grutopia_extension/interactive_navigation/`、Go2 控制器/机器人/传感器：导航与交互逻辑。
- `grutopia/core/` 及注册模块：运行所需框架依赖，保留其公共配置、语义模块接口和测试；并非只有一个独立导航脚本。
- `reproduction/`：面向新机器的资产准备、启动、结果检查入口。
- `delivery/`：模型权重、固定输入、机器人资产、文档和原始验收证据。
- `tests/`：相关算法、控制器、框架回归测试。历史训练和诊断入口不作为交付运行入口。

Git 历史用于追溯，当前分支根目录 README 是新部署入口。历史证据中的绝对路径是来源记录；下面的新复现入口不依赖原服务器实验目录。

## 1. 新机器环境

验证基准为 Ubuntu 22.04 x86_64。使用 Python **3.10**、Isaac Sim **4.2.0.2**、PyTorch **2.5.1**。不要直接替换成最新 Isaac Sim。准备 NVIDIA RTX GPU（至少 14 GiB 空闲显存，建议 24 GiB 显卡）、至少 56 GiB 空闲主机内存。任务串行执行，单任务 24 GiB 内存上限、禁止 swap。系统需支持 cgroup v2 与 `systemd-run --user`。

场景原始数据由上游以分卷压缩包发布，下载约 51 GB；连同解压、仿真环境和复制资产，建议预留至少 200 GB 磁盘。网络需能访问 NVIDIA Python 索引和 Hugging Face。

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg p7zip-full libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 libgl1 libgomp1
git clone --branch stage1-go2-simulation-release --single-branch \
  https://github.com/zkj623/inter-nav-project.git
cd inter-nav-project
conda create -n go2-simulation python=3.10 -y
conda activate go2-simulation
python -m pip install 'isaacsim==4.2.0.2' \
  'isaacsim-extscache-physics==4.2.0.2' 'isaacsim-extscache-kit==4.2.0.2' \
  'isaacsim-extscache-kit-sdk==4.2.0.2' --extra-index-url https://pypi.nvidia.com
python -m pip install -r reproduction/requirements.txt
python -m pip install --no-deps -e .
export ISAAC_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD"
# 在按 NVIDIA 提示阅读并接受 Isaac Sim 许可后配置其运行环境。
# 安装系统工具：ffmpeg、p7zip-full；验证 nvidia-smi 和 systemd-run --user 可用。
```

Isaac Sim 安装遵循 [NVIDIA 官方文档](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)，这里的锁定版本用于保持旧 API 与物理行为。语义视觉识别/训练不在本次复现流程内，不需要另外部署 Qwen 服务。

## 2. 下载并准备场景资产

机器人 USD、本次箱体场景覆盖层和 policy 随分支交付。大场景采用上游 [GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes)，固定版本 `4760b9031881c985e3582075cb2d8dbde1530a68`。数据许可为 **CC BY-NC-SA 4.0**，使用和分享需遵守其非商业、署名与相同方式共享条件。机器人资产来自 Unitree/IsaacLab，保留其上游许可约束。

```bash
python reproduction/download_dataset.py --download-dir "$HOME/go2-data-download"
# 查看解压结果，找到同时包含 models/、Materials/、scenes/ 的 home_scenes 目录。
export GO2_DATASET_HOME="$HOME/go2-data-download/extracted/home_scenes"
python reproduction/prepare_assets.py \
  --dataset-home "$GO2_DATASET_HOME" --output "$HOME/go2-simulation-assets"
```

`GO2_DATASET_HOME` 应指向实际解压层级，不能把 `scenes/MV7...` 子目录当作根目录。准备程序检查 1680 个依赖文件的原始 SHA256；任一缺失或不一致会停止。随后复制必要依赖、恢复相对引用、改写服务器绝对引用，检查全部 1107 个 USD 层与 6252 个资产依赖均位于新的输出目录。输出目录必须不存在，原始数据不会被修改。`OmniPBR.mdl` 属于 Isaac 自带材质库。

## 3. 检查安装与代码回归

```bash
python delivery/verify_delivery.py
QWEN3_WORKER_LOG="$PWD/.qwen-worker-test.log" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -rs tests/ \
  --deselect=tests/test_interaction_navigation_camera.py::ThirdPersonCameraPlacementTest::test_g1_third_person_is_close_and_elevated
```

G1 摄像机用例不属于本 Go2 交付范围。若当前测试类名与上述不同，使用 `--collect-only` 核对后指定节点；历史测试结果不能替代本机的真实输出。

## 4. 执行任务

先验证普通导航 T01，再验证包含推箱的 T10。每次必须使用新的输出目录：

```bash
python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
  --output "$HOME/go2-runs/T01-first" --task T01 --gpu 0
python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
  --output "$HOME/go2-runs/T10-first" --task T10 --gpu 0
```

`--gpu` 指 `nvidia-smi` 的设备编号；不要设置 `CUDA_VISIBLE_DEVICES`。程序等待主机内存/显存条件连续满足60秒后启动；全机使用 `/tmp/go2-simulation-reproduction.lock` 串行锁。输出目录中的 `run.log` 为实时仿真日志。需要五路录像时添加 `--record-video`，综合视频输出在 `<output>/<task>/video/combined.mp4`。

全部十项按顺序复现：

```bash
for task in T01 T02 T03 T04 T05 T06 T07 T08 T09 T10; do
  python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
    --output "$HOME/go2-runs/${task}-full" --task "$task" --gpu 0 || break
done
```

入口复制该任务原始输入到新输出目录，仅修改场景所在路径；机器人比例、policy、地图频率、物理/驱动与 stage1-go2-simulation-release 参数保持一致。输入守卫禁止把离线评估图提供给导航器。

## 5. 结果判定与验证范围

自动生成 `reproduction-check.json`：`passed=true` 表示任务成功到达、误差≤0.5m、实测停稳、倾角/启动/健康检查和在线输入守卫通过；交互项还要求技能成功和实际箱体移动>0.25m。原始结果为 `<task>/result.json`，完整轨迹、事件和物理接触日志一并保存。耗时仅记录，不设合格门槛。

历史 stage1-go2-simulation-release 十项均通过完整功能审计；包括真实接触、新扫描与整机穿越，详见 `delivery/evidence/`。新入口的自动结果检查范围以其代码和报告明确列出的检查为准，不能将历史审计冒充新机器的审计。跨 GPU/驱动版本不能承诺位级一致或完全相同耗时。

独立目录验证记录见 `reproduction/validation.json`；完整回归1008项通过、4项外部资产检查跳过、1项G1用例排除。跳过原因见 `reproduction/validation-evidence/tests.log`。只有实际执行成功的任务才标记为通过；未实际测试的新机器安装条件会明确保留，不宣称已在用户的新硬件上运行。

## 文档

- [交付说明与测试用例](delivery/01-交付说明与测试用例.docx)
- [测试报告与可视化材料](delivery/02-测试报告与可视化材料.docx)
- [路径与场景复杂度](delivery/task-paths-and-complexity.csv)

Word/PDF保留历史交付时的环境说明；跨机器部署以本README和`reproduction/`为准。

### 本次复现验证结果

独立代码目录和重新定位的资产副本：T01通过（目标误差1.11cm），T10通过（15.27cm，实际推箱并完成导航）。验证系统Ubuntu 22.04.5、RTX 4090 D、驱动550.107.02；66份核心运行源码与本分支一致。完整回归1008通过、4项旧资产检查跳过、1项G1用例排除。

已从GitHub重新克隆并完成交付文件哈希检查。本次复用现有Isaac/Python安装，没有重新下载51GB数据或从零安装Conda；接收方新硬件上的安装与执行尚未实测。安装流程与数据校验入口均已提供，详细验证范围见`reproduction/validation.json`。
