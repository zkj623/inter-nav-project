# 第一阶段仿真交付材料

交付分支：`stage1-go2-simulation-release`。新机器从[安装与下载指南](../reproduction/INSTALL.md)开始，全部运行命令见[根目录README](../README.md)。

## 文件与资产来源

| 内容 | 仓库位置或下载方式 |
| --- | --- |
| 交付说明与测试用例 | `01-交付说明与测试用例.docx`、`01-交付说明与测试用例-V4.pdf` |
| 测试报告与可视化材料 | `02-测试报告与可视化材料.docx`、`02-测试报告与可视化材料-V4.pdf` |
| 算法设计文档 | `03-算法设计文档.docx`、`03-算法设计文档-V4.pdf` |
| policy与部署契约 | `model/`，随仓库提供 |
| 机器人与场景覆盖层 | `assets-source/`，随仓库提供 |
| 任务输入 | `inputs/T01`至`T10`，随仓库提供 |
| 运行支持脚本 | `runtime-support/`，由`../reproduction/run_task.py`自动引用 |
| 场景几何与纹理 | [GRScenes公开数据集](https://huggingface.co/datasets/InternRobotics/GRScenes)，使用`../reproduction/download_dataset.py`下载固定版本，随后执行`prepare_assets.py` |
| 仿真与Python环境 | 按`../reproduction/INSTALL.md`从NVIDIA和Python包索引安装 |
| 原始验收证据 | `evidence/`，用于结果和源码追溯 |
| 截图与来源 | `figures/`及`figures/frame-sources.json` |

## 新机器运行入口

在仓库根目录执行：

```bash
python delivery/verify_delivery.py
python reproduction/download_dataset.py --download-dir "$HOME/go2-data-download"
# 按reproduction/INSTALL.md第5节设置GO2_DATASET_HOME。
python reproduction/prepare_assets.py \
  --dataset-home "$GO2_DATASET_HOME" --output "$HOME/go2-simulation-assets"
python reproduction/run_task.py --assets "$HOME/go2-simulation-assets" \
  --output "$HOME/go2-runs/T01-first" --task T01 --gpu 0
```

`$HOME`是接收方本机用户目录。资产与输出目录均由接收方选择；准备程序检查原始文件SHA256、修复引用并检查依赖闭包，启动器复制任务输入并替换资产路径。

## 文档与验收

三份正式文档均采用宋体、A4黑白排版。算法设计文档包含方法设计和实现细节；测试报告包含十项任务的实际路径与最短几何路径对比，以及20张任务截图。固定T01–T10均通过交付版本测试，任务完成为验收依据，无耗时门槛。

路径曲折度均大于1.3，初始杂乱度38.195603%大于20%。数据见`task-paths-and-complexity.json`和`.csv`；离线资格几何不输入导航器。

原始输入快照和证据中的历史机器路径用于追溯。新部署统一调用`reproduction/`入口；交付目录内保留的历史审计／启动脚本供证据复核。独立目录复现记录及实际执行范围见`../reproduction/validation.json`。
