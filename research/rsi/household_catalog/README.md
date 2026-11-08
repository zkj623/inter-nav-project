# R1 Lite：20种家居物体加载与原地操作入口

本模块可加载选定的20种原比例GRScenes模型，建立与物体局部坐标一致的碰撞网格，并启动接近、夹持、抬升与持物测试。它为后续机器人操作提供入口；**不表示20种物体都已通过抓取、稳定抬升或运输。**

## 当前能力与验证边界

- 20种模型均完成实际碰撞几何的坐标、尺寸和位置核对。
- 将源模型的层级变换烘焙到米制碰撞顶点；手指规划和执行检查使用PhysX实际烘焙的凸分解。身体使用覆盖真实碰撞形状的保守外包围。
- 根据实际表面截面生成双手位置、宽度和抓取高度。当前有限搜索为14种找到双侧接触候选，其余6种需要补充候选或使用其他抓法。
- 当前操作范围为原地接近、夹持、抬升、持物；没有主动运输、放下、软体或桶盖/轮子关节验证。适配器在独立进程内安装，不用于共用进程里的混合任务。
- 质量统一假定1 kg，静/动摩擦系数1.0/0.9。2 mm手指穿透等安全门槛未放宽。
- 三个代表物体的新版复测尚未通过完整抬升持物：白箱在后续抬升规划中触发实际碰撞几何穿透门槛；加固角箱接触力不足；小桶的姿态可达性/法向位置精度未通过。旧白箱3/3结果不能沿用为本版通过。

`evidence/geometry_validation_summary.json`保存观察结果；`evidence/actual_geometry_grasp_sections.png`展示实际截面与整体外包围的区别。旧15个停止位姿的回查确认为代理误报，只补充诊断，不改写旧测试成绩。

## 环境与外部资产

在仓库根目录执行下列命令，使用**已经配置好的Isaac Sim 4.2 Python环境**。示例中的`python`应替换为该环境的Python。安装仓库基础依赖，再安装额外碰撞查询依赖：

```bash
python -m pip install -r requirements/r1lite-household-geometry.txt
```

如果使用已有的独立依赖目录，可在各入口传入`--collision-deps /path/to/python-fcl-directory`，无需修改共享环境。

机器人需使用本项目已适配的R1 Lite USD、同名URDF及其网格/材质依赖：

```bash
export R1LITE_USD_PATH=/path/to/r1lite_navigation.usd
export R1LITE_CATALOG_ROOT=/path/to/household_catalog20
```

也可在原地操作入口显式传入`--robot-usd`与`--root`。机器人资产和GRScenes模型二进制文件不放入Git。

当前项目服务器可直接复用已有目录，无需重新导入：

```bash
export R1LITE_CATALOG_ROOT=/data20t/embodied/zkj/inter-nav-results/r1lite-household-catalog20-20261028
export R1LITE_USD_PATH=/data20t/embodied/zkj/inter-nav-results/r1lite-household-v5/candidate-35/robot_snapshot/r1lite_navigation.usd
```

### 校验已有20物体包

```bash
python research/rsi/household_catalog/catalog_support.py --root "$R1LITE_CATALOG_ROOT"
```

该命令不启动仿真，检查20个ID、原模型版本、每件USD入口和209项源文件依赖哈希。它可以防止只有清单、没有模型文件的目录被误认为可加载。

### 从已有GRScenes数据安装生成资产包

数据来源：[OpenRobotLab/GRScenes](https://huggingface.co/datasets/OpenRobotLab/GRScenes)。安装根目录需包含`scenes/GRScenes-100/home_scenes/models/object`、`benchmark/meta`及原README。下载/展开数据集后运行：

```bash
python research/rsi/household_catalog/prepare_catalog.py \
  --source /path/to/inter-nav-assets \
  --output /path/to/new-household-catalog20 \
  --gpu 0
```

入口使用当前`selection.json`中的20个模型，复制USD/MDL/贴图依赖、统一米制和Z-up、落地归零，并生成`asset_manifest.json`和5×4展示场景。输出必须是新目录；不会覆盖现有资产包。准备过程不推进物理时间，结束后会校验源依赖并保存`prepare_validation.json`。

模型数据许可为CC BY-NC-SA 4.0，模型ID和源哈希见`expected_catalog.json`，上游说明见`GRScenes_SOURCE_README.md`；代码遵循仓库根许可证。模型数据的许可与代码许可证分别适用。

## 加载并开始操作

### 加入已打开的Isaac Sim场景

```python
from research.rsi.household_catalog.load_asset import add_asset

# stage必须为米制、Z-up；默认只加载外观，不启动时间轴。
prim = add_asset(stage, 'bin_round_pedal', '/World/Target',
                 position=(0.65, 0, 0), package_root='/path/to/household_catalog20')
```

若用于动态操作，不要仅添加可视引用。必须显式指定质量与摩擦，再调用`bake_collision_meshes`建立物体局部坐标的碰撞网格。下方的原地测试入口会完整执行这些准备和查询适配，推荐以它作为后续操作起点。

### 只核对20种实际碰撞几何，不执行抓取

```bash
python research/rsi/household_catalog/export_catalog_geometry.py \
  --root "$R1LITE_CATALOG_ROOT" \
  --output /tmp/household-geometry-new-run --gpu 0
```

输出每个模型的实际凸体、变换烘焙校验、候选截面、`summary.json`和零物理步计数。`geometry_ready=true`与“找到候选”都不代表物理抓取成功。

### 单个物体原地抓取、抬升、持物

```bash
python research/rsi/household_catalog/lift_probe.py \
  --root "$R1LITE_CATALOG_ROOT" \
  --robot-usd "$R1LITE_USD_PATH" \
  --asset box_white --gpu 0 \
  --output /tmp/white-box-new-run
```

输出目录必须不存在。入口加载新物体并添加网格碰撞，选择真实接触面，检查整段轨迹，再执行实际接触抓取和分段抬升；目标抬升22 cm，通过要求离地至少15 cm、连续持物3秒、双手保持接触、相对滑移不超过3 mm、相对旋转不超过0.05 rad。失败立即停止并保留原因，**不执行底盘运输**。

输出包括`result.json`、`surface_grasp.json`、`cooked_object_geometry.json`、碰撞审计、状态动作记录和MP4。初始化/规划失败时可能没有动作录像，不能把这类结果描述为已经执行了抓取。

### 批量原地测试（Linux）

```bash
python research/rsi/household_catalog/run_batch.py \
  --root "$R1LITE_CATALOG_ROOT" --robot-usd "$R1LITE_USD_PATH" \
  --only box_white,bin_round_pedal,box_reinforced \
  --gpus 0,2,3 --output /tmp/household-batch-new-run
```

不传`--only`会测试全部20种，但没有合适候选或初始安全检查失败的物体会明确停止。每GPU一次一个独立进程、启动前至少8000 MiB空闲，不设置`CUDA_VISIBLE_DEVICES`。默认每例600秒上限。已有结果默认拒绝复用；显式`--resume`也会核对代码、运行时、目录清单与机器人USD/URDF指纹，避免把旧结果当新结果。

## 回归测试与源码基线

```bash
python research/rsi/household_catalog/run_tests.py \
  --robot-usd "$R1LITE_USD_PATH" --result /tmp/household-tests.json
```

测试不执行物理rollout。`r35_test_modules.json`列出对应控制器回归，另运行9项几何查询检查和资产包完整性检查。控制器源码来自已记录的R35工作快照，原快照指纹见`r35_runtime_manifest.json`，仅换行规范化后的文件指纹见`published_runtime_normalization.json`；R35离线检查通过不代表已完成物理稳定性验收。

后续应先解决实际抓取点、接触校准、位姿精度和持物重规划，再考虑运输与释放。当前分支提供可运行的加载和原地操作框架，不提供20种物体已成功搬运的承诺。

## 本次发布检查

仓库内266项回归检查通过（无跳过）；20物体几何导出通过；从原始数据重新生成20物体并通过209项源依赖校验；机器人与白箱的初始化/几何接入通过。后两项不是抓取或稳定持物成功，详见`evidence/publication_validation.json`。
