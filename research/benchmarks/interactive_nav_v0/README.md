# GRScenes Interactive Navigation Benchmark v0.1

初版基于 GRScenes-100 的家庭场景和 `genie2-rsi` 的 15 种刚体物体。包含 10 个场景、每场景 3 个物体与 6 个独立起终点任务，共 60 个任务。用于后续交互式导航开发。场景与任务固定，不把当前机器人策略的通过结果作为筛选条件。

## 设计与数据定义

- 从服务器已有的全部 69 个家庭场景测量地面水平投影的并集面积，按面积排序后等间隔抽取 10 个秩位置。覆盖小、中、大家庭空间；这不等同于覆盖商场、办公室等语义类别。
- 面积采用 0.1 m 栅格，选取 z 在 ±0.18 m 内的近水平地面。它是近似地面投影面积，不是房产建筑面积。`inventory.json` 保留所有候选的统计。
- 场景原始坐标统一缩放 0.01 至米；家居资产已是米制，不再次缩放。物体使用已有实际凸分解碰撞资产，没有用方箱替换外观或物理几何。
- 为几何筛选投影 0.12–1.15 m 高度带内的障碍三角形，并使用半径 0.30 m 的圆形机器人近似。此网格仅用于任务构建与离线参考评价，不能直接提供给未知环境导航策略。
- 从多条可通行参考路线中选择较窄的中间位置作为通行关键点，要求物体外包围与静态障碍分离、不同放置点相隔至少 1.5 m，并保留一条平面无碰撞的物体移位见证。这些点是几何瓶颈候选；尚未逐个人工确认是门口或路口。
- 每个物体对应一条往返任务，三个物体同时存在于初始场景。正反方向分别作为独立任务，每次重置整个场景。15 种物体各出现两次。
- `reference_path_m` 为移除本次新增物体后的圆形机器人网格最短路径；`initial_detour_path_m` 为保留新增物体时的最短路径，null 表示该近似网格不连通。两者均不是包含推移代价的全局最优解。

## 物理与能力边界

沿用 genie2-rsi 的整体刚体假设：每件质量 1 kg，静/动摩擦 1.0/0.9；编织篮使用半刚体近似，椅轮与桶盖没有独立关节。这些是受控实验参数，不是各类物体的真实质量。

生成器只证明离线几何筛选通过。`physics_smoke.json` 单独报告 Isaac Sim 中 240 步、1/120 s 步长的静置落地检查；没有该文件就没有该场景的静置验证。静置通过不等于机器人能推、抓、搬或完成导航。当前 60 个任务的机器人闭环执行均待验证。

## 构建

需要 Linux、numpy、scipy、OpenCV、OpenUSD；物理加载使用现有 Isaac Sim 4.2 Python。提供三个外部目录：GRScenes `home_scenes`，genie2-rsi 的 `object_assets`，以及源码中的 `rigid_catalog.json`。物体 USD 引用的原家居资产依赖也必须保留。数据来源与许可证见上游 GRScenes（CC-BY-NC-SA-4.0）和 genie2-rsi 文档。

```bash
python scan.py --home "$GRSCENES_HOME" --out "$CACHE"
python select_scenes.py --data "$CACHE"
python scan.py --home "$GRSCENES_HOME" --out "$CACHE" --maps
python build.py --data "$CACHE" --home "$GRSCENES_HOME" \
  --objects "$G2_OBJECT_ASSETS" --catalog "$RIGID_CATALOG" --out "$OUTPUT"
python -m unittest test_geometry.py
python prepare_physics.py --out "$OUTPUT"
python smoke.py --scene-dir "$OUTPUT/S01" --gpu 0
```

已有冻结任务时，优先直接实例化，避免重新采样：

```bash
python materialize.py --suite fixtures --home "$GRSCENES_HOME" \
  --objects "$G2_OBJECT_ASSETS" --out "$OUTPUT"
python prepare_preview.py --out "$OUTPUT"
```

`materialize.py` 校验 15 个物体 USD 的 SHA256。原始物体资产包及其依赖需要从已有 genie2-rsi 工作区取得；这里尚未提供独立下载的资产包。`preview.usda` 使用材质兼容近似；`scene.usda` 保留原材质。`prepare_physics.py` 生成只用于物理检查的展开副本，去除着色器并隐藏可视网格，保留碰撞与物理材质，约需额外数 GB 存储。它不能用来评估视觉感知。

材质兼容函数沿用项目交付源码 aad62d7 中的 `material_fallback.py` 与 `mdl_preview_materials.py`，许可证沿用仓库根 LICENSE。

`OUTPUT` 使用专属新目录。生成的 `scene.usda` 通过相对路径引用传入的外部资产；整体移动后资产相对布局改变时，请重新生成。原始 USD 不被修改。每个场景包含：

- `scene.usda`：可直接加载的环境与 3 个动态刚体。
- `episodes.json`：物体状态、起终点、路径长度、几何任务类型和验证状态。
- `evaluation_grid.npz`：离线筛选网格。
- `layout.png`：布局与参考路径俯视图。

运行一个任务时，在停止时间轴后重新打开该场景，初始化机器人至任务起始位置与朝向，等待物理稳定，然后开始计时。重新开场景将还原所有物体位姿及速度；不要只重置机器人。相机/LiDAR、机器人型号与完整外形、控制器、观测接口和动作空间需由后续 baseline 适配器固定。

评测至少记录成功率、实际路径、交互次数、物体位移和跌倒/碰撞；耗时可记录，不设交付式耗时门槛。正式跨场景划分和多次随机扰动评测留待场景人工审查后冻结，避免将同一场景的不同任务作为独立泛化场景。
