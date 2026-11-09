# deploy 对齐仿真发布分支（2026-11-09）

基线是 `origin/stage1-go2-simulation-release` 的 `67112e4606ee42dae70ba0bc37d9f74b46deb30e`。
旧 deploy HEAD 为 `4457c74`。操作前的 8 个未提交文件已保存为本地提交 `54e27ca`，
包含命令过期保护、arm 后重新采集、当时的高度射线修改及测试；未丢弃现场修改。

两个分支没有共同祖先。合并保留双方历史，共享目录完整采用仿真发布分支，删除旧分支独有的共享实现；
再在 `deploy/` 重新适配。合并不把旧 deploy 的共享算法补丁叠到新实现上。
共享源码只有一个额外修复：精确射线端点在浮点栅格边界上时，已到达目标格的轴不再越过目标，
避免 `world_ray_cells` 无法终止。另有回归测试和部署输出目录忽略规则。

## 与原 deploy 的差别

| 项目 | 原 deploy | 当前 deploy |
|---|---|---|
| 导航核心 | 旧 PointNavigationComponent / MapNavigationRuntime | OnlineNavigationMap、OnlineGoalSelector、SidePushPlanner、Voronoi 路线和连续跟踪器 |
| 未知目标 | 尚未观测的目标直接拒绝 | 接受地图范围内的目标，沿已观测可达前沿逐步探索，未知格仍不可直接穿越 |
| 路线选择 | 直达优先补丁，之后 Voronoi/A* | 使用新仿真的路线、局部几何检查、姿态/缓冲区恢复，以及绕行与推移比较 |
| 交互决策 | deploy 没有推移执行入口 | 类别确认、技能许可、主动观察、PushAsideSkill、停稳交接和通道重新观测接入任务链 |
| 推移执行 | 无 | 实测姿态/速度/物体反馈进入新闭环技能，输出机身速度；失去新鲜目标即结束并停止 |
| 几何来源 | LiDAR + 可选旧语义地图 | LiDAR 在线分层地图；交互几何来自验收后的真实 RGB-D 可见点，不使用仿真物体真值 |
| 完成判定 | 到目标约 0.30 m 后结束 | 小于 0.25 m 且实测稳定约 0.4 s，再触发 SDK 停止确认 |
| 找物 | `e` 按类别寻找目标，含旧视觉记忆/Qwen3 评分 | 删除旧入口和服务；保留坐标任务中的前沿探索及交互对象识别 |
| 地图实现 | deploy 独有按高度累计障碍证据 | 新仿真的实际射线证据、局部/全局更新、独立物体层；旧高度实现不再保留 |
| 显示点云 | 旧 voxel 清除接口 | 新版 voxel 累积历史点，标明只供显示；通行判断看实时地图 |
| 任务切换 | 保留地图、重置旧组件路径 | 保留结构地图，重置执行/识别权限；旧物体专属区域回未知，防止新 ID 与历史 ID 叠加 |
| 记录 | 观测/地图/任务事件 | 加入在线决策、技能状态、分层图、真实分类裁剪图证据，记录不依赖临时图片路径 |

## 保留的真机接口

- ROS 点云/里程计解码、时间配对、单次 TF 转换、地面/LiDAR 外参与固定 experiment 坐标。
- SportClient 作为唯一运动输出，继续保留 MCF，不导入仿真关节策略、不释放模式。
- 独立运动进程的 TTL、限速、加速度限制、接管锁存、异常停止和停止反馈。
- 开始任务前的验收配置；arm 后重新取观测，规划耗时超期丢弃命令。
- run / view / record 分工、独立显示进程、离线回放、短时 SDK 验收、控制流录制与校时工具。
- 此机的已保存话题、标定和障碍高度设置。没有把仿真测试成绩解释成新的真机验收。

## 尚未完成的硬件验收

当前默认设备只有单目预览，`camera_mode=unavailable`，因此交互推移默认关闭。
软件已经接入新技能，但要实际启用还缺：对齐 RGB-D、可靠完整物体包络、足够低延迟的目标位姿更新，
以及 SportClient 接触/推移/侧移/退让验收。具体配置见 [README.md](README.md)。

感知适配器从分割点计算可见范围，不能自动恢复隐藏表面；`geometry_verified=true` 是真实传感器包络验收声明，
不是“相信模型”开关。默认不会把不完整几何注入物体层。定位漂移、DDS/固件兼容及真实停止距离仍须设备验证。

本次没有连接机器人、没有发运动命令、没有运行 Isaac 全场景或现场推移实验。

## 迁移使用

旧配置中 `direct_path_tracking_margin`、`lidar_height_resolution`、`voxel_clear_observations`、
旧 frontier/visual memory 参数和 `topology_scorer` 已失效；加载会明确提示迁移。
用新 `deploy/configs` 为模板，保留设备实际标定/网卡/话题/验收值，设置新版地图刷新和速度边界。
配置会检查导航速度/加速度不超过底层 SDK 保护限制。

同步脚本不删除远端文件，应同步到新目录，防止旧模块残留。此次工作只作本地合并提交，未推送远端。

## 本次验证结果

测试解释器：`/home/seeker/miniconda3/envs/grutopia/bin/python`，Python 3.10.21。

| 验证 | 结果与边界 |
|---|---|
| `python -m pytest deploy/tests -q --disable-warnings` | **292 passed**；含控制过期/接管、传感器适配、两帧视觉许可、技能接入、身份过期、地图/可视化与真实子进程记录回放 |
| 共享框架 CPU 回归 | **985 passed, 2 skipped, 3 deselected**；见下面的完整命令 |
| `python deploy/run.py --check` | 通过 CPU 导入/配置检查；本机没有 ROS/SDK，输出明确 interaction_enabled=false |
| 历史实机点云离线回放 | **519/519 帧**处理成功，初始地图全未知，完成后 2594 个已观测格；无错误、不发运动命令、不含推移实验 |
| Python 3.8 兼容性 | 57 个实际导入的本仓库模块通过 3.8 语法检查；未在 Orin/Python 3.8 运行安装 |
| shell / 修改格式 | 安装、同步脚本 `bash -n` 通过；相对 release 的适配差异 whitespace 检查通过 |

共享 CPU 测试命令：

```bash
python -m pytest tests -q -p no:nose --disable-warnings \
  --ignore=tests/test_carton_materials.py \
  --ignore=tests/test_material_compaction.py \
  --ignore=tests/test_mdl_preview_materials.py \
  --ignore=tests/test_task_difficulty.py \
  --ignore=tests/test_texture_compaction.py \
  -k 'not small_composed_asset_is_honored_before_sibling_or_download and not asset_validation_requires_go2_joint_structure and not g1_third_person_is_close_and_elevated'
```

这些排除项是本机缺少 `pxr`/`shapely` 的资产/网格测试，以及 release README 已排除的 G1 摄像机用例。
首次直接全仓库测试因此出现导入错误/资产用例失败；没有把它算作“全仓库全部通过”。
`-p no:nose` 避免本机旧 pytest 的 nose 插件把测试中的 `setup(...)` 辅助函数误当模块初始化。
继承的 `delivery/assets-source/after_scene.usda` 与 `tests/test_fluent_tracking.py` 原有末尾空行未作无关修改。

实机记录输入（只读）：
`results/deploy/handoff-20261105/extracted/go2-mapping-review-20261105/repo/results/deploy/record-copy-20261105/19700113T114010_809990Z`。
回放输出：`results/deploy/release-migration-20261109/20261109T055238_576866Z/run_summary.json`，
同目录 `final_map/layered_map.npz` / `layered_map.json` 保存新分层图；结果目录保留本地且不提交。
该回放验证已有传感器数据可以进入新建图链，不验证定位精度、真实导航或推移执行效果。
