# Go2 真机部署

`deploy` 已合入 `stage1-go2-simulation-release`（基线 `67112e4`）。共享算法以该分支为准，
真机层负责传感器、时间与坐标转换、任务生命周期和宇树 SDK 输出。迁移说明见 [MIGRATION.md](MIGRATION.md)。

## 运行入口

| 入口 | 职责 |
|---|---|
| `python run.py` | 采集定位/点云、维护在线地图；默认只观察 |
| `python view.py` | 读取 run 的 RGB、导航地图、历史显示点云，提供浏览器界面 |
| `python record.py` | 从 run 保存观测、地图、决策、事件和结果 |
| `python serve_models.py …` | 在 GPU 主机运行 DINO / SAM / Qwen-VL 服务 |
| `python sync_time.py` | 独立的时间诊断/维护工具 |

只有 `run.py --allow-motion` 的任务键能启用自主输出；view/record 不发运动命令。
运动后端仍是 **SportClient 高层机身速度**，MCF 保持运行，不使用仿真的关节策略或 LowCmd。
`--check`、`--demo`、`--replay` 均不连接机器人运动接口。

## 新的导航与推移链路

```text
ROS odom + LiDAR + 可选对齐 RGB-D
    → normalize → 本轮固定 ExperimentFrame
    → Tasks → NavigationEngine
        ├─ OnlineNavigationMap：局部/全局/物体分层，未知区域保持未知
        ├─ OnlineGoalSelector：目标直达/绕行、目标导向前沿探索、推移比较
        ├─ ExplorationVoronoiRoutes + SmoothExplorationTracker：路线和机身运动检查
        └─ RecognitionStore + InteractionCatalog → PushAsideSkill
             → 接近、接触、推移、退让、重新观测、通道证书、重新导航
    → 命令有效期检查 → 独立 MotionProcess → SportClient.Move/StopMove

run 内存会话 → view / record（只读）；历史显示点云在独立进程维护
```

- 输入 `n` 后填写本轮 experiment 坐标 `x,y`。地图内尚未观测的目标允许提交，
  系统沿已观测可达区域选择前沿探索；不会把未知区域直接变成通路。
- 已观测的障碍内目标拒绝。地图更新后检查目标与运动路径，不能因为上一帧安全便继续盲走。
- 绕行与推移复用新仿真的比较方法；未获视觉类别许可的物体始终作为障碍。
- 到达距离默认 **小于 0.25 m**，还必须通过连续约 **0.4 s** 的实测停稳检查才报告成功；
  底层 StopMove 后的 SDK 停止反馈仍单独记录。
- `p` 暂停、`r` 恢复、空格取消、`o` 观察、`m` 停止自主输出、`q` 退出。
  暂停/切换保留地图；恢复重新规划并清除旧接触状态，不从中断的推移动作继续。
- 已移除旧 `e` 按名称找物入口、旧 Qwen 拓扑评分、直达优先补丁和旧高度分层建图分支。
  坐标导航中的“前沿探索”仍保留，采用新在线框架。

## 推移输入与当前边界

默认 `navigation.interaction_enabled: false`。当前这台 Go2 的 `camera_mode: unavailable`，
SDK 视频只用于预览；它没有对齐深度和物体三维几何，不能满足推移闭环。

软件适配已包含异步 DINO→SAM→Qwen-VL、实测 RGB-D 点的空间跟踪、两独立帧类别确认、
`RecognitionStore` 和 `PushAsideSkill`。感知队列有界，任务重置、帧重复和过期结果会被过滤。
几何时效按**采集时间**检查，不因模型返回得晚而刷新；推移过程中失去新鲜物体几何或许可就结束任务并停止。
没有把仿真场景的物体 ID、尺寸、类别、真值位姿注入真机。

启用推移需同时满足配置中的真实输入条件：

1. `robot.camera_mode: rgbd`，明确 RGB、对齐深度、CameraInfo 话题，验证内外参和时间配对。
2. `runtime.perception.geometry_verified: true`：已验证检测点集给出的物体包络覆盖实际物体。
   单视角分割点只能给出**可见表面范围**，通常不能覆盖隐藏背面；仅有一个检测框或类别结果不满足此条件。
   默认 false 时不向规划器注入未经验证的完整物体几何；当前默认 run 不启动交互识别。
   感知适配器可在独立测试中保留候选诊断。
3. `robot.interaction_motion_verified: true`：另行验证 SportClient 接触、侧移、退让与停止响应。
   仿真使用的步态、摩擦和推移参数不能作为此项实机验收证据。
4. `navigation.interaction_enabled: true`，以及原有自主导航验收项全部满足。

默认几何有效期 0.75 s，输入新鲜度 0.5 s，视觉请求总预算 8 s。
模型结果可以晚于几何有效期返回，此时分类证据不等于可用的实时接触位姿；需要足够低延迟的
实测几何更新。离线测试通过不代表已经完成 RGB-D、SDK 接触控制或真机推移验收。

## 输入坐标和时间

当前默认话题是 `/utlidar/robot_odom` 和 `/utlidar/cloud_deskewed`。
点云已在 `odom` 时不重复施加机身位姿；其他坐标系按采集时间查 TF。
点云原点采用采集时刻插值的机身位姿与 LiDAR 安装偏移，不能用 odom 的固定原点清除自由空间。

- 单位为米、秒、弧度；机器人四元数为 **wxyz**。
- 控制输出为机身前/左轴的 `[vx, vy, wz]`；Odometry 完整的线速度 xyz、角速度 xyz 另外保留给停稳与推移反馈。
  `wz` 不作为竖直线速度，旧录像缺少完整 twist 时不会获得推移许可。
- 首组标定且新鲜的观测固定 experiment 原点：起点 XY=(0,0)、X=初始前方、Y=初始左方、Z=标定地面。
  暂停、换目标和启动 view/record 都不重设它。该变换不消除里程计漂移。
- 默认固定地图 X/Y 为 [-10,10] m。定位跳变停止，需重新建立本轮地图；不提供全局重定位或回环重建。
- 当前保存的 `ground_z: -0.012` 来自此机平地记录，`lidar_origin_offset_base_m: [0.28945,0,-0.046825]`
  来自用户接受的 URDF base/radar 对应关系，不是重新进行的物理标定；odom 重置或设备改变后需复核。
- 导航障碍高度仍为距地面 0.15～0.63 m，历史显示点高度 [-0.1,2.0] m；低于/高于该导航范围的物体不由此层表达。
- 新在线地图使用实际射线与局部/全局刷新，不再采用旧 deploy 的按高度累计障碍证据。
  真机输入过滤量程边缘点，避免把它们误当作 PhysX 的“无命中射线”。
- 仅对已核对的 deskewed 话题移除连续 10,000 个零填充前缀，不普遍删除真实原点。
- 采集启动时建立相对时间参照，监控后续延迟；不能检测启动前已存在的共同固定延迟。
  默认数据最大年龄 0.4 s，命令有效期 0.35 s；耗时规划超期后直接丢弃命令并暂停，恢复前重新采集。

## 安装与同步

开发机在仓库根目录执行，替换设备地址和路径：

```bash
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project --apply
```

第一条预览，第二条写入；不会删除远端旧文件。此次移除了旧模块，建议同步到**新的目标目录**，
避免旧文件残留。设备配置单独放到仓库外，通过 `--config-dir` 指定；用新版配置为模板迁移，
保留该机经确认的标定、话题、网卡和验收值，不直接套用旧 YAML。过时字段会明确报错。

Go2 Orin 上使用与已安装 ROS 兼容的系统 Python（已知配置为 Ubuntu 20.04 / Foxy / Python 3.8）：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
bash deploy/scripts/install.sh
source .venv-deploy/bin/activate
cd deploy
python run.py --check
python run.py
```

安装脚本只安装 CPU Python 依赖，保留 system-site-packages；不安装 Isaac、ROS、SLAM、传感器驱动或模型。
还需与设备匹配的 `unitree_sdk2_python`、ROS 消息和 DDS。`--check` 是离线导入/配置检查，返回 0 不代表硬件就绪。

## 观察、记录与离线验证

先启动 run，再分别启动只读客户端：

```bash
python view.py
python record.py --output /你的实验目录
```

run 的数据服务默认 `127.0.0.1:8765`，view 默认 `127.0.0.1:8080`。远程访问用 SSH 转发：

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.18
```

浏览器显示 RGB、导航地图、历史点云；地图保留决策、技能状态和路径数据。
历史点云只作场景回看，可能保留已移动物体的旧位置，**不能作为当前碰撞地图**。
当前碰撞/通路判断来自在线地图及新鲜物体观测。
run 不自动写实验文件，record 保存 JSON/NPZ 观测、场景、事件和来源版本。

无硬件检查（从仓库根目录；pytest 需开发测试环境，安装脚本只装运行依赖）：

```bash
python deploy/run.py --check
python deploy/run.py --demo --camera none --demo-seconds 5
python deploy/tests/make_fixture.py --output /tmp/go2-fixture
python deploy/run.py --replay /tmp/go2-fixture/生成目录 --task navigate --goal 1 0 --output /tmp/go2-replay
python -m pytest deploy/tests -q
```

回放不初始化 ROS/SDK，也不重放视觉模型推理或动力学。记录缺少完整物体输入时只验证几何导航。
真机自主导航在验收配置满足后运行 `python run.py --allow-motion`，再按任务键。
`python run.py --motion-test` 保留独立的短时 SDK 验收流程，不启用建图导航。
