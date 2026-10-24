# Go2 真机部署

入口是 `python run.py`，默认只观察。复用仓库里的导航算法，不提取 core、不复制算法。

**当前后端是宇树 SportClient 高层速度控制。保留 MCF/主运控服务运行。**
不调用 `ReleaseMode()`、不切换主运控、不发布关节 `LowCmd`，也不自动修改遥控响应。
自己的 loco 策略不在这个版本中。

## 目前可以做什么

- 默认接入实测可接收的 `/utlidar/robot_odom` 和 `/utlidar/cloud_deskewed`。
- RGB-D 是可选输入；当前实机配置关闭它，几何点导航不依赖视觉模型，语义找物暂不开放。
- 使用原有 `PointNavigationComponent` / `SemanticExplorationComponent` 做点导航与目标探索。
- 点导航先检查安全直达：保留障碍膨胀，再加 `direct_path_tracking_margin`（默认 0.10 米）；机身半径加该余量覆盖的通行带必须全部已观测、无障碍，包含起终点和地图边界。栅格检查还保守考虑单元格尺寸。
  不满足时走原有 Voronoi，Voronoi 失败仍回退 A*。执行直达时持续检查余量，地图更新导致余量不足时重新规划。
  此规则不改动语义探索的路线选择；设为 `null` 可恢复点导航原来的 Voronoi 优先行为。余量是待实机验收的参数，不代表已经消除了跟踪误差。
  任务统计的 `direct_plans` / `voronoi_plans` / `voronoi_plan_fallbacks` 分别记录直达、骨架成功与骨架失败回退次数。
- 使用现有 DINO、MobileSAM、Qwen-VL 服务；异步感知不阻塞运动输出，旧任务/过期结果丢弃，证据使用真实帧 ID 去重。新任务不能仅凭旧地图中的目标票数宣布找到。
- 独立进程维护命令期限、限速、加速度限制、机器人状态检查和遥控接管按键监听。
- 暂停保留目标和地图，恢复重新检查与规划；任务结束保留控制台。
- JSON/NPZ 观测记录与离线点导航回放，回放绝不初始化 SDK。

**软件链路可离线验证，不等于已经通过真机验收。**默认配置依据用户提供的 SSH 只读检查：

| 项目 | 已确认 / 尚缺条件 |
|---|---|
| 计算机 | Go2 EDU 随身 Orin NX，约 16 GB；Ubuntu 20.04 ARM64、L4T 35.3.1 |
| ROS | Foxy、CycloneDDS、eth0；使用系统 Python 3.8，不能直接替换为 Conda Python |
| 局部位姿 | `/utlidar/robot_odom` 实际收到，`odom → base_link`；不是全局定位 |
| 默认点云 | `/utlidar/cloud_deskewed` 实际收到，点已在 `odom` 中，不再重复施加机身位姿 |
| 其他输入 | `/utlidar/cloud`、`cloud_base`、`imu`、`robot_pose` 实际收到；默认导航不重复融合这些输入 |
| 视频 | `/frontvideostream` 的 `unitree_go/msg/Go2FrontVideoData` 实际收到；不是 RGB-D；导航不解码该自定义话题，独立可视化通过 SDK 获取 RGB |
| 全局定位 | `/uslam/frontend/odom`、`/uslam/localization/odom`、`/lio_sam_ros2/mapping/odometry` 检查时均无发布者 |
| 未确定 | 本体固件、雷达 L1/L2 型号、相机型号、雷达外参、地面高度、自回波过滤和遥控仲裁 |
| DDS | 检查时出现 `invalid data size`，尚未证明消息定义/中间件兼容，运动验收标记保持 false |

存在 MID360 配置不代表连接了 MID360；未检测到 RealSense，也未发现标准 RGB-D 话题。
默认不启动新雷达驱动、SLAM 或相机节点。

首版限制：平整室内地面、固定地图范围；定位跳变后停止并重新启动建立新地图，尚不处理回环重建。
不包含传感器驱动、SLAM 算法安装、推箱/推门和自研步态。
探索默认使用现有几何 frontier 评分；可选 Qwen3 拓扑评分异步调用，返回前、失败或结果过期时沿用几何评分。
回放目前验证几何导航，不重放历史视觉推理结果，也不模拟动力学。

## 目录与调用关系

```text
view.py → visualization/server.py → 浏览器点云 / RGB 界面
             └─ visualization/sources.py → ROS 只读订阅 + 独立 SDK 取图进程
run.py --diagnose → robot/diagnostics.py（只读订阅）
run.py → app/runtime.py
           ├─ app/keyboard.py → app/tasks.py
           ├─ robot/sensors.py → localization.py → observation.py
           ├─ 原有 grutopia_extension/interactive_navigation 算法
           ├─ perception/client.py → 远端模型服务
           ├─ robot/motion.py（独立进程）→ safety.py → SportClient
           └─ recording/recorder.py

serve_models.py → perception/server.py → 原有 serve_semantic_perception.py
```

观测采用米、秒、弧度；机器人四元数为 **wxyz**，速度为机身前/左/上轴对应的 `[vx, vy, wz]`。
配置中的 `map_frame` 表示规划坐标系，当前为 **odom**；仅其他坐标系的点云才按采集时刻查询 TF 转换。相机反投影使用光学坐标系（右、下、前），
视觉覆盖模块的相机朝向则转换为前、左、上，不能混用这两个约定。
`ground_z` 是平地在规划坐标系中的 Z；适配器减去它，使原有算法的高度检查相对地面。
不能把里程计初始 Z=0 解释为机器人趴地，也不能直接猜一个高度填入。
`lidar_origin_offset_base_m` 是雷达测距原点相对 base_link 的实测 `[x,y,z]`。
射线起点采用点云采集时刻的机身位姿和雷达安装位置；用时间邻近位姿插值，超出误差限制便等待新数据。
世界坐标点云的 frame 原点不是雷达原点，不能从固定的 `[0,0,0]` 清除自由空间。

两项标定默认都是 `null`：此时可接收、记录原始坐标观测，但不会生成导航地图或开放运动。
每次启动建立新地图，不跨次复用 odom 地图；长时间里程计漂移仍可能使地图失准，当前未实现全局重定位。

## 1. 开发机同步代码

在**笔记本本地终端**、仓库根目录执行。下列用户名、地址和目录需替换。
先确保机器人上的目标目录已存在，正在运行的任务已停止。

```bash
bash deploy/scripts/sync.sh user@go2 /home/user/inter-nav-project
bash deploy/scripts/sync.sh user@go2 /home/user/inter-nav-project --apply
```

第一条仅预览，第二条才同步。包含 Git 跟踪与非忽略的新文件，不同步 `.git`、环境、
忽略的场景/模型/结果目录，不删除远端文件，并写入来源版本 `.deploy-source.json`。正式更新建议同步到新目录，避免已删除旧文件残留。设备自己的配置放到仓库外，运行时用
`--config-dir /home/user/go2-config` 指定，避免代码同步覆盖标定。

## 2. Go2 计算机安装

通过 SSH 登录。在安装的 ROS 2 环境中执行（ROS 版本必须与机器人端实际系统匹配）：

```bash
source ~/unitree_ros2/setup.sh
cd /home/user/inter-nav-project
bash deploy/scripts/install.sh
source .venv-deploy/bin/activate
cd deploy
python run.py --check
```

安装脚本仅创建 `.venv-deploy` 并安装 CPU 依赖，不安装 Isaac、模型、ROS 或 SDK，不修改系统网络。
环境使用 system-site-packages 以访问系统 ROS；安装结果保存在 `.venv-deploy/installed.txt`。
安装脚本先检查 Foxy 和系统 Python 3.8，并实际导入 ROS 消息；依赖只安装二进制 wheel，避免在 Orin 上隐式编译。
Python 3.8 使用 NumPy 1.24.4、SciPy 1.10.1、Requests 2.32.4；CPU OpenCV 固定为 4.10.0.84，NumPy 不升级到 2。
这些版本的 Python 支持见 [NumPy](https://pypi.org/project/numpy/1.24.4/)、[SciPy](https://pypi.org/project/scipy/1.10.1/)、[Requests](https://pypi.org/project/requests/2.32.4/)。
还需与本机 Python/ROS 匹配的 `rclpy`、`tf2_ros`、标准消息，以及已核对版本的
`unitree_sdk2_python`。如果用本地 SDK 源码，在已激活的部署环境中执行：

```bash
python -m pip install -e /你的路径/unitree_sdk2_python
```

`--check` 是**离线依赖/配置检查**，不连接机器人。它会分别显示 CPU 依赖、可选硬件依赖和运动配置状态，
退出码 0 不代表设备已就绪。机器人端运行不用调用整个仓库的 `setup.py`。

## 3. 配置真实输入

编辑三份配置，或复制到自己的配置目录：

- `configs/robot.yaml`：实测话题、odom/base 坐标系、地面高度、雷达外参、相机模式、时间误差和设备验证结果。
- `configs/navigation.yaml`：地图边界、机器人包络、规划与探索参数，超时采用秒。
- `configs/runtime.yaml`：速度限制、模型服务地址和记录设置。

ROS 输入契约：

| 输入 | 类型 | 必须满足 |
|---|---|---|
| 定位 | `nav_msgs/Odometry` | 直接使用 pose；header 为 odom、child 为 base_link 时无需 TF；其他父坐标系才需转换。twist 为机身速度 |
| 点云 | `sensor_msgs/PointCloud2` | FLOAT32/FLOAT64 x/y/z、米制；支持字段偏移、行填充和大小端。odom 点云无需 TF；去畸变与自身回波仍需实测核对 |
| 彩色图 | `sensor_msgs/Image` | rectified rgb8/bgr8/rgba8/bgra8 |
| 对齐深度 | `sensor_msgs/Image` | 32FC1 米或 16UC1 毫米，与 RGB 同分辨率、同光学坐标系 |
| 内参 | `sensor_msgs/CameraInfo` | 对应上述校正后的图像，校正后投影 P 与分辨率匹配（无有效 P 时使用 K） |

前三项几何参数（地面、雷达外参、时间配对）必须先核对。`ground_z` 应根据静止时的地面点云和实际机身离地高度确认；
例如已测得 base 原点离地 h、同一时刻里程计 z 为 z_base，则平地参考为 z_base−h。这里 h 必须测量，不能直接套默认站立高度。
雷达安装位置应以设备标定/实际 TF 和坐标定义为准，不猜 L1/L2 的通用外参。
如后续接入 RGB-D，再将 `camera_mode` 改为 `rgbd` 并填写三项相机话题，验证内外参和同步。
前置视频经网络送到 Qwen 只能提供图像判断，尚不能给现有语义地图提供可靠的目标三维坐标；本版不伪造深度或把单目视频当作找物闭环。

点云不能把未知的无回波区域伪装成清空射线。默认最大数据年龄 0.4 秒、RGB-D 配对误差 0.04 秒。
这些是起始配置，需要根据实测频率、延迟和停止距离调整。真实输入使用同步的系统时间，不能开启仿真时钟。
固定地图范围必须覆盖实际工作区域；定位大跳变会锁定错误，要求重新建图，不能直接继续沿旧地图行走。

### 只读诊断与 DDS 排查

在 Go2 的已初始化 ROS、已激活部署环境中，从 `deploy/` 执行：

```bash
python run.py --diagnose --duration 10 --output /tmp/go2-ros-diagnostic.json 2>/tmp/go2-dds-errors.log
```

它只订阅，不创建 SportClient、不发送控制命令。记录实际收到的消息数、频率、最大接收间隔、帧名、时间戳、
点云字段/字节数、当前发布者数量、Python/ROS 路径，以及本机 `.msg` / `.idl` 文件哈希。
无法加载自定义视频类型时，会保留错误并继续检查点云与里程计；不直接输出巨大的点云/视频载荷。
退出码 0 只说明本次收到默认位姿和点云，不代表 DDS、全局定位或运动验收通过。

`invalid data size` 可能发生在进入 Python 回调之前，因此额外保留 stderr；JSON 不会自动宣称没有 DDS 错误。
先根据错误日志确认受影响话题，对比发布端与订阅端 `unitree_go` 消息定义和生成环境，检查 ROS 1/ROS 2 环境混用、
Python 路径及 CycloneDDS 版本。只有本机定义哈希不足以证明两端一致。不要仅因收到了几个包就勾选通信通过。
诊断不修改 DDS XML、网卡或系统服务，不凭错误文字自动判定具体原因。

## 4. GPU 机器启动模型服务

**当前视频输入条件不足，先完成几何接入即可，无需现在启动模型。**配置为 RGB-D 后才创建感知客户端。
GPU 机器也放一份同版本仓库。模型使用已有推理环境，**不要把机器人 CPU 环境当成模型环境**。
模型和 MobileSAM checkpoint 应提前准备；没有权重时不宣称服务可用。
在三个终端分别运行，以下相对路径从 `deploy/` 目录执行：

```bash
python serve_models.py grounding-dino --host 0.0.0.0 --device cuda:0
python serve_models.py mobile-sam --host 0.0.0.0 --device cuda:0 --mobile-sam-checkpoint /实际路径/mobile_sam.pt
python serve_models.py qwen-vl --host 0.0.0.0 --device cuda:0 --qwen-vl-model /实际路径/Qwen3-VL-8B-Instruct
```

是否能共享同一 GPU 取决于显存；按机器配置分配 device。默认端口依次为 12181、12183、12185。
将 Go2 的 `runtime.yaml` 中地址改成 GPU 机器的可达 IP。`127.0.0.1` 只表示调用端本机。
服务沿用现有 HTTP 接口，没有额外认证层，应放在受控网络内，不直接暴露到公网。

可选的 Qwen3 拓扑评分（与 Qwen-VL 分类是两个模型）也复用已有实现：

```bash
python serve_models.py qwen3 --model /实际路径/Qwen3-8B --host 0.0.0.0 --device cuda:0
```

在 `runtime.yaml` 的 `topology_scorer` 中设置 `enabled: true` 和 GPU 机器地址。
服务接收拓扑评分提示并返回评分文本；机器人端沿用现有评分解析器。
只采用目标、候选点位置与拓扑内容匹配且未过期的结果，不阻塞导航等待模型。

## 5. 观察，再启用运动

Go2 启动并能正常遥控后，保留 MCF 运行。在 SSH 终端加载 `source ~/unitree_ros2/setup.sh`；本次检查的点云和里程计已存在，无需另启驱动。激活部署环境后执行：

```bash
python run.py
```

默认只观察，**不创建 SDK 运动进程**。未填几何标定时只录制数据；填写并核对地面/雷达原点后才持续更新地图。记录保存到仓库 `results/deploy/`。
当前是终端状态界面，没有实时图像窗口；通过录制的 NPZ、地图和已有工具检查观测对齐。

运动前必须实际完成：外参/坐标/时间核对，点云去畸变和自身回波核对，遥控与 SDK 指令仲裁核对。
还须完成 DDS 消息兼容与接收稳定性排查。然后设置 `calibration_verified`、`deskew_and_self_filter_verified`、
`communication_verified`、`remote_arbitration_verified`，
填写网卡与经过验证的遥控接管按键 mask，才允许：

```bash
python run.py --allow-motion
```

这些字段是设备验收记录，不能为了绕过报错直接改成 true。
没有关闭遥控响应：必须先验证具体固件的原生摇杆与 SDK 行为；程序并不保证“随便推摇杆就接管”。
接管键通过 `rt/lowstate.wireless_remote` 的第 2、3 字节位掩码读取，具体按键不能在未知遥控器上猜测。

| 按键 | 行为 |
|---|---|
| n | 先停止当前任务，再输入地图 `x,y`，例如 `2.0,1.0`；Enter 提交 |
| e | 先停止，再输入英文目标；多词用下划线。当前无 RGB-D，提交后明确拒绝找物，不发送行走指令 |
| p / r | 暂停 / 明确恢复；暂停继续更新地图，不消耗任务时间 |
| 空格 | 在菜单或输入目标时均停止并取消任务 |
| o / m | 停止自主任务并停止持续自主指令；保持观测，不修改遥控器设置 |
| q / Ctrl+C | 先关闭运动输出并尝试确认停稳，再保存地图退出 |

Esc 取消目标输入。任务成功或耗尽后只结束任务，控制台仍在；停止请求不等于已停稳，
以独立运动进程报告的 `stopped` 为反馈：它要求停止请求之后的新状态反馈，且速度连续满足静止阈值达到配置的停稳时长。`m` 没有虚构一个“宇树模式切换”：
因为未关闭 MCF、未禁用遥控，它只是停止本程序的自主输出，遥控实际响应需实机验收。

控制进程独立于导航、视觉和记录。命令过期、机器人反馈过期、倾角异常、接管键触发后锁定停止，
后来的旧命令不会自动恢复行走。恢复需要显式操作且机器人状态新鲜、已静止。
停止反馈来自 `rt/sportmodestate`，状态或网络失效时不声称停稳。整个机器人侧进程/计算机故障或彻底断网，
仍依赖经过验证的机器人端保护，Python 进程不能保证这种情况下的物理停止。
普通 SSH 断开通过 SIGHUP 走退出流程；使用 tmux 时断开 SSH 不会退出任务，必须显式停止或遥控接管。

## 6. 离线验证与回放

无需 Go2、ROS、SDK、GPU 或 Isaac。以下生成的是**合成输入轨迹，不是物理仿真**：

```bash
python tests/make_fixture.py --output /tmp/inter-nav-deploy-fixtures
python run.py --replay /上条命令打印的目录 --task navigate --goal 1.0 0.0
```

未标定的现场记录可以用 observe 模式回放，但不能直接用来离线点导航；需先完成标定并重新录制。
回放输出 `sent_to_robot=false`，通过已有导航算法生成决策、保存最终地图。
`--replay` 与 `--allow-motion` 互斥。记录用数值 NPZ 与 JSON，禁用 pickle。
记录结束但任务没到点时报告 `recording_exhausted`，不能当作导航成功。

从仓库根目录运行测试：

```bash
python -m pip install pytest
python -m pytest deploy/tests -q
```

## 输出与验收边界

每次创建新的 `results/deploy/时间戳/`：

```text
config.json                # 配置、Git commit、工作树是否有未提交改动
events.jsonl               # 操作、规划速度请求、运动状态及错误
observations.jsonl         # 观测索引和采集时间
observations/*.npz         # 原始适配后的数值观测
final_map.json/.npz        # 仅已完成几何标定并实际建图时生成
run_summary.json           # 任务结果、结束原因、停止反馈
```

观测写盘使用有界后台队列；磁盘跟不上时丢弃观测并在报告中计数，不阻塞控制。
记录的速度请求不是电机执行确认；停稳反馈也不等于长期无漂移。机身实测速度、低速跟踪、
混合转弯、停止距离、脚部稳定性、窄门通过、断网/接管与模型超时都需要实机分级验收。
先做只读对齐，再验证停止与接管，再做空旷点导航，最后做绕障和语义探索。


## 实时点云与 RGB 可视化

另开一个 SSH 终端运行 `python view.py`，与 `python run.py` 独立运行。
界面借鉴 zkj 版本的点云视角和 RGB 分屏；原版渲染器依赖仿真对象并用于录像，
这里改成直接读取真机传感器的浏览器界面。没有 Isaac、Qt、Open3D、Node 或外部 CDN 依赖。

**笔记本终端：** 将 `<用户>`、`<Go2地址>` 和下面的项目路径替换为实际值。

```bash
ssh -L 8080:127.0.0.1:8080 <用户>@<Go2地址>
```

**进入 Go2 的 SSH 终端后：**

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
source .venv-deploy/bin/activate
cd deploy
python view.py
```

然后在**笔记本浏览器**打开 `http://127.0.0.1:8080`。如果导航已经在另一个
SSH 终端运行，保持它运行即可。此处无需 `--allow-motion`，不修改标定验收开关。
服务只监听 Go2 的回环地址，通过 SSH 转发访问；不需要 SSH 图形转发或在 Orin 上接显示器。
端口占用时，用 `--port 8081`，SSH 的两处端口及浏览器地址也一起改成 8081。

界面功能：

- 最新一帧三维点云，按高度着色；拖动旋转、滚轮缩放、Shift + 拖动平移，支持俯视与复位。
- 点云保留消息本身坐标系，不重复叠加里程计。不做历史点云累积，不把它当作导航占据地图。
- 位姿与点云坐标系相同、时间差不超过 0.1 秒且均未过期时，才叠加机器人位置/朝向。
- RGB 和点云独立刷新；不要求 RGB-D、地面标定或导航运行，不把图像与点云强行融合。
- 显示接收时间、点数和错误。数据过期会变暗并提示，服务断开也不会把最后画面当作实时数据。
- 断流阈值为 `max(max_data_age_s, 2/fps)`（默认 0.4 秒），与运动安全阈值独立。
- 默认上限 5 Hz、12000 点、RGB 最大 960×540；单帧缓存，不积压历史。可用
  `--fps 2 --max-points 6000` 降低 CPU / 带宽占用。这不是运动控制频率。

默认 RGB 使用宇树官方
[VideoClient.GetImageSample](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/video/video_client.py)，
需要安装 `unitree_sdk2py` 且视频服务可用。它运行在独立进程，避免与 Foxy 的 DDS 库混用；
仅请求图像，不初始化 SportClient，也不发布运动命令。
**此前实机只确认 `/frontvideostream` 收到消息，SDK 取图还未实机验收。**
本实现不猜测 `Go2FrontVideoData` 的编码；取图失败会显示 SDK 返回错误，点云继续显示。
SDK 图像没有采集时间戳，界面只能报告接收距今时间，不能证明相机端没有重复旧帧。

其他输入选项：

```bash
# 仅显示点云，排查相机问题时使用
python view.py --camera none

# 已有标准彩色 sensor_msgs/Image 话题时，不需要深度或相机内参
python view.py --camera ros --rgb-topic /camera/color/image_raw

# 本地预览界面：合成点云与 RGB，完全不连接机器人
python view.py --demo
```

关闭浏览器或对 `view.py` 按 Ctrl+C，只关闭可视化；**不会停止另一个终端的导航**。
它不显示任务状态、规划路径或模型识别结果，也没有运动按钮。原 `run.py` 的键盘操作保持不变。

真机验收顺序：先运行只读界面确认点云、坐标系与 RGB；临时停止输入或断开 SSH，
确认界面能提示断流；最后与已验收的导航同时运行，检查 Orin 负载与导航频率。
只读可视化通过，不等于 DDS 通信、导航标定和运动接管已经验收。
