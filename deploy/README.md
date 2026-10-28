# Go2 真机部署

四个入口分工明确：

| 入口 | 职责 | 常用位置 |
|---|---|---|
| `python run.py` | 唯一的数据采集、建图与导航主程序，默认只观察 | Go2 随身计算机 |
| `python view.py` | 从 run 读取 RGB、真实俯视地图、累计点云，提供三图界面 | Go2；笔记本浏览器访问 |
| `python serve_models.py …` | DINO / SAM / Qwen 等远端推理服务 | GPU 计算机 |
| `python record.py` | 从 run 记录这一轮观测、三幅图、地图、事件与结果 | Go2，或经转发连接的笔记本 |

复用仓库里的导航算法，不提取 core、不复制算法。**只有 run 采集和建图；view、record 均为只读客户端。**
主程序不再自动把实机实验写盘，需要单独运行 record。离线 `run.py --replay` 仍会保存回放报告。
`sync_time.py` 是独立维护工具，保留原有校时流程，不是实验运行入口。

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
| 未确定 | 本体固件、雷达 L1/L2 型号、相机型号、自回波过滤和遥控仲裁；地面/雷达原点的本机依据见下文 |
| DDS | 检查时出现 `invalid data size`，尚未证明消息定义/中间件兼容，运动验收标记保持 false |

存在 MID360 配置不代表连接了 MID360；未检测到 RealSense，也未发现标准 RGB-D 话题。
默认不启动新雷达驱动、SLAM 或相机节点。

首版限制：平整室内地面、固定地图范围；定位跳变后停止并重新启动建立新地图，尚不处理回环重建。
不包含传感器驱动、SLAM 算法安装、推箱/推门和自研步态。
探索默认使用现有几何 frontier 评分；可选 Qwen3 拓扑评分异步调用，返回前、失败或结果过期时沿用几何评分。
回放目前验证几何导航，不重放历史视觉推理结果，也不模拟动力学。

## 目录与调用关系

```text
run.py → app/runtime.py（资源创建、逐帧采集、退出清理）
           ├─ robot/sensors.py → observation.normalize → ExperimentFrame
           │       → app/controller.py → safety.check_observation
           │       → app/tasks.py → 原有导航算法 / 唯一地图
           ├─ app/keyboard.py → app/controller.py（任务切换、运动请求、反馈处理）
           ├─ perception/client.py → GPU 上的 serve_models.py
           ├─ robot/motion.py（独立进程）→ safety.py → SportClient
           ├─ visualization/sources.py → 独立 SDK 取图进程（仅预览 RGB）
           └─ app/session.py（有界内存数据服务，127.0.0.1:8765）
                   ├─ view.py → visualization/server.py → 浏览器三图界面
                   └─ record.py → recording/capture.py → 本轮实验目录

run.py --motion-test → app/motion_test.py → 同一个 robot/motion.py（短时验收，无导航）
run.py --diagnose → robot/diagnostics.py（只读订阅）
serve_models.py → perception/server.py → 原有模型实现
```

运行职责：`runtime` 管资源和采集顺序；`controller` 把操作、观测与运动反馈转成明确状态变更；
`tasks` 只适配原有算法、保留本轮地图。任务取消会回到 observe 并清掉旧目标；暂停保留任务，必须显式恢复。
运动进程仍是唯一 SDK 发令者；session 只发布数据，view / record 不能发运动请求。

观测采用米、秒、弧度；机器人四元数为 **wxyz**，速度为机身前/左/上轴对应的 `[vx, vy, wz]`。
配置中的 `map_frame` 表示上游传感器对齐坐标系，当前为 **odom**；仅其他坐标系的点云才按采集时刻查询 TF 转换。相机反投影使用光学坐标系（右、下、前），
视觉覆盖模块的相机朝向则转换为前、左、上，不能混用这两个约定。
`ground_z` 是平地在上游坐标系（默认 odom）中的 Z；适配器减去它，使原有算法的高度检查相对地面。
不能把里程计初始 Z=0 解释为机器人趴地，也不能直接猜一个高度填入。
`lidar_origin_offset_base_m` 是雷达测距原点相对 base_link 的 `[x,y,z]`，依据实测或已确认对应关系的设备外参填写。
射线起点采用点云采集时刻的机身位姿和雷达安装位置；用时间邻近位姿插值，超出误差限制便等待新数据。
世界坐标点云的 frame 原点不是雷达原点，不能从固定的 `[0,0,0]` 清除自由空间。

仓库配置保存这台 Go2 于 2026-10-25 站立在水平地面时拟合的 `ground_z: -0.012` 米，odom 重置或地面/参考系变化后需复核。
2026-10-27 按用户提供的官方 `radar_joint` URDF，采用 `lidar_origin_offset_base_m: [0.28945, 0.0, -0.046825]` 米；
用户接受将 URDF 的 base 视为运行时 base_link、radar 原点视为测距原点，并明确要求将 `calibration_verified` 设为 `true`。
此项验收依据为已有地面测量与用户认可的 URDF 对应关系，不是独立物理测量确认。
URDF 的 `rpy=[0, 2.8782, 0]` 不再应用到已处于 odom 的点云；适配器只按采集时刻机身位姿转换雷达原点。
此前 raw/base 点云配准得到的 `[0.28216, 0, 0]` 是发布流的有效变换，与上述物理安装参数不同，保留为历史证据。
上述 2026-10-27 标定确认未包含去畸变/自身过滤、通信、遥控仲裁验收；后续单项验收见下文。
这些值不适用于未经核对的其他机器人。若任一标定值设为 `null`，可接收、记录原始坐标观测，但不会生成导航地图或开放完整自主导航。
`run.py` 在收到首组已标定、时间匹配且新鲜的观测后，固定本轮 `experiment` 坐标系：
- XY 原点为此时机身位置，X 指向初始前方的水平投影，Y 指向初始左侧；Z=0 为标定地面，保留机身实际离地高度。
- `robot/observation.py::ExperimentFrame` 在任务与边界检查前，统一转换机身、雷达原点、点云，以及 RGB-D 三维点和相机位姿。机身速度、RGB 像素和深度值保持原定义。
- 地图默认 X/Y 各为 `[-10, 10]` 米，围绕本轮起点；输入目标 `2,1` 指初始前方 2 米、初始左侧 1 米的固定位置，并非相对当前朝向。
- 暂停、换任务、开关 view/record 不重设坐标。未标定时仍只显示/记录上游观测，不初始化实验原点。上游定位跳变仍触发既有停止保护。
- `record.py` 的每帧观测包含 `coordinate_transform`，其中 `origin_source_m` 和 `yaw_source_rad` 可还原原坐标；初始化事件及运行总结也保留该参数，晚启动记录仍可从观测获取。
- 回放直接使用记录中的坐标，不再做初始化或重复变换。旧记录缺少该字段时保持原坐标；回放目标须采用该记录对应的坐标系。

每次启动 run 建立新地图，不跨次复用实验坐标；长时间里程计漂移仍可能使地图失准，当前未实现全局重定位。

## 1. 开发机同步代码

在**笔记本本地终端**、仓库根目录执行。下列用户名、地址和目录需替换。
先确保机器人上的目标目录已存在，正在运行的任务已停止。

```bash
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project --apply
```

第一条仅预览，第二条才同步。包含 Git 跟踪与非忽略的新文件，不同步 `.git`、环境、
忽略的场景/模型/结果目录，不删除远端文件，并写入来源版本 `.deploy-source.json`。正式更新建议同步到新目录，避免已删除旧文件残留。设备自己的配置放到仓库外，运行时用
`--config-dir /home/unitree/go2-config` 指定，避免代码同步覆盖标定。

## 2. Go2 计算机安装

通过 SSH 登录。在安装的 ROS 2 环境中执行（ROS 版本必须与机器人端实际系统匹配）：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
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
`/utlidar/cloud_deskewed` 曾实测每帧以 10,000 个 `(0,0,0)` 填充点开头。
输入适配器仅在该话题、`odom` 坐标系且完整前缀匹配时移除这段填充；
不会普遍删除世界原点处的真实点，也不以此认定去畸变和机身过滤已验收。
如后续接入 RGB-D，再将 `camera_mode` 改为 `rgbd` 并填写三项相机话题，验证内外参和同步。
前置视频经网络送到 Qwen 只能提供图像判断，尚不能给现有语义地图提供可靠的目标三维坐标；本版不伪造深度或把单目视频当作找物闭环。

点云不能把未知的无回波区域伪装成清空射线。默认最大数据年龄 0.4 秒、RGB-D 配对误差 0.04 秒。
这些是起始配置，需要根据实测频率、延迟和停止距离调整。真实输入使用同步的系统时间，不能开启仿真时钟。
固定地图范围必须覆盖实际工作区域；定位大跳变会锁定错误，要求重新建图，不能直接继续沿旧地图行走。

### 开机后检查与持续校时

在 **Go2 SSH 终端**加载 ROS 和部署环境后、启动 `run.py` / `view.py` 之前执行：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
source .venv-deploy/bin/activate
cd deploy
python sync_time.py --watch
```

保持这个终端运行，看到“连续校时已启动”后，再在其他终端启动 `run.py` 和 `view.py`。
它以配置中的里程计话题（默认 `/utlidar/robot_odom`）采集时间为参考，先对齐 Orin 系统时间并复测，
随后默认每10秒检查一次，用 Linux `adjtime` 渐进补偿小幅漂移，不在运行中跳变系统时间。
单次连续修正超过20ms、传感器时间不一致或系统 NTP 已同步时，停止校正并报告原因。
停止后需要退出 run/view，再重新启动校时器；不放宽数据年龄或未来时间阈值。
需要调整时会通过 sudo 请求系统权限；不会发送运动命令、修改验收标记或放宽时间校验。
只检查可用 `python sync_time.py --check`；仅一次性检查/对齐仍用 `python sync_time.py`。
独立标定配置用同样的 `--config-dir` 指定。一个用户只允许一个校时写入进程；Ctrl+C 停止连续校时。

这不是互联网标准时间同步：它依赖已确认的里程计时间源，单向消息无法精确测出网络延迟。
没有安装开机服务或修改 RTC；每次重启后重新启动校时器。单次校时不能抵消长期频率漂移。
渐进调整使用系统接口，见 [Linux adjtime 文档](https://man7.org/linux/man-pages/man3/adjtime.3.html)。
不同传感器的时间基准不一致、样本不足或时序波动过大时拒绝校时；不会把消息改成接收时间。
时间检查通过不代表 DDS、标定或运动验收通过。

### 只读诊断与 DDS 排查

在 Go2 的已初始化 ROS、已激活部署环境中，从 `deploy/` 执行：

```bash
python run.py --diagnose --duration 10 --output /tmp/go2-ros-diagnostic.json 2>/tmp/go2-dds-errors.log
```

它只订阅，不创建 SportClient、不发送控制命令。记录实际收到的消息数、频率、最大接收间隔、帧名、时间戳、
点云字段/字节数、当前发布者数量、Python/ROS 路径，以及本机 `.msg` / `.idl` 文件哈希。
无法加载自定义视频类型时，会保留错误并继续检查点云与里程计；不直接输出巨大的点云/视频载荷。
退出码 0 表示本次收到默认位姿和点云，且收到的样本未触发诊断解析错误；不代表 DDS、全局定位或运动验收通过。

当前实机的 `/frontvideostream` 存在同名消息的线格式差异：本机与官方 `Go2FrontVideoData`
声明为 `uint64 + 三路 uint8[]`，实收为 `uint64 + uint32 流标记 + 一路 uint8[]`，
已观测到标记 360、720 和 Annex B 起始码。旧类型解析会把流标记误当数组长度，导致
`invalid data size` / `unable initialize generic sequence`，并拖慢同一诊断进程中的其他回调。
诊断现在对该话题使用 `raw=True` 接收，再按有界 CDR 解析器逐包验证长度、对齐和封装；
同时兼容声明的三路格式。JSON 分别记录 `received`、`validated_messages`、`sample_error_count`、
`wire_layout_counts` 和流标记计数，兼容布局会明确标为 `declared_schema_matches=false`。
未知或损坏包仍报告错误并使诊断退出码为 2；没有关闭错误日志或跳过视频订阅。
这只修复本项目诊断的接收适配，没有替换系统消息包；直接用旧类型执行 `ros2 topic echo`
仍可能报错。视频包结构通过验证不代表图像解码或 RGB-D 验收通过，`time_frame` 不当作 ROS 时间使用。

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
机器人端使用捕获时刻计算整帧感知预算（当前 8 秒），DINO / SAM / Qwen 的后续请求共用剩余时间。
切换或取消目标时清掉排队工作；当前 HTTP 请求返回后，旧任务不会继续处理候选或覆盖新任务结果。
这是请求之间的协作式取消，不强杀请求或撤销远端 GPU 推理，也不保证严格的墙钟截止时间。
具体网络异常会保留在诊断中；服务超时不能当成“没有找到目标”。

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

默认只观察，**不创建 SDK 运动进程**；默认另起 SDK 取图进程，失败不会开启运动。
填写并核对地面/雷达原点后才持续更新导航地图；未标定时，view 显示原始点云并提示地图未就绪。
终端负责键盘操作，另开终端分别运行 `python view.py` 和 `python record.py`。
view 和 record 使用 run 发布的同一个会话编号、配置和数据，不启动第二套 ROS 订阅或导航算法。
若使用独立标定配置，只需给 run 传 `--config-dir`，view 和 record 自动沿用该会话实际数据。

主程序的 RGB 选项：

```bash
python run.py --camera none                      # 不启动预览取图，只看几何
python run.py --camera ros --rgb-topic /实际彩色图像话题
```

`--camera ros` 的预览只需要彩色 Image，不代表启用语义找物。导航 RGB-D 仍由 robot.yaml 单独配置。
配置 RGB-D 后优先显示同一份导航相机观测，不再额外启动 SDK 相机。
旧版 view 的 `--camera` / `--rgb-topic` 移到 run；不会再由可视化进程采集第二路数据。

### 受控单项验收

先结束其他 run，另开终端 `python record.py` 等待记录，然后运行：

```bash
python run.py --motion-test
```

这个模式只建立 SDK 状态/遥控订阅和同一个受保护的运动进程，不启动 ROS、建图、相机或模型。
启动不发速度；`record.py` 保存状态、按键、运动调用与停稳反馈事件，本模式没有三幅图或导航观测帧。
它允许三个运动验收标记尚未完成，也不依赖几何标定；完整 `--allow-motion` 仍要求全部验收。

1. **先看反馈。** 终端持续显示 `keys`、设备状态值、健康/静止判断、实测速度。
   `takeover_button_mask: 0` 时只能看，任何动作均拒绝。用已知可安全操作的遥控按钮核对位值，
   填写 `robot.yaml` 的 mask，退出后重启本模式。程序不猜按钮、不自动修改配置。
2. **准备。** 按 `a`，检查反馈新鲜健康、机器人静止、接管键已释放；准备有效期 5 秒，此时仍不发运动。
3. **一次动作。** 按 `f` 前行（上限 0.10 m/s），或按 `j` 左转（上限 0.25 rad/s）。
   worker 独立执行不可续期的 3 秒指令窗口，并保留加速度限制与较低的设备配置速度上限。
   每次结束后重新按 `a`；按住动作键不会连续执行。这个模式不提供避障，使用平整空地和现场遥控监护。
4. **停止与接管。** 动作期间按空格或已配置遥控接管键。查看 `stopping` 原因，等待 `stopped`。
   后者必须由停止请求之后的新状态反馈连续满足静止阈值确认；没有有效反馈会报告故障，不宣称停稳。
   遥控触发后，要先松开按键；第一次启用请求只清锁存，需再按 `a` 与动作键才可重新测试。
5. **命令超时。** 按 `a` 再按 `t`：以 0.05 m/s 请求前行，约 0.12 秒后停止喂指令，保持 SDK/DDS 通信。
   默认命令有效期仍为 0.35 秒；本项由命令超时提前停止，不持续前行 3 秒。
   预期出现 `command_timeout` 再到 `stopped`。这是主程序不再产生命令的测试，不是物理断网测试。
6. **退出。** 按 `q` 或 Ctrl+C，先停止运动进程，再发布结果供 record 收尾。

3 秒是软件的发令窗口；同步 SDK 调用和停止反馈都有耗时，不表示真机恰好在 3 秒内物理静止。
控制台喂命令目标 20 Hz，运动 worker 目标 50 Hz；实际频率受同步 RPC（单次超时 0.15 秒）和系统负载影响。
同用户控制锁只防止本部署重复发令，不是机器人全局控制权仲裁；MCF 和遥控优先级没有被切换。

记录区分 `command_requested`（导航请求排队）、`move_accepted`（SDK 调用成功返回）、
`stop_requested`、`stopped`（新反馈确认），测试停止事件还保存速度峰值与状态年龄。
SDK 返回成功不证明机器实际移动；低速死区也可能导致没有动作，需结合现场观察和速度反馈判断。
纯事件记录完整同样不代表运动验收通过，程序不会自动设置任何 verified 字段。
主循环已经收到的运动 fault 会保留到最终汇总，不会因 close 阶段未重复上报而丢失。

| 标记 | 对应验收证据 |
|---|---|
| `deskew_and_self_filter_verified` | 默认观察模式下遥控走停/转身，查看并记录墙体重影、机身附近伪障碍；不能靠 SDK 测试判定 |
| `communication_verified` | 核对传感器连续接收、控制/停止反馈与命令超时测试；物理断网/机器人计算机故障的保护另行实测 |
| `remote_arbitration_verified` | 多次动作中接管、松键后不自动恢复，再核对原生遥控操作；不能仅凭 mask 非零判定 |

2026-10-28 本机通信单项检查已由用户接受，`communication_verified: true`。
依据为此前传感器记录与 `20261028T142816_831350Z` 完整 motion-test 记录：9 次动作均有 SDK 调用成功与停稳反馈，
其中 3 次到时停止、1 次空格停止、3 次接管停止、2 次命令超时停止；用户现场确认实际动作正常。
这些是已执行的次数，不表示所有项目均已完成三次重复验收，也不包含真实断网或计算机故障保护。
同日 `remote_arbitration_verified: true`：上述记录有 3 次 `4096` 接管停止，松键后不自动恢复，
重新启用先清除锁存、再次明确操作才开始动作；用户另行确认接管后原生摇杆仍能正常控制。
同日用户明确要求先进行有人接管的点导航测试，`deskew_and_self_filter_verified` 改为 `true`。
这是用户指定的测试放行设置，**不代表动态去畸变与机身过滤已验收**；相关几何证据仍待补齐。
普通导航终端会显示运动进程的停止/拒绝原因（如 `command_timeout`、`remote_takeover`），
出现 `paused` 时先核对原因；放行标记不会取消命令超时、观测新鲜度或接管锁存保护。
`command_timeout` 表示有效速度指令未及时续上，不等于 ROS 时钟不同步。
事件中的 `generated_at` 是本轮决策起始时间，`expires_at` 同时受 0.35 秒命令期限和观测期限限制；
建图规划计算也消耗这段期限。`decision_duration_s` 记录算法耗时，运动事件的 `worker_at`
记录运动进程实际产生事件的单调时间；不要把主循环稍后收到事件的 `at` 当作物理停止时刻。
静止条件拒绝会附带当时状态/遥控年龄、速度等反馈，便于区分仍在移动与反馈失效。

通过这些单项实验后，再进行短距离点导航和完整场景验收。

完整自主导航前必须实际完成：外参/坐标/时间核对，点云去畸变和自身回波核对，遥控与 SDK 指令仲裁核对。
还须完成 DDS 消息兼容与接收稳定性排查。然后设置 `calibration_verified`、`deskew_and_self_filter_verified`、
`communication_verified`、`remote_arbitration_verified`，
填写网卡与经过验证的遥控接管按键 mask，才允许：

```bash
python run.py --allow-motion
```

这些字段是设备验收记录，不能为了绕过报错直接改成 true。
没有关闭遥控响应：必须先验证具体固件的原生摇杆与 SDK 行为；程序并不保证“随便推摇杆就接管”。
接管键通过 `rt/lowstate.wireless_remote` 的第 2、3 字节位掩码读取，具体按键不能在未知遥控器上猜测。

`sport_state_motion_codes` 指定这台机器人允许进行速度控制的 `SportModeState.error_code` 值。
2026-10-27 用户明确确认当前 Go2 的 `100` 可以运动，因此本机配置为 `[100]`；用户说明它对应灵动状态，
此依据与固件版本号、厂家独立确认分别记录，不能推广为所有型号/固件的通用故障码表。
运动进程按该列表检查状态，缺少列表或收到其他值（包括未配置的 `0`）都会阻止运动。
消息新鲜度、四元数、倾斜、速度与停稳反馈检查仍然生效。`Move()` / `StopMove()` 的调用返回值仍单独检查为 `0`。
配置这一状态值不会自动通过其他验收，也不会启动运动或切换宇树模式。

| 按键 | 行为 |
|---|---|
| n | 先停止当前任务，再输入地图 `x,y`，例如 `2.0,1.0`；Enter 提交 |
| e | 先停止，再输入英文目标；多词用下划线。当前无 RGB-D，提交后明确拒绝找物，不发送行走指令 |
| p / r | 暂停 / 明确恢复；暂停继续更新地图，不消耗任务时间 |
| 空格 | 在菜单或输入目标时均停止并取消任务 |
| o / m | 停止自主任务并停止持续自主指令；保持观测，不修改遥控器设置 |
| q / Ctrl+C | 先关闭运动输出并尝试确认停稳，再发布结束结果供 record 收尾 |

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

## 7. 本轮实验记录

在已激活环境的另一个终端、`deploy/` 下运行：

```bash
python record.py
```

可以先启动 record（默认等待 run 最多 30 秒），再启动 run，覆盖整个实验。
若 run 已经开始，record 会读取仍在缓存中的数据；更早数据无法补回，报告会记录缺失数量。
默认输出到**执行 record 的计算机**的仓库 `results/deploy/时间戳/`；指定其他目录：

```bash
python record.py --output /实际结果目录
```

保存内容：

```text
config.json                # run 所用配置、run 所在机器的 Git / 依赖信息
session.json               # 会话编号、开始时间、demo / 实机模式
observations.jsonl         # 可供 run --replay 读取的观测索引
observations/*.npz         # 未按显示点数降采样的适配后原始观测
frames/0000001.jpg         # 该采样时刻可用的 RGB（缺失时不生成）
frames/0000001.png         # 实际导航俯视图（未建图时不生成）
frames/0000001.cloud.png   # 点云三维视图导出
frames/0000001.npz         # 观测 + 地图 masks/log-odds + 轨迹 + 采样体素点云
frames/0000001.json        # 上述数组索引、坐标、路径、RGB 数据年龄等元数据
events.jsonl               # 操作、速度请求、运动反馈、错误与结束事件
run_summary.json           # 记录结果、丢帧计数、是否收到 run 的最终结果
```

实机输出不再由 run 直接产生 `final_map.*`；每个 frames 包保存当时的真实地图状态。
最后一包是**最后采样到的地图**，不伪称精确退出时刻的地图；退出时任务结果保存在 summary。
离线回放仍可重新生成原有 `final_map.json/.npz`。几何标定不足时不能用原始观测直接做导航回放。
三幅图以带元数据的逐帧图片保存，本版不额外编码 MP4。NPZ 禁用 pickle。

默认显示/记录采样目标 5 Hz，与导航 10 Hz 和运动进程配置 50 Hz 独立。
这些是循环目标频率；算法耗时、同步 SDK 调用和设备负载都会影响实际频率。
RGB 和点云来自各自采集时刻，记录接收年龄；不冒充严格同步 RGB-D。
run 只保留有界内存缓存（4 包、4096 个事件），后台编码队列容量 1；客户端和慢磁盘不反压导航。
同一帧供多个 record 读取时只压缩一次，压缩不占用显示发布锁。
编码队列满、迟加入或网络/写盘落后会产生缺失，分别在 `producer_dropped`、`missing_frames`、
`missing_events` 中计数。它是实验采样记录，不是传感器原频率 rosbag。

停止顺序：先在 run 按 q，使运动停止流程完成；run 发布结束结果并留 2 秒给 record 排空，
record 收到同会话结束标记后自动退出。再关闭 view。
结束结果固定 `final_frame_id` / `final_event_id`；record 核对是否读到这两个位置。
编码线程未能按时排空时，会记录丢帧并设置 `telemetry_complete=false`，迟到帧不会越过结束边界。
单独对 record 按 Ctrl+C，只停止记录，run 继续；此时 `run_final_received=false`。
断网、主程序异常消失或会话编号变化时记录会结束并说明原因，不混合两轮实验。
若未收到最终结果，不能把该记录当作完整的正常结束。

## 8. 三图可视化

参考 zkj 仿真的占据图、轨迹/路径叠加与累计点云表现；不复制 Isaac 相机或第三人称画面。
浏览器界面无需机器人桌面、Qt、Open3D、Node 或外部 CDN。

在**笔记本**另开一个 SSH 终端；已有导航终端继续运行：

```bash
ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 8080:127.0.0.1:8080 go2
```

未配置别名时将 `go2` 替换为实机 SSH 地址，例如此前使用的 `unitree@192.168.123.18`。
**进入 Go2 后：**

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
source .venv-deploy/bin/activate
cd deploy
python view.py
```

笔记本浏览器打开 `http://127.0.0.1:8080`。另一个 Go2 终端中的 `python run.py` 必须已经运行。
view 只监听回环地址，默认读取 `http://127.0.0.1:8765` 的 run 会话。
端口占用时分别用 `run.py --session-port`、`view.py --source` / `--port`、`record.py --source` 指定。
如果希望 view 和 record 都运行在笔记本：先 `ssh -N -L 8765:127.0.0.1:8765 go2`，
再在笔记本的 CPU 依赖环境运行这两个入口；无需笔记本安装 ROS 或 Unitree SDK。

三幅图：

1. **RGB**：run 接收的前置画面；无深度时仍只是预览图。
2. **俯视图**：run 实际占据栅格，保留未知区、自由区、障碍、膨胀区，叠加已有 Voronoi 骨架、
   frontier、实际轨迹、当前规划路径、机器人与目标。没有对应算法结果时不虚构标记。
3. **累计点云图**：同一导航体素地图的采样点，最多 12000 点，默认以透视镜头从 Go2 后上方看向前方，跟随机器人位置和朝向。
   镜头在机器人后方 3.2 m、机身上方 2.7 m，朝向前方 1.2 m、机身上方 0.45 m；只跟随平面航向，不随机身俯仰/横滚晃动。
   橙色机身和腿是位置/朝向示意，不是关节实测或 Go2 三维模型。
   滚轮可缩放；点击“显示全图”切换到全图自动适配，此时支持旋转、平移、俯视；“回到背后”恢复固定跟随。
   有效同坐标系位姿到来前暂示全图；位姿过期时冻结镜头并提示，不使用无效位姿驱动视角。
   标定后距地面 ±8 cm 的点显示浅灰，其他点按高度渐变；未标定时不区分地面。此规则用于着色，不是地形分割，也不改变导航判障。
   同一轮 run 中，离开视野的已观测点仍保留在体素地图里；关闭/重开 view 不会清除它。重启 run 会新建地图。
   点导航配置启用 `mapping.voxel_clear_observations: 3`：新的命中持续融合；旧 LiDAR 点只有在三次不同扫描中，
   被实际三维射线近距离穿过且在更远处收到回波后才清除。当前命中会重置清除计数；未看到、被遮挡、
   接近量程上限的回波均不能作为清除依据。未被射线覆盖的旧物体可能仍会保留，不按时间强行过期。
   这项更新复用原有体素地图，仿真默认值为 0（关闭）；二维导航占据图继续按原有占据证据更新。
   record 保存的点云 PNG 同样显示全图。点云按配置的高度范围过滤并体素融合，不是全部原始激光点；
   当前依赖 odom 对齐，没有额外增加回环校正，里程计漂移仍可能导致重影。
   未标定时降级为“当前原始点云”，明确提示没有建图。不是把逐帧点云当作累计地图。
   显示采样放在 deploy 内，通过原有 `voxels.points()` 接口取点；超过显示上限后按稳定空间键采样，
   避免点数增长改变全局采样间隔而造成历史场景闪烁。界面区分累计体素数与实际显示点数。
   体素中心在新点融合时更新，显示直接读取；12000点仍只限制显示和传输量，大地图需继续测量耗时。

地图与点云的 XY 使用规划坐标系；已标定观测的 Z 已按 ground_z 归一到地面高度。
界面保持等比例俯视投影，不把正方形栅格拉伸成矩形。机器人位姿只在同坐标系、时间近邻且新鲜时叠加。
显示目标为5Hz，实际帧率受建图负载影响，界面显示实际发布频率和点云年龄。
观察/暂停时拓扑维护按 `topology_interval_s` 执行，不推进任务超时；录制NPZ仅在 record 请求时编码。
地图更新年龄使用最后一次成功融合的点云时间，不因收到一个未处理的新观测而刷新。
`/api/frame` 的 `performance` 包含输入年龄、捕获/排队/编码耗时、发布频率及显示更新年龄。
输入过期与显示更新延迟分别提示，原有0.4秒输入校验、50ms未来容差、1秒画面过期阈值保留。
地图更新年龄与主程序心跳都会检查。界面断流后停止轮询，恢复后点击“重新连接”；run 重启后清除旧会话画面。
原有 SSH 保活与主动重连行为保留，避免持续请求已关闭的转发端口。

默认 RGB 仍使用独立进程的 Unitree `VideoClient.GetImageSample`。
此前实机只确认自定义前置视频收到消息，SDK 取图仍需实机验收；未猜测 Annex B 的具体编解码格式。
SDK 没有提供采集时间戳，界面与记录只能报告接收年龄，不能证明相机端没有重复旧帧。

关闭浏览器或 view，不停止导航；record 退出也不停止导航。所有运动操作仍在 run 控制台。

## 9. 离线检查四入口协作

不连接机器人，在安装了 deploy CPU 依赖的环境中分别打开三个终端，从 `deploy/` 运行：

```bash
# 终端 1：先等待主程序，保存演示数据
python record.py --output /tmp/go2-demo-records

# 终端 2：合成传感器输入，经过实际 Tasks / 建图算法
python run.py --demo

# 终端 3：与实机相同的三图界面
python view.py
```

浏览器打开 `http://127.0.0.1:8080`。页面始终标注合成演示；demo 不初始化 ROS、SDK 或运动。
可给 run 加 `--demo-seconds 20` 自动结束。旧的 `view.py --demo` 改为由 run 统一供数。
无需启动 serve_models；它只在远端模型联调时使用已有 GPU 环境启动，接口与命令保持原样。

录制成功、单元测试和演示界面通过都不代表真机通过。仍需验证传感器/时钟、地图、
SDK 取图、低速停稳、遥控接管，以及 run/view/record 并行时 Orin 的实际负载和控制频率。
