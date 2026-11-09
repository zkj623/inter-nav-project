# Go2 点导航 Demo

本轮目标：输入平面坐标，让 Go2 自主到达、绕过障碍并停稳，支持暂停、继续和人工接管。
使用 ROS 定位与 LiDAR 建图，通过 SportClient 输出机身速度，MCF 保持运行。
**不需要语义识别、GPU 模型服务或 RGB-D；交互识别与推移保持关闭。**

以下设备端命令假定仓库位于 `~/inter-nav-project`，运行环境已安装；首次使用见文末。
机器人需站稳，定位与雷达驱动正常运行。操作 `run` 的终端须支持交互，SSH 登录时保留 TTY。

## 启动与记录

### 终端 1：运行点导航

在 Go2 的 Orin 上执行：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
source .venv-deploy/bin/activate
python deploy/run.py --allow-motion --camera none
```

启动后先观察，不会立即行走。等待出现“实验坐标系已固定”，且状态为
`观测=ready 建图=True 最近错误=None`，再在地图中确认目标附近和行进通道。
按 `n`，输入 `1,0`，按回车，开始前方 1 米的点导航。

仅查看传感器和地图时，将最后一条命令换为：

```bash
python deploy/run.py --camera none
```

仅观察模式不能通过按键启用运动，需退出并用 `--allow-motion` 重新启动。
需要前置相机画面时，把 `--camera none` 换为 `--camera sdk`；这只启用视频预览。

### 终端 2：打开地图

在同一台 Orin 上另开终端：

```bash
cd ~/inter-nav-project
source .venv-deploy/bin/activate
python deploy/view.py
```

浏览器打开 <http://127.0.0.1:8080>。如果浏览器在开发机上，在开发机执行 SSH 转发，
将地址换成实际设备地址，然后打开同一网址：

```bash
ssh -N -L 8080:127.0.0.1:8080 unitree@192.168.123.18
```

界面上方并排显示全局路线/局部碰撞地图和历史点云，下方显示 RGB。地图使用与仿真一致的
分层配色、机器人轮廓、目标与路线标记；按“放大地图”可展开查看。
点云默认背后跟随，使用 Viridis 高度配色，只显示距地面 `0.1–0.8 m` 的切片。
点击高度按钮可切换到“全部高度”，也可选择全图、俯视和滚轮缩放。切片只影响显示，原始记录与导航高度范围不变。
**判断当前障碍以导航地图为准**；历史点云可能保留已移动物体的位置。
关闭相机时没有 RGB 画面，不影响点导航。

### 终端 3：保存验收记录

在同一台 Orin 上另开终端，开始导航前启动记录：

```bash
cd ~/inter-nav-project
source .venv-deploy/bin/activate
python deploy/record.py --output ~/go2-demo-records
```

记录目录会打印到终端，每次运行自动创建独立子目录。
`run` 不自动保存实验文件；`view` 和 `record` 都不控制机器人。
结束时先在 **run 终端按 `q`**，等待停止反馈和 `record` 自动保存结束，再关闭 `view`。
在 record 终端按 `Ctrl+C` 只会结束记录，不会停止机器人。

## run 按键

普通模式下按键立即生效，无需回车；只有坐标输入需要回车确认。

| 按键 | 功能 |
|---|---|
| `n` | 停止并取消当前任务，输入 `x,y`，回车启动新目标 |
| `p` | 暂停，保留当前目标 |
| `r` | 继续暂停的任务，重新规划 |
| **空格** | 停止并取消任务；输入坐标时也有效 |
| `o` / `m` | 停止并取消任务，回到观察，保留地图 |
| `Esc` | 退出坐标输入，不恢复刚才的任务 |
| `q` / `Ctrl+C` | 停止自主输出并退出 run |

**坐标输入中不要加空格。** 此时 `q`、`p` 等字母会作为输入文本；先按 `Esc` 返回普通模式，
再按命令键。`Ctrl+C` 可以直接退出。

## 坐标怎么填

每次启动 run，用第一组有效观测固定本轮坐标系：起点为 `(0,0)`，
**X 正方向是启动时的前方，Y 正方向是启动时的左方，单位为米。**
机器人转向、暂停、换目标都不会改变这个坐标系；重新启动 run 会建立新的原点和地图。

| 输入 | 目标位置，相对本轮起点 |
|---|---|
| `1,0` | 前方 1 米 |
| `2,1` | 前方 2 米、左方 1 米 |
| `0,0` | 返回起点 |

默认地图 X、Y 范围都是 `[-10,10]` 米。障碍内或地图外的目标会被拒绝。
地图内尚未观测的目标会触发朝目标方向的探索；**首轮验收选已观测的空闲目标和通道**。
坐标基于里程计，固定坐标变换不消除定位漂移。

## 点导航验收顺序

全程开启 record，操作者持遥控器监护；以下是现场验收步骤，不代表设备已经通过验收。

1. **观察地图**：静止时检查墙面与障碍位置；短距离运动和转向后再检查，确认没有影响通路判断的错位。
2. **短距往返**：`n` → `1,0` → 回车，等待到达并停稳；再 `n` → `0,0` → 回车。建议重复三轮，用现场标记核对实际落点。
3. **转向与绕障**：选已观测且有足够通路的目标，例如 `2,1`，确认能绕开固定障碍并到达。
4. **暂停与接管**：途中测试 `p` / `r`、空格取消，以及本机已配置的遥控接管键。接管后不应自行恢复自主运动。
5. **结束并核对记录**：在 run 按 `q`，等待 record 完成，检查该次记录目录中的结果。

默认到达条件是距离目标 **小于 0.25 m**，并通过连续约 **0.4 s** 的实测停稳检查。
还需要运动进程报告 `stopped`，并在现场确认机器人停止。

| 记录 | 核对内容 |
|---|---|
| `events.jsonl` | 每个目标的 `task_result` 事件中 `result.success=true`；对应的 `motion_state` 事件中 `state=stopped` |
| `run_summary.json` | `recording_complete=true`；最后任务的 `run_summary.task_result.success=true`；`run_summary.last_autonomous_stop_confirmed=true` |
| `observations/`、`frames/` | 定位、点云、导航地图、路径和现场对应情况 |

`completed` 只表示流程结束，不等于导航成功。最终 summary 只保留最后一个任务结果，多目标验收看 events。
`hardware_validated` 不会自动变为 true，真机验收结论需要结合现场表现和记录给出。
配置中的 `verified` 开关也不能替代验收；运动中的点云配准与地图一致性仍需现场确认。

## 常用配置与排查

配置位于 `deploy/configs/`；改完后重新启动 run。

| 配置文件 | 当前点导航设置 |
|---|---|
| `robot.yaml` | 网卡 `eth0`；定位 `/utlidar/robot_odom`；雷达 `/utlidar/cloud_deskewed` |
| `navigation.yaml` | `interaction_enabled: false`；到达距离 `0.25 m`；地图分辨率 `0.1 m` |
| `navigation.yaml` | 速度上限：前向 `0.8 m/s`、侧向 `0.3 m/s`、转向 `0.9 rad/s` |
| `navigation.yaml` | 导航障碍高度：距地面 `0.15–0.63 m`；障碍膨胀半径 `0.4 m` |
| `robot.yaml` / `runtime.yaml` | 数据最大年龄 `0.4 s`；命令有效期 `0.35 s` |

运动进程还会按 `runtime.yaml` 的上限限速；调整速度时同时核对两处配置。
导航高度范围外的障碍不由此地图层表达，现场应使用平地、无台阶的验收区域。
更换设备、重置里程计或改变地面参考后，复核地面高度和 LiDAR 外参。

在已加载环境的设备终端中，可先做以下检查；两条命令均不发运动指令：

```bash
python deploy/run.py --check
python deploy/run.py --diagnose --duration 10 --output /tmp/go2-diagnostic.json
```

`--check` 检查依赖和配置，退出码为 0 不等于 ROS、DDS 或硬件已就绪；
`--diagnose` 实际订阅 ROS，检查收到的数据。

| 现象 | 先检查 |
|---|---|
| `观测=waiting` / `missing_sensor_data` | 定位和雷达驱动、话题、TF、时间配对及数据新鲜度 |
| `建图=False` | 有效点云、地面高度和 LiDAR 原点标定 |
| 目标被拒绝 | 坐标格式、地图边界、障碍占据，以及终端给出的具体原因 |
| `运动未启用` / `rejected` | SDK 反馈、网卡、机身状态、接管状态与配置验收项 |
| `decision_expired` / `command_timeout` | 记录中的观测年龄、规划耗时、命令时限和运动状态；查明原因后再恢复 |
| 运动后墙面重影或通道错位 | 定位、点云配准和时间同步；固定实验坐标系不会修正这些偏差 |

## 首次安装与其他工具

设备端使用 Ubuntu 20.04 / ROS 2 Foxy 对应的系统 Python 3.8：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project
bash deploy/scripts/install.sh
```

安装脚本创建 `.venv-deploy`，安装 CPU 运行依赖。ROS、传感器驱动、定位和
`unitree_sdk2_python` 需在设备上就绪；脚本不安装或启动这些组件。

需要同步代码时，在开发机仓库根目录执行，替换实际设备地址和路径。
先预览，再应用；应用前停止设备上的 run：

```bash
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project
bash deploy/scripts/sync.sh unitree@192.168.123.18 /home/unitree/inter-nav-project --apply
```

同步会覆盖同名文件，不删除目标目录中的其他文件。设备专用配置可放在仓库外，
启动时用 `--config-dir /绝对路径/配置目录` 指定，目录内需包含三份 YAML 配置。

在仓库根目录、已激活环境的终端中，其他常用入口为：

| 用途 | 命令 |
|---|---|
| 无硬件合成演示，5 秒后退出 | `python deploy/run.py --demo --camera none --demo-seconds 5` |
| 实际记录离线回放，不发送运动指令 | `python deploy/run.py --replay /绝对路径/单次记录目录 --task navigate --goal 1 0 --output /tmp/go2-replay` |
| 独立 SDK 短时运动验收 | `python deploy/run.py --motion-test` |

`--goal` 仅用于离线回放；真机目标通过 `n` 输入。回放使用保存的观测，不模拟机器人对命令的响应，不能代替闭环验收。
`--motion-test` 与正常 run 分开运行，不启用地图或避障；按 `a` 准备，再按 `f` 前行、`j` 左转或 `t` 测试命令超时停车，空格停止、`q` 退出。
