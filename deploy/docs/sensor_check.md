# Go2 点云与 IMU 测试

目标：分别检查静止墙距、IMU 输出和转向时的点云对齐。先保持当前驱动和定位配置，
用同一套配置完成全部测试；环境物体保持静止。录包只订阅传感器，不发送运动命令。

本轮使用 `ros2 bag record` 同时保存 Swarm-LIO 配准后的稀疏点云、里程计和机载 IMU。
`record.py` 用于保存点导航运行时的观测。

## 开测前检查与实验顺序

这一轮确认四件事：静止墙距偏差是否稳定、实际使用的点云与定位发布者、IMU 是否有有效数据、
转向时是否出现新的畸变或跨帧重影。假定环境物体静止；这不是完整的 IMU 标定或点导航验收。

- [ ] Orin 已同步本说明和 `configs/sensor_record_qos.yaml`，仓库路径与下面命令一致。
- [ ] 雷达、定位、站立服务正常，配准点云、里程计和 IMU 都有发布者。
- [ ] `run.py` 已退出；需要运动时由现场操作员用原生遥控器控制，已确认能停止。
- [ ] 同一面实体墙、地面标记和卷尺准备好；测量基准为雷达中心投影到主墙面。
- [ ] 保存当前代码版本和话题发布者信息；记录本轮使用的驱动、定位启动方式，组间不切换配置。
- [ ] 记录盘有可用空间；确认第一组录制已收到点云和 IMU，再继续后续实验。

| 顺序 | 记录 | 操作 | 主要检查 |
|---|---|---|---|
| 1 | `wall_080` | 约 80 cm，静止 30 秒，立即检查 bag | 数据是否录齐、近距离墙距、静止 IMU |
| 2 | `wall_120` | 约 120 cm，静止 30 秒 | 墙距误差是否随距离变化 |
| 3 | `wall_160` | 约 160 cm，静止 30 秒 | 固定偏移与比例误差的区别 |
| 4 | `turn_90` | 静止、缓慢转约 90°、停、转回、停 | 运动畸变、姿态与时间对齐 |

每组记录实际墙距或转角，不把目录名称当作实测值。前三组已包含静止 IMU 检查，
本轮无需单独增加一个静止 IMU 实验。
点导航的启动、暂停和接管流程见 [deploy README](../README.md)。

## 1. 登录与准备

先将本说明和 `configs/sensor_record_qos.yaml` 同步到 Go2 的对应 deploy 目录。
开发机登录：

```bash
ssh go2
```

在 Orin 新终端执行，不需要激活 Python 虚拟环境：

```bash
bash ~/fishbot_multirobot_sim/deploy/go2_real/scripts/run_mid360s_lio.sh start
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project/deploy
SENSOR_TEST_DIR="$PWD/../results/deploy/sensor-check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SENSOR_TEST_DIR"
df -h "$SENSOR_TEST_DIR"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$SENSOR_TEST_DIR/source-commit.txt"
  git status --short > "$SENSOR_TEST_DIR/source-status.txt"
elif test -f ../.deploy-source.json; then
  cp ../.deploy-source.json "$SENSOR_TEST_DIR/"
fi
ros2 topic list -t | tee "$SENSOR_TEST_DIR/topics.txt"
ros2 node list > "$SENSOR_TEST_DIR/nodes.txt"
for SENSOR_TOPIC in /bot1/cloud_registered_sparse /bot1/lidar_slam/odom /livox/imu; do
  ros2 topic info "$SENSOR_TOPIC" --verbose \
    > "$SENSOR_TEST_DIR/${SENSOR_TOPIC//\//_}-publisher.txt" 2>&1
done
cp configs/robot.yaml configs/sensor_record_qos.yaml "$SENSOR_TEST_DIR/"
```

把本轮实际使用的雷达、定位启动命令及已知版本写入记录目录中的 `setup.txt`，
已有驱动或定位配置文件也一并保存。不了解版本时注明未知，不用猜测。
用 MID-360S 的光心量墙距。`/bot1/cloud_registered_sparse` 的点已经在 `bot1/world` 中。
点云和 `/bot1/lidar_slam/odom` 的时间戳都是本次扫描的结束时间。
`/tf`、`/tf_static` 按实际发布情况记录。

宇树给 Go2 配 MID-360 的标称位置以**机身内部 IMU** 为原点，向前 0.1870 m、向左 0 m、
向上 0.0803 m，绕 +Y 轴转 13°。本机 MID-360S 保护架的圆环、拱架及前背部安装位置
与示意图相似。静止时测得 LIO 与 Go2 的俯仰角相差约 13°，平移量仍待核实。
程序用 `mid360_mount` 把 LIO 的 IMU 位姿换算成机身位姿。射线起点是 `lidar_origin_offset_base_m` 指定的 MID-360S 光心。
[Go2 URDF](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf) ·
[宇树安装说明](https://support.unitree.com/home/en/developer/SLAM%20and%20Navigation_service) ·
[宇树安装图](https://doc-cdn.unitree.com/static/2023/11/13/492e14d9686a4322ad5502b743d707e1_2933x3088.png)

退出 `run.py` 后进行测试，保留原有雷达、定位和机器人站立服务。
运动阶段由现场操作员使用原生遥控器控制。

## 2. 三组静止墙距

选择同一面平直实体墙，地面平整，机器人朝向、站立姿态保持一致。
从**雷达中心的地面投影**量到**主墙面的地面投影**，取水平垂直距离，
不要沿倾斜视线测距，也不要把墙脚凸起当作主墙面。每次摆好后稳定约 5 秒，再开始录制。

| 记录名称 | 建议墙距 | 录制期间 |
|---|---|---|
| `wall_080` | 约 80 cm | 静止 30 秒 |
| `wall_120` | 约 120 cm | 静止 30 秒 |
| `wall_160` | 约 160 cm | 静止 30 秒 |

距离不必刚好等于建议值，**必须记下每次卷尺实测值**。空间不足时可选其他三个距离，
尽量拉开间隔。记录主墙面与墙脚的关系，保留能看清测量起止点的现场照片。

在刚才同一终端定义录制命令：

```bash
record_sensors() {
  ros2 bag record \
    --qos-profile-overrides-path configs/sensor_record_qos.yaml \
    -o "$SENSOR_TEST_DIR/$1" \
    /bot1/cloud_registered_sparse /bot1/lidar_slam/odom /livox/imu \
    /tf /tf_static
}
```

先摆好第一组位置，然后执行：

```bash
record_sensors wall_080
```

确认出现配准点云、里程计和 IMU 的 `Subscribed to topic` 日志后，保持静止 30 秒，
按 `Ctrl+C`，等待录制器保存退出。接着检查：

```bash
ros2 bag info "$SENSOR_TEST_DIR/wall_080"
```

先核对第一组。配准点云、里程计和 IMU 应都有非零消息数。
如果 IMU 缺失，先检查话题名称、发布者和录制日志；解决后重录第一组，再继续完整测试。
若现场暂时无法获得 IMU，仍可检查墙距，并在记录里注明“仅点云检查”。

后两组分别摆好、测量并站稳后，再各执行一条，按同样方式结束和检查：

```bash
record_sensors wall_120
```

```bash
record_sensors wall_160
```

在下面提示中依次填写三组**实际**墙距（单位 cm），不要照抄名义距离：

```bash
read -r -p 'wall_080 实测墙距(cm): ' WALL_080_CM
read -r -p 'wall_120 实测墙距(cm): ' WALL_120_CM
read -r -p 'wall_160 实测墙距(cm): ' WALL_160_CM
printf 'wall_080_cm=%s\nwall_120_cm=%s\nwall_160_cm=%s\n' \
  "$WALL_080_CM" "$WALL_120_CM" "$WALL_160_CM" > "$SENSOR_TEST_DIR/measurements.txt"
```

同名 bag 目录不能重复使用；重录时换名，例如 `record_sensors wall_080_repeat`，
同时注明新的实际距离。不要删除已有记录来覆盖。

## 3. 缓慢转向一次

在有足够转身空间、能同时看到墙和拐角的位置，用地面标记记录起始位置和朝向。
执行：

```bash
record_sensors turn_90
```

收到点云及 IMU 后按顺序操作：

1. 静止 10 秒。
2. 用原生遥控器缓慢转向约 90°，用约 10 秒完成，然后静止 10 秒。
3. 缓慢转回起始朝向，用约 10 秒完成，再静止 10 秒。
4. 按 `Ctrl+C` 保存退出，执行 `ros2 bag info "$SENSOR_TEST_DIR/turn_90"`。

记录实际转向方向、转角及最终是否回到原来的位置。以现场标记作为参考，
不要把 odom 显示的角度当作独立真值。机器人转向时可能伴随小幅平移，要一并记录。

## 4. 录完核对与下载

每个 bag 都应有配准点云和里程计。要判断 IMU，还需要连续的 IMU 数据。
如果点云有数据而 IMU 没有，本轮只能分析点云，不能据此判定 IMU 好坏。
可选的 TF 缺失时注明即可。消息计数非零不代表完全没有丢帧；
离线还需检查频率、时间间隔、时间戳回退和有效字段。

录制结束后在 Orin 打印目录：

```bash
printf '%s\n' "$SENSOR_TEST_DIR"
```

在开发机另开终端，进入本地 deploy，将下面的 `sensor-check-实际时间` 替换为打印的目录名：

```bash
cd ~/inter-nav-project/deploy
mkdir -p ../results/deploy
scp -r go2:~/inter-nav-project/results/deploy/sensor-check-19700120-194654 ../results/deploy/
```

复制整个目录，保留 `metadata.yaml`、全部 `.db3` 文件、话题清单和测量说明。
本组命令已检查 shell 语法及 QoS 配置结构；仍需在 Orin 验证实际话题发现和录制结果。

## 数据如何判断

| 现象 | 优先检查 |
|---|---|
| 三个距离都相差近似固定值 | 话题原点与物理雷达中心定义、固定偏移；不能仅凭这点判定硬件测距错误 |
| 误差随距离明显增长 | 距离比例、标定、驱动输出定义及测量方法 |
| 静止正常，转向时墙面变厚或重影 | 单帧运动畸变、IMU/点云时间对齐、外参、上游姿态估计 |
| 单帧几何稳定，跨帧累积位置变化 | odom、姿态与时间同步；换 SLAM 可能改善这一部分 |

墙距使用每帧主墙面的平面拟合，避开墙脚与边缘杂点；比较三个实测距离与点云距离，
估计 `点云距离 ≈ a × 实测距离 + b`。以卷尺实测值作为距离基准。
这一轮的转向数据用于定位问题，不能单凭一次转向完成完整 IMU 标定。

IMU 检查加速度、角速度、坐标系和时间戳，不以“姿态已经融合”为不可用依据。
按 ROS 的标准定义，静止加速度模长约为 9.8 m/s²，角速度接近零；若加速度模长约为 1，
先核对是否使用 g 为单位；若接近零，核对是否去除了重力或字段无效。
不能仅根据一个模长值判定好坏。转向时须先明确 IMU 轴向和外参，再比较角速度积分与实测角度，
不能直接把倾斜安装的 IMU 的 z 轴角速度当成机器人 yaw 角速度。

此机 Livox 驱动的原始加速度单位为 g，角速度单位为 rad/s。
接收时间对齐不等于采样时刻同步，雷达到 base 的外参也不能直接当作雷达到 IMU 的外参。
bag 接收时间与消息 header 可能不在同一个时间域，不能直接相减解释为传输延迟。

参考：[ROS IMU 数据约定](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst)、
[ROS 2 Foxy rosbag2](https://github.com/ros2/rosbag2/tree/foxy)。
