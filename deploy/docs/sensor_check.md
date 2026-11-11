# Go2 点云与 IMU 测试

目标：分别检查静止墙距、IMU 输出和转向时的点云对齐。先保持当前驱动和定位配置，
用同一套配置完成全部测试；环境物体保持静止。录包只订阅传感器，不发送运动命令。

本轮使用 `ros2 bag record` 保存原始 ROS 消息。`record.py` 保存的是运行程序输出，
不能替代三路原始点云与 IMU 的同步记录。

## 1. 登录与准备

先将本说明和 `configs/sensor_record_qos.yaml` 同步到 Go2 的对应 deploy 目录。
开发机登录：

```bash
ssh go2
```

在 Orin 新终端执行，不需要激活 Python 虚拟环境：

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project/deploy
SENSOR_TEST_DIR="$PWD/../results/deploy/sensor-check-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SENSOR_TEST_DIR"
ros2 topic list -t | tee "$SENSOR_TEST_DIR/topics.txt"
ros2 topic info /utlidar/cloud --verbose > "$SENSOR_TEST_DIR/cloud-publisher.txt"
ros2 topic info /utlidar/imu --verbose > "$SENSOR_TEST_DIR/imu-publisher.txt"
cp configs/robot.yaml configs/sensor_record_qos.yaml "$SENSOR_TEST_DIR/"
```

确认三路 `/utlidar/cloud`、`/utlidar/cloud_base`、`/utlidar/cloud_deskewed`、
`/utlidar/robot_odom` 有发布者。IMU 重点记录 `/utlidar/imu` 和 `/lowstate`；
后者的 `imu_state` 是机身 IMU。如果 IMU 话题缺失，可以完成墙距测试，但这份数据不能
完成对应 IMU 的检查。话题存在也不代表持续有数据，录完还要核对消息数量。

`/go2/imu_lowstate` 仅在桥接程序已经运行时录入；`/tf`、`/tf_static` 也可能没有发布者。
录制器会等待这些可选话题，不必为补齐它们启动另一套 SLAM 或修改当前运行配置。
如果实际话题名称不同，须同步修改下面的录制命令和 QoS 文件。

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
    /utlidar/cloud /utlidar/cloud_base /utlidar/cloud_deskewed \
    /utlidar/robot_odom /utlidar/imu /lowstate /go2/imu_lowstate \
    /tf /tf_static
}
```

先摆好第一组位置，然后执行：

```bash
record_sensors wall_080
```

确认出现三路点云和所需 IMU 的 `Subscribed to topic` 日志后，保持静止 30 秒，
按 `Ctrl+C`，等待录制器保存退出。接着检查：

```bash
ros2 bag info "$SENSOR_TEST_DIR/wall_080"
```

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

每个 bag 都应有三路点云和 odom；要判断 IMU，还必须有对应 IMU 的非零、持续数据。
如果点云有数据而 IMU 没有，本轮只能分析点云，不能据此判定 IMU 好坏。
`/go2/imu_lowstate` 和 TF 缺失时注明即可。消息计数非零不代表完全没有丢帧；
离线还需检查频率、时间间隔、时间戳回退和有效字段。

录制结束后在 Orin 打印目录：

```bash
printf '%s\n' "$SENSOR_TEST_DIR"
```

在开发机另开终端，进入本地 deploy，将下面的 `sensor-check-实际时间` 替换为打印的目录名：

```bash
cd ~/metabot-workspace/inter-nav-project/deploy
mkdir -p ../results/deploy
scp -r go2:~/inter-nav-project/results/deploy/sensor-check-实际时间 ../results/deploy/
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
估计 `点云距离 ≈ a × 实测距离 + b`。三路点云来自同一雷达，互相一致不能代替卷尺真值。
这一轮的转向数据用于定位问题，不能单凭一次转向完成完整 IMU 标定。

IMU 检查加速度、角速度、坐标系和时间戳，不以“姿态已经融合”为不可用依据。
按 ROS 的标准定义，静止加速度模长约为 9.8 m/s²，角速度接近零；若加速度模长约为 1，
先核对是否使用 g 为单位；若接近零，核对是否去除了重力或字段无效。
不能仅根据一个模长值判定好坏。转向时须先明确 IMU 轴向和外参，再比较角速度积分与实测角度，
不能直接把倾斜安装的 IMU 的 z 轴角速度当成机器人 yaw 角速度。

机身 `/lowstate`、雷达 `/utlidar/imu` 和桥接后的 `/go2/imu_lowstate` 分别检查。
接收时间对齐不等于采样时刻同步，雷达到 base 的外参也不能直接当作雷达到 IMU 的外参。
bag 接收时间与消息 header 可能不在同一个时间域，不能直接相减解释为传输延迟。

参考：[ROS IMU 数据约定](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst)、
[ROS 2 Foxy rosbag2](https://github.com/ros2/rosbag2/tree/foxy)。
