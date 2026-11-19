# Go2 单箱前视 AprilTag 与 LiDAR 观测实验

本轮用一个直立刚性箱，目标是找到从斜向接近到接触、再到退出时，两种传感器分别在哪段距离提供有效观测。操作员控制接近位置。浏览器在前置 RGB 预览上识别标签，LiDAR 使用真实的登记箱体跟踪器。两路观测独立显示，结果不向运动进程授予推移许可。

## 打印与布置

使用仓库中的 [两页 A4 标签](../../output/pdf/go2-box-apriltag-36h11-120mm.pdf)。单面、哑光、100% 实际大小打印。打印后量黑色正方形最外沿，两方向均应为 120 mm。另量页面的 100 mm 校验线。沿浅色虚线裁切后约 152 mm 见方，留下每侧约 16 mm 白边。PDF 中的码图来自 [AprilRobotics 的 tag36h11 图库](https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11)。

- ID 0 贴在预定接触的竖直面，横向居中。先用纸胶带临时定位，狗正常站立、离箱约 30–50 cm 时查看画面，把码放在画面中部且避开实际顶推点。优先调高度，避免贴太高后近距离被画面上沿裁掉。保持整张纸平整。
- ID 1 贴在初始斜向接近时可见的相邻竖直面，尽量同高、整张贴在一个平面。用尺记录两张标签中心离地面、各自离同一个箱角的距离，标注所在面，并拍一张布置照片。若箱面容不下完整白边，应按实际箱体重新选择标签规格。
- 量箱体真实长、宽、高，单位米。用胶带标出箱体初始轮廓和朝向，另标一条 20 cm 的移动线。选好机身前端的固定测量点，各距离档统一量这个点到箱面的水平间隙。

## 创建只观察会话

以下命令假定 Orin 已有本版查看器，仓库在 `~/inter-nav-project`。先从开发机登录并转发可视化端口。

```bash
ssh -L 8080:127.0.0.1:8080 go2
```

在这个 SSH 终端执行下列命令。首次关联区域的示例值适合把箱体中心放在起点前方约 1.2–1.5 m、左右 0.8 m 以内的布局。实验坐标原点在 run 启动时固定，X 正向为初始前方，Y 正向为初始左方。按实际布置调整区域，只放一个匹配尺寸的箱体。

```bash
source ~/unitree_ros2/setup.sh
cd ~/inter-nav-project/deploy
source ../.venv-deploy/bin/activate
trial_cfg=$(mktemp -d "$HOME/go2-tag-observe.XXXXXX")
cp -a configs/. "$trial_cfg/"
read -rp '输入实测长 宽 高（米，空格分隔） ' box_l box_w box_h
python - "$trial_cfg" "$box_l" "$box_w" "$box_h" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = Path(sys.argv[1])
dims = list(map(float, sys.argv[2:]))
settings = {
    'source': 'registered_box_lidar',
    'tracking_enabled': True,
    'geometry_verified': False,
    'registry_reference': 'round-1, box-1, measured dimensions, front tag 0, adjacent tag 1',
    'boxes': [{'id': 'box-1', 'dimensions_m': dims,
               'search_region': [0.5, 2.2, -0.8, 0.8], 'push_allowed': False}],
    'tracking': {'max_age_s': 0.25},
    'contact_profile': None,
}
(cfg / 'interaction.yaml').write_text(yaml.safe_dump(settings))
nav_path = cfg / 'navigation.yaml'
nav = yaml.safe_load(nav_path.read_text())
nav['interaction_enabled'] = False
nav_path.write_text(yaml.safe_dump(nav))
PY
python run.py --config-dir "$trial_cfg" --check &&
python run.py --config-dir "$trial_cfg" --observe-only --camera sdk
```

在 Orin 的第二个 SSH 终端启动记录，等它打印出单次记录目录后再开始移动。

```bash
cd ~/inter-nav-project/deploy
source ../.venv-deploy/bin/activate
python record.py --output ~/go2-tag-trials
```

在 Orin 的第三个 SSH 终端打开可视化。开发机浏览器打开 [现场观察页](http://127.0.0.1:8080)，点击「前置 RGB」栏的「AprilTag 识别框」。第一个终端的 SSH 转发需保持连接。

```bash
cd ~/inter-nav-project/deploy
source ../.venv-deploy/bin/activate
python view.py
```

## 三段观测

1. **建立身份。** 在箱体仍处于登记区域时，让机载雷达斜看两个相邻竖面，建议从与箱面法向约 30–45° 开始。保持 3–5 秒，确认现场画面中是否出现 ID 0 或 1 的四角，以及图下方是否显示 `box-1` 已跟踪。可在记录中核对两个可见面和拟合中心。标签与 LiDAR 登记项的对应关系在这轮由单箱布置确定。
2. **固定箱体，移动机器人。** 用原生遥控缓慢转正、接近和退出。在约 1 m、50 cm、30 cm、15 cm、5 cm 各停 3–5 秒，按箱体和机身几何调整站位。零间隙可让机器人先站稳，再从侧面将箱子轻靠预定接触点。逐档记下哪个 ID 可读、首次四角不全的位置、LiDAR 从双面转为单面的位置和第一次丢失原因。比较世界坐标中的箱体中心和朝向与地面标记，检查机器人自运动造成的假位移。
3. **固定机器人，移动箱体。** 在身份有效且视野好的站位，把箱子沿标线慢移 20 cm 并停住，再转一个量过的角度。检查 LiDAR 中心变化的方向、距离及停下后的漂动。另做一次遮挡测试，记录标签和 LiDAR 各自何时失效。当前 LiDAR 跟踪器在已确认身份丢失后会锁住该身份；下一轮退出并重开只观察会话、重新斜看两面建轨。每次重启需重新对应实验坐标和新记录目录。

## 查看与判读

单帧点云较稀疏时，可以选择 1 秒滑动窗口。在第一终端的启动命令末尾加 `--box-window`，其余配置、录制和端口相同。开始时让机器人停稳、斜看两个箱面；建轨后再逐档接近。普通 view 的诊断栏及记录查看器均显示轨迹编号。记录查看器中的 `confirming` 表示等待不共用扫描的窗口复核，`tracked` 表示本次拟合有效，`missing` 表示临时保留编号且本次没有有效位置。双面实测支持过期即撤下位置，最长保留轨迹身份 1 秒。重新确认后使用新的编号。箱面方向按 90° 周期表示。多帧输出只用于观察，不提供推移动作的目标或许可。

离线重跑使用同一观测器和当次的登记配置：

```bash
python run.py --config-dir "$trial_cfg" --replay "$trial_run" --task observe --box-window --output ~/go2-tag-replay
```

输出目录中的 `events.jsonl` 保存每次判定。窗口末帧的 `scan_id`、`captured_at` 用于匹配记录原始扫描。`detections[].captured_at` 取两个箱面最新支持时刻中较早的那个，`expires_at` 按此时刻计算。只有仍然有效的双面拟合会输出中心；运行时还计入处理耗时。

现场观察页同时显示前置 RGB 标签框、导航图和点云。箱体跟踪诊断在图下方，未核验几何的登记箱体只出现在诊断中。

需要逐帧核对时，在 Orin 的 `deploy` 目录执行 `python scripts/review_box_tags.py "<单次记录目录>"`，通过 `ssh -L 8081:127.0.0.1:8081 go2` 转发后打开[记录回看页](http://127.0.0.1:8081)。回看页显示标签四角及最短边像素数。LiDAR 栏按原始 scan ID 和采集时间，从 `events.jsonl` 中取同一次扫描的拟合，显示箱体中心、朝向、可见面、拟合残差、两侧边缘射线或失效原因。没有对应结果时显示缺失。

SDK 预览默认约 2 Hz，仅有收包时间，多个显示帧可能共用同一张 JPEG。用各距离档停留期间的观测作对照。页面的“暂停在此帧”只暂停浏览；“上一帧 / 下一帧”可以检查丢失前后。先在 run 终端按 `q`，等 record 写完，再退出查看器。稍后对同一记录目录运行同一脚本可再次逐帧查看。

若标签近处丢失而 LiDAR 单面跟踪仍有效，记录可交接的距离区间。若两路在接触前同时失效，先移动标签位置或调整观察角重测。若标签可读但箱体中心在机器人移动时明显漂移，优先检查点云、里程计和地面参考。带载试验前，需要用测得的位姿误差、连续丢失时长和实际接触余量决定控制参数。
