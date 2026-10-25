# HANDOFF

更新时间：2026-10-26（Asia/Shanghai）
发布分支：`codex/go2-interactive-navigation-20261026`
服务器发布工作树：`/data20t/embodied/zkj/inter-nav-publish-go2-20261026`
实验工作树：`/data20t/embodied/zkj/inter-nav-project`

<a id="go2-interactive-navigation"></a>

## Go2 原尺寸 MV7：可交互箱子点到点导航（2026-10-26）

入口为 `grutopia/demo/go2_side_clearance_navigation.py`，本次发布分支为 `codex/go2-interactive-navigation-20261026`。场景和 Go2 均为原尺寸，辅助碰撞地面高 0.01 m；程序判断是否需要交互，需要时将挡路箱子侧推到通路一边，再穿过通路到达终点。

### 连接服务器并运行一组推箱任务

在本机终端连接：

```bash
ssh zkj@162.105.195.166
```

随后在服务器的 Bash 中执行以下命令。示例进入已准备好的发布工作树；若自行 clone，请先检出上述分支，并把 `cd` 路径改为自己的仓库目录。示例运行原始起终点的 **T05**，使用 GPU 1；可按空闲 GPU 情况修改 `--gpu`。

```bash
cd /data20t/embodied/zkj/inter-nav-publish-go2-20261026
source /data20t/embodied/share/inter-nav/env.sh

go2_inputs=/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/collision-corrected-inputs
go2_out="/data20t/embodied/zkj/inter-nav-results/go2-interactive-nav-$(date +%Y%m%d-%H%M%S)"

set -o pipefail
"$ISAAC_PYTHON" -u grutopia/demo/go2_side_clearance_navigation.py \
  --benchmark-dir "$go2_inputs" \
  --task-ids T05 \
  --gpu 1 \
  --floor-top 0.01 \
  --max-steps 30000 \
  --record-video --material-mode preview \
  --route-planner voronoi --navigation-heading path \
  --path-controller pure --roadmap-shape footprint --route-clearance 0.22 \
  --push-max-speed 0.95 --turn-speed 1.0 \
  --contact-centering-speed 0.25 --contact-yaw-tolerance 0.10 --contact-front-gap 0.40 \
  --output "$go2_out" 2>&1 | tee "${go2_out}.log"
```

使用共享环境的 `$ISAAC_PYTHON`，GPU 通过 `--gpu` 选择，不设置 `CUDA_VISIBLE_DEVICES`。每次使用新的输出目录。上述命令前台运行，日志同步保存为 `${go2_out}.log`。

输入目录已包含箱子布局、门口通道、修正后的碰撞栅格及任务：`scene_profile.json`、`layout.json`、`passages.json`、`navigation_grid.npz`、`original_metadata.json`、`tasks_preflight.json`，另有地面配置和碰撞修正审计。场景配置引用服务器资产 `/data20t/embodied/zkj/inter-nav-results/rectangular-boxes-five-scenes-xy100-20261021-v2/mv7/after_scene.usda`；复现时需要保留这些外部数据和共享 Go2 模型/policy。

### 任务选择和主要参数

这些是同一个 MV7 场景中的五组点到点任务；坐标单位为米，起点/终点在地图上显示为 S/G。下表交互类型为当前初始布局下的决策，运行时会根据实际位姿重新判断。

| 任务 | 起点 `(x, y)` | 终点 `(x, y)` | 初始决策 |
|---|---|---|---|
| T01 | `(2.11, -0.78)` | `(12.0, -0.5)` | 普通导航 |
| T02 | `(13.6, -0.4)` | `(2.11, -0.78)` | 普通导航 |
| T03 | `(7.74, -4.7)` | `(8.0, 4.2)` | 普通导航 |
| T04 | `(2.11, -0.78)` | `(10.3, 1.75)` | 侧推 B06 |
| T05 | `(13.6, -0.4)` | `(3.3, 1.4)` | 侧推 B07 |

- 只运行两组推箱任务：将命令中的 `--task-ids T05` 改为 `--task-ids T04,T05`。
- 运行全部五组：改为 `--task-ids T01,T02,T03,T04,T05`。`--max-steps 30000` 是每组任务的控制循环上限，录制索引还包含初始化后的停步/对齐帧，二者不能直接等同。
- 无录像的物理调试：去掉 `--record-video`，将 `--material-mode preview` 改为 `--material-mode simple`。
- 普通导航复用纯探索的 Voronoi 骨架规划和原路径跟随控制，机身朝向随路径变化。规划图使用已有碰撞先验和仿真箱子位姿，LiDAR 在线建图同时运行；图不连通时的栅格回退记录在结果中。
- `--contact-centering-speed 0.25` 是接触对齐的纵向速度请求；当前控制器横向纠偏为 0.35 m/s、后退至少 0.35 m/s。`--contact-front-gap 0.40` 为鼻端预接触目标间距，转向容差 `0.10` 的单位是 rad。这些是当前实验参数，尚未证明完整任务稳定成功。
- 录像为完整 25 fps，第三视角正上方 2.5 m、俯角 90°、FOV 80°，狗前方朝画面上方并随实际 yaw 转动。

### 输出与结果判断

运行输出保存于 `$go2_out`：

```text
summary.json                       # 各任务 success、reason 和完成情况
recording_config.json               # 本次实际录像/控制配置
source/、source_manifest.json        # 启动源码快照及哈希
T05/result.json                     # 本组结果、侧推/穿门状态、规划统计
T05/events.jsonl、trajectory.json    # 事件和轨迹
T05/video/combined.mp4              # 四画面完整录像
T05/video/third_person.mp4          # 头朝上第三视角
T05/video/robot_rgb.mp4             # 第一视角
T05/video/map_topdown.mp4           # 带 S/G 的俯视地图
T05/video/pointcloud_map.mp4        # LiDAR 点云
```

多任务会分别生成 `T01/` 至 `T05/` 子目录。`summary.json` 的 `complete` 表示本批任务已经结束；判断导航成功需看各任务的 `success`、`reason`，推箱任务还应核对箱子侧停、清路和 `full_body_passage_crossed`。退出码 0 或视频正常保存不代表成功。

最近结果截至 2026-10-26：T01 的独立正式录像已成功；箱前局部 T05 校准成功（4531 步、终点误差约 0.031 m），但该校准修改了起点且未录像。原始起终点的正式 v19 批次中，T04 侧推 B06 约 0.808 m 并清路，随后在 `cross_passage` 阶段失稳；T05 侧推 B07 约 0.655 m，尚未清路，在 `push_center` 阶段失稳。该批次 T02/T03 也未成功，不能将这条命令描述为五组全部成功的复现。

正式 v19 的完整诊断录像和结果：

- 服务器：`/data20t/embodied/zkj/inter-nav-results/go2-voronoi-heading-up-20261025/full-voronoi-final-remaining-v19/{T02,T03,T04,T05}/video/`。
- 本机：`E:\inter-nav-results\go2-voronoi-heading-up-20261025\full-voronoi-final-remaining-v19\{T02,T03,T04,T05}\video\`。
- T01 已验证录像：本机 `E:\inter-nav-results\go2-voronoi-heading-up-20261025\T01-current-verified\T01\video\`。

### 场景准备与代码范围

`grutopia/demo/prepare_interactive_box_scenes.py` 和 `grutopia_extension/interactive_navigation/interactive_box_scenarios.py` 保存五场矩形箱子布局及 USD 叠加层的生成逻辑；`grutopia/demo/render_interactive_box_previews.py` 提供前后对比图与物理检查。这些准备工具使用服务器保留的 20261019 五场 profile、原始资产和架构栅格。上面的导航命令直接复用已生成的 MV7 输入目录。

本次推送包含 Go2 导航、箱子交互、可配置辅助地面、录像及相关测试。机器人/场景资产、模型权重、生成的输入数据和视频继续保存在上述服务器及 E 盘目录，不随 Git 推送。仅 clone 仓库还需要安装共享环境并取得这些外部数据。

`grutopia/demo/go2_walk_demo.py` 提供无路径规划的恒速步态诊断，可用于区分 locomotion 与导航控制问题。

### 本次发布验证

在基于 `origin/main` 的独立 Go2 发布工作树中，265 项测试通过。主分支已有的 `test_g1_third_person_is_close_and_elevated` 依赖未包含在该分支的 G1 配置文件，本次未运行该项；其余测试均通过。测试日志和命令预检记录位于服务器结果目录的 `publish-20261026/`。

T05 单组与全部五组的文档命令均通过 `--preflight-only` 检查，确认参数、输入数据及所选原始起终点可加载；这不包含新的物理复跑。实际物理结果仍以上述 v19 记录为准。录像编码依赖 `imageio-ffmpeg`，已补入 `requirements/runtime.txt`。
