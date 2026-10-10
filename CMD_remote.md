# 远程运行（zengyi@arl）

使用与 [CMD.md](CMD.md) 相同的 main 入口和参数。服务器已有共享环境及资产；本说明不启动 GroundingDINO、Qwen-VL 或 Qwen3 服务。

2026-10-11 核对时，`/data/zengyi/inter-nav-project` 仍在 `planner` 分支且有未提交修改。下面将本地已提交的 main 打包为 Git bundle，在 `/data20t` 创建独立运行仓库，保留远端原工作区，也无需先 push GitHub。

## 1. 本地：上传已提交的 main

```bash
cd /home/seeker/metabot-workspace/inter-nav-project
git status --short --branch
git bundle create /tmp/inter-nav-main.bundle main
scp /tmp/inter-nav-main.bundle zengyi@arl:/data20t/embodied/zengyi/inter-nav-main.bundle
ssh zengyi@arl
```

bundle 只包含提交；若本地还有希望运行的修改，先提交后重新打包。

## 2. 远程：创建运行目录并进入 tmux

以下命令均在 SSH 终端执行。每次创建带时间戳的目录，确保运行本次上传的版本。

```bash
REMOTE_REPO="/data20t/embodied/zengyi/inter-nav-main-$(date +%Y%m%d-%H%M%S)"
git clone -b main /data20t/embodied/zengyi/inter-nav-main.bundle "$REMOTE_REPO"
git -C "$REMOTE_REPO" log -1 --oneline
tmux new-session -s "go2-main-$(date +%H%M%S)" -c "$REMOTE_REPO"
```

这个仓库的 origin 是上传的 bundle；用于运行与核对版本，GitHub 提交仍在本地仓库管理。

## 3. 远程 tmux 内：启动并录制

```bash
source /data20t/embodied/share/inter-nav/env.sh
nvidia-smi
```

根据服务器分配和当前占用，修改下面的 GPU 编号；不要另外设置 `CUDA_VISIBLE_DEVICES`。共享脚本会设置解释器、`PYTHONPATH` 并链接 `grutopia/assets`。

```bash
GPU=5
INTERNAV_RUN_DIR="/data20t/embodied/zengyi/inter-nav-results/main-$(date +%Y%m%d-%H%M%S)"
INTERNAV_KIT_ROOT=/data20t/embodied/zengyi/inter-nav-kit
mkdir -p "$(dirname "$INTERNAV_RUN_DIR")" "$INTERNAV_KIT_ROOT"
echo "$INTERNAV_RUN_DIR"
"$ISAAC_PYTHON" -u grutopia/demo/go2_semantic_exploration.py \
  --portable-root "$INTERNAV_KIT_ROOT" \
  --scene grscene \
  --profile grutopia/demo/profiles/go2_grscene_mv7_autonomous_exploration.json \
  --material-mode simple --target 'fire hydrant' \
  --detection-mode isaac --no-qwen --gpu "$GPU" --headless \
  --max-steps 12000 --record-every 4 --video-fps 30 \
  --record-dir "$INTERNAV_RUN_DIR"
```

`--portable-root` 将 Kit 缓存、配置和日志放到自己的可写目录。并行运行多个实例时，为各实例使用不同的 Kit 目录。`simple` 材质适合几何探索；此命令使用仿真标签，尚未自动接入推箱。

按 `Ctrl-b`，松开后按 `d` 可离开 tmux，任务继续运行。重连：

```bash
ssh zengyi@arl
tmux ls
tmux attach -t 上面列出的会话名
```

## 4. 停止、检查和取回视频

提前停止时，在另一 SSH 终端执行，用实际结果路径替换占位路径，然后等待程序退出：

```bash
touch /完整远程运行目录/STOP
```

程序结束后，在原 tmux 中执行；新终端需重新 source 环境并设置 `INTERNAV_RUN_DIR`。服务器没有系统 ffmpeg 时，使用环境自带的编码器：

```bash
cat "$INTERNAV_RUN_DIR/run_summary.json"
FFMPEG_BIN="$("$ISAAC_PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
"$FFMPEG_BIN" -hide_banner -loglevel warning -n -i "$INTERNAV_RUN_DIR/combined.mp4" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  "$INTERNAV_RUN_DIR/combined_review.mp4"
```

确认 `completion_recorded`、`artifacts_complete` 为 true，并查看 `status` 和 `reason`。`global_step_limit` 只是到达步数上限，frontier 耗尽也不能证明全屋覆盖。地图、轨迹和事件文件含义见 [CMD.md](CMD.md)。

最后在**本地终端**下载完整结果并打开视频，替换为实际远程目录名：

```bash
cd /home/seeker/metabot-workspace/inter-nav-project
REMOTE_RUN=/data20t/embodied/zengyi/inter-nav-results/main-实际时间戳
LOCAL_RUN="$PWD/grutopia/results/$(basename "$REMOTE_RUN")"
mkdir -p "$LOCAL_RUN"
rsync -av --progress "zengyi@arl:$REMOTE_RUN/" "$LOCAL_RUN/"
xdg-open "$LOCAL_RUN/combined_review.mp4"
```

共享环境说明见[服务器环境文档](docs/en/get_started/lab-shared-environment.md)。本次仅核对远程路径和环境；上述导航效果已在本地验证，未重新进行远程仿真验收。
