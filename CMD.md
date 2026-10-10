# 本地运行（main）

从仓库根目录运行。下面复现本次 GRScene / Go2 找不到目标时的自主探索：几何 frontier 选点、Voronoi 优先导航、相机读取 Isaac 自带语义标签，不需要启动视觉模型服务。入口仍是 main 原来的 `go2_semantic_exploration.py`。

## 1. 环境与资源

本机已安装 `grutopia` 环境和 MV7 场景；新机器先完成[环境安装](docs/en/get_started/environment-setup.md)。

```bash
cd /home/seeker/metabot-workspace/inter-nav-project
conda activate grutopia
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

缺少资源时，在这个环境中依次运行（已有资源可以跳过）：

```bash
python grutopia/demo/download_mv7_scene.py
python grutopia/demo/download_go2_asset.py
python grutopia/demo/download_go2_policy.py
```

## 2. 启动探索并录制 combined 视频

```bash
RUN_DIR="$PWD/grutopia/results/main-exploration-$(date +%Y%m%d-%H%M%S)"
echo "$RUN_DIR"
python -u grutopia/demo/go2_semantic_exploration.py \
  --scene grscene \
  --profile grutopia/demo/profiles/go2_grscene_mv7_autonomous_exploration.json \
  --material-mode simple --target 'fire hydrant' \
  --detection-mode isaac --no-qwen --gpu 0 --headless \
  --max-steps 12000 --record-every 4 --video-fps 30 \
  --record-dir "$RUN_DIR"
```

- `fire hydrant` 是本次测试中未找到的标签；找冰箱可改为 `refrigerator`。
- `--gpu 0` 按实际可用 GPU 修改；有桌面时改为 `--no-headless` 可看实时窗口。
- 此 profile 仅调整了跟随相机，便于观察步态。`simple` 简化场景材质、保留碰撞与语义，适合本次几何探索；评估视觉模型时应使用原始或 preview 材质。
- 输出目录必须是新目录。12000 是运行步数上限，不保证走遍全屋。此入口尚未自动接入遇阻推箱。

## 3. 提前停止

在另一个终端中，用启动时打印的完整路径替换下面的占位路径：

```bash
touch /完整运行目录/STOP
```

等待程序保存地图、关闭录像后退出；tmux 或窗口关闭不等于正常保存。

## 4. 查看结果

回到启动终端执行；若是新终端，先将 `RUN_DIR` 设为实际结果目录。

```bash
cat "$RUN_DIR/run_summary.json"
ffmpeg -hide_banner -loglevel warning -n -i "$RUN_DIR/combined.mp4" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart \
  "$RUN_DIR/combined_review.mp4"
xdg-open "$RUN_DIR/combined_review.mp4"
```

`combined_review.mp4` 是便于播放器查看的 H.264 视频；`final_map.json/.npz` 保存地图和探索统计，`trace.csv`、`events.jsonl` 用于定位动作与规划问题。运行期间可查看 `progress.json`。

以 `run_summary.json` 的 `status`、`reason`、`completion_recorded` 和 `artifacts_complete` 判断结果：`global_step_limit` 表示达到预算，`stop_requested` 表示主动停止，frontier 耗尽也不代表已证明全屋覆盖。进程退出码不能单独说明找到目标。

模型识别与更多参数见[语义探索说明](docs/en/get_started/go2-semantic-exploration.md)；服务器运行见 [CMD_remote.md](CMD_remote.md)。
