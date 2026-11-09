# Go2底层运动policy训练记录

此目录记录交付权重对应的训练过程。策略在项目中使用PPO训练，配置改编自Unitree RL Lab，不是直接下载的官方预训练权重。

- `source/train_go2_official_baseline.py`：原训练入口。
- `source/go2_official_training/`：环境、观测、指令课程和奖励配置；`upstream_manifest.json`记录改编来源。
- `records/source_manifest.json`：训练时保存的9份源码哈希，交付时已与源码逐一校验。
- `records/ppo_config.json`：训练算法参数。
- `records/training_contract.json`：机器人、观测、动作和物理配置。
- `records/summary.json`：1024环境、2000轮训练及Actor导出等价性检查。
- `records/live.json`：最后一轮训练指标，累计49,152,000条状态转移。

来源：<https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3>。保留各源码文件的来源标注，原始文件对应关系见`source/go2_official_training/upstream_manifest.json`。

训练环境为Isaac Sim 4.2.0.2、IsaacLab 1.4.0（提交dd044a4977722d85e966c5b63949093c9b14667f）、RSL-RL 2.3.1。导航部署使用已交付权重及`reproduction/requirements.txt`，无需重新训练或将部署环境升级为训练环境。

原始摘要中的`original_task_accepted=false`等字段记录训练结束时的状态；后续导航任务验收证据位于`delivery/evidence/T01`至`T10`。算法设计文档第2章说明训练方法和policy到推箱skill的衔接。
