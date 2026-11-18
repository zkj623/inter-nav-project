# Benchmark v0.1 状态

10 个家庭场景，15 种 genie2-rsi 刚体物体，30 个放置实例，60 个固定任务。

- 10 个场景的离线几何/碰撞配置检查通过；15 种物体的 USD 与网格 SHA256 均与 genie2-rsi 清单一致。
- 物理静置检查：10/10 个已检查场景通过。每次 240 步，步长 1/120 秒。物体平面位移 <0.25 m、竖直变化 <0.20 m、最终速度 <0.10 m/s、倾斜 <15°。
- 几何初始不连通任务 26 个，初始仍连通任务 34 个。该分类基于 0.30 m 圆形机器人近似，不等同于机器人实际交互能力。
- 参考路径长度 3.96–11.88 m；每场景 6 组独立起终点，按三个任务的双向运行设置。
- 尚未执行 60 个任务的机器人导航/交互闭环；尚未冻结正式训练/验证/测试划分。

| 场景 | 原场景 ID | 地面投影面积 m² | 初选分组 |
|---|---|---:|---|
| S01 | MWAX5JYKTKJZ2AABAAAAAEA8_usd | 57.98 | small |
| S02 | MVUHLWYKTKJ5EAABAAAAACY8_usd | 88.69 | small |
| S03 | MWAX5JYKTKJZ2AABAAAAAAY8_usd | 98.30 | small |
| S04 | MV7J6NIKTKJZ2AABAAAAADY8_usd | 101.03 | medium |
| S05 | MWBGLKQKTKJZ2AABAAAAABI8_usd | 104.37 | medium |
| S06 | MV7J6NIKTKJZ2AABAAAAAEA8_usd | 108.13 | medium |
| S07 | MVUCSQAKTKJ5EAABAAAAADA8_usd | 113.75 | medium |
| S08 | MWBGLKQKTKJZ2AABAAAAABQ8_usd | 120.53 | large |
| S09 | MVUHLWYKTKJ5EAABAAAAABI8_usd | 142.46 | large |
| S10 | MWAX5JYKTKJZ2AABAAAAAAA8_usd | 201.35 | large |

面积为近地面水平投影的 0.1 m 栅格并集估计；仅涵盖家庭环境。物体物理参数沿用源实验的 1 kg、摩擦 1.0/0.9 假设。

本地查看 `fixtures/index.html`；固定任务重新实例化见 README 的 materialize.py 命令。视觉原场景与兼容预览分别生成，物理检查使用去除着色器的几何副本，未验证完整材质渲染效果。
