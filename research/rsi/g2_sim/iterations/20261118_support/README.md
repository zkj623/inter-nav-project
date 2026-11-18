# 2026-11-18：镜像侧夹、托底与接触力平衡

本轮针对原来未完成的圆凳、曲面木凳和两把椅子尝试不同交互方式。通过记录、规划淘汰与失败试验全部保留；没有通过放松姿态/离地验收标准制造成功。

## 物理结果

| 物体 / 控制版本 | 通过/已完成 | 包含站姿下降 | 最低保持抬升 cm | 最大保持倾斜 ° |
| --- | --- | --- | --- | --- |
| stool_small / mirror | 3/3 | True | 19.61 | 4.77 |
| stool_rectangular / supportbalance | 3/3 | False | 19.85 | 0.28 |
| stool_rectangular / supportpath | 3/3 | True | 19.85 | 0.08 |
| stool_small / mirroroffset | 3/3 | True | 19.64 | 1.54 |
| stool_rectangular / supportpathoffset | 1/1 | True | 19.85 | 0.43 |

`mirror` 将右手方向镜像，使用原来的物体 40% 高度侧面区域，避免左右手以不匹配方向加载。圆凳为刚体替代模型，侧面软包的真实刚性未确认，暂不作为已验证的真实刚体交互训练标签。

`supportbalance` 从低位预姿态开始：双手张开，分别以两条手指分支托住曲面木凳座面下方；根据内外指接触力差微调腕部滚转，每次最多 0.15°、累计不超过 4°，让四个承力面共同承重。它不等价于完整下降动作。

`supportpath` 使用相同托底控制，加入经过机器人自碰撞和全身/物体碰撞检查的下降路径：双手保持在外侧和高位下降，再沿凳体外侧下移，最后伸入座面下方。先做 2.5 cm 承力验证，四个接触建立后再完成 20 cm 抬升；这样可以在接触不均时停止。

托底的四接触定义为每只手两条手指分支（包含近端指节）各承力 >0.5 N，保持覆盖率至少 90%；不是要求四个末端 link4 同时接触。结果另存末端四指比例。侧夹仍使用原末端四指判据。高度至少 15 cm、真实保持 3 s、倾斜不超过 20°、相对漂移不超过 2 cm 及接触表面一致性检查不变。

`offset` 仅改变物体初始位置 3 mm，机器人轨迹和验收标准不变；所有已完成结果见 robustness_screen.json。未完成随机多维扰动、不同质量/摩擦系数的统计验证。同初态三次运行不能视为鲁棒成功率。

## 两把椅子与失败方案

尝试了不同开度、上下掌面方向、座面下四点支撑、单手托底配另一手夹结构件等候选。淘汰原因记录在 plans 中，包括手指/腕部与椅身相交、双臂逆解不可达、躯干/手臂碰撞。规划没有找到合格方案不表示数学上证明不可抓取；它意味着本轮搜索未找到可安全执行的轨迹，不能计入物理抓取成功。

直接下落和预设收臂路径曾发生腕部碰撞或双手自碰撞；这些失败规划保留。木凳最初四个低姿态物理试验在 2.5 cm 承力验证阶段因为一个指面不承力而停止，之后才加入腕部接触力平衡。

## 复现与边界

全部仍为 1 kg 刚体筛查和理想躯干驱动，保留手臂/夹爪力矩上限。没有物体焊接、抬升期间物体位姿写入、机器人基座固定。验证范围为原地抬升，未验证负载起身、搬运移动或实机。

本目录叠加在 ../20261118 和原 snapshot 上；原归档不修改。result_index.json 将结果关联至源码 SHA256、外部完整视频和样本路径；manifest.json 验证归档字节。ground_clearance_check.json 用实际可视网格所有顶点计算最低离地距离，避免把物体原点抬高误认为完全离地。

```bash
python replay.py verify
python replay.py prepare --workspace /absolute/new-workspace --assets-from /absolute/g2-migration-20261111 --collision-deps /absolute/r1lite-collision-deps
python replay.py run --workspace /absolute/new-workspace --python /absolute/isaaclab/bin/python --asset stool_small --gpu 0 --repeat 1
```

prepare 需要原外部资产与前一迭代的基础依赖。baselines.json 只列入已完成三次完整下降并通过的方案，低姿态单独通过不会被列入。
