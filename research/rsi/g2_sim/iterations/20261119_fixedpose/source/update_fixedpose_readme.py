from pathlib import Path
R=Path(__file__).resolve().parent
p=R.parent/'g2-publish-20261118/research/rsi/g2_sim/README.md';s=p.read_text()
note='''## 2026-11-19 更新

[固定低姿态、只训练双臂抬升的姿态筛选与验证](iterations/20261119_fixedpose/README.md)：腰部五关节 [-50, 150, -85, 0, 0] 度，肩高约 0.791 m、整体前倾 15 度，固定底盘/腰/头/轮组，仅 16 维手臂与夹爪动作。11 种物体通过几何路径审计，3 种代表物体各完成一次固定支撑物理抬升通过；其余未在该姿态完成物理重测。统一的是非手臂姿态，双臂仍按物体预抓取初始化。尚未开始训练。

'''
assert '## 2026-11-19 更新' not in s
p.write_text(s.replace('## 2026-11-18 更新',note+'## 2026-11-18 更新',1))
