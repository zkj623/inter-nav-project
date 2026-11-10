"""Draw original monochrome engineering diagrams for the design document."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent/'algorithm-figures'
FONT = FontProperties(family='SimSun')


def canvas(height=6):
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    return fig, ax


def label(ax, x, y, value, size=12, **kwargs):
    ax.text(x, y, value, ha='center', va='center', fontproperties=FONT, fontsize=size, color='black', **kwargs)


def box(ax, x, y, width, height, value, size=12, dashed=False):
    ax.add_patch(Rectangle((x, y), width, height, facecolor='white', edgecolor='black', linewidth=1.25, linestyle='--' if dashed else '-'))
    label(ax, x+width/2, y+height/2, value, size)


def arrow(ax, start, end, value=None, at=None, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12, linewidth=1.1, color='black', linestyle='--' if dashed else '-'))
    if value:
        at = at or ((start[0]+end[0])/2, (start[1]+end[1])/2)
        label(ax, *at, value, 10, bbox=dict(facecolor='white', edgecolor='none', pad=1))


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+'.png'), dpi=240, bbox_inches='tight', pad_inches=.12, facecolor='white')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight', pad_inches=.12, facecolor='white')
    plt.close(fig)


def architecture():
    fig, ax = canvas(7)
    box(ax, .08,.87,.84,.09,'输入：LiDAR、机器人状态、RGB-D与任务目标')
    box(ax,.08,.68,.39,.12,'几何通道\n全局地图1 Hz／局部地图10 Hz')
    box(ax,.55,.68,.37,.12,'语义通道\n视觉确认或布局授权')
    arrow(ax,(.27,.87),(.27,.80));arrow(ax,(.74,.87),(.74,.80))
    box(ax,.23,.49,.54,.11,'行为决策\n导航／探索／主动观察／推箱')
    arrow(ax,(.28,.68),(.38,.60),'可行空间',(.25,.63))
    arrow(ax,(.73,.68),(.63,.60),'技能授权',(.77,.63))
    box(ax,.08,.30,.39,.11,'路线规划与跟踪\n目标、前沿或观察位置')
    box(ax,.55,.30,.37,.11,'push_aside skill\n对齐、接触、推动与退让')
    arrow(ax,(.35,.49),(.28,.41));arrow(ax,(.65,.49),(.73,.41))
    box(ax,.23,.11,.54,.11,'同一底层运动policy\n机体速度指令＋机器人观测 → 十二关节动作')
    arrow(ax,(.28,.30),(.38,.22));arrow(ax,(.73,.30),(.63,.22))
    label(ax,.5,.045,'物理执行后的机器人状态、箱体运动和新扫描反馈至地图与决策',11)
    ax.plot([.77,.97,.97,.92],[.165,.165,.915,.915],color='black',lw=1,linestyle='--')
    arrow(ax,(.97,.915),(.92,.915),dashed=True)
    save(fig,'01-system-architecture')


def decisions():
    fig, ax = canvas(7)
    box(ax,.29,.86,.64,.10,'已观测地图可形成参考通路？')
    box(ax,.02,.65,.25,.12,'前沿探索\n或等待观测')
    arrow(ax,(.29,.91),(.14,.77),'否',(.16,.86))
    box(ax,.39,.64,.54,.12,'现有路线存在，且\nL_nav ≤ 1.6 L_ref + 0.5 m？')
    arrow(ax,(.66,.86),(.66,.76),'是',(.70,.81))
    box(ax,.02,.40,.25,.12,'普通导航\n跟踪现有路线')
    arrow(ax,(.39,.70),(.145,.52),'是',(.28,.62))
    box(ax,.39,.43,.54,.12,'搜索推箱候选\n8个方向 × 离散推距与碰撞检查')
    arrow(ax,(.66,.64),(.66,.55),'否',(.70,.595))
    box(ax,.39,.22,.54,.13,'存在可行候选，且推箱代价更低\n或当前没有普通导航路线？',11)
    arrow(ax,(.66,.43),(.66,.35))
    box(ax,.65,.03,.29,.10,'执行推箱skill')
    box(ax,.08,.03,.43,.10,'有路线则导航\n否则探索／等待')
    arrow(ax,(.76,.22),(.80,.13),'是',(.84,.18))
    arrow(ax,(.49,.22),(.34,.13),'否',(.38,.18))
    save(fig,'02-behavior-selection')


def policy():
    fig, ax = canvas(6)
    label(ax,.245,.95,'离线训练',15);label(ax,.75,.95,'导航与交互部署',15)
    box(ax,.03,.72,.42,.13,'1024个并行速度跟踪环境\n采样机器人状态、动作与奖励')
    box(ax,.03,.45,.42,.15,'PPO：Actor–Critic\nActor观测45维／Critic观测60维\n2000轮训练',11)
    arrow(ax,(.24,.72),(.24,.60),'状态转移',(.32,.66))
    box(ax,.03,.18,.42,.13,'导出并冻结Actor\n权重与部署契约哈希校验')
    arrow(ax,(.24,.45),(.24,.31))
    box(ax,.56,.72,.41,.13,'路线跟踪器／推箱skill\n生成速度指令(v_x, v_y, ω_z)')
    box(ax,.56,.45,.41,.15,'同一固定Actor\n速度指令＋机器人观测\n输出12维关节位置偏移',11)
    arrow(ax,(.76,.72),(.76,.60))
    arrow(ax,(.45,.245),(.56,.51),'部署权重',(.50,.36))
    box(ax,.56,.18,.41,.13,'PD／电机模型与接触动力学\n执行关节动作，反馈实测状态')
    arrow(ax,(.76,.45),(.76,.31))
    label(ax,.5,.07,'物体几何和推箱目标由skill处理；底层policy学习并执行机体速度跟踪',11)
    save(fig,'03-policy-and-skill')


def geometry():
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.set(xlim=(0,8),ylim=(-1.45,2.4),aspect='equal');ax.axis('off')
    # Schematic corridor and side parking area, not benchmark geometry.
    ax.plot([.3,7.7],[-.65,-.65],color='black',lw=2)
    ax.plot([.3,3.1],[.65,.65],color='black',lw=2)
    ax.plot([5.2,7.7],[.65,.65],color='black',lw=2)
    ax.plot([3.1,3.1,5.2,5.2],[.65,2.05,2.05,.65],color='black',lw=2)
    ax.plot([.5,7.4],[0,0],color='black',ls='-.',lw=1)
    ax.add_patch(Rectangle((3.7,-.40),.8,.8,fill=False,ls='--',lw=1.6))
    ax.add_patch(Rectangle((3.7,1.10),.8,.8,facecolor='white',edgecolor='black',lw=1.6))
    arrow(ax,(4.1,.43),(4.1,1.07))
    ax.add_patch(Rectangle((.7,-.22),.82,.44,fill=False,lw=1.4))
    ax.add_patch(Rectangle((.57,-.35),1.08,.70,fill=False,ls=':',lw=1.3))
    arrow(ax,(1.7,0),(2.6,0));arrow(ax,(5.3,0),(6.9,0))
    label(ax,4.1,2.23,'侧方停放区域',14)
    label(ax,6.35,1.45,'实线：候选停放位置',12)
    label(ax,5.8,.48,'虚线：初始箱体',12)
    label(ax,1.15,-.9,'机身占据区域\n及安全裕度',12)
    label(ax,4.4,-.95,'箱体移动后仍需新扫描确认\n并检查机身沿通路的扫掠空间',12)
    save(fig,'04-interaction-geometry')


def main():
    architecture();decisions();policy();geometry()


if __name__ == '__main__':
    main()
