"""Add the recorded Go2 locomotion-policy training and its skill interface."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from report_methods import table


def section(doc, title, content):
    doc.add_heading(title, 2)
    doc.add_paragraph(content)


def add_training(doc):
    doc.add_heading('02  底层运动policy训练与技能封装', 1)
    section(doc, '2.1  训练任务与算法',
        '底层policy学习Go2的机体速度跟踪，将前向速度、横向速度及偏航角速度指令映射为十二个关节的运动目标。训练基于Unitree RL Lab的Go2速度任务配置进行适配，在Isaac Sim 4.2.0.2、IsaacLab 1.4.0和RSL-RL 2.3.1环境中采用PPO算法完成。交付权重由本项目训练产生，训练记录中official_pretrained_policy=false。')
    doc.add_paragraph('采用非对称Actor–Critic结构：Actor根据部署时使用的观测输出动作，Critic在训练时额外使用机体线速度与关节力矩估计状态价值。PPO通过截断策略比值更新Actor，并联合优化价值函数及熵项。训练任务是速度跟踪，物体位置、推箱目标和箱体位移不作为Actor输入或推箱奖励。')
    section(doc, '2.2  观测、网络与动作定义',
        'Actor观测按固定顺序组成45维向量，Critic观测为60维。Actor和Critic均采用512、256、128三个隐藏层及ELU激活函数，部署时仅加载确定性Actor。')
    table(doc, [
        ('Actor观测（45维）', '机体角速度3维、投影重力3维、速度指令3维、相对关节位置12维、相对关节速度12维、上一时刻动作12维。'),
        ('Critic观测（60维）', '机体线速度3维，加上上述45维观测对应项和关节力矩12维；训练时用于价值估计。'),
        ('预处理', '角速度缩放0.2、关节速度缩放0.05，其余Actor项缩放1；各项缩放前截断至[−100,100]；不使用经验观测归一化。'),
        ('动作（12维）', '输出相对于默认关节姿态的位置偏移：q_des = q_default + 0.25 × a。'),
        ('关节执行', 'PD刚度25、阻尼0.5；电机饱和力矩23.5 N·m，关节速度上限30 rad/s；部署采用与训练契约对应的explicit_dc执行。'),
    ])
    doc.add_page_break()
    section(doc, '2.3  奖励、随机化与终止条件',
        '奖励由指令跟踪项及姿态、关节和足端约束项组成。平面线速度与偏航角速度跟踪均使用指数形式，误差尺度参数为0.5；奖励项及权重记录于训练环境配置。')
    table(doc, [
        ('速度跟踪', '平面线速度跟踪 +1.5；偏航角速度跟踪 +0.75。'),
        ('机体运动与姿态', '竖直线速度 −2.0；横滚／俯仰角速度 −0.05；水平姿态偏差 −2.5。'),
        ('关节运动', '关节速度 −0.001；加速度 −2.5×10⁻⁷；力矩 −2×10⁻⁴；动作变化 −0.1。'),
        ('关节位置与能量', '关节限位 −10；相对关节位置约束 −0.7；能量项 −2×10⁻⁵。'),
        ('足端与接触', '足端腾空时间 +0.1；腾空时间方差 −1.0；足端滑动 −0.1；非期望接触 −1.0。'),
    ])
    doc.add_paragraph('训练场景使用平面地形。启动时随机化机器人摩擦系数、恢复系数及机体附加质量：静／动摩擦系数均在[0.3,1.2]采样，恢复系数在[0,0.15]采样，机体附加质量在[−1,3] kg采样。回合重置随机化机体平面位置、偏航与关节速度；每5–10秒施加一次平面速度扰动。')
    doc.add_paragraph('速度指令每10秒重新采样，初始平面速度范围为[−0.1,0.1] m/s，并通过课程调整至前向[−1,1] m/s、横向[−0.4,0.4] m/s；偏航角速度范围为[−1,1] rad/s。单回合时长20秒，机身非法接触或倾角超过30°时提前终止。重置仅属于独立强化学习训练回合。')
    doc.add_page_break()
    section(doc, '2.4  训练配置、导出与来源',
        '记录的训练使用1024个并行环境、随机种子42，物理步长0.005秒，每4个物理步执行一次policy。每轮采集24步，完成2000轮，共49,152,000条状态转移。')
    table(doc, [
        ('PPO更新', '每轮5个学习epoch、4个mini-batch；截断系数0.2；初始学习率0.001，使用自适应调度。'),
        ('回报与优势', '折扣因子γ=0.99；GAE参数λ=0.95；价值损失系数1.0，使用截断价值损失。'),
        ('正则与梯度', '熵系数0.01；目标KL为0.01；最大梯度范数1.0；初始动作噪声标准差1.0。'),
        ('模型保存', '每100轮保存checkpoint，导出Actor及导航加载器使用的actor_checkpoint.pt；部署权重与契约逐一校验。'),
    ])
    doc.add_paragraph('训练摘要记录完成2000轮，导出等价性误差为1.7881×10⁻⁷。交付actor_checkpoint.pt的SHA256为8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8，与逐项导航运行的policy来源记录一致。')
    doc.add_paragraph('训练源码、PPO参数、训练摘要、最终指标及原始源码哈希清单分别位于delivery/training/source/和delivery/training/records/。来源配置改编自Unitree RL Lab提交4960b84732b0c2ec593dccbfe963fda1bcd7b1e3，训练源码提交为8ab8daaaf925d4180a455343b46cd402b4b0e861。训练依赖版本与导航部署依赖分别记录。')
    section(doc, '2.5  基于policy的推箱skill封装',
        '推箱采用分层技能封装：行为决策给出箱体、推向、目标停放位置和接近路线；push_aside skill按反馈状态组织就位、转向、对齐、接触、推动、制动和退让，将阶段目标转换为机体速度指令(v_x,v_y,ω_z)。同一底层policy接收这些指令及机器人观测，生成十二关节动作，通过物理接触推动箱体。')
    doc.add_paragraph('导航与推箱复用同一份冻结权重。箱体位移和通路状态由skill用于阶段切换及动作修正，policy本身不接收箱体目标；推箱skill没有单独训练的推箱网络。交互后重新扫描、验证通路并恢复导航，具体执行参数见第3章。')
    doc.add_page_break()


def update_algorithm(directory):
    directory = Path(directory)
    path = directory/'03-算法设计文档.docx'
    doc = Document(path)
    assert not any(p.text.startswith('02  底层运动policy') for p in doc.paragraphs)
    anchor = next(p for p in doc.paragraphs if p.text == '02  实现细节')
    for p in doc.paragraphs:
        if p.text == '02  实现细节':
            p.text = '03  实现细节'
        elif p.style.name.startswith('Heading') and p.text.startswith('2.'):
            p.text = '3.' + p.text[2:]
        elif '具体阈值及执行参数见第2章' in p.text:
            p.text = p.text.replace('具体阈值及执行参数见第2章', '具体阈值及执行参数见第3章')
    chapter = Document()
    add_training(chapter)
    chapter.paragraphs[0].paragraph_format.page_break_before = True
    for el in list(chapter.element.body):
        if el.tag != qn('w:sectPr'):
            anchor._p.addprevious(el)
    doc.save(path)


if __name__ == '__main__':
    update_algorithm(Path(__file__).resolve().parent)
