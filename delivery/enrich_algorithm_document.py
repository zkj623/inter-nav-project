"""Add research context, design interfaces, original diagrams and references."""
from pathlib import Path
import json
import re
from docx import Document
from docx.shared import Cm
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from report_methods import table

ROOT = Path(__file__).resolve().parent
REFERENCES = [
    ('Yamauchi B.', 'A Frontier-Based Approach for Autonomous Exploration', 'IEEE CIRA, 1997', 'https://www.robotfrontier.com/papers/cira97.pdf'),
    ('Stilman M., Kuffner J. J.', 'Navigation Among Movable Obstacles: Real-Time Reasoning in Complex Environments', 'International Journal of Humanoid Robotics, 2005', 'https://publications.ri.cmu.edu/navigation-among-movable-obstacles-real-time-reasoning-in-complex-environments-2'),
    ('Schulman J., et al.', 'Proximal Policy Optimization Algorithms', 'arXiv:1707.06347, 2017', 'https://arxiv.org/abs/1707.06347'),
    ('Rudin N., Hoeller D., Reist P., Hutter M.', 'Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning', 'Conference on Robot Learning, PMLR 164, 2022', 'https://proceedings.mlr.press/v164/rudin22a.html'),
    ('Ahn M., et al.', 'Do As I Can, Not As I Say: Grounding Language in Robotic Affordances', 'arXiv:2204.01691, 2022', 'https://arxiv.org/abs/2204.01691'),
    ('Liu S., et al.', 'Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection', 'arXiv:2303.05499, 2023', 'https://arxiv.org/abs/2303.05499'),
    ('Kirillov A., et al.', 'Segment Anything', 'arXiv:2304.02643, 2023', 'https://arxiv.org/abs/2304.02643'),
    ('Zhang C., et al.', 'Faster Segment Anything: Towards Lightweight SAM for Mobile Applications', 'arXiv:2306.14289, 2023', 'https://arxiv.org/abs/2306.14289'),
    ('Bai S., et al.', 'Qwen3-VL Technical Report', 'arXiv:2511.21631, 2025', 'https://arxiv.org/abs/2511.21631'),
]


def chapter(doc, value):
    p = doc.add_heading(value, 1)
    p.paragraph_format.page_break_before = True


def section(doc, value, content):
    doc.add_heading(value, 2)
    doc.add_paragraph(content)


def background(doc):
    chapter(doc, '01  研究背景与设计目标')
    section(doc, '1.1  应用场景与任务需求',
        '室内移动任务同时涉及空间探索、目标导航和障碍物处理。环境布局尚未完整观测时，机器人需要根据连续传感输入建立可通行区域；通道被可移动物体阻挡时，任务过程还包含物体交互及其后的环境重新确认。交互式导航将机器人运动与环境状态变化共同纳入任务过程，相关问题可与可移动障碍导航（NAMO）的研究表述相联系[2]。')
    doc.add_paragraph('本项目以Go2四足机器人为执行平台，在MV7室内仿真环境中完成给定起终点任务。普通导航任务依赖在线建图、路线生成及稳定运动；交互任务还需确定对象是否具有可调用技能、如何接近和侧推，以及何时恢复通行。底层速度跟踪policy提供运动接口，上层以几何分析、语义授权和阶段反馈组织任务。')
    section(doc, '1.2  状态、观测与行为表述',
        '用q_t表示机器人位姿与运动状态，B_t表示已观测物体及其几何属性，M_t^g和M_t^l分别表示全局与局部地图，C_t表示对象类别和交互授权。系统根据当前观测更新这些状态，并从导航、探索、主动观察和推箱四类行为中选择下一步动作。导航改变机器人位姿，探索和观察补充环境信息，推箱同时改变机器人及物体状态。')
    doc.add_paragraph('候选运动需满足机器人几何尺寸与安全裕度约束；对象操作需满足类别授权、接近可达性、推动扫掠空间及停放后的通行条件。任务终止时同时检查目标位置误差与实际停稳状态。动作执行产生的新观测继续更新地图和决策输入，形成任务级闭环。')
    section(doc, '1.3  设计目标与功能分解',
        '系统接口围绕“可观测、可规划、可执行、可确认”组织：观测模块提供几何和语义证据，规划模块给出路线与交互参数，控制模块执行机体运动，验证模块依据实测状态决定阶段切换。图6-1至图6-4给出模块关系、决策条件、policy与skill接口以及交互几何示意。')
    table(doc, [
        ('环境表示', '区分已观测空闲、障碍和未知区域，维护全局结构与局部更新。'),
        ('行为选择', '结合目标连通性、可达前沿、语义授权和导航／交互代价生成行为。'),
        ('分层执行', '导航跟踪器与推箱skill共享机体速度接口及固定底层policy。'),
        ('结果确认', '以新扫描、物体运动和机器人状态驱动重规划、通路确认与目标停稳。'),
    ])


def related(doc):
    chapter(doc, '02  相关技术进展与设计关联')
    section(doc, '2.1  未知环境探索与环境重构导航',
        '前沿探索将已知空闲区域与未探索区域的边界作为观测目标。Yamauchi的工作给出了在证据栅格中识别前沿并导航至前沿的探索方式[1]。本项目在已观测地图中筛选可达前沿，并将到达距离、目标方向、邻域未知面积和初始转向纳入候选评分；前沿目标用于补充观测，原始任务目标保持不变。')
    doc.add_paragraph('Stilman与Kuffner将移动障碍、清理空间和机器人路线联系起来，形成NAMO问题的规划框架[2]。本项目围绕侧推箱体构造具体操作：根据参考通路选择对象，枚举推向和停放位置，检查接近及运动过程，再重新确认通道。物体几何变化由物理执行产生，动作结果重新进入地图更新和后续规划。')
    section(doc, '2.2  强化学习运动策略与分层技能',
        'PPO以与环境交互采样和多轮小批量策略更新组成训练循环，并通过截断代理目标组织策略优化[3]。Rudin等将大规模并行仿真和课程机制用于四足机器人运动策略训练[4]。相关研究为“速度跟踪训练—策略导出—上层指令调用”提供了技术背景；本项目的实际并行环境数量、训练轮数、观测和奖励参数均依据交付训练记录，在第4章给出。')
    doc.add_paragraph('分层技能将任务目标与底层动作接口联系起来。SayCan研究将语言层面的任务关联与技能可执行性结合，用于选择机器人技能[5]。本项目的语义模块生成对象类别和技能授权，几何决策模块计算导航与交互候选，push_aside skill将具体操作阶段转换为速度指令，再交由同一运动policy执行。这里的行为排序由已实现的几何代价计算完成。')
    doc.add_page_break()
    section(doc, '2.3  开放词汇感知与视觉语言模型',
        'GroundingDINO将文本输入引入开放集目标检测，以类别名称或指代表达生成图像候选[6]；SAM采用可提示分割方式生成对象掩膜[7]，MobileSAM通过轻量化图像编码器构建相应的实例分割流程[8]。本项目视觉分支将候选框、实例掩膜、同步深度和三维对象关联依次衔接，以建立图像实例与规划对象的对应关系。')
    doc.add_paragraph('Qwen3-VL技术报告描述了图文联合理解及多模态推理模型[9]。在本项目中，模型接收对象局部图像及候选类别集合，输出类别和模型来源记录；识别存储模块依据不同帧的结果形成类别确认，技能目录再将确认类别映射为允许调用的动作。类别推断、几何参数搜索及物理执行分别由相应模块处理。')
    section(doc, '2.4  相关技术在本项目中的组织方式',
        '下表将文献方向与本项目接口对应。文献用于说明技术背景，项目使用方式以现有源码、冻结配置及训练／测试记录为依据。视觉输入与布局输入共享导航和执行模块；本次T01–T10验收使用布局授权，视觉输入分支的处理过程在第3章及图6-1中说明。')
    table(doc, [
        ('探索与可移动障碍规划[1–2]', '前沿候选生成、任务连通性分析、箱体侧推位置搜索和交互后重规划。'),
        ('PPO与并行四足训练[3–4]', '速度跟踪Actor–Critic训练、观测／动作契约及确定性Actor部署。'),
        ('语义与技能衔接[5]', '以类别授权控制技能调用，并与几何可行性及动作代价联合使用。'),
        ('检测、分割与视觉语言理解[6–9]', 'RGB候选、实例分割、深度关联、类别判别和跨帧确认。'),
    ])
    doc.add_paragraph('本项目的输入配置还包含仿真提供的机器人状态及经LiDAR可见性筛选的对象几何。相关信息在地图、接触几何和控制反馈中的用途分别保留，使类别确认、对象位置和动作结果在设计说明中具有明确来源。')


def picture(doc, filename, caption):
    p = doc.add_paragraph()
    p.add_run().add_picture(str(ROOT/'algorithm-figures'/filename), width=Cm(15.5))
    doc.add_paragraph(caption, 'Caption')


def diagrams(doc):
    chapter(doc, '06  系统图解与模块接口设计')
    section(doc, '6.1  系统架构与状态反馈',
        '系统将几何状态与语义授权作为决策的两类输入。全局和局部地图保存可通行信息，交互目录提供对象可调用的技能；行为层选择任务目标、前沿、观察位置或推箱方案。路线跟踪和推箱skill分别形成速度请求，再通过同一底层policy执行。')
    picture(doc, '01-system-architecture.png', '图6-1  本项目系统架构及反馈关系（自行绘制）。RGB-D语义链路为视觉输入配置；本次验收采用布局授权。')
    table(doc, [
        ('地图→决策', '已观测范围、结构占据、可移动对象快照及局部障碍信息。'),
        ('语义→决策', '对象ID、确认类别、允许技能、图像证据或布局授权来源。'),
        ('执行→地图与决策', '机器人实测状态、箱体运动、新扫描、阶段状态和通路检查结果。'),
    ])
    doc.add_page_break()
    section(doc, '6.2  导航与交互行为选择',
        '图6-2对应已实现的顺序判定：先检查已观测区域中的任务参考通路，再判断现有路线是否满足绕行阈值；需要比较交互时生成推箱候选，并结合可行性和代价选择行为。不能形成可执行任务方案时转入前沿选择，没有可达前沿时等待观测。')
    picture(doc, '02-behavior-selection.png', '图6-2  导航、探索与推箱的判定流程。L_nav与L_ref的定义、候选惩罚项及完整公式见第5章。')
    doc.add_paragraph('同一轮决策产生明确的行为类型和目标参数。地图和机器人状态更新后，目标选择器重新检查任务通路或前沿候选；推箱执行期间由skill状态机负责接触操作。只有阶段完成并通过重新观测，行为层才恢复目标通路导航。')
    doc.add_page_break()
    section(doc, '6.3  policy训练、部署与skill接口',
        '底层训练与任务执行采用不同的数据流。训练时由并行环境采集状态转移，PPO联合更新Actor和Critic；部署时导出并冻结Actor，导航与推箱通过统一的速度指令接口调用该网络。Critic观测用于训练价值估计，部署Actor输入仍按45维契约组织。')
    picture(doc, '03-policy-and-skill.png', '图6-3  并行训练、Actor导出及导航／推箱复用关系。配置数值取自本项目训练记录。')
    doc.add_paragraph('对时刻t，skill输出c_t=(v_x,v_y,ω_z)，机器人状态与上一动作组成o_t；Actor生成a_t=π_θ(o_t)，关节目标为q_des=q_default+0.25a_t。PD及电机模型执行关节动作，skill依据实际箱体位移、机器人运动及接触几何推进阶段。物体目标由skill处理，底层网络维持速度跟踪接口。')
    doc.add_page_break()
    section(doc, '6.4  交互几何与机身运动约束',
        '侧推候选需要同时满足箱体停放和机器人运动条件。候选位置须让出预期通路，并检查机器人接近、推动及退让时的占据区域。图6-4使用通道与侧方停放区域说明几何关系；实际规划使用在线地图和已观测对象几何。')
    picture(doc, '04-interaction-geometry.png', '图6-4  箱体侧推与机身扫掠空间示意（自行绘制，非测试场景地图或运行结果）。')
    doc.add_paragraph('以B(q)表示机器人位姿q对应的机身区域，O表示当前检查使用的障碍区域，δ为相应安全裕度；运动检查沿候选动作采样机身位置和朝向，检查B(q)及其安全扩展是否与障碍相交。箱体扫掠区域由其几何范围沿推向移动得到，停放约束再检查侧方空间与后续通路。')
    doc.add_paragraph('箱体移动后的原占据区域先恢复为待确认状态，再由晚于最后物体运动的新扫描更新。最终通路确认结合当前路线和完整机身通过空间，并记录实际穿越。静态几何参考路线仅用于离线资格评估，其长度与实际运动轨迹分别记录在测试报告中。')


def bibliography(doc):
    chapter(doc, '07  参考文献')
    if 'Reference Entry' not in doc.styles:
        doc.styles.add_style('Reference Entry', WD_STYLE_TYPE.PARAGRAPH)
    for number, (authors, title, venue, url) in enumerate(REFERENCES, 1):
        doc.add_paragraph(f'[{number}] {authors} {title}. {venue}.\n{url}', 'Reference Entry')
    doc.add_paragraph('以上为原始论文或机构／作者发布页面，查阅日期为2026年11月10日。文中图6-1至图6-4依据项目实现自行绘制；训练参数与验收数据分别以delivery/training/和delivery/evidence/中的记录为准。')


def enrich_algorithm(directory):
    directory = Path(directory)
    path = directory/'03-算法设计文档.docx'
    doc = Document(path)
    assert not any(p.text == '01  研究背景与设计目标' for p in doc.paragraphs), 'Already enriched'
    anchor = next(p for p in doc.paragraphs if p.text == '01  方法设计')
    for p in doc.paragraphs:
        if p.style.name == 'Heading 1':
            match = re.match(r'^(\d+)  (.+)$', p.text)
            if match:
                p.text = f'{int(match[1])+2:02d}  {match[2]}'
        elif p.style.name.startswith('Heading'):
            p.text = re.sub(r'^(\d+)\.', lambda m: str(int(m[1])+2)+'.', p.text)
        if '第3章' in p.text:
            p.text = p.text.replace('第3章', '第5章')
        if p.text.startswith('编制日期：'):
            p.text = '编制日期：2026年11月10日'
    # Append in the same document so image relationship IDs remain valid.
    old_elements = set(doc.element.body)
    background(doc)
    related(doc)
    for element in list(doc.element.body):
        if element not in old_elements and element.tag != qn('w:sectPr'):
            anchor._p.addprevious(element)
    diagrams(doc)
    bibliography(doc)
    doc.save(path)
    payload = [{'id': i, 'authors': a, 'title': t, 'publication': v, 'url': u, 'accessed': '2026-11-10'} for i, (a,t,v,u) in enumerate(REFERENCES,1)]
    (directory/'algorithm-references.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    enrich_algorithm(ROOT)
