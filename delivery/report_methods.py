"""Editable method design and implementation chapters for the delivery report."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def text(doc, value):
    doc.add_paragraph(value)


def section(doc, title, value):
    doc.add_heading(title, 2)
    text(doc, value)


def table(doc, rows):
    result = doc.add_table(rows=0, cols=2)
    for index, values in enumerate(rows):
        cells = result.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = value
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), 'EAF1F5' if index % 2 == 0 else 'F7F9FB')
            cell._tc.get_or_add_tcPr().append(shade)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(3)
                for run in paragraph.runs:
                    run.font.name = '微软雅黑'
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor.from_string('18354D')
                    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '微软雅黑')
        result.rows[-1]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))


def add_method_chapter(doc):
    doc.add_heading('02  方法设计', 1)
    section(doc, '2.1  任务建模与闭环框架',
        '研究对象为部分可观测环境中的四足机器人交互式导航。机器人通过在线观测建立环境表示，在满足机身通行约束的条件下到达给定目标；当可移动障碍限制通行时，允许通过物理交互改变局部障碍分布。系统采用感知、规划、控制与交互反馈相结合的分层框架，以环境可通行性作为导航和交互之间的共同决策依据。')
    section(doc, '2.2  全局—局部双层环境表征',
        '全局地图汇总已观测环境结构，为长程路线生成及连通性分析提供依据；局部地图维护机器人邻域的最新障碍状态，为短程运动检查和交互变化感知提供支持。两者采用不同更新周期，以协调全局规划的稳定性与近场响应的及时性。地图均由在线观测建立，未知空间不直接作为可通行区域；物体移动后的空闲空间须由新观测重新确认。')
    section(doc, '2.3  净空约束规划与分层运动控制',
        '在已观测环境中构建Voronoi骨架路线，通过净空加权兼顾路径长度与障碍间距。上层依据路线、机器人状态和局部环境生成速度指令，下层利用固定四足运动策略实现运动。路线跟踪与运动可行性检查构成反馈闭环，环境变化或执行受阻时触发重新规划。该方法不保证连续真实环境中的全局最短路径。')
    section(doc, '2.4  导航与交互的联合决策',
        '决策层比较保持环境不变的导航方案与移动障碍后的交互方案。对允许交互且阻挡候选通路的物体，搜索满足接近、推动和停放约束的动作，以接近距离、推移距离与后续导航距离构成启发式代价。当绕行具有较低代价时优先导航；当可行交互能够恢复通路或降低代价时执行交互。若当前观测尚不能支持可执行任务方案，则选择可达前沿补充观测。')
    section(doc, '2.5  交互反馈与适用范围',
        '物体位移不等同于通路恢复。交互后重新观测环境、检验机身通行净空并更新路线，将动作结果反馈至规划层。实验使用仿真位姿和预先给定的箱体交互授权，箱体候选须通过LiDAR可见性检查；不将该设置表述为独立SLAM或自主视觉语义识别。任务完成由目标到达与稳定停止共同判定。')
    doc.add_page_break()
    doc.add_heading('03  实现细节', 1)
    section(doc, '3.1  运行配置与地图更新',
        '以下数值对应交付启动器与其调用的规划、执行模块。更新周期以仿真时间计；初始化、决策及交互结束可额外刷新地图，因而不是严格固定次数。')
    table(doc, [
        ('全局地图', '1 Hz；维护已观测环境快照，支持目标路线与连通性分析。'),
        ('局部地图', '10 Hz；局部窗口半径参数3 m，窗口随机器人移动，支持近场运动检查。'),
        ('地图与路线参数', '在线栅格分辨率0.05 m；路线净空参数0.30 m；机身半宽参数0.22 m；Voronoi净空权重0.30。'),
        ('机器人执行', '物理步进200 Hz、policy执行50 Hz；explicit_dc执行器，使用交付的固定步态policy。'),
    ])
    section(doc, '3.2  导航与推箱的阈值判定',
        '令L_nav为保留当前全部箱体时的可行路线长度，L_ref为仅忽略允许交互箱体后的参考路线长度。未授权物体和固定障碍保留，未知空间始终阻塞。若L_ref不存在，不能据此产生推箱任务方案；若存在，则依次执行以下判定。')
    text(doc, '① 若存在导航路线且L_nav ≤ 1.6 × L_ref + 0.5 m，直接导航，不搜索推箱候选。例如L_ref=5 m时，L_nav不超过8.5 m即满足该条件。')
    text(doc, '② 否则搜索推箱候选，代价为J_push = L_approach + 2.5 × d_push + L_remaining。若L_nav ≤ 最佳J_push，仍导航；若导航不可行或其代价更高，则选择最佳可行推箱方案。没有推箱候选但仍有导航路线时，使用导航路线。')
    text(doc, '③ 上述代价是人工设计的距离等效评分，不是实测耗时或能耗。第一步提前返回意味着该实现不始终求两类行为的全局最小代价。候选若暂不能形成完整后续路线，使用15的有限惩罚替代后续路径项，允许保留继续清障的可能性；未显式求解完整多物体动作序列。')
    doc.add_page_break()
    doc.add_heading('03  实现细节（续）', 1)
    section(doc, '3.3  探索触发与前沿选择',
        '探索没有“未知面积超过某比例”的触发阈值。在线目标选择器先检查已观测地图中的忽略箱体路径，再调用导航／推箱判定；若得不到非失败的任务方案，则转入前沿选择。这既可能由观测不充分导致，也可能由当前缺少安全交互方案导致。没有安全区域或可达前沿时返回等待，不把未知区域直接作为路线。')
    text(doc, '在已知—未知区域边界附近生成候选，并筛除距离机器人不足0.45 m的点。候选先粗排，最多检查前32个，收集最多8个具有安全可达路线的方案。最终评分为：')
    text(doc, 'J_frontier = L_reach + 1.25 × D_goal − 0.8 × G + 0.30 × |Δθ|。')
    text(doc, 'L_reach为到前沿的路径长度，D_goal为前沿至目标的直线距离；G为约0.8 m邻域内的未知栅格面积，作为潜在观测收益近似；Δθ为初始转向角，单位为弧度。选择评分最低者。已经访问的候选在已观测格数增加不足30时不重复选择；失败区域可设置6秒冷却，抑制无效重复尝试。')
    section(doc, '3.4  推向、推距与停放位置搜索',
        '仅对已观测、具有交互授权且阻挡参考通路的箱体生成候选。推向枚举世界坐标系的8个单位方向：0°、45°、90°、135°、180°、225°、270°、315°。设推向u、通路方向a，二者归一化后要求|u·a| ≤ 0.4，使推向接近通路侧方。')
    text(doc, '沿每个保留方向，以0.35 m起步、0.10 m步长搜索距离。参数上限1.6 m，实际采样为0.35、0.45、…、1.55 m。候选箱体中心为p_target = p_box + d × u。每个方向只保留第一个满足约束的距离，再跨方向、跨箱体比较J_push。')
    text(doc, '检查包括：机器人接近路径可达；箱体扫掠空间不与固定障碍及其他箱体相交；机器人推动位置可行；终点满足侧方停放净空；移后参考通路不再被该箱体阻挡。在线设置下沿推移过程按0.05 m栅格步长采样。侧停规则要求箱体中心横向偏移至少覆盖箱体投影半宽、规划半径0.25 m、附加0.08 m及0.025 m余量；执行后的整机通路验证另行进行。')
    doc.add_page_break()
    doc.add_heading('03  实现细节（续）', 1)
    section(doc, '3.5  分阶段执行与反馈',
        '方案包含箱体编号、推向、目标位置、计划推距、接近点和接近路线。状态机依次完成接近就位、停稳、转向、横向／前后对齐、接触推动、制动、释放退让及新扫描验证；阶段转换由观测反馈决定，不能仅按定时动作宣布完成。')
    table(doc, [
        ('接触准备', '前方间距参数0.35 m；航向容限0.10 rad（约5.7°）；横向、前后对齐使用独立反馈。'),
        ('对齐速度请求', '横向0.35 m/s、前后0.25 m/s；通过原步态policy执行，并受运动检查限制。'),
        ('推箱速度请求', '接触起始0.60 m/s；推动阶段以0.15 m/s²增大，最高0.95 m/s；不是保证达到的实际速度。'),
        ('进度与制动', '使用箱体沿推向的实测位移，结合机器人运动估计制动距离，避免按固定时长过推。'),
        ('停滞与恢复', '单次接触超过12秒未建立会重试；推动时6秒无超过0.015 m的进展会重试；恢复受次数和时限约束。'),
    ])
    section(doc, '3.6  交互完成与恢复导航',
        '推动阶段结合通路证书或侧停几何条件判断是否进入制动，并要求沿推向位移接近计划推距（容许0.035 m差值）。制动后重新核对箱体位置；释放退让后获取晚于最后箱体运动的新扫描，重新检查整机通路及后续路线。仅在验证通过后恢复通路导航；观测或空间不足时继续观测、调整或恢复。箱体通过真实接触动力学移动，程序不直接设置物体位置。')
    text(doc, '任务级验收要求到达目标、误差≤0.5 m并满足实测停稳、启动／运行倾角、健康及在线输入检查。历史交互证据还审计接触、箱体位移、新扫描与实际整机穿越；新机器检查脚本的范围以其检查项为准。耗时不设验收门槛。')
    section(doc, '3.7  实现定位与验证边界',
        '配置入口：reproduction/run_task.py。决策：online_navigation_decision.py、box_push_navigation.py；侧停约束：side_push_navigation.py；地图：online_navigation_map.py、layered_navigation_map.py；规划：go2_voronoi_navigation.py；执行：push_aside_skill.py、go2_contact_control.py；控制：isaac_go2_ctrl.py。运行编排为go2_side_clearance_navigation.py。')
    text(doc, '上述阈值为当前交付实现的启发式选择，未通过参数消融证明最优性。本次修改仅完善方法与实现说明，不新增物理试验、不改变算法；原测试结果及新机器复现验证范围仍按对应证据记录。')
    doc.add_page_break()


def update_existing_report(path):
    doc = Document(path)
    start = next(p for p in doc.paragraphs if p.text == '02  方法介绍')
    anchor = next(p for p in doc.paragraphs if p.text.startswith('03  stage1-'))
    node = start._p
    while node is not anchor._p:
        following = node.getnext()
        node.getparent().remove(node)
        node = following
    for paragraph in doc.paragraphs:
        if paragraph.style.name == 'Heading 1':
            number, separator, rest = paragraph.text.partition('  ')
            if separator and number.isdigit() and int(number) >= 3:
                paragraph.text = f'{int(number)+1:02d}  {rest}'
    chapter = Document()
    add_method_chapter(chapter)
    for element in list(chapter.element.body):
        if element.tag != qn('w:sectPr'):
            anchor._p.addprevious(element)
    doc.save(path)


if __name__ == '__main__':
    from pathlib import Path
    update_existing_report(Path(__file__).resolve().parent / '02-测试报告与可视化材料.docx')
