"""Build the report's method chapter using editable Word paragraphs and tables."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def add_method_chapter(doc):
    doc.add_heading('02  方法介绍', 1)
    doc.add_paragraph('本阶段采用“在线感知建图—净空约束规划—反馈运动控制—必要时推箱—重新观测与规划”的闭环方法。T01–T05验证普通导航，T06–T10验证推箱后通行；任务均在同一MV7仿真场景中执行。')
    doc.add_heading('2.1  总体流程', 2)
    flow = [
        ('① 输入与初始化', '加载起终点、Go2机器人和固定policy；地图从空状态启动。'),
        ('↓ ② 在线感知与建图', '采集仿真LiDAR及机器人位姿，更新占据与可通行信息。'),
        ('↓ ③ 规划与跟踪', '在已观测区域生成净空加权Voronoi路线；持续检查前方运动与制动空间。'),
        ('↓ ④ 受阻时交互', '对已授权且满足可见性条件的箱体就位、对齐、接触和侧推；无须交互时继续导航。'),
        ('↓ ⑤ 重新扫描与通行', '箱体移动后退让、获取新扫描并确认整机通路；未确认则继续观测或恢复。'),
        ('↓ ⑥ 目标停稳与审计', '沿更新路线到达目标，实测停稳；保存轨迹、接触、箱体位移和验收结果。'),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.autofit = True
    for index, (label, text) in enumerate(flow):
        cells = table.add_row().cells
        for cell, value in zip(cells, (label, text)):
            cell.text = value
            shade = OxmlElement('w:shd')
            shade.set(qn('w:fill'), 'EAF1F5' if index % 2 == 0 else 'F7F9FB')
            cell._tc.get_or_add_tcPr().append(shade)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(5)
                for run in paragraph.runs:
                    run.font.name = '微软雅黑'
                    run.font.size = Pt(10)
                    run.font.color.rgb = RGBColor.from_string('18354D')
                    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '微软雅黑')
        table.rows[-1]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    doc.add_paragraph('图 2-1  导航与推箱交互闭环。重新观测后的结果反馈给规划器；推箱完成不直接等同于通路已经可安全穿越。', 'Caption')
    doc.add_heading('2.2  在线建图与位姿来源', 2)
    doc.add_paragraph('使用仿真LiDAR测距构建在线占据表示，区分已观测空闲、障碍和未知区域；未知区域不能直接作为已确认通路。全局地图按1Hz、局部地图按10Hz更新。位姿来自仿真状态，本报告不将其表述为已实现独立定位或完整SLAM。离线资格几何和参考路径仅供评估，不输入在线导航器。')
    doc.add_page_break()
    doc.add_heading('02  方法介绍（续）', 1)
    doc.add_heading('2.3  路径规划与运动控制', 2)
    doc.add_paragraph('规划使用已观测占据地图上的Voronoi骨架路线，并以净空权重0.30增加狭窄路段的代价。连续路径跟踪结合机器人朝向、路线进度、局部障碍和制动条件给出速度指令；地图或执行条件变化时刷新路线。路线选择兼顾距离与净空，不宣称获得真实场景的全局最短路径。')
    doc.add_paragraph('Go2使用交付的固定步态policy，采用IsaacLab训练模型对应的物理配置及explicit_dc执行器模式；物理步进200Hz、policy执行50Hz。上层负责速度、转向、就位、退让和停稳决策，下层policy驱动腿部运动。本次交付未重新训练policy。')
    doc.add_heading('2.4  交互资格与侧推箱体', 2)
    doc.add_paragraph('本次验收使用ground_truth交互来源：布局文件提供箱体类别与允许技能的授权名单，再结合LiDAR可见性和运行条件决定是否交互。它不用于初始化全局地图，也不等同于由RGB图像自主识别箱体；视觉识别模式不属于本次已验证方法。')
    doc.add_paragraph('交互按就位、航向修正、接触、侧推、释放和退让等阶段执行，结合机器人状态与箱体实际运动反馈。推箱后不能只依据预期位移宣布成功：需要真实接触记录、足够的实际箱体位移，以及晚于最后箱体运动的新扫描。随后结合机器人完整机身净空确认通路，再实际穿越；不足时重新观测或进入受限恢复。')
    doc.add_heading('2.5  目标停稳与验证依据', 2)
    doc.add_paragraph('到达目标附近后执行制动和实测停稳检查，而非仅判断距离。任务完成标准为目标误差≤0.5m并通过停稳、倾角、健康和在线输入检查；历史交互验收还审计接触、新扫描及整机穿越。耗时仅作记录，不设验收门槛。固定十项成功不代表任意场景或重复试验的统计成功率。')
    doc.add_paragraph('方法对应源码：go2_side_clearance_navigation.py（编排与参数）；mapping.py、layered_navigation_map.py（建图）；go2_voronoi_navigation.py（规划）；push_aside_skill.py、go2_passage_confirmation.py（交互与通路）；isaac_go2_ctrl.py（控制）。模块位于grutopia/demo/、grutopia_extension/interactive_navigation/及controllers/，可结合reproduction/run_task.py核对冻结配置。')
    doc.add_page_break()


def update_existing_report(path):
    doc = Document(path)
    if any(p.text == '02  方法介绍' for p in doc.paragraphs):
        raise ValueError('Method chapter already exists')
    anchor = next(p for p in doc.paragraphs if p.text.startswith('02  '))
    for paragraph in doc.paragraphs:
        if paragraph.style.name == 'Heading 1':
            number, separator, rest = paragraph.text.partition('  ')
            if separator and number.isdigit() and int(number) >= 2:
                paragraph.text = f'{int(number)+1:02d}  {rest}'
        if paragraph.text.startswith('版本 V3.0'):
            paragraph.text = '更新日期：2026年11月6日'
    chapter = Document()
    add_method_chapter(chapter)
    for element in list(chapter.element.body):
        if element.tag != qn('w:sectPr'):
            anchor._p.addprevious(element)
    doc.save(path)


if __name__ == '__main__':
    from pathlib import Path
    update_existing_report(Path(__file__).resolve().parent / '02-测试报告与可视化材料.docx')
