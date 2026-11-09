"""Detailed method design, including the implemented visual interaction branch."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn


def paragraph(doc, value):
    doc.add_paragraph(value)


def heading(doc, value):
    doc.add_heading(value, 2)


def add_design(doc):
    doc.add_heading('02  方法设计', 1)
    heading(doc, '2.1  任务表述与系统框架')
    paragraph(doc, '将任务表述为部分可观测环境中的四足机器人交互式导航。给定机器人初始状态与目标位置，系统通过连续观测更新环境表示，并在机身通行约束下生成运动。当可移动物体阻挡路线时，系统将物体位姿变化纳入动作结果，通过物理交互改变局部障碍分布，再利用更新后的环境状态继续导航。')
    paragraph(doc, '系统由环境感知、双层建图、语义交互授权、行为决策、路线规划及运动执行组成。感知与建图维护当前可用的几何信息；语义模块确定对象对应的可调用技能；决策层选择导航、探索、观察或交互；控制层执行动作并向上层反馈机器人及物体状态。各模块通过共享对象标识和地图状态构成闭环。')
    heading(doc, '2.2  全局—局部双层环境表征')
    paragraph(doc, '采用占据栅格表示已观测空闲、障碍和未知区域。LiDAR测距及仿真提供的机器人位姿用于将传感观测投影到统一坐标系，并依据射线经过区域与障碍返回更新占据信息。结构占据与可移动对象占据分别维护；未知区域在路径与运动检查中保持阻塞。')
    paragraph(doc, '全局地图保存已观测环境的空间结构与对象快照，为目标路线生成、绕行比较及通路连通性分析提供依据；局部地图随机器人移动，维护邻域障碍的最新状态，为局部避障、接触接近、制动和退让提供依据。全局与局部地图分别以1 Hz和10 Hz更新，交互或决策事件可额外触发刷新。物体移动后，原占据区域由新扫描重新确认其空闲状态。')
    heading(doc, '2.3  净空约束路线规划与分层控制')
    paragraph(doc, '在已观测可通行区域构建Voronoi骨架图，以路径长度与障碍净空共同确定路线代价。全局路线提供面向目标的行进方向，局部运动检查结合机器人朝向和完整机身范围，检验前进、转向、制动及恢复动作对应的空间。地图发生变化或执行受阻时，系统刷新路线及局部运动目标。')
    paragraph(doc, '运动执行采用分层结构：上层路径跟踪器依据位置误差、航向误差与路线进度生成机体速度指令；下层固定步态policy根据机器人状态输出关节运动目标。接近交互对象和目标停稳采用分阶段反馈，使路线跟踪、接触操作与停止控制使用统一的机器人状态。')
    doc.add_page_break()
    heading(doc, '2.4  视觉语言模型驱动的交互对象识别')
    paragraph(doc, '视觉输入分支将“对象是什么”与“对象位于何处”分别处理。首先，GroundingDINO根据类别文本查询在RGB图像中生成候选框；随后，MobileSAM提取候选实例掩膜，并利用同步深度及世界坐标点图，将实例像素对应的三维点与已观测几何候选关联，建立图像实例和规划对象之间的标识对应关系。')
    paragraph(doc, '对关联后的实例构造局部图像：保留目标前景，弱化周围背景，并保留一定上下文。将该图像与候选类别集合输入Qwen3-VL，由模型输出对象类别及模型来源记录。识别记录包含对象标识、类别、帧标识、观测时刻、检测分数和图像证据；系统依据不同图像帧中的一致类别结果形成对象语义状态。')
    paragraph(doc, '语义类别通过交互技能目录映射为可调用动作。例如，箱体类别对应侧推技能；未确认类别或目录中未配置相应技能的对象继续作为障碍处理。当影响通路的对象尚未完成语义确认时，观察规划模块生成可达观察位置和面向对象的视角，机器人移动并停稳后获取新图像，再更新类别及交互授权。视觉语言模型负责类别判别，推向、推距和碰撞约束由几何规划模块计算。')
    heading(doc, '2.5  导航、探索与交互的联合决策')
    paragraph(doc, '行为选择首先检查当前已观测地图能否支持目标任务方案。保持物体不动时，生成普通导航路线；对具有交互授权的对象，进一步构造移开对象后的参考通路。依据绕行阈值与动作代价比较，选择直接导航或推箱。若当前无法形成可执行任务方案，则在已知—未知区域边界选择可达前沿，并综合接近距离、目标方向、局部未知面积和初始转向确定探索目标。')
    paragraph(doc, '交互规划针对阻挡候选通路的对象枚举侧推方向与停放位置，检验接近可达性、箱体扫掠空间、机器人推动空间和停放后的通路净空。以接近路径、加权推距及后续导航路径构成交互代价，选择对应对象与动作参数。执行层根据实测相对位置和物体运动依次完成就位、对齐、接触、推动、制动及退让。')
    heading(doc, '2.6  交互后验证与实验输入配置')
    paragraph(doc, '完成推动后，系统获取晚于物体最后运动时刻的新扫描，更新双层地图并重新生成目标路线。通路验证结合完整机身范围及当前局部障碍，确认机器人可通过后切回导航，并记录实际穿越。目标处采用距离和实测停稳联合判据结束任务。')
    paragraph(doc, '同一实现支持视觉识别授权和布局授权两种输入方式。本次T01–T10验收使用布局授权，视觉识别流程作为已实现的方法组成在本章说明。两种方式共享导航与推箱执行模块；视觉分支中的对象几何关联仍使用经LiDAR可见性筛选的仿真候选位置和尺寸，类别由图像识别链路产生。具体阈值及执行参数见第3章。')
    doc.add_page_break()


def update_design(path):
    doc = Document(path)
    start = next(p for p in doc.paragraphs if p.text == '02  方法设计')
    anchor = next(p for p in doc.paragraphs if p.text == '03  实现细节')
    node = start._p
    while node is not anchor._p:
        following = node.getnext()
        node.getparent().remove(node)
        node = following
    chapter = Document()
    add_design(chapter)
    chapter.paragraphs[0].paragraph_format.page_break_before = True
    for element in list(chapter.element.body):
        if element.tag != qn('w:sectPr'):
            anchor._p.addprevious(element)
    doc.save(path)


if __name__ == '__main__':
    update_design(Path(__file__).resolve().parent/'02-测试报告与可视化材料.docx')
