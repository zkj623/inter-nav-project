"""Use explicit geometric collision-constraint terminology in delivery prose."""
from pathlib import Path
from docx import Document

TITLE = '2.3  考虑机器人几何约束的路径规划与分层控制'
DESIGN = ('在已观测可通行区域构建Voronoi骨架图，以路径长度与障碍间距共同确定路线代价。'
          '路径规划与运动执行同时考虑机器人几何尺寸及安全裕度，通过检查机身沿候选轨迹的扫掠区域'
          '与障碍物之间的空间关系，筛选满足无碰撞约束的运动方案。全局路线提供面向目标的行进方向，'
          '局部运动检查结合机器人位置、朝向和完整机身范围，检验前进、转向、制动及恢复动作。'
          '地图发生变化或执行受阻时，系统刷新路线及局部运动目标。')
REPLACEMENTS = [
    ('净空约束路线规划与分层控制', '考虑机器人几何约束的路径规划与分层控制'),
    ('净空约束规划与分层运动控制', '考虑机器人几何约束的路径规划与分层运动控制'),
    ('路线净空参数', '路线规划半径参数'),
    ('Voronoi净空权重', 'Voronoi障碍间距代价权重'),
    ('侧方停放净空', '侧方停放所需的通行空间'),
    ('通路净空', '通路的机身通过空间'),
    ('机身通行净空', '机身通行空间'),
    ('整机净空', '整机通行空间'),
    ('机器人净空栅格可行路径', '考虑机器人尺寸的栅格可行路径'),
    ('机器人净空路径', '满足机器人尺寸约束的可行路径'),
    ('障碍净空', '障碍间距'),
    ('净空约束', '机器人几何尺寸与安全裕度约束'),
    ('净空', '通行空间'),
]


def revise_terminology(path):
    doc = Document(path)
    paragraphs = list(doc.paragraphs)
    paragraphs += [p for table in doc.tables for row in table.rows for cell in row.cells for p in cell.paragraphs]
    for paragraph in paragraphs:
        old = paragraph.text
        if old.startswith('在已观测可通行区域构建Voronoi骨架图'):
            new = DESIGN
        else:
            new = old
            for first, second in REPLACEMENTS:
                new = new.replace(first, second)
        if old != new:
            paragraph.text = new
    doc.save(path)


if __name__ == '__main__':
    for path in Path(__file__).resolve().parent.glob('*.docx'):
        revise_terminology(path)
