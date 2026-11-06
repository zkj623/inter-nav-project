"""Populate the acceptance table from recorded trajectories and qualification data."""
import json
from pathlib import Path
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

NOTE = ('注：实际路径长度取各任务result.json中的actual_path_length_m，按机器人采样轨迹在XY平面的相邻位移累计，'
        '包含探索、接近、推箱及退让等运动。最短几何路径长度取task-paths-and-complexity.json中的'
        'shortest_geometry.length_m，为初始场景未膨胀几何的离线参考路径；该参考值不输入在线导航器。')


def add_path_comparison(doc, root):
    root = Path(root)
    qualification = {x['id']: x for x in json.loads((root/'task-paths-and-complexity.json').read_text(encoding='utf-8'))}
    table = doc.add_table(rows=1, cols=6)
    for cell, value in zip(table.rows[0].cells, ['任务', '类型', '目标误差/cm', '实际路径/m', '最短几何路径/m', '验收']):
        cell.text = value
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    for task in sorted(qualification):
        result = json.loads((root/'evidence'/task/'result.json').read_text(encoding='utf-8'))
        actual = float(result['actual_path_length_m'])
        shortest = float(qualification[task]['shortest_geometry']['length_m'])
        assert actual > 0 and shortest > 0 and result['success']
        values = [task, '导航' if task <= 'T05' else '推箱交互', f"{result['goal_error_m']*100:.2f}", f'{actual:.4f}', f'{shortest:.4f}', '通过']
        for cell, value in zip(table.add_row().cells, values):
            cell.text = value
    # Set useful column proportions before the shared layout normalizes width.
    from docx.shared import Cm
    for row in table.rows:
        for cell, width in zip(row.cells, [1.3, 2.1, 2.6, 3.1, 4.0, 1.3]):
            cell.width = Cm(width)
    return table


def update_report(path):
    path = Path(path)
    doc = Document(path)
    old = next(t for t in doc.tables if any('到目标制动' in c.text for c in t.rows[0].cells))
    new = add_path_comparison(doc, path.parent)
    old._tbl.addprevious(new._tbl)
    old._tbl.getparent().remove(old._tbl)
    for p in list(doc.paragraphs):
        if p.text.startswith('上述时刻为仿真时间') or '比历史基线慢' in p.text:
            p._p.getparent().remove(p._p)
    paragraph = doc.add_paragraph(NOTE)
    new._tbl.addnext(paragraph._p)
    doc.save(path)


if __name__ == '__main__':
    update_report(Path(__file__).resolve().parent/'02-测试报告与可视化材料.docx')
