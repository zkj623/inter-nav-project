"""Move the method chapters from the test report into a standalone document."""
from copy import deepcopy
from pathlib import Path
import re
from docx import Document
from docx.oxml.ns import qn


def split_documents(directory):
    directory = Path(directory)
    report_path = directory/'02-测试报告与可视化材料.docx'
    guide_path = directory/'01-交付说明与测试用例.docx'
    report = Document(report_path)
    start = next(p for p in report.paragraphs if p.text == '02  方法设计')
    end = next(p for p in report.paragraphs if p.text.startswith('04  stage1-'))
    algorithm = Document(guide_path)
    # Keep only the common cover from the guide, which has no screenshot media.
    first_table = algorithm.tables[0]._tbl
    node = first_table
    while node is not None and node.tag != qn('w:sectPr'):
        following = node.getnext()
        node.getparent().remove(node)
        node = following
    for p in algorithm.paragraphs:
        if p.style.name == 'Title':
            p.text = '算法设计文档'
        elif p.text.startswith('源码交接、'):
            p.text = '算法框架、方法设计与实现细节'
        elif p.text.startswith('项目交付文件'):
            p.text = '项目交付文件  /  GO2-ALG-003'
        elif p.text.startswith('编制日期：'):
            p.text = '编制日期：2026年11月9日'
    for section in algorithm.sections:
        for p in section.footer.paragraphs:
            for run in p.runs:
                run.text = run.text.replace('GO2-DLV-001', 'GO2-ALG-003')
    section_properties = algorithm.element.body.find(qn('w:sectPr'))
    node = start._p
    moved = []
    while node is not end._p:
        following = node.getnext()
        moved.append(deepcopy(node))
        section_properties.addprevious(deepcopy(node))
        node.getparent().remove(node)
        node = following
    for p in algorithm.paragraphs:
        if p.style.name == 'Heading 1':
            p.text = p.text.replace('02  方法设计', '01  方法设计').replace('03  实现细节', '02  实现细节')
        elif p.style.name.startswith('Heading'):
            p.text = re.sub(r'^([23])\.', lambda m: str(int(m[1])-1)+'.', p.text)
        if '第3章' in p.text:
            p.text = p.text.replace('第3章', '第2章')
    for p in report.paragraphs:
        if p.style.name == 'Heading 1':
            number, separator, rest = p.text.partition('  ')
            if separator and number.isdigit() and int(number) >= 4:
                p.text = f'{int(number)-2:02d}  {rest}'
    reference = report.add_paragraph('算法框架、方法设计及实现细节见配套《算法设计文档》。本报告记录测试范围、验收结果、路径对比及可视化材料。')
    end._p.addprevious(reference._p)
    report.save(report_path)
    algorithm.save(directory/'03-算法设计文档.docx')
    guide = Document(guide_path)
    for table in guide.tables:
        for row in table.rows:
            if row.cells[0].text == '配套文件':
                row.cells[1].text = '《算法设计文档》《测试报告与可视化材料》、源码、模型、场景输入、测试证据'
    guide.save(guide_path)


if __name__ == '__main__':
    split_documents(Path(__file__).resolve().parent)
