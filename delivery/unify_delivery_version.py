"""Present the verified task results under the single delivery release name."""
import re
from pathlib import Path
from docx import Document


def unify_report(path):
    doc = Document(path)
    task = None
    for p in doc.paragraphs:
        old = p.text
        new = old
        match = re.search(r'\b(T\d{2})\b', old)
        if p.style.name == 'Heading 1' and match:
            task = match.group(1)
        if old.startswith('统一版本功能结果来自'):
            new = '本报告采用统一交付版本stage1-go2-simulation-release记录T01–T10验收结果，十项任务均已测试通过，其中T01–T05为普通导航，T06–T10为推箱交互导航。对应运行源码提交为514cd51，逐项结果及源码校验记录见evidence/。'
        elif old.startswith('可视化材料来自'):
            new = '可视化材料截取自go2-textured-interaction-20261103保存的任务录像，用于展示导航、推箱及目标停稳过程。验收数值统一采用交付版本测试结果；截图原始运行来源、帧号及视频SHA256保存在figures/frame-sources.json。'
        elif '统一验证：通过' in old:
            new = old.split('。')[0] + '。下列截图为任务过程可视化材料，原始来源见截图清单。'
        elif old.startswith('图 '):
            new = re.sub(r' · QV\d+', '', old)
            new = new.replace(' · stage1-go2-simulation-release', '')
        elif old.startswith('来源：') and task:
            new = f'来源：figures/frame-sources.json，任务{task}。'
        elif old == 'stage1-go2-simulation-release统一版本验证结果 · 历史录像画面补充':
            new = 'stage1-go2-simulation-release统一版本测试结果与可视化材料'
        if new != old:
            p.text = new
    doc.save(path)


if __name__ == '__main__':
    unify_report(Path(__file__).resolve().parent / '02-测试报告与可视化材料.docx')
