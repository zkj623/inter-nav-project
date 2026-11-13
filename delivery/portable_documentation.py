"""Replace historical server deployment instructions with portable setup steps."""
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from report_methods import table


def update_guide(path):
    doc = Document(path)
    start = next(p for p in doc.paragraphs if p.text == '02  运行环境与外部依赖')
    anchor = next(p for p in doc.paragraphs if p.text == '03  测试方法与判定标准')
    node = start._p
    while node is not anchor._p:
        following = node.getnext()
        node.getparent().remove(node)
        node = following
    chapter = Document()
    from docx.enum.style import WD_STYLE_TYPE
    for target in (doc, chapter):
        if 'Deployment Command' not in target.styles:
            target.styles.add_style('Deployment Command', WD_STYLE_TYPE.PARAGRAPH)
    heading = chapter.add_heading('02  运行环境与外部依赖', 1)
    heading.paragraph_format.page_break_before = True
    chapter.add_paragraph('本章面向接收方的新机器。所有相对路径均以仓库根目录为基准；$HOME表示接收方本机用户目录。运行环境通过公开安装源准备，模型和任务配置随仓库提供，场景模型与纹理从公开数据集下载。')
    chapter.add_heading('2.1  依赖来源与资产位置', 2)
    table(chapter, [
        ('系统与硬件', 'Ubuntu 22.04 x86_64；NVIDIA RTX GPU，空闲显存≥14 GiB、空闲内存≥56 GiB；建议磁盘空间≥200 GB。驱动下载：https://www.nvidia.com/Download/index.aspx。'),
        ('Python环境', 'Python 3.10；Miniconda官方安装器：https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh。安装与激活步骤见reproduction/INSTALL.md第2节。'),
        ('仿真及Python依赖', 'Isaac Sim 4.2.0.2及extscache扩展从https://pypi.nvidia.com安装；PyTorch 2.5.1等依赖见仓库reproduction/requirements.txt。安装命令见reproduction/INSTALL.md第3节。'),
        ('机器人与场景覆盖层', '仓库delivery/assets-source/go2_training_flattened.usdc及after_scene.usda；prepare_assets.py将其复制至本机资产目录并修复引用。'),
        ('策略及支持脚本', '仓库delivery/model/actor_checkpoint.pt及部署契约；支持脚本位于delivery/runtime-support/，由reproduction/run_task.py自动引用。'),
        ('场景几何与纹理', '公开来源：https://huggingface.co/datasets/InternRobotics/GRScenes；固定revision见reproduction/download_dataset.py。下载三个home_scenes分卷约51 GB，由脚本解压。'),
        ('任务输入', '仓库delivery/inputs/T01至T10；启动器复制输入并替换场景路径，移除离线障碍元数据入口。'),
    ])
    from docx.shared import Cm
    for row in chapter.tables[0].rows:
        row.cells[0].width = Cm(3)
        row.cells[1].width = Cm(13)
    chapter.add_paragraph('策略权重SHA256：8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8。完整安装指南为reproduction/INSTALL.md，仓库README.md提供全部十项运行与结果检查命令。')
    chapter.add_page_break()
    chapter.add_heading('2.2  新机器部署与执行', 2)
    chapter.add_paragraph('按reproduction/INSTALL.md完成系统工具、Miniconda和显卡驱动准备，克隆私有仓库zkj623/rsi-nav的rsi-nav-v1分支并进入仓库根目录。以下命令在Ubuntu Bash中执行，所有输出均保存在接收方本机。')
    commands = [
        'conda create -n go2-simulation python=3.10 -y',
        'conda activate go2-simulation',
        "python -m pip install 'isaacsim==4.2.0.2' \\",
        "  'isaacsim-extscache-physics==4.2.0.2' \\",
        "  'isaacsim-extscache-kit==4.2.0.2' \\",
        "  'isaacsim-extscache-kit-sdk==4.2.0.2' \\",
        '  --extra-index-url https://pypi.nvidia.com',
        'python -m pip install -r reproduction/requirements.txt',
        'python -m pip install --no-deps -e .',
        'python -m pip check',
        'python delivery/verify_delivery.py',
        'python reproduction/download_dataset.py \\',
        '  --download-dir "$HOME/go2-data-download"',
    ]
    for command in commands:
        if command.endswith('\\'):
            command = command.rstrip('\\') + chr(92)
        chapter.add_paragraph(command, 'Deployment Command')
    chapter.add_paragraph('下载后，按reproduction/INSTALL.md第5节定位同时包含models/、Materials/、scenes/的解压目录，并将其设置为GO2_DATASET_HOME。随后准备资产并启动任务：')
    for command in [
        'python reproduction/prepare_assets.py \\',
        '  --dataset-home "$GO2_DATASET_HOME" \\',
        '  --output "$HOME/go2-simulation-assets"',
        'python reproduction/run_task.py \\',
        '  --assets "$HOME/go2-simulation-assets" \\',
        '  --output "$HOME/go2-runs/T01-first" --task T01 --gpu 0',
    ]:
        chapter.add_paragraph(command, 'Deployment Command')
    chapter.add_paragraph('本机资产目录包含after_scene.usda、go2_training_flattened.usdc、home_scenes/及asset-preparation.json。准备程序校验1680个原始文件的SHA256并检查引用闭包。资产及任务输出目录均须使用新路径；启动器自动串行执行并等待资源条件。T10及全部任务命令见仓库README.md。')
    chapter.add_page_break()
    for element in list(chapter.element.body):
        if element.tag != qn('w:sectPr'):
            anchor._p.addprevious(element)
    for p in doc.paragraphs:
        if p.text.startswith('保留原框架依赖与导航算法'):
            p.text = '交付包含运行代码、任务输入、策略权重、Go2机器人USD与场景覆盖层。大型场景几何和纹理按第2章从公开数据集下载；Isaac Sim及Python依赖按安装指南准备。'
        elif p.text == '□ 完成交付文件哈希校验，恢复外部资产路径。':
            p.text = '□ 完成交付文件哈希校验、公开资产下载和本机资产准备。'
    doc.save(path)


if __name__ == '__main__':
    from docx.enum.style import WD_STYLE_TYPE
    # The temporary chapter needs this style before its paragraphs are built.
    update_guide(Path(__file__).resolve().parent/'01-交付说明与测试用例.docx')
