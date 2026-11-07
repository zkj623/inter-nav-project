"""Apply the common monochrome SimSun layout to the two delivery documents."""
from pathlib import Path
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

FONT = 'SimSun'


def set_font(item, size=None, bold=None):
    item.font.name = FONT
    item.font.italic = False
    item.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None:
        item.font.size = Pt(size)
    if bold is not None:
        item.font.bold = bold
    rpr = item._element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    fonts.attrib.clear()
    for key in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        fonts.set(qn('w:' + key), FONT)
    color = rpr.find(qn('w:color'))
    if color is not None:
        color.attrib.clear()
        color.set(qn('w:val'), '000000')


def apply_document_format(path):
    doc = Document(path)
    for style in doc.styles:
        if hasattr(style, 'font'):
            set_font(style)
    specs = {'Normal': (12, False), 'Title': (22, True), 'Subtitle': (12, False),
             'Heading 1': (16, True), 'Heading 2': (14, True), 'Heading 3': (12, True),
             'Caption': (9, False), 'Header': (9, False), 'Footer': (9, False)}
    if 'Deployment Command' in doc.styles:
        specs['Deployment Command'] = (10.5, False)
        pf = doc.styles['Deployment Command'].paragraph_format
        pf.first_line_indent = Pt(0)
        pf.line_spacing = 1.15
        pf.space_before = pf.space_after = Pt(0)
    for name, (size, bold) in specs.items():
        set_font(doc.styles[name], size, bold)
    normal = doc.styles['Normal'].paragraph_format
    normal.line_spacing = 1.5
    normal.space_before = Pt(0)
    normal.space_after = Pt(0)
    normal.first_line_indent = Pt(24)
    for name in ('Heading 1', 'Heading 2', 'Heading 3'):
        pf = doc.styles[name].paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        pf.keep_with_next = True
        pf.line_spacing = 1.25
    for name in ('Caption', 'Header', 'Footer', 'Title', 'Subtitle'):
        doc.styles[name].paragraph_format.first_line_indent = Pt(0)
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2.3)
        section.left_margin = section.right_margin = Cm(2.5)
        section.header_distance = section.footer_distance = Cm(1.1)
        for paragraph in section.header.paragraphs:
            paragraph.text = '第一阶段仿真交付文件'
            paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for paragraph in section.footer.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Clear direct font colors/sizes so semantic paragraph styles govern text.
    for paragraph in doc.paragraphs:
        name = paragraph.style.name
        size, bold = specs.get(name, (12, False))
        if name == 'Normal':
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            if re.match(r'^\d+\.\s', paragraph.text) or '_' in paragraph.text or 'python ' in paragraph.text or 'SHA256' in paragraph.text:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if name.startswith('Heading'):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if name == 'Caption':
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.space_after = Pt(5)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in paragraph.runs:
            set_font(run, size, bold)
        if paragraph._p.findall('.//' + qn('w:drawing')):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.line_spacing = 1
    for tbl in doc.tables:
        tbl.autofit = False
        widths = [cell.width or Cm(16 / len(tbl.columns)) for cell in tbl.rows[0].cells]
        total = sum(widths)
        for index, column in enumerate(tbl.columns):
            column.width = int(Cm(16) * widths[index] / total)
        props = tbl._tbl.tblPr
        for tag in ('tblBorders', 'shd'):
            for old in list(props.findall(qn('w:' + tag))):
                props.remove(old)
        borders = OxmlElement('w:tblBorders')
        for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            edge = OxmlElement('w:' + side)
            for key, value in [('val', 'single'), ('sz', '4'), ('color', '000000')]:
                edge.set(qn('w:' + key), value)
            borders.append(edge)
        props.append(borders)
        for row in tbl.rows:
            header = row._tr.get_or_add_trPr().find(qn('w:tblHeader')) is not None
            for index, cell in enumerate(row.cells):
                cell.width = int(Cm(16) * widths[index] / total)
                cellprops = cell._tc.get_or_add_tcPr()
                for old in list(cellprops.findall(qn('w:shd'))):
                    cellprops.remove(old)
                for paragraph in cell.paragraphs:
                    pf = paragraph.paragraph_format
                    pf.first_line_indent = Pt(0)
                    pf.line_spacing = 1.15
                    pf.space_before = pf.space_after = Pt(3)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for run in paragraph.runs:
                        set_font(run, 10.5, header)
    # A dedicated title page; keep existing document identifiers and subtitles.
    if not any(p.text == '交付版本：stage1-go2-simulation-release' for p in doc.paragraphs):
        cover = doc.paragraphs[:4]
        for paragraph in cover:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.space_after = Pt(20)
        cover[0].paragraph_format.space_before = Pt(90)
        cover[1].paragraph_format.space_before = Pt(35)
        cover[3].text = '编制日期：2026年11月6日'
        set_font(cover[3].runs[0], 12)
        anchor = cover[3]._p
        for content in ('交付版本：stage1-go2-simulation-release', ''):
            p = doc.add_paragraph(content)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            if not content:
                p.add_run().add_break(__import__('docx.enum.text', fromlist=['WD_BREAK']).WD_BREAK.PAGE)
            anchor.addnext(p._p)
            anchor = p._p
    project = '项目名称：面向未知环境的交互式导航项目'
    if not any(p.text == project for p in doc.paragraphs):
        title = next(p for p in doc.paragraphs if p.style.name == 'Title')
        item = doc.add_paragraph(project)
        title._p.addprevious(item._p)
    anchor = next(p for p in doc.paragraphs if p.text == '交付版本：stage1-go2-simulation-release')._p
    for label in ('编制人：周康杰', '审核人：刘畅'):
        existing = next((p for p in doc.paragraphs if p.text == label), None)
        if existing is None:
            existing = doc.add_paragraph(label)
            anchor.addnext(existing._p)
        anchor = existing._p
        existing.paragraph_format.space_before = Pt(8)
        existing.paragraph_format.space_after = Pt(8)
    # Preserve centered cover metadata on repeated format applications.
    metadata = [p for p in doc.paragraphs if p.text.startswith(('交付版本：', '编制日期：', '项目名称：', '编制人：', '审核人：'))]
    for paragraph in doc.paragraphs[:4] + metadata:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Pt(0)
        if paragraph.text.startswith('项目名称：'):
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(12)
            for run in paragraph.runs:
                set_font(run, 16, True)
        elif paragraph in metadata:
            for run in paragraph.runs:
                set_font(run, 12, False)
    # Convert standalone breaks to heading page breaks to avoid empty pages.
    from docx.text.paragraph import Paragraph
    for paragraph in list(doc.paragraphs):
        breaks = paragraph._p.findall('.//' + qn('w:br'))
        if not paragraph.text.strip() and any(b.get(qn('w:type')) == 'page' for b in breaks):
            following = paragraph._p.getnext()
            if following is not None and following.tag == qn('w:p'):
                Paragraph(following, doc._body).paragraph_format.page_break_before = True
                paragraph._p.getparent().remove(paragraph._p)
            else:
                pf = paragraph.paragraph_format
                pf.space_before = pf.space_after = Pt(0)
                pf.line_spacing = Pt(1)
                for run in paragraph.runs:
                    set_font(run, 1)
    # Remove theme borders, colored rules, and all remaining run colors.
    for root in [doc.element, doc.styles.element]:
        for border in list(root.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    for element in doc.element.iter(qn('w:color')):
        element.attrib.clear()
        element.set(qn('w:val'), '000000')
    doc.save(path)


if __name__ == '__main__':
    for path in Path(__file__).resolve().parent.glob('*.docx'):
        apply_document_format(path)
