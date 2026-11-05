from pathlib import Path
import json,csv
from docx import Document
from docx.shared import Inches,Pt,RGBColor,Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
D=Path('E:/inter-nav-workspaces/go2-delivery-qv23/delivery');A=json.loads((D/'evidence/uniform-qv23-acceptance.json').read_text(encoding='utf-8'));F=json.loads((D/'figures/frame-sources.json').read_text(encoding='utf-8'));FONT='微软雅黑';NAVY='18354D';TEAL='217A86'
def font(style,size,bold=False,color=NAVY):
 style.font.name=FONT;style.font.size=Pt(size);style.font.bold=bold;style.font.color.rgb=RGBColor.from_string(color)
 rp=style.element.get_or_add_rPr();fs=rp.find(qn('w:rFonts'))
 if fs is None:fs=OxmlElement('w:rFonts');rp.insert(0,fs)
 for k in ['ascii','hAnsi','eastAsia','cs']:fs.set(qn('w:'+k),FONT)
def new(code,title,sub):
 d=Document();s=d.sections[0];s.page_width=Cm(21);s.page_height=Cm(29.7);s.top_margin=Cm(1.8);s.bottom_margin=Cm(1.7);s.left_margin=Cm(2);s.right_margin=Cm(2);s.header_distance=Cm(.7);s.footer_distance=Cm(.7)
 for name,size,bold,color in [('Normal',10,False,'263746'),('Title',27,True,NAVY),('Subtitle',12,False,TEAL),('Heading 1',17,True,NAVY),('Heading 2',12,True,TEAL),('Heading 3',10,True,NAVY),('Caption',8,False,'596A78')]:font(d.styles[name],size,bold,color)
 n=d.styles['Normal'].paragraph_format;n.space_after=Pt(5);n.line_spacing=1.15
 for name in ['Heading 1','Heading 2','Heading 3']:d.styles[name].paragraph_format.keep_with_next=True
 h=s.header.paragraphs[0];h.text='Go2 在线建图与交互导航                                      项目交付文件';font(d.styles['Header'],8,color='647685')
 p=s.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.RIGHT;p.add_run(code+'  |  第 ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');p._p.append(field);p.add_run(' 页');font(d.styles['Footer'],8,color='647685')
 d.add_paragraph('项目交付文件  /  '+code,'Subtitle');d.add_paragraph(title,'Title');d.add_paragraph(sub,'Subtitle');d.add_paragraph('版本 V3.0  ·  2026年11月4日')
 return d

def p(d,t,style=None):return d.add_paragraph(t,style)
def h(d,t):d.add_heading(t,1)
def table(d,heads,rows,widths=None):
 t=d.add_table(rows=1,cols=len(heads));t.autofit=False
 for c,v in zip(t.rows[0].cells,heads):c.text=str(v)
 for row in rows:
  for c,v in zip(t.add_row().cells,row):c.text=str(v)
 for idx,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  for j,c in enumerate(row.cells):
   if widths:c.width=Cm(widths[j])
   cp=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),NAVY if idx==0 else ('F0F5F7' if idx%2 else 'FFFFFF'));cp.append(sh)
   for para in c.paragraphs:
    para.paragraph_format.space_after=Pt(2);para.paragraph_format.space_before=Pt(2)
    for r in para.runs:r.font.name=FONT;r.font.size=Pt(9);r.font.bold=idx==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if idx==0 else '263746')
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'));return t

Q=json.loads((D/'task-paths-and-complexity.json').read_text(encoding='utf-8'))
def xy(v):return '('+', '.join(f'{x:.4f}' for x in v)+')'
def qualification(doc,number):
 doc.add_page_break();h(doc,number+'  路径与场景复杂度验收')
 p(doc,'T01–T10是同一MV7场景中的十组起终点任务，场景比例均为1:1。路径因起终点不同而不同，初始场景杂乱度相同。坐标以场景XY平面表示，单位为米。')
 p(doc,'原验收要求：路径曲折度严格大于1.3，场景杂乱度大于20%。曲折度τ=L最短几何路径 / D起终点直线距离，是无量纲比值；不是路径长度大于1.3米，也不是实际运行轨迹绕行比。τ>1.3表示最短几何路线也比直线至少长30%。')
 table(doc,['任务','直线/m','最短路径/m','曲折度τ','杂乱度','判定'],[[x['id'],f"{x['euclidean_m']:.4f}",f"{x['shortest_geometry']['length_m']:.4f}",f"{x['tortuosity']:.4f}",f"{x['clutter']*100:.4f}%",'均达标'] for x in Q],[1.7,2.6,3.1,2.7,3.6,2.1])
 p(doc,'十项曲折度范围1.3172–1.5830，全部严格大于1.3；最接近阈值的T10为1.3171657，仍达标。按原始精度判断，表中数值仅为显示舍入。')
 p(doc,'杂乱度=初始不可通行低高度碰撞投影面积 / 固定原始地面支撑面积=68.105625 / 178.3075=38.195603%。外部空域排除，初始箱体计入，不做机器人半径膨胀。38.195603%>20%，十项均满足。')
 p(doc,'几何最短路由固定栅格派生碰撞多边形上的可见性图计算，使用未膨胀几何；它是该占据表示下有限尺寸运动的下界。机器人可行性另行检查：T01–T05存在机器人净空路径；T06–T10初始机器人通路被阻挡，侧推后连通，并经真实运行的接触、箱体位移、新扫描和穿越证据验收。')
 p(doc,'资格数据仅用于离线评估，不向在线导航器输入这些路线。qualification.json中的navigation_success_not_yet_tested是资格评估时点的字段；任务成功结论以其后的stage1-go2-simulation-release功能证据为准。原始完整路径坐标见task-paths-and-complexity.json / .csv，资格文件见evidence/qualification/。')
d=new('GO2-DLV-001','交付说明与测试用例','源码交接、环境配置、运行方式与验收方法')
table(d,['项目','说明'],[['交付版本','stage1-go2-simulation-release；运行源码 514cd51dc7a4400a3e7db431b1f18a7aba7e3f50'],['交付分支','stage1-go2-simulation-release'],['验收口径','以任务成功完成为准，不设耗时门槛'],['配套文件','《测试报告与可视化材料》、源码、模型、场景输入、测试证据']],[3,14])
p(d,'本文件将交付说明与测试用例合并，供接收方理解交付范围、准备运行环境，并执行后续验证。原始证据保留历史记录，耗时比较仅供参考。')
h(d,'01  交付范围')
table(d,['内容','位置与用途'],[['运行入口','grutopia/demo/go2_online_interactive_navigation.py'],['导航与交互模块','grutopia_extension/interactive_navigation/：建图、探索、路径跟随、执行安全、推箱与停稳'],['Go2控制器','grutopia_extension/controllers/isaac_go2_ctrl.py'],['自动测试与审计','tests/、tools/；delivery/regression-case-index.csv 提供定位索引'],['输入、模型与证据','delivery/inputs/、model/、evidence/；figures/ 包含截图与来源清单']],[4,13])
p(d,'保留原框架依赖与导航算法，避免为拆分目录改变已验证行为。大型场景、纹理、机器人引用资产及 Isaac 安装不在源码压缩包中，需按下页清单准备。')
d.add_page_break();h(d,'02  运行环境与外部依赖')
p(d,'使用原 Linux Isaac Sim / IsaacLab 环境。模型、机器人尺度和物理契约应与验证记录一致。普通 Windows Python 不能直接替代仿真环境。')
table(d,['依赖','路径或要求'],[['环境初始化','/data20t/embodied/share/inter-nav/env.sh'],['场景入口','/data20t/embodied/zkj/inter-nav-results/rectangular-boxes-five-scenes-xy100-20261021-v2/mv7/after_scene.usda 及引用资源'],['机器人资产','go2-online-smooth-20261027/official-policy-review-20261030/nvidia_isaaclab_4_5/go2_training_flattened.usdc 及引用资源'],['策略权重','delivery/model/actor_checkpoint.pt'],['权重SHA256','8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8'],['运行支持脚本','delivery/runtime-support/；原启动器仍引用既有服务器路径'],['资产目录','grutopia/assets 为原服务器资产链接，交接时需恢复其目标目录']],[3.3,13.7])
d.add_heading('部署与执行步骤',2)
for text in ['1. 解压源码并准备上述资产，在原 Linux 环境 source 初始化脚本。','2. 进入源码根目录，设置 PYTHONPATH="$PWD:$PYTHONPATH"。','3. 运行 python delivery/verify_delivery.py，检查交付文件完整性。','4. 使用 "$ISAAC_PYTHON" -m grutopia.demo.go2_online_interactive_navigation --help 查看入口参数。','5. 参照 delivery/launch_uniform_qv23_diagnostic.py 的冻结参数准备新输出目录。原脚本含固定路径与覆盖保护，迁移时需显式映射，不能覆盖历史试验。','6. 检查共享锁和资源约束后串行运行；24GiB内存上限、零swap，空闲内存56GiB持续60秒、GPU空闲14GiB。','7. 运行对应功能审计，保存结果、配置、日志和版本信息。']:p(d,text)
p(d,'离线评估图仅供资格与审计，不得输入导航器。跨机器迁移或路径变更后，需要重新执行验证。')
d.add_page_break();h(d,'03  测试方法与判定标准')
p(d,'场景用例检查完整任务能否成功完成；自动回归用例检查关键代码逻辑。测试方法与实际测试结果分开记录，结果见配套报告。')
table(d,['检查项','预期结果'],[['任务完成','success=true，reason=goal_reached，目标误差≤0.5m并满足实测停稳条件'],['运行安全','启动倾角<15°，运行最大倾角<30°，原执行安全与健康检查通过'],['在线输入','初始地图为空，使用真实LiDAR，离线评估几何未传入导航器'],['交互任务附加检查','真实接触、实际箱体位移、新于最后箱体运动的扫描、整机通路穿越审计通过'],['耗时','记录仿真耗时供参考，不设验收门槛']],[4,13])
d.add_heading('通用前置条件与步骤',2)
p(d,'冻结stage1-go2-simulation-release源码、指定policy、场景资产、机器人比例及物理参数；使用新的输出目录。交互资格为授权真值箱体名单加LiDAR可见性，不能表述为新视觉识别模型。')
p(d,'步骤：加载对应场景和起终点 → 空地图启动 → LiDAR建图与导航 → 交互项执行推箱、重新扫描和穿越 → 目标处停稳 → 执行功能审计并归档。')
d.add_heading('T01–T10 场景用例索引',2)
table(d,['编号','类型','输入与结果要求'],[[f'T{i:02d}','在线导航' if i<=5 else '推箱交互导航',f'delivery/inputs/T{i:02d}/tasks_preflight.json；满足通用标准'+('及交互附加检查' if i>=6 else '')] for i in range(1,11)],[2,3.5,11.5])
p(d,'每个输入目录同时保存 scene_profile.json、layout.json、floor_config.json，明确起终点与场景配置。test-cases.csv 可用于补充执行日期、人员与复测结果。')
d.add_page_break();h(d,'04  自动回归与交接核对')
table(d,['测试文件','验证内容'],[['test_online_navigation.py','空地图、LiDAR可见性与在线地图决策'],['test_final_push_release.py','推箱退出与最终释放'],['test_passage_certificate.py','通路有效性证书'],['test_release_reserve_cycle.py','T10释放余量循环回归']],[8,9])
p(d,'定向复核命令：python -m pytest -q tests/test_online_navigation.py tests/test_final_push_release.py tests/test_passage_certificate.py tests/test_release_reserve_cycle.py')
p(d,'完整回归索引见 regression-case-index.csv。静态函数条目数量与pytest参数化展开后的数量不同；完整历史结果为1012 passed、1 deselected，执行环境与原日志应一并保留。')
d.add_heading('交接检查清单',2)
for text in ['□ 接收源码ZIP与Git bundle，确认交付分支。','□ 确认模型权重、输入快照与依赖清单。','□ 完成交付文件哈希校验，恢复外部资产路径。','□ 阅读场景用例和配套测试报告，明确无耗时验收门槛。','□ 后续复测使用新目录，保留新版本及新结果，不覆盖历史证据。']:p(d,text)
p(d,'文档说明：本次仅整理交付材料与已有视频截图，未启动新的训练或物理仿真。')
qualification(d,'05')
for offset in range(0,len(Q),3):
 d.add_page_break();h(d,'06  各任务路径坐标'+('（续）' if offset else ''))
 for t in Q[offset:offset+3]:
  d.add_heading(t['id']+' · '+('推箱交互导航' if t.get('requires_interaction') else '在线导航'),2)
  p(d,'起点S='+xy(t['start'])+'；终点G='+xy(t['goal'])+'。')
  p(d,f"直线距离{t['euclidean_m']:.4f}m；最短几何路径{t['shortest_geometry']['length_m']:.4f}m；曲折度{t['tortuosity']:.4f}>1.3；杂乱度{t['clutter']*100:.4f}%>20%。")
  p(d,'离线几何参考路径：'+' → '.join(xy(pt) for pt in t['shortest_geometry']['path']))
  if t.get('requires_interaction'):
   iq=t['interaction_qualification'];p(d,f"交互资格：{iq['box_id']}；侧推前机器人不连通，侧推后连通；资格评估的侧推后机器人路径{iq['robot_path_after_m']:.4f}m。")
  else:p(d,f"机器人净空栅格可行路径：{t['robot_clearance_grid_path_m']:.4f}m。")
d.save(D/'01-交付说明与测试用例.docx')

r=new('GO2-TST-002','测试报告与可视化材料','stage1-go2-simulation-release统一版本验证结果 · 历史录像画面补充')
table(r,['结论','依据'],[['任务完成验收','10 / 10 通过；以成功完成任务为标准'],['最大停稳误差','19.13cm（stage1-go2-simulation-release统一版本固定十项）'],['代码回归','历史完整1012 passed、1 deselected；交接定向复核43 passed'],['耗时口径','仅供参考，不影响验收'],['图片材料','十项任务，每项过程与停稳两帧，共20张原视频截图']],[4,13])
h(r,'01  测试范围与版本说明')
p(r,'统一版本功能结果来自运行提交514cd51，T01–T09为无录像试验，T10沿用同提交已有录像。固定任务各一次成功验证不等于统计成功率。')
p(r,'可视化材料来自 go2-textured-interaction-20261103 已验收历史录像：T01–T05为QV13，T06–T09为QV22，T10为stage1-go2-simulation-release。图片用于展示任务过程，不将历史录像冒充stage1-go2-simulation-release无录像运行。每页分别列示两组结果，便于核对。')
p(r,'截图保留原画面，不改写状态叠字。图片本身不能证明所有安全检查；实际验收依据为日志、契约及功能审计。图像来源、帧号、视频SHA256见 figures/frame-sources.json。')
r.add_page_break();h(r,'02  stage1-go2-simulation-release统一版本结果')
table(r,['任务','类型','误差/cm','到目标制动/s','验收'],[[t['task'],'导航' if t['task']<='T05' else '推箱交互',f'{t["goal_error_m"]*100:.2f}',f'{t["duration_to_goal_braking_s"]:.2f}','通过'] for t in A['tasks']],[2,3,3,5,2.8])
p(r,'上述时刻为仿真时间，不是墙钟执行时间。T02–T05比历史基线慢约3–10秒，主要增加在就位、转向与制动，作为工程参考保留。')
dummy=0
h(r,'03  验证证据与复核')
p(r,'历史完整测试见 evidence/tests-qv23-historical.log；本次交接定向复核见 evidence/tests-handover-targeted.log，43 passed in 7.82s。Windows缺少cv2的收集失败不计入通过结果。')
p(r,'十项source_manifest与交付源码按LF行尾规范化比对一致。逐项输入、物理和执行器契约、功能结果见 evidence/T01–T10/。')
p(r,'交互任务的验收还包括真实接触冲量、箱体位移、新扫描及整机穿越。截图仅作可视化补充，不能替代这些检查。')
qualification(r,'04')
for i,t in enumerate(A['tasks'],1):
 r.add_page_break();task=t['task'];items=[x for x in F if x['task']==task];v=items[0]['version'].upper();record=items[0]
 h(r,f'{i+4:02d}  {task} '+('在线导航' if i<=5 else '推箱交互导航'))
 p(r,f'stage1-go2-simulation-release统一验证：通过，停稳误差{t["goal_error_m"]*100:.2f}cm。下列录像版本：{v}；该录像运行成功，停稳误差{record["recorded_goal_error_m"]*100:.2f}cm。')
 qt=next(x for x in Q if x['id']==task)
 p(r,'S='+xy(qt['start'])+' → G='+xy(qt['goal'])+f"；最短几何路径{qt['shortest_geometry']['length_m']:.4f}m；直线{qt['euclidean_m']:.4f}m；τ={qt['tortuosity']:.4f}>1.3；杂乱度38.1956%>20%，满足原路径与复杂度要求。")
 for j,x in enumerate(items,1):
  pp=r.add_paragraph();pp.paragraph_format.space_after=Pt(2);pp.add_run().add_picture(str(D/x['image']),width=Cm(15.6))
  label='推箱过程' if i>=6 and j==1 else ('导航过程' if j==1 else '目标停稳')
  p(r,f'图 {i}-{j}  {label}｜{task} · {v}｜视频 {x["video_time_s"]:.2f}s · 帧 {x["frame_index"]}｜{x["state"]}','Caption')
 p(r,'来源：'+Path(items[0]['source_video']).relative_to(Path('E:/inter-nav-results/go2-textured-interaction-20261103')).as_posix(),'Caption')
r.save(D/'02-测试报告与可视化材料.docx')
print('Built two formal documents')
