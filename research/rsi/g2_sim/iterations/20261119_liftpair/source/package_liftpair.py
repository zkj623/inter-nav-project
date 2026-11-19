from pathlib import Path
import ast,hashlib,json,shutil
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_liftpair';I.mkdir(exist_ok=True)
rows=json.loads((R/'liftpair_verified_results.json').read_text());previous=json.loads((root/'iterations/20261119_liftfix/fixtures/liftfix_ground_lift_config.json').read_text());config=json.loads(json.dumps(previous));best={}
def group(r):return r['result'].split('_repeat')[0]
counts={group(r):sum(s['verified_success'] and group(s)==group(r) for s in rows) for r in rows}
for r in sorted(rows,key=lambda r:(-counts[group(r)],r.get('max_hold_tilt_deg') if r.get('max_hold_tilt_deg') is not None else 999)):
 if r['verified_success'] and r['asset'] not in best:best[r['asset']]=r
assert {r['asset'] for r in rows}=={'box_reinforced','case_handheld'}
config['validated_assets']=sorted(set(previous['validated_assets'])|set(best));config['pending_adaptation_assets']=sorted({'box_reinforced','case_handheld'}-set(best))
for a,r in best.items():
 parts=Path(r['result']).parts
 config['validated_simulation_baselines'][a]=dict(iteration='20261119_liftpair',harness=parts[0].replace('_trials','_trial.py'),plan_dir=parts[1],candidate=int(parts[3].split('_')[0].replace('candidate','')),repeat=int(parts[3].split('_repeat')[1]))
config.update(source_iteration='iterations/20261119_liftpair',previous_config_archive='iterations/20261119_liftfix/fixtures/liftfix_ground_lift_config.json',status=f"{len(config['validated_assets'])}/12 rigid candidates validated across archived fixed-posture baselines; no policy training started")
config['per_asset_plan_directory_overrides']={a:b['plan_dir'] for a,b in config['validated_simulation_baselines'].items()}
config['controller_scope']='Per-asset archived baseline; new pair trials test continuous load balancing, gradual total-load control, and geometry changes. Original acceptance and physical parameters unchanged.'
config['validation_scope']='Deterministic fixed-initialization screening; per-asset trial counts in archives; 1 kg assumed mass, fixed root/non-arm joints, per-object pregrasp arm pose. No randomized robustness, common arm reset, navigation or hardware validation.'
config['latest_baseline_verified_repeat_counts']={a:counts[group(r)] for a,r in best.items()}
(R/'liftpair_ground_lift_config.json').write_text(json.dumps(config,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(config,indent=2))
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*liftpair*.py'):
 ast.parse(p.read_text());copy(p,Path('source')/p.name)
for p in R.glob('liftpair*.json'):copy(p,Path('fixtures')/p.name)
for folder in {'liftpair_geometry_plans'}|{Path(r['result']).parts[1] for r in rows}:
 for p in (R/folder).glob('*.json'):copy(p,Path('fixtures')/folder/p.name)
for p in R.glob('liftpair*plans/*.json'):copy(p,Path('fixtures')/p.relative_to(R))
for r in rows:
 p=R/r['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('liftpair*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(rows,indent=2))
table='\n'.join(f"| {r['asset']} | {r['result']} | {r['verified_success']} | {r['failure']} | {r['min_hold_lift_m']} | {r['max_hold_tilt_deg']} | {r['four_contact_fraction']} |" for r in rows)
doc=f'''# 加固箱与手提硬箱：固定姿态继续验证

本轮针对上一轮两个四接触承力覆盖仅 60% 的物体，完成 {len(rows)} 次物理试验，新增 {len(best)} 种通过独立复核；当前总计 {len(config['validated_assets'])}/12 种拥有通过基线。

## 观察与调整

上一轮保持阶段高度、倾斜变化较小，但内外指负载反复切换，尤其左内指瞬时卸载。原控制器只在平均弱指载荷低于阈值时改变腕部平衡，可能漏掉这种偏载。本轮做控制与几何两类对照：

- roll 版本连续按内外指归一化载荷差调整腕部，最大角速度系数 0.6 度/秒，角度范围仍为 ±8 度；按真实仿真时间缩放更新量。
- soft 版本在上述基础上把每只夹爪的总接触载荷目标设为 20 N，以最大 0.8 mm/s 的挤压位移变化率缓慢调节，挤压范围仍为 0～12 mm，保留强指超 35 N 时释放规则。这里的载荷是接触力向量模之和，不是单独测量的法向力。
- range 版本在观察到 12 mm 位移补偿饱和后，将有界挤压范围扩到 20 mm；仍保留缓慢调节、35 N 释放、原驱动力矩上限和全部碰撞检查。该变化是手臂位置补偿范围，不是提高电机力矩上限。
- 重新按几何生成候选，尝试 35 度腕部倾角与更高侧面接触区，并验证前 10 cm 抬升同时前移 6 cm 的完整手臂路径。物体位姿没有被驱动。

躯干仍为 [-40,142,-87,0,0] 度，约 10 度最小原始限位余量。物体质量 1 kg、摩擦系数、驱动力矩上限、碰撞资产和最终通过标准均未改变。

## 全部物理结果

| 物体 | 试验 | 独立复核通过 | 失败原因 | 保持最低抬高 m | 最大倾斜 deg | 四接触覆盖 |
| --- | --- | --- | --- | --- | --- | --- |
'''+table+'''

通过仍要求至少抬升 15 cm、保持至少 3 秒、倾斜不超过 20 度、相对漂移不超过 2 cm、四个末端指节承力覆盖至少 90%。运行时非手部接触冲量、可视表面、碰撞、关节锁定与物理时钟检查保留；成功轨迹再独立复核整体离地、非手部几何穿透、根部漂移、原始关节余量。

当前配置优先选择同一控制器/计划/候选重复通过的基线，再比较保持倾斜；同初始条件重复不等同于随机鲁棒性测试。

## 复现和范围

按 ../20261119_liftfix/README.md 准备工作区，覆盖本目录 source/ 与 fixtures/。共享资产只读。OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1；使用 Isaac Sim 4.2 对应 Python。每个通过物体的 iteration、harness、plan_dir、candidate、repeat 见 ../../ground_lift_training_config.json。此前十种物体沿用各自已归档基线，不声称在本轮控制器上全部回归。

根部和非手臂关节固定，双臂按物体预抓取姿态初始化。确定初始条件验证不等同于随机鲁棒性或实机验证，尚未开始 policy 训练。失败记录不进入成功示范集。trial_index.json 保存全部成功与失败记录，以及完整视频、采样的服务器路径与 SHA256。
'''
(I/'README.md').write_text(doc,encoding='utf-8');(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
p=root/'README.md';s=p.read_text();marker='<!-- liftpair-evaluation -->'
if marker not in s:p.write_text(marker+f"\n两种箱体继续验证后，当前 {len(config['validated_assets'])}/12 种抬升候选拥有通过基线：[本轮对照结果](iterations/20261119_liftpair/README.md)。\n<!-- /liftpair-evaluation -->\n\n"+s)
print(json.dumps(dict(new_passed=sorted(best),total=len(config['validated_assets']),pending=config['pending_adaptation_assets'],trials=len(rows)),indent=2))
