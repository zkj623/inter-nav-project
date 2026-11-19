from pathlib import Path
import ast,hashlib,json,shutil
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_liftfix';I.mkdir(exist_ok=True)
rows=json.loads((R/'liftfix_verified_results.json').read_text());previous=json.loads((root/'iterations/20261119_liftall/fixtures/liftall_ground_lift_config.json').read_text());config=json.loads(json.dumps(previous))
best={}
for r in rows:
 if r['verified_success'] and r['asset'] not in best:best[r['asset']]=r
assets=['basket_rectangular','basket_woven','bin_round_pedal','box_reinforced','case_handheld','bin_pedal']
assert set(r['asset'] for r in rows)==set(assets)
config['validated_assets']=sorted(set(previous['validated_assets'])|set(best))
config['pending_adaptation_assets']=sorted(set(assets)-set(best));config['planning_excluded_assets']=[]
config['geometry_screen_pass_assets']=sorted(set(previous['geometry_screen_pass_assets'])|{'bin_pedal'});config['geometry_screen_fail_assets']=[]
config['excluded_from_initial_lift_set'].pop('bin_pedal',None)
for b in config['validated_simulation_baselines'].values():b['iteration']='20261119_liftall'
for a,r in best.items():
 parts=Path(r['result']).parts
 config['validated_simulation_baselines'][a]=dict(iteration='20261119_liftfix',harness=parts[0].replace('_trials','_trial.py'),plan_dir=parts[1],candidate=int(parts[3].split('_')[0].replace('candidate','')),repeat=1)
config.update(source_iteration='iterations/20261119_liftfix',previous_config_archive='iterations/20261119_liftall/fixtures/liftall_ground_lift_config.json',status=f"{len(config['validated_assets'])}/12 rigid candidates validated across archived fixed-posture baselines; no policy training started")
config['plan_directory']='liftall_validated_plans'
config['plan_resolution']='Use each validated_simulation_baselines entry for authoritative iteration, harness, plan_dir and candidate; plan_directory is the legacy default.'
config['per_asset_plan_directory_overrides']={a:b['plan_dir'] for a,b in config['validated_simulation_baselines'].items()}
config['controller_scope']='Per-asset archived harness; balance variant adds weak-finger squeeze and active hold feedback. Acceptance criteria unchanged. Baseline trial harness remains frozen.'
(R/'liftfix_ground_lift_config.json').write_text(json.dumps(config,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(config,indent=2))
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*liftfix*.py'):
 ast.parse(p.read_text());copy(p,Path('source')/p.name)
for pattern in ['liftfix*.json','liftfix*plans/*.json']:
 for p in R.glob(pattern):copy(p,Path('fixtures')/p.relative_to(R))
for r in rows:
 p=R/r['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('liftfix*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(rows,indent=2))
table='\n'.join(f"| {r['asset']} | {r['result']} | {r['verified_success']} | {r['failure']} | {r['min_hold_lift_m']} | {r['four_contact_fraction']} |" for r in rows)
doc=f'''# 固定 G2 姿态：六种失败物体的抓取调整

继续使用 [-40,142,-87,0,0] 度躯干姿态及约 10 度最小关节限位余量。本轮针对此前六种未通过候选调整，共保留 {len(rows)} 次物理试验。新增通过 {len(best)} 种；加上上一轮六种，当前共 {len(config['validated_assets'])}/12 种通过各自归档基线。

## 调整方法

- 方篮减浅夹爪插入深度，避免腕部接触；编织篮调整右夹爪沿闭合轴的对中位置。
- 圆形脚踏桶与加固箱调整物体前后站位和侧面接触高度；加固箱在升高接触区时双手各外移 4 mm，避免预抓取时指面穿透。
- 手提硬箱提高侧面接触区；额外搜索上部特征夹持，但开放夹爪几何检查没有合格候选，保留失败原因。
- 脚踏桶重新基于几何射线生成倾斜腕部四指支撑策略，固定原躯干重新求解完整路径，首次得到可物理执行候选。
- 独立 balance 控制器：弱指承力不足且强指低于 30 N 时继续有限挤压，保持原有 12 mm 上限与腕部平衡约束；保持阶段继续手臂承力反馈。夹持类可在 +/-8 mm 内对中，并要求四分支预载后才抬升。执行过的源码不覆盖，结果绑定源码 SHA256。
- preload 变体进一步把侧面支撑控制目标提高至每指 5 N，开始保持前要求各指 >4 N、物体速度 <5 mm/s 连续八次更新；最终成功判据不变。
- hold 变体针对高预载门控反复等待，保留 5 N 承力目标及主动保持反馈，将进入保持的门控恢复为原基线各指 >1 N、速度 <15 mm/s 连续四次更新。最终 15 cm / 3 秒 / 20 度 / 2 cm / 四接触覆盖 90% 验收未改变，必须实际执行保持才能判成功。
- 箱桶避让路径还测试了前 10 cm 抬升同步向前移出、随后继续竖直抬升的方案。前移量仅用于手臂目标和名义碰撞审计，未控制物体位姿；是否通过仍以运行时接触与保持结果为准。

判据没有放宽：抬高至少 15 cm，保持至少 3 秒，倾斜 <=20 度，相对夹爪漂移 <=2 cm，四接触承力覆盖率 >=90%；非手部接触冲量、可视表面、碰撞和固定关节检查均保留。独立复核要求整体可视模型离地 >5 mm、采样非手部穿透不超过 1 mm、锁定误差 <0.005 rad、根部漂移 <1 mm。

| 物体 | 试验路径 | 独立复核通过 | 失败原因 | 保持最低抬高 m | 四接触覆盖 |
| --- | --- | --- | --- | --- | --- |
'''+table+'''

## 复现和训练边界

先按 ../20261119_liftall/README.md 准备工作区，再覆盖本目录 source/ 和 fixtures/；共享资产保持只读。运行前设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1，使用 Isaac Sim 4.2 对应 Python。每个通过物体的 harness、plan_dir、candidate、iteration 见 ../../ground_lift_training_config.json 的 validated_simulation_baselines。旧六种沿用上一轮冻结基线，不声称已在新控制器上重新回归。

全部物体统一假设 1 kg，根部与非手臂关节固定，理想躯干驱动。双臂仍按物体预抓取姿态初始化，未验证统一 reset、随机初始误差或实机能力。没有物体绑定或运行中位姿写入。仅通过独立复核的轨迹可成为后续训练成功示范，尚未开始 policy 训练。

trial_index.json 保存所有成功与失败记录，以及完整采样和原始视频服务器路径、SHA256。关节姿态以当前 q 与物理采样为准，历史候选 envelope 字段保留用于追溯。
'''
(I/'README.md').write_text(doc,encoding='utf-8');(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
p=root/'README.md';s=p.read_text();marker='<!-- liftfix-evaluation -->'
if marker not in s:p.write_text(marker+f"\n固定姿态迭代后 {len(config['validated_assets'])}/12 种抬升候选通过各自归档基线：[六种物体的调整结果](iterations/20261119_liftfix/README.md)。\n<!-- /liftfix-evaluation -->\n\n"+s)
print(json.dumps(dict(new_passed=sorted(best),total=config['validated_assets'],pending=config['pending_adaptation_assets'],trials=len(rows)),indent=2))
