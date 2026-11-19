from pathlib import Path
import ast,hashlib,json,shutil
R=Path(__file__).resolve().parent;root=R.parent/'g2-publish-20261118/research/rsi/g2_sim';I=root/'iterations/20261119_liftall';I.mkdir(exist_ok=True)
records=json.loads((R/'liftall_verified_results.json').read_text());latest={}
for row in records:
 if row['failure']=='HARNESS_ERROR':continue
 latest[row['asset']]=row
assert len(latest)==11,len(latest)
pedal=json.loads((R/'liftall_pedal_search.json').read_text())
passed=sorted(a for a,r in latest.items() if r['verified_success'])
config=json.loads((R/'ground_lift_training_config.json').read_text())
config.update(status='fixed posture screened on all 12 rigid lift candidates; partial success',validated_assets=passed,pending_adaptation_assets=sorted(set(latest)-set(passed)),planning_excluded_assets=['bin_pedal'],source_iteration='iterations/20261119_liftall',previous_config_archive='iterations/20261119_marginpose/fixtures/margin_ground_lift_config.json')
config['validated_simulation_baselines']={}
config['plan_directory']='liftall_validated_plans'
config['per_asset_plan_directory_overrides']={'bin_round_lidded':'liftall_metadata_plans'}
for a in passed:
 parts=Path(latest[a]['result']).parts
 config['validated_simulation_baselines'][a]=dict(harness=parts[0].replace('_trials','_trial.py'),plan_dir=parts[1],candidate=0,repeat=1)
config['validation_scope']='One deterministic trial per accepted asset, 1 kg common assumed mass, fixed root and non-arm joints, per-object pregrasp initialization; no common arm reset, randomized robustness, navigation or hardware validation.'
config['additional_acceptance']='Non-hand contact impulse <=0.001 Ns; sampled non-hand penetration <=1 mm; held visual mesh floor clearance >5 mm. Four loaded contacts: distal fingers for spread, inner/outer branches for pinch and under-support.'
(R/'liftall_ground_lift_config.json').write_text(json.dumps(config,indent=2));(root/'ground_lift_training_config.json').write_text(json.dumps(config,indent=2))
def copy(p,rel):
 dst=I/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for p in R.glob('*liftall*.py'):
 ast.parse(p.read_text());copy(p,Path('source')/p.name)
for pattern in ['liftall*.json','liftall*plans/*.json']:
 for p in R.glob(pattern):copy(p,Path('fixtures')/p.relative_to(R))
for row in records:
 p=R/row['result']
 for name in ['result.json','protocol.json','self_contacts.json','nonhand_contacts.json']:
  f=p.with_name(name)
  if f.exists():copy(f,Path('results')/f.relative_to(R))
for p in R.glob('liftall*.log'):
 if p.stat().st_size<1000000:copy(p,Path('diagnostics')/p.name)
(I/'trial_index.json').write_text(json.dumps(records,indent=2))
table='\n'.join(f"| {a} | {'PASS' if r['verified_success'] else 'FAIL'} | {r['failure'] or ''} | {r['min_hold_lift_m']} | {r['max_hold_tilt_deg']} |" for a,r in sorted(latest.items()))
text='''# 固定姿态全物体抬升验证（2026-11-19）

躯干仍为 [-40,142,-87,0,0] 度，肩高约 0.823 m；全部躯干关节保留至少约 10 度原始 URDF 限位余量。仅双臂和夹爪运动，根部固定。不是统一双臂初始姿态：每个物体从各自预抓取姿态开始。

本轮 12 个刚体抬升候选，11 个完成物理试验；脚踏桶 bin_pedal 在 81 个已有抓法/站位组合中没有合格路径（固定躯干逆解或底盘碰撞），未运行物理抬升。此有限搜索不证明所有抓法都不可能。两把大椅子保留推移任务，软包圆凳 stool_small 排除。

| 物体 | 复核结果 | 失败原因 | 保持最低抬高 m | 保持最大倾斜 deg |
| --- | --- | --- | --- | --- |
'''+table+'''
| bin_pedal | PLAN_FAIL | no admissible path in 81 attempts | — | — |

## 检查范围

至少抬高 15 cm，保持 3 秒，倾斜不超过 20 度，相对夹爪漂移不超过 2 cm，四接触承力覆盖率至少 90%。侧面张爪支撑要求四个末端指节；把手/边缘夹持及下托要求左右内外四分支。把手夹持的四分支要求比历史仅左右双侧合力检查更严格，不直接混用历史成功标签。

沿用可视表面接触检查及自碰撞检查；本轮增加运行时非手部接触冲量检查，并独立复核成功试验的采样几何穿透、原始关节余量、固定关节误差、根部漂移以及保持期间整个可视模型离地间隙。采样 10 Hz，物理步长 1/120 秒，采样峰值不等同于全物理步峰值。

四腿木凳、长方木凳、边桌、方篮统一使用已有修正表面碰撞资产；其他使用原始物体碰撞资产。本次没有修改共享资产。圆盖桶首轮因旧候选缺少 contact_centers_local 报 HARNESS_ERROR；根据 q_close 时真实指尖朝内表面补齐元数据后重跑，关节目标未改变，首轮错误保留但不计为物理失败。

统一物体质量假设 1 kg，理想躯干驱动、原手臂/夹爪力矩上限。没有物体绑定或运行中物体位姿写入。每个通过物体仅有一次确定初始条件试验，不等同于随机鲁棒性、真实质量或实机验证。尚未训练 policy；失败轨迹不作为成功示范。

## 复现

按 ../20261119_marginpose/README.md 准备工作区，再覆盖本目录 source/ 和 fixtures/。共享资产只读。运行前设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1，使用 Isaac Sim 4.2 对应 Python。

具体通过入口见 ../../ground_lift_training_config.json 的 validated_simulation_baselines。例如：

```bash
python liftall_general_trial.py --asset bin_lidded --plan-dir liftall_validated_plans --gpu 0 --repeat 2
```

长方木凳使用 liftall_under_trial.py；圆盖桶使用 liftall_metadata_plans。trial_index.json 记录完整采样和视频服务器路径及 SHA256，结果/协议/接触事件已归档。历史 envelope 字段沿用候选来源，当前姿态以重求后的 q 值和实际采样为准。
'''
(I/'README.md').write_text(text,encoding='utf-8')
(I/'.gitattributes').write_text('*.json -text\nsource/** -text -whitespace\ndiagnostics/** -text -whitespace\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(I/'manifest.json').write_text(json.dumps({str(p.relative_to(I)):sha(p) for p in I.rglob('*') if p.is_file() and p.name!='manifest.json'},indent=2))
p=root/'README.md';s=p.read_text();marker='<!-- liftall-evaluation -->'
if marker not in s:p.write_text(marker+'\n固定姿态全物体抬升结果：'+str(len(passed))+'/12 候选通过当前检查，详见 [逐物体验证](iterations/20261119_liftall/README.md) 和 [当前训练配置](ground_lift_training_config.json)。\n<!-- /liftall-evaluation -->\n\n'+s)
print(json.dumps(dict(passed=passed,failed=config['pending_adaptation_assets'],records=len(records)),indent=2))
