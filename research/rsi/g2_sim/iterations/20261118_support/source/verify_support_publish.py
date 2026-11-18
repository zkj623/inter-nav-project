from pathlib import Path
import ast,hashlib,json,subprocess
R=Path(__file__).resolve().parent
repo=R.parent/'g2-publish-20261118';root=repo/'research/rsi/g2_sim/iterations/20261118_support'
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest.items():
 p=root/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,name
 staged=subprocess.check_output(['git','show',':'+str(p.relative_to(repo))],cwd=repo)
 assert hashlib.sha256(staged).hexdigest()==expected,('staged mismatch',name)
 if p.suffix=='.py':ast.parse(p.read_text())
 if p.suffix=='.json':json.loads(p.read_text())
rows=json.loads((root/'result_index.json').read_text())
for row in rows:
 d=json.loads((root/row['result']).read_text())
 assert hashlib.sha256((root/row['source_file']).read_bytes()).hexdigest()==d['source_sha256']
 if not d['success']:continue
 assert d['min_hold_lift_m']>=.15 and d['max_hold_tilt_deg']<=20 and d['hold_relative_drift_m']<=.02
 assert d['hold_four_finger_contact_fraction']>=.9 and d['max_contact_surface_error_m']<=.0025
 assert d['self_collision_enabled'] and not d['object_attachment'] and not d['robot_fixed_to_world']
for aid,b in json.loads((root/'baselines.json').read_text()).items():
 selected=[r for r in rows if r['asset']==aid and r['variant']==b['result_directory'] and r['candidate']==b['candidate'] and r['repeat'] in b['verified_repeats']]
 assert len(selected)==3 and all(r['success'] and r['includes_descent'] for r in selected)
ground=json.loads((root/'ground_clearance_check.json').read_text())
assert len(ground)==sum(r['success'] for r in rows)
assert all(r['passed_10mm'] for r in ground)
print('Verified staged manifest:',len(manifest),'files;',len(rows),'trial records;',sum(r['success'] for r in rows),'passes; all source hashes and mesh ground clearances match')
