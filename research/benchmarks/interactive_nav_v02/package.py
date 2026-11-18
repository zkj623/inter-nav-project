"""Freeze portable code/configuration/verification evidence; exclude external binaries."""
import argparse,json,shutil,hashlib,zipfile,csv,subprocess,sys,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--old',type=Path,required=True);ap.add_argument('--fine',type=Path,required=True);ap.add_argument('--selected',type=Path,required=True);a=ap.parse_args();code=Path(__file__).resolve().parent;fixtures=code/'fixtures';fixtures.mkdir(exist_ok=True)
for name in ('object_catalog.json','source_asset_audit.json','environment.json'):shutil.copy2(a.old/name,a.out/name)
shutil.copy2(a.selected,fixtures/'selected.json');fine=fixtures/'fine_maps';fine.mkdir(exist_ok=True)
for f in a.fine.glob('S*.npz'):shutil.copy2(f,fine/f.name)
for f in a.fine.glob('S*.floor_support.json'):shutil.copy2(f,fine/f.name)
rows=[]
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());report=ep.parent/'physics_smoke.json'
 if report.exists():
  result=json.loads(report.read_text());assert len(result['objects'])==len(d['objects'])
  for obj,test in zip(d['objects'],result['objects']):
   assert obj['asset_id']==test['asset_id']
   assert math.dist(obj['position_m'][:2],test['initial_position'][:2])<.001,'Stale physical placement report'
  d['physics_validation']='passed_initial_passive_settling' if result['passed'] else 'failed_initial_passive_settling';ep.write_text(json.dumps(d,ensure_ascii=False,indent=2))
 rows.append(d)
subprocess.run([sys.executable,str(code/'build_viewer.py'),'--out',str(a.out)],check=True)
with (a.out/'tasks.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['task','scene','class','start_x','start_y','goal_x','goal_y','reference_route_m','reference_state','minimum_relocations_in_recorded_model','length_band','start_region','goal_region','initial_start_component','initial_goal_component','selected_clearing_objects'])
 for d in rows:
  for t in d['tasks']:w.writerow([t['id'],d['id'],t['class'],*t['start_position_m'][:2],*t['goal_position_m'][:2],t['cleared_route_length_m'],'all_parked' if t['class']=='interaction_required' else 'initial',t['minimum_relocations_in_optimistic_model'],t.get('length_band'),t.get('start_region'),t.get('goal_region'),t.get('initial_start_region'),t.get('initial_goal_region'),'|'.join(d['objects'][j]['instance_id'] for j in t.get('optimistic_relocation_subset',[]))])
coverage={}
for d in rows:
 for j,o in enumerate(d['objects']):
  x=coverage.setdefault(o['asset_id'],{'name':o['name'],'scenes':[],'tasks_using_object_in_one_recorded_relocation_witness':[]});x['scenes'].append(d['id']);x['tasks_using_object_in_one_recorded_relocation_witness'].extend(t['id'] for t in d['tasks'] if j in t.get('optimistic_relocation_subset',[]))
(a.out/'coverage.json').write_text(json.dumps(coverage,indent=2,ensure_ascii=False))
for f in a.out.iterdir():
 if f.is_file() and f.suffix in ('.json','.html','.js','.csv'):shutil.copy2(f,fixtures/f.name)
for ep in sorted(a.out.glob('S*/episodes.json')):
 dest=fixtures/ep.parent.name;dest.mkdir(exist_ok=True)
 for name in ('episodes.json','evaluation_grid.npz','physics_smoke.json','map.png','connectivity.png'):
  f=ep.parent/name
  if f.exists():shutil.copy2(f,dest/name)
  elif (dest/name).is_file():(dest/name).unlink()
for p in fixtures.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.html','.js','.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
files={str(p.relative_to(fixtures)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fixtures.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(fixtures/'manifest.json').write_text(json.dumps({'version':'0.2.0','revision':rows[0].get('revision','fixed_layout'),'robot':'Genie2/G2 CRS','scenes':10,'fixed_layouts':10,'tasks':100,'object_types':15,'upstream_commit':'c9ce1a5ba405905f4d626fe630c485ca60fe128a','files':files},indent=2))
archive=a.out/'grscenes-interaction-benchmark-v0.2.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(code.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.name!='inspect_stats.py' and (p.name=='LICENSE' or p.suffix in ('.py','.md','.json','.png','.html','.js','.npz','.csv')):z.write(p,str(Path('benchmark')/p.relative_to(code)))
print(str(archive),archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
