"""Freeze reviewable configuration and build a small code/configuration archive."""
import argparse,json,shutil,hashlib,zipfile,csv
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);a=ap.parse_args();code=Path(__file__).resolve().parent;fixtures=code/'fixtures';fixtures.mkdir(exist_ok=True)
for name in ('inventory.json','selected.json'):shutil.copy2(a.cache/name,fixtures/name)
for p in a.out.iterdir():
 if p.is_file() and p.suffix in ('.json','.html','.png'):shutil.copy2(p,fixtures/p.name)
for scene in sorted(a.out.glob('S*/episodes.json')):
 dest=fixtures/scene.parent.name;dest.mkdir(exist_ok=True)
 for name in ('episodes.json','evaluation_grid.npz','layout.png','physics_smoke.json','preview_conversion.json'):
  p=scene.parent/name
  if p.exists():shutil.copy2(p,dest/name)
rows=[json.loads(p.read_text()) for p in sorted(a.out.glob('S*/episodes.json'))]
with (fixtures/'scenes.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['id','scene_id','floor_area_m2','size_band','free_area_m2','robot_component_m2','tasks','objects'])
 for d in rows:w.writerow([d['id'],d['scene_id'],d['floor_area_m2'],d['size_band'],d['free_area_m2'],d['largest_robot_component_m2'],len(d['tasks']),','.join(o['asset_id'] for o in d['objects'])])
with (fixtures/'tasks.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['id','scene','start_xyz_m','goal_xyz_m','reference_path_m','initial_detour_path_m','geometric_class'])
 for d in rows:
  for t in d['tasks']:w.writerow([t['id'],d['id'],t['start_position_m'],t['goal_position_m'],t['reference_path_m'],t['initial_detour_path_m'],t['geometric_class']])
catalog=json.loads((fixtures/'object_catalog.json').read_text())
actual={x['asset_id']:x['dimensions_m'] for d in rows for x in d['objects']}
for item in catalog['objects']:
 item['upstream_documented_dimensions_m']=item['dimensions_m'];item['dimensions_m']=actual[item['asset']]
for target in (fixtures/'object_catalog.json',a.out/'object_catalog.json'):target.write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
scene_summary=[{k:v for k,v in d.items() if k not in ('tasks','objects')} for d in rows]
for target in (fixtures/'scenes.json',a.out/'scenes.json'):target.write_text(json.dumps(scene_summary,ensure_ascii=False,indent=2))
# File hashes describe packaged fixtures, not the external assets or temporary physics copies.
files={str(p.relative_to(fixtures)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(fixtures.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(fixtures/'manifest.json').write_text(json.dumps({'version':'0.1.0','upstream_branch':'genie2-rsi','upstream_commit':'c9ce1a5ba405905f4d626fe630c485ca60fe128a','files':files},indent=2))
runtime_files={str(p.relative_to(a.out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.out.rglob('*')) if p.is_file() and p.suffix in ('.json','.png','.npz','.usda','.html') and p.name not in ('manifest.json','physics_only.usda')}
(a.out/'manifest.json').write_text(json.dumps({'version':'0.1.0','scenes':10,'tasks':60,'object_types':15,'files':runtime_files,'excludes':'temporary flattened physics copies and logs'},indent=2))
archive=a.out/'grscenes-interaction-benchmark-v0.1.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(code.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and (p.name=='LICENSE' or p.suffix in ('.py','.md','.json','.png','.html','.npz','.csv')):z.write(p,str(Path('benchmark')/p.relative_to(code)))
print(archive,archive.stat().st_size,flush=True)
