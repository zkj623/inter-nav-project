from pathlib import Path
import json,hashlib,importlib.metadata as im,subprocess
R=Path(__file__).resolve().parent;report=dict(workspace=str(R),baseline_commit='52f4d1d1141f2634a4d1cfa26d3fe668e8cf149f',physics_hz=120,control_hz=20,initialization='public common low home; first white-box pilot includes full standby/preparation',root_pinned=True,waist_ideal_cap_nm=1e7,waist_locked_after_settle=True,object_mass_kg=1.,nominal_gpus=[0,3,2,4,6],versions={},configs={},assets=[])
for name in ['numpy','scipy','h5py','opencv-python','isaacsim']:
 try:report['versions'][name]=im.version(name)
 except im.PackageNotFoundError:report['versions'][name]='not packaged under this name'
for name in ['ground_lift_training_config.json','standby_preparation_config.json']:
 p=R/name;report['configs'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
selected=set(json.loads((R/'standby_preparation_config.json').read_text())['execution'])
paths=[R/'generated/g2_crs_omnipicker.urdf']
for folder in ['robot_assets','object_assets','strategy15_surface_assets','strategy15_surface_proxies']:
 for p in (R/folder).glob('*'):
  if not p.is_file() or p.suffix not in ['.npz','.json','.usdc','.usd','.usda']:continue
  if folder!='robot_assets' and not any(p.name==a+p.suffix or p.name.startswith(a+'_') for a in selected):continue
  paths.append(p)
for p in paths:report['assets'].append(dict(relative_path=str(p.relative_to(R)),resolved_path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'collection_provenance.json').write_text(json.dumps(report,indent=2));print('PROVENANCE_INPUTS',len(paths))
