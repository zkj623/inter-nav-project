"""Serial isolated Isaac processes; reports failures instead of silently skipping them."""
import argparse,subprocess,sys,json,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',type=int,default=0);a=ap.parse_args();results=[]
for p in sorted(a.out.glob('S*/episodes.json')):
 scene=p.parent;report=scene/'physics_smoke.json'
 newest_stage=max((scene/'scene.usda').stat().st_mtime,(scene/'physics_only.usdc').stat().st_mtime)
 if not report.exists() or report.stat().st_mtime<newest_stage:
  with (scene/'physics_smoke.log').open('w') as log:
   try:r=subprocess.run([sys.executable,str(Path(__file__).with_name('smoke.py')),'--scene-dir',str(scene),'--gpu',str(a.gpu)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
   except subprocess.TimeoutExpired:r=None
  if not report.exists():results.append({'scene':scene.name,'passed':False,'error':'no result; inspect log','returncode':r.returncode if r else 'timeout'});continue
 result=json.loads(report.read_text());results.append(result);print(scene.name,result['passed'],flush=True)
 data=json.loads(p.read_text());data['physics_validation']='passive_settling_passed' if result['passed'] else 'passive_settling_failed';data['physics_validation_report']='physics_smoke.json';p.write_text(json.dumps(data,indent=2,ensure_ascii=False))
 (a.out/'physics_validation.json').write_text(json.dumps({'scenes':results,'passed':len(results)==10 and all(x['passed'] for x in results)},indent=2))
(a.out/'physics_validation.json').write_text(json.dumps({'scenes':results,'passed':len(results)==10 and all(x['passed'] for x in results)},indent=2))
