"""Serial passive object settling checks; never represents a robot rollout."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',type=Path,required=True);ap.add_argument('--gpu',type=int,default=6);a=ap.parse_args();results=[]
for ep in sorted(a.out.glob('S*/episodes.json')):
 folder=ep.parent;report=folder/'physics_smoke.json'
 if not report.exists() or report.stat().st_mtime<ep.stat().st_mtime:
  with (folder/'physics_smoke.log').open('w') as log:
   try:p=subprocess.run([sys.executable,str(a.smoke),'--scene-dir',str(folder),'--gpu',str(a.gpu)],stdout=log,stderr=subprocess.STDOUT,timeout=420)
   except subprocess.TimeoutExpired:print(folder.name,'TIMEOUT',flush=True);continue
 if report.exists():
  r=json.loads(report.read_text());results.append(r);d=json.loads(ep.read_text());d['physics_validation']='passed_initial_passive_settling' if r['passed'] else 'failed_initial_passive_settling';ep.write_text(json.dumps(d,indent=2,ensure_ascii=False));print(folder.name,r['passed'],flush=True)
 (a.out/'physics_validation.json').write_text(json.dumps({'scope':'initial-layout passive settling only; robot execution and parked-state settling not tested','scenes':results,'passed':len(results)==10 and all(r['passed'] for r in results)},indent=2))
