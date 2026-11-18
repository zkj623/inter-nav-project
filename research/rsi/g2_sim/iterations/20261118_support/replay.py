"""Replay the support iteration, preserving the previous evidence archive."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,shutil,os,importlib.util
HERE=Path(__file__).resolve().parent
def verify():
 for name,sha in json.loads((HERE/'manifest.json').read_text()).items():
  if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=sha:raise ValueError(name)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['verify','prepare','run'])
 p.add_argument('--workspace',type=Path);p.add_argument('--assets-from',type=Path);p.add_argument('--collision-deps',type=Path)
 p.add_argument('--python');p.add_argument('--asset');p.add_argument('--gpu',default=0,type=int);p.add_argument('--repeat',default=1,type=int)
 a=p.parse_args();verify()
 if a.action=='verify':print('Support manifest OK');return
 if a.workspace is None:p.error('--workspace required')
 if a.action=='prepare':
  if a.assets_from is None or a.collision_deps is None:p.error('--assets-from and --collision-deps required')
  subprocess.run([sys.executable,str(HERE.parent/'20261118/replay.py'),'prepare','--workspace',str(a.workspace),'--assets-from',str(a.assets_from),'--collision-deps',str(a.collision_deps)],check=True)
  for folder in ['source','plans']:
   for src in (HERE/folder).rglob('*'):
    if not src.is_file():continue
    dest=a.workspace/src.relative_to(HERE/folder)
    if dest.exists() and dest.read_bytes()!=src.read_bytes():raise ValueError('Refusing conflicting overlay: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  print('Support workspace prepared:',a.workspace);return
 if not a.python or not a.asset:p.error('--python and --asset required')
 b=json.loads((HERE/'baselines.json').read_text()).get(a.asset)
 if not b:p.error('No repeated full-path passing baseline for this asset')
 for folder in ['source','plans']:
  for src in (HERE/folder).rglob('*'):
   if src.is_file() and src.read_bytes()!=(a.workspace/src.relative_to(HERE/folder)).read_bytes():raise ValueError('Changed workspace source/plan: '+str(src))
 output=a.workspace/b['result_directory']/b['plan_dir']/a.asset/f"candidate{b['candidate']}_repeat{a.repeat}"
 if output.exists():raise ValueError('Choose an unused repeat number')
 spec=importlib.util.spec_from_file_location('g2_workspace',HERE.parents[1]/'workspace.py')
 base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 subprocess.run([a.python,str(a.workspace/b['harness']),'--asset',a.asset,'--plan-dir',b['plan_dir'],'--candidate',str(b['candidate']),'--repeat',str(a.repeat),'--gpu',str(a.gpu)],cwd=a.workspace,env=base.environment(a.workspace),check=True)
 d=json.loads((output/'result.json').read_text());print({k:d.get(k) for k in ['asset','success','failure','min_hold_lift_m','max_hold_tilt_deg']})
 if not d['success']:raise SystemExit(1)
if __name__=='__main__':main()
