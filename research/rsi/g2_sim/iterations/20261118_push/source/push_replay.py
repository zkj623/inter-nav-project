"""Verify, prepare, or replay a repeatedly validated rigid-chair push."""
from pathlib import Path
import argparse,ast,json,hashlib,shutil,subprocess,sys,os
HERE=Path(__file__).resolve().parent
def verify():
 for name,digest in json.loads((HERE/'manifest.json').read_text()).items():
  p=HERE/name
  assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
  if p.suffix=='.py':ast.parse(p.read_text())
  if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 print('Push archive verified')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['verify','prepare','run']);p.add_argument('--workspace',type=Path);p.add_argument('--assets-from',type=Path);p.add_argument('--collision-deps',type=Path);p.add_argument('--python');p.add_argument('--asset');p.add_argument('--gpu',type=int,default=0);p.add_argument('--repeat',type=int,default=1);a=p.parse_args();verify()
 if a.action=='verify':return
 if not a.workspace:p.error('--workspace required')
 if a.action=='prepare':
  if not a.assets_from or not a.collision_deps:p.error('--assets-from and --collision-deps required')
  subprocess.run([sys.executable,str(HERE.parent/'20261118_support/replay.py'),'prepare','--workspace',str(a.workspace),'--assets-from',str(a.assets_from),'--collision-deps',str(a.collision_deps)],check=True)
  for archive in [HERE.parent/'20261118_chairs',HERE.parent/'20261118_collision',HERE]:
   for folder in ['source','plans']:
    for src in (archive/folder).rglob('*'):
     if not src.is_file():continue
     dest=a.workspace/src.relative_to(archive/folder);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  print('Prepared push workspace',a.workspace);return
 if not a.python or not a.asset:p.error('--python and --asset required')
 b=json.loads((HERE/'baselines.json').read_text()).get(a.asset)
 if not b:p.error('No repeatedly validated push baseline for this object')
 for folder in ['source','plans']:
  for src in (HERE/folder).rglob('*'):
   if src.is_file():assert src.read_bytes()==(a.workspace/src.relative_to(HERE/folder)).read_bytes(),str(src)
 output=a.workspace/b['result_directory']/b['plan_dir']/a.asset/f"candidate{b['candidate']}_repeat{a.repeat}"
 if output.exists():p.error('Trial already exists; choose an unused repeat number')
 subprocess.run([a.python,str(a.workspace/b['harness']),'--asset',a.asset,'--plan-dir',b['plan_dir'],'--candidate',str(b['candidate']),'--repeat',str(a.repeat),'--gpu',str(a.gpu)],cwd=a.workspace,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True)
 subprocess.run([sys.executable,str(a.workspace/'verify_push_results.py')],cwd=a.workspace,stdout=subprocess.DEVNULL,check=True)
 relative=str(output.relative_to(a.workspace)/'result.json');rows=json.loads((a.workspace/'push_result_verification.json').read_text());r=next(r for r in rows if r['result']==relative);print(r)
 if not r['review_accepted']:raise SystemExit(1)
if __name__=='__main__':main()
