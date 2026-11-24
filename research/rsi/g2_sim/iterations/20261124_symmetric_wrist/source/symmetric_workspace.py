from pathlib import Path
import argparse,subprocess,sys,shutil
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[3]);a=p.parse_args();pkg=a.package.resolve();dst=a.workspace.absolute();it=pkg/'iterations/20261124_symmetric_wrist'
subprocess.run([sys.executable,str(pkg/'iterations/20261124_realpose/source/realpose_workspace.py'),'--workspace',str(dst),'--package',str(pkg)],check=True)
for folder in ['source','fixtures']:
 for f in (it/folder).iterdir():
  if f.is_dir():shutil.copytree(f,dst/f.name,dirs_exist_ok=True)
  else:shutil.copy2(f,dst/f.name)
print('SYMMETRIC_WRIST_WORKSPACE_READY',dst)
