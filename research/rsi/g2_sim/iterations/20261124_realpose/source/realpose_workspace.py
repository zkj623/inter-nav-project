from pathlib import Path
import argparse,subprocess,sys,shutil
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[3]);args=p.parse_args();pkg=args.package.resolve(strict=True);iteration=pkg/'iterations/20261124_realpose';destination=args.workspace.absolute()
subprocess.run([sys.executable,str(pkg/'iterations/20261122_collection/source/collection_workspace.py'),'--workspace',str(destination),'--package',str(pkg)],check=True)
for f in (iteration/'source').glob('*.py'):shutil.copy2(f,destination/f.name)
for f in (iteration/'fixtures').iterdir():
    if f.is_dir():shutil.copytree(f,destination/f.name,dirs_exist_ok=True)
    else:shutil.copy2(f,destination/f.name)
old=Path('/data20t/embodied/zkj/inter-nav-results/g2-act-collect-20261122/home_floor_assets')
if old.exists() and not (destination/'home_floor_assets').exists():(destination/'home_floor_assets').symlink_to(old,target_is_directory=True)
print('REALPOSE_WORKSPACE_READY',destination)
