from pathlib import Path
import argparse,shutil,json,ast
p=argparse.ArgumentParser(description='Prepare an isolated G2 collection/replay workspace; never edits shared assets.')
p.add_argument('--workspace',type=Path,required=True);p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--assets-from',type=Path,default=Path('/data20t/embodied/zkj/inter-nav-results/g2-roundretry-20261120'));args=p.parse_args();pkg=args.package.resolve(strict=True);base=args.assets_from.resolve(strict=True);work=args.workspace.absolute();work.mkdir(parents=True,exist_ok=False)
for x in base.iterdir():
 if x.is_symlink():(work/x.name).symlink_to(x.resolve(strict=True),target_is_directory=x.is_dir())
 elif x.is_file() and x.suffix in ('.py','.json'):shutil.copy2(x,work/x.name)
 elif x.is_dir() and x.name.endswith('_plans'):shutil.copytree(x,work/x.name)
for name in ('20261120_symhome','20261120_restpose','20261120_narrowrest','20261120_directprep','20261120_roundretry'):
 it=pkg/'iterations'/name
 for x in (it/'source').glob('*.py'):shutil.copy2(x,work/x.name)
 for x in (it/'fixtures').iterdir():
  if x.is_dir():shutil.copytree(x,work/x.name,dirs_exist_ok=True)
  else:shutil.copy2(x,work/x.name)
for name in ['ground_lift_training_config.json','standby_preparation_config.json']:shutil.copy2(pkg/name,work/name)
source=Path(__file__).resolve().parent
for x in source.glob('*.py'):shutil.copy2(x,work/x.name)
for version_dir in ['source_versions','dependency_versions']:
 if (source/version_dir).exists():shutil.copytree(source/version_dir,work/version_dir)
for x in work.glob('*.py'):ast.parse(x.read_text(encoding='utf-8-sig'))
config=json.loads((work/'standby_preparation_config.json').read_text())
for asset in config['execution']:assert (work/('collect_'+asset+'_trial.py')).is_file(),asset
print('G2_COLLECTION_WORKSPACE_READY',work)

