"""Prepare a relocatable MV7 copy from the original dataset; never edit source assets."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from pxr import Sdf,UsdUtils
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-home',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.dataset_home.resolve();out=a.output.resolve()
 if out.exists():raise SystemExit('Use a new empty output directory: '+str(out))
 manifest=json.loads((R/'reproduction/dataset-files.json').read_text());bad=[n for n,h in manifest.items() if not (src/n).is_file() or sha(src/n)!=h['sha256']]
 if bad:raise SystemExit('Dataset missing or differs from tested assets: '+str(bad[:20]))
 out.mkdir(parents=True);home=out/'home_scenes'
 for n in manifest:
  dest=home/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/n,dest)
 for n in ['after_scene.usda','go2_training_flattened.usdc']:shutil.copy2(R/'delivery/assets-source'/n,out/n)
 # Keep the native dataset hierarchy; restore aliases used by original relative USD paths.
 scene=home/'scenes/MV7J6NIKTKJZ2AABAAAAADA8_usd'
 for name in ['models','Materials']:
  if not (scene/name).exists():(scene/name).symlink_to(Path('../..')/name,target_is_directory=True)
 layers=[home/n for n in manifest if Path(n).suffix in ('.usd','.usda','.usdc')]+[out/'after_scene.usda',out/'go2_training_flattened.usdc']
 for path in layers:
  alias=path.parent/'Materials'
  if path.is_relative_to(home/'models') and not alias.exists():alias.symlink_to(os.path.relpath(home/'Materials',path.parent),target_is_directory=True)
 rewritten=[]
 for path in layers:
  layer=Sdf.Layer.FindOrOpen(str(path))
  def resolve(asset):
   if '/home_scenes/' in asset:
    target=home/asset.split('/home_scenes/',1)[1]
    if not target.exists():raise FileNotFoundError(target)
    rewritten.append(asset);return os.path.relpath(target,path.parent)
   return asset
  UsdUtils.ModifyAssetPaths(layer,resolve);layer.Save()
 all_layers,assets,missing=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(out/'after_scene.usda')))
 if missing:raise RuntimeError('Unresolved scene assets: '+repr(missing[:20]))
 outside=[str(x.realPath) for x in all_layers if not Path(x.realPath).resolve().is_relative_to(out)]+[x for x in assets if not Path(x).resolve().is_relative_to(out)]
 if outside:raise RuntimeError('Nonportable dependency: '+repr(outside[:20]))
 report={'source_dataset_hashes_verified':True,'all_dependencies_inside_output':True,'layers':len(all_layers),'assets':len(assets),'rewritten_absolute_paths':len(rewritten),'scene':'after_scene.usda','robot':'go2_training_flattened.usdc'}
 (out/'asset-preparation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
