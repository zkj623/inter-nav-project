"""Fetch the pinned public split archive; upstream dataset is CC BY-NC-SA 4.0."""
from pathlib import Path
import argparse,subprocess
from huggingface_hub import hf_hub_download
p=argparse.ArgumentParser();p.add_argument('--download-dir',type=Path,required=True);a=p.parse_args();d=a.download_dir.resolve();d.mkdir(parents=True,exist_ok=True)
for name in ['home_scenes.z01','home_scenes.z02','home_scenes.zip']:
 hf_hub_download(repo_id='InternRobotics/GRScenes',repo_type='dataset',revision='4760b9031881c985e3582075cb2d8dbde1530a68',filename='scenes/GRScenes-100/'+name,local_dir=str(d))
archive=d/'scenes/GRScenes-100/home_scenes.zip';output=d/'extracted';output.mkdir(exist_ok=True)
subprocess.run(['7z','x',str(archive),'-o'+str(output),'-y'],check=True)
print('Extraction complete. Locate the directory containing models/, Materials/ and scenes/ (home_scenes) under',output)
