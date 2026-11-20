from pathlib import Path
import shutil,json,numpy as np
P=Path('/data20t/embodied/zkj/inter-nav-results');src=P/'g2-roundretry-20261120';dst=P/'g2-effortpose-20261120';dst.mkdir(exist_ok=True)
for p in src.iterdir():
 if p.name in ['directprep_plans','narrowrest_plans','restpose_v2_plans','commonstart_base_plans']:
  if not (dst/p.name).exists():shutil.copytree(p,dst/p.name)
 elif p.suffix in ['.py','.json'] and p.is_file():shutil.copy2(p,dst/p.name)
 elif p.is_symlink() and not (dst/p.name).exists():(dst/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
print(dst)
