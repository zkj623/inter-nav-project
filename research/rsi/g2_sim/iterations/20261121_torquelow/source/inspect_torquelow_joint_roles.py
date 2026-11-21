from pathlib import Path
import re
R=Path('/data20t/embodied/zkj/inter-nav-results/g2-torquelow-20261121/vendor/genie_sim_drive_audit/source/geniesim_ros');found=[]
for p in R.rglob('*'):
 if p.suffix not in ['.py','.yaml','.yml','.md','.xacro','.cpp','.hpp','.html']:continue
 if p.stat().st_size>500000:continue
 try:s=p.read_text(errors='ignore')
 except:continue
 if 'body_joint1' in s and re.search(r'ankle|knee|脚踝|膝盖',s,re.I):
  lines=s.splitlines();hits=[(i+1,l[:220]) for i,l in enumerate(lines) if re.search(r'ankle|knee|脚踝|膝盖',l,re.I)];found.append(dict(file=str(p.relative_to(R)),hits=hits[:5]))
  if len(found)>=12:break
print(found)
