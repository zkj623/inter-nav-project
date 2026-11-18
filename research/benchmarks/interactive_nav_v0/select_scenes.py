import json,argparse
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);p=ap.parse_args().data
rows=sorted(json.loads((p/'inventory.json').read_text()),key=lambda r:r['floor_area_m2'])
selected=[]
for i,k in enumerate(np.rint(np.linspace(0,len(rows)-1,10)).astype(int)):
 r=rows[k];r['id']=f'S{i+1:02}';r['area_rank']=int(k+1);r['size_band']='small' if i<3 else 'medium' if i<7 else 'large';selected.append(r)
(p/'selected.json').write_text(json.dumps(selected,indent=2))
print('inventory',len(rows));print(json.dumps(selected,indent=2))
