"""Render metric map layers and a self-contained local review website."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();scenes=[]
colors=np.array([[89,142,181],[146,172,119],[192,151,109],[155,135,180],[104,170,160],[187,126,138]])
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());g=np.load(ep.parent/'evaluation_grid.npz');floor=g['floor'];blocked=g['blocked'];h,w=floor.shape;rgb=np.full((h,w,3),245,np.uint8);rgb[floor]=[226,229,227];rgb[floor&blocked]=[82,91,95];rgb[g['nominal']]=[252,252,247]
 Image.fromarray(rgb).save(ep.parent/'map.png');labels,n=ndi.label(g['initial_optimistic'],np.ones((3,3),bool));colored=rgb.copy()
 for i in range(1,n+1):colored[labels==i]=colors[(i-1)%len(colors)]
 Image.fromarray(colored).save(ep.parent/'connectivity.png')
 d['map']={'width':w,'height':h,'origin':g['origin'].tolist(),'resolution':float(g['resolution'])};scenes.append(d)
(a.out/'viewer_data.json').write_text(json.dumps({'version':'0.2.0','scenes':scenes},ensure_ascii=False,indent=2))
shutil.copy2(Path(__file__).with_name('index.html'),a.out/'index.html')
print('Viewer:',len(scenes),'scenes')
