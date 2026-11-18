"""Render metric map layers and a self-contained local review website."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();scenes=[]
colors=np.array([[140,180,210],[170,195,137],[220,175,128],[185,155,211],[122,193,181],[218,148,160],[180,184,214],[199,187,131],[152,190,159],[205,166,192]])
for ep in sorted(a.out.glob('S*/episodes.json')):
 d=json.loads(ep.read_text());g=np.load(ep.parent/'evaluation_grid.npz');floor=g['floor'];blocked=g['blocked'];h,w=floor.shape;rgb=np.full((h,w,3),245,np.uint8);rgb[floor]=[226,229,227];rgb[floor&blocked]=[82,91,95];rgb[g['nominal']]=[252,252,247]
 Image.fromarray(rgb).save(ep.parent/'map.png');labels,n=ndi.label(g['initial_optimistic'],np.ones((3,3),bool));colored=rgb.copy()
 active=[x['component_id'] for x in d.get('component_regions',[])] or list(range(1,n+1))
 for j,i in enumerate(active):colored[labels==i]=colors[j%len(colors)]
 Image.fromarray(colored).save(ep.parent/'connectivity.png')
 d['map']={'width':w,'height':h,'origin':g['origin'].tolist(),'resolution':float(g['resolution'])};scenes.append(d)
(a.out/'viewer_data.json').write_text(json.dumps({'version':'0.2.0','revision':scenes[0].get('revision','fixed_layout'),'scenes':scenes},ensure_ascii=False,indent=2))
shutil.copy2(Path(__file__).with_name('index.html'),a.out/'index.html')
shutil.copy2(Path(__file__).with_name('viewer.js'),a.out/'viewer.js')
print('Viewer:',len(scenes),'scenes')
