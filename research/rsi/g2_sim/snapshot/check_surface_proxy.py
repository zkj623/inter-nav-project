from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,'/data20t/embodied/zkj/inter-nav-results/r1lite-collision-deps')
import trimesh
R=Path(__file__).resolve().parent;aid=sys.argv[1]
def read(folder):
 d=np.load(R/folder/(aid+'.npz'));return trimesh.Trimesh(d['vertices'],d['faces'],process=False)
a=read('object_assets');b=read('strategy15_surface_proxies');np.random.seed(42)
out={}
for name,source,target in [('proxy_to_visual',b,a),('visual_to_proxy',a,b)]:
 pts=np.concatenate([source.vertices,trimesh.sample.sample_surface(source,5000)[0]])
 if len(pts)>20000:pts=pts[np.random.choice(len(pts),20000,replace=False)]
 target=target.copy();target.apply_scale(1000.);dist=[]
 for start in range(0,len(pts),500):dist.extend(trimesh.proximity.closest_point(target,pts[start:start+500]*1000.)[1]/1000.)
 out[name]=dict(samples=len(pts),max_m=float(max(dist)),p99_m=float(np.quantile(dist,.99)))
out['passed_sampled_check']=all(v['max_m']<.0008 for v in out.values() if isinstance(v,dict))
(R/'strategy15_surface_proxies'/(aid+'_check.json')).write_text(json.dumps(out,indent=2));print(out,flush=True)
assert out['passed_sampled_check']
