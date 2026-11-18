from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
CAT=BASE/'r1lite-household-catalog20-20261028'
ids=[a['id'] for a in json.loads((CAT/'asset_manifest.json').read_text())['assets']]
roots=[BASE/n for n in ['r1lite-adaptive-retry-20261111','r1lite-remaining17-corrected-20261110','r1lite-continuous-retry-20261111','r1lite-new-assets-geometry-v2-20261108']]
roots += [BASE/'r1lite-contact-geometry-20261109',BASE/'r1lite-all20-lift-20261109']
files=[p for root in roots for p in root.rglob('cooked_object_geometry.json')]
out={}
for aid in ids:
    for p in sorted([p for p in files if any(part==aid or part.startswith(aid+'_v') for part in p.parts)],key=lambda p:p.stat().st_mtime,reverse=True):
        d=json.loads(p.read_text())
        if d.get('contact_representation_version')!=4 or not d['transform_validation']['valid_metric_transform']:continue
        out[aid]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),hulls=d['hull_count']);break
(ROOT/'object_cache_sources.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'found':len(out),'missing':[a for a in ids if a not in out],'hulls':{k:v['hulls'] for k,v in out.items()}},indent=2))
