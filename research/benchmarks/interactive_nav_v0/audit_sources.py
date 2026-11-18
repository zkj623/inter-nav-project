import argparse,json,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--upstream-manifest',type=Path,required=True);ap.add_argument('--objects',type=Path,required=True);ap.add_argument('--home',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
expected={x['path']:x['sha256'] for x in json.loads(a.upstream_manifest.read_text())};ids=json.loads((a.out/'object_catalog.json').read_text())['active_assets'];rows=[]
for aid in ids:
 for suffix in ('.usdc','.npz'):
  name=aid+suffix;sha=hashlib.sha256((a.objects/name).read_bytes()).hexdigest();ref=expected.get('object_assets/'+name);rows.append({'file':name,'sha256':sha,'upstream_sha256':ref,'matches_upstream':sha==ref})
scenes={}
for p in a.out.glob('S*/episodes.json'):
 d=json.loads(p.read_text());scenes[d['scene']]=hashlib.sha256((a.home/d['scene']).read_bytes()).hexdigest()
report={'files':rows,'all_match_upstream':all(r['matches_upstream'] for r in rows),'scene_entry_sha256':scenes,'scope':'entry USD and object meshes; not a complete dependency lock'};(a.out/'source_asset_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
