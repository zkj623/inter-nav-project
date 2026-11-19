from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
for p in sorted(R.glob('*validated_plans/bin_pedal.json')):
 d=json.loads(p.read_text());print(p.parent.name,len(d['plans']),[(x.get('mode'),x.get('strategy')) for x in d['plans'][:2]])
row=next(x for x in json.loads((R/'fixedpose_inventory.json').read_text()) if x['asset']=='bin_pedal')
print('CANDIDATE',json.dumps(row['plan']['candidate'],indent=2))
d=np.load(R/'object_assets/bin_pedal.npz');print('BOUNDS',d['vertices'].min(0),d['vertices'].max(0))
