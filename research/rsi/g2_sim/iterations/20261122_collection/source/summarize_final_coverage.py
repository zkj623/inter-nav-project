from pathlib import Path
import json,collections
R=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (R/'dataset/manifest.jsonl').read_text().splitlines()]
for a in sorted({x['asset'] for x in rows}):
 s=[x for x in rows if x['asset']==a]
 print(a,'dx_mm', [round(fn(x['dx_m'] for x in s)*1000,2) for fn in (min,max)],'dy_mm',[round(fn(x['dy_m'] for x in s)*1000,2) for fn in (min,max)],'yaw_deg',[round(fn(x['dyaw_deg'] for x in s),2) for fn in (min,max)])
print('TOTAL_FRAMES',sum(x['frames'] for x in rows),'SPLITS',dict(collections.Counter(x['split'] for x in rows)))
