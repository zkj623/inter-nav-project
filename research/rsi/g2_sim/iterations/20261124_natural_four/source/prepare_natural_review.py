from pathlib import Path
import json
R=Path(__file__).resolve().parent;selected=json.loads((R/'natural_selected.json').read_text());rows=[]
for a,path in selected.items():
 T=Path(path);res=json.loads((T/'result.json').read_text());ss=json.loads((T/'samples.json').read_text());v=json.loads((T/'verification.json').read_text());pose=json.loads((T/'posture_verification.json').read_text());assert v['verified_success'] and pose['natural_verified']
 samples={}
 for phase in ['contact_refine','hold']:
  s=[x for x in ss if x['phase']==phase]
  if s:samples[phase]=s[len(s)//2]
 rows.append({'asset':a,'trial':str(T),'plan':res['plan'],'render_samples':samples})
(R/'all_posture_audit.json').write_text(json.dumps(rows,indent=2))
s=(R.parent/'g2-basket-posture-20261124/render_all_posture_preview.py').read_text();s=s.replace("assets=list(cfg['validated_simulation_baselines'])","assets=list(json.loads((R/'natural_selected.json').read_text()))").replace("O=R/'all_posture_preview'","O=R/'natural_posture_preview'")
(R/'render_natural_review.py').write_text(s)
