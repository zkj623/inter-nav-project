from pathlib import Path
import argparse,json,cv2
W=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();summary=json.loads((W/'evaluations'/a.run/'summary.json').read_text());out=[]
for row in summary['results']:
 T=Path(row['trial']);r=json.loads((T/'result.json').read_text());report={k:v for k,v in r.items() if any(s in k for s in ['nonhand','failure','max_lift'])};report['asset']=row['asset'];out.append(report)
 cap=cv2.VideoCapture(str(T/'review.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[]
 for fraction in [.25,.5,.75,.95]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,int((n-1)*fraction));ok,im=cap.read()
  if ok:tiles.append(cv2.resize(im,(640,360)))
 cap.release()
 if len(tiles)==4:
  import numpy as np
  cv2.imwrite(str(W/(a.run+'_'+row['asset']+'_review.jpg')),np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)]))
(W/'evaluations'/a.run/'failure_details.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2)[:12000])
