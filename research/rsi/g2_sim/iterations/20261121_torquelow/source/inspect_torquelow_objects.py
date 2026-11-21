from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent
for a in ['basket_woven','basket_rectangular','case_handheld','bin_lidded','stool_wood_four_leg']:
 p=json.loads((R/'directprep_plans'/f'{a}.json').read_text())['plans'][0];d=np.load(R/'object_assets'/f'{a}.npz');v=d['vertices'];print(a,'shape',v.shape,d['faces'].shape,'bounds',v.min(0),v.max(0),'mode',p['mode'],'candidate',p['candidate'],flush=True)
