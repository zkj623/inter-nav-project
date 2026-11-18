from pathlib import Path
import json,numpy as np
from pxr import Usd,UsdPhysics
R=Path(__file__).resolve().parent
for aid in ('stool_small','stool_rectangular','chair_dining','chair_wheeled'):
 p=R/('strategy15_surface_assets' if aid=='stool_rectangular' else 'object_assets')/(aid+'.usdc');s=Usd.Stage.Open(str(p))
 for prim in s.Traverse():
  if prim.HasAPI(UsdPhysics.MassAPI):
   a=UsdPhysics.MassAPI(prim);print(aid,str(prim.GetPath()),'mass',a.GetMassAttr().Get(),'com',a.GetCenterOfMassAttr().Get())
p=R.parent/'g2-iterate-20261118/strategy15_loadclock_trials/strategy15_clear_validated_plans/stool_small/candidate1_repeat1/samples.json'
ss=json.loads(p.read_text())
for phase in ('descent_hold','approach','close','contact_refine','lift','contact_settle','hold'):
 rows=[s for s in ss if s['phase']==phase]
 if rows:
  s=rows[-1];print(phase,s['t'],'lift',s['lift_m'],'tilt',s['tilt_deg'],'object',s['object_position'],'quat',s['object_quaternion'])
