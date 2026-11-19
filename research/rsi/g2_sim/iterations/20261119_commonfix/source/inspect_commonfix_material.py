from pathlib import Path
from pxr import Usd
R=Path(__file__).resolve().parent
for f in [R/'robot_assets/g2_contact_inertia_fixed.usdc']+list(R.glob('commonfix_precise_trials/**/prepared_scene.usdc')):
 stage=Usd.Stage.Open(str(f));print('SCENE',f,flush=True)
 for p in stage.Traverse():
  if 'gripper_l_outer_link4' not in str(p.GetPath()):continue
  rels=[(x.GetName(),[str(v) for v in x.GetTargets()]) for x in p.GetRelationships() if 'material' in x.GetName()]
  props=[(x.GetName(),str(x.Get())) for x in p.GetAttributes() if 'friction' in x.GetName().lower()]
  if rels or props:print(str(p.GetPath()),rels,props,flush=True)
