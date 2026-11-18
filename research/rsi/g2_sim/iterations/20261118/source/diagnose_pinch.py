from pathlib import Path
import json,numpy as np,trimesh
R=Path(__file__).resolve().parent
for aid,pd,c in [('chair_wheeled','strategy15_lowpose_validated_plans',0),('stool_rectangular','strategy15_guard_validated_plans',1)]:
 p=R/'strategy15_forcefeature4_trials'/pd/aid/f'candidate{c}_repeat1'
 if not p.exists():p=R/'strategy15_forcefeature5_trials'/pd/aid/f'candidate{c}_repeat1'
 ss=json.loads((p/'samples.json').read_text());rows=[s for s in ss if s['phase']=='force_close']
 if not rows:continue
 s=rows[-1];print(aid,'prelift contact centers in object frame')
 data=np.load(R/'object_assets'/(aid+'.npz'));mesh=trimesh.Trimesh(data['vertices'],data['faces'],process=False);normals={}
 for w in s['contact_witnesses']:
  if w['link'].endswith('link4'):
   v=np.array(w['points_object']);center=v.mean(0);_,_,face=trimesh.proximity.closest_point(mesh,[center]);normals[w['link']]=mesh.face_normals[face[0]]
   print(w['link'],'n',len(v),'mean',np.round(center,4).tolist(),'normal',np.round(normals[w['link']],3).tolist())
 for side in ('l','r'):
  a=normals.get(f'gripper_{side}_inner_link4');b=normals.get(f'gripper_{side}_outer_link4')
  if a is not None and b is not None:print(side,'surface_normal_dot',round(float(a@b),3))
