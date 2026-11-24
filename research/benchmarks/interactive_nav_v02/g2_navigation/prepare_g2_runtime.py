import sys,json,hashlib,os
from pathlib import Path
import numpy as np
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':int(os.environ.get('G2_GPU','6')),'physics_gpu':int(os.environ.get('G2_GPU','6')),'multi_gpu':False})
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,Gf
ROOT=Path(os.environ['G2_MODEL_ROOT']);sys.path.insert(0,str(ROOT))
from g2_kinematics import Kinematics,transform
OUT=Path(os.environ['G2_OUT']);OUT.mkdir(exist_ok=True)
scene=Path(os.environ['G2_SCENE_RUNTIME']);doc=json.loads((scene/'episodes.json').read_text());task=next(t for t in doc['tasks'] if t['id']==os.environ.get('G2_TASK_ID','S01-T09'))
# Only task start/goal enter the runner; reference path and evaluation grids are excluded.
spec={k:task[k] for k in ('id','start_position_m','start_yaw_rad','goal_position_m','goal_tolerance_m')};spec['scene']=doc['id'];center=np.round(np.asarray(spec['start_position_m'][:2])/10.)*10.;spec['map_limits']=[[float(v-15.),float(v+15.)] for v in center]
(OUT/'task.json').write_text(json.dumps(spec,indent=2))
from g2_standby_pose import load_standby,collision_footprint
k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf');config_path=Path(os.environ.get('G2_STANDBY_CONFIG',str(Path(__file__).with_name('standby_preparation_config.json'))));qpre,pose_source=load_standby(config_path,k)
(OUT/'standby_preparation_config.json').write_bytes(config_path.read_bytes())
(OUT/'pose-source.json').write_text(json.dumps(pose_source,indent=2))
stage=Usd.Stage.Open(str(ROOT/'robot_assets/g2_contact_inertia_fixed.usdc'));fk=k.fk(qpre,transform((*spec['start_position_m'][:2],.04),(0,0,spec['start_yaw_rad'])))
for p in stage.Traverse():
 if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in fk:UsdGeom.Xformable(p).MakeMatrixXform().Set(Gf.Matrix4d(fk[p.GetName()].T.tolist()))
adj={n:set() for n in k.links}
for j in k.joints:adj[j['parent']].add(j['child']);adj[j['child']].add(j['parent'])
for j in k.joints:
 if j['kind']=='fixed':
  a,b=j['parent'],j['child'];ns=(adj[a]|adj[b])-{a,b};adj[a]|=ns;adj[b]|=ns
  for n in ns:adj[n]|={a,b}
rigid=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
for i,p in enumerate(rigid):
 a=p.GetName();targets=[]
 for other in rigid[i+1:]:
  b=other.GetName();near=b in adj.get(a,set()) or any(b in adj.get(x,set()) for x in adj.get(a,set()))
  hand=any((a.startswith('gripper_'+s+'_') or a=='arm_'+s+'_link7') and (b.startswith('gripper_'+s+'_') or b=='arm_'+s+'_link7') for s in ('l','r'))
  head=(a.startswith('head_') or a=='body_link5') and (b.startswith('head_') or b=='body_link5')
  if near or hand or head:targets.append(other.GetPath())
 if targets:UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(targets)
joints={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
for p in stage.Traverse():
 j=k.by_name.get(p.GetName())
 if not j or not p.IsA(UsdPhysics.Joint):continue
 n=j['name'];m=j['mimic'];v=qpre.get(m['joint'],0)*float(m.get('multiplier',1))+float(m.get('offset',0)) if m else qpre.get(n,0.)
 drive=UsdPhysics.DriveAPI.Apply(p,'angular');wheel='wheel' in n and n.endswith('joint2');drive.CreateTypeAttr('force');drive.CreateMaxForceAttr(0 if m else j['effort']);drive.CreateStiffnessAttr(0 if m or wheel else 100000 if 'body_joint' in n else 1000);drive.CreateDampingAttr(0 if m else 5000 if 'body_joint' in n else 100);drive.CreateTargetPositionAttr(float(np.degrees(v)));drive.CreateTargetVelocityAttr(0.)
 state=PhysxSchema.JointStateAPI.Apply(p,'angular');state.GetPositionAttr().Set(float(np.degrees(v)));state.GetVelocityAttr().Set(0.)
 if m:
  for schema in list(p.GetAppliedSchemas()):
   if schema.startswith('PhysxMimicJointAPI:'):p.RemoveAPI(PhysxSchema.PhysxMimicJointAPI,schema.split(':',1)[1])
  mimic=PhysxSchema.PhysxMimicJointAPI.Apply(p,'rotX');mimic.GetReferenceJointRel().SetTargets([joints[m['joint']].GetPath()]);mimic.GetGearingAttr().Set(-float(m.get('multiplier',1)));mimic.GetOffsetAttr().Set(-np.degrees(float(m.get('offset',0))))
for p in list(stage.Traverse()):
 if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False:
  for schema in (UsdPhysics.CollisionAPI,UsdPhysics.MeshCollisionAPI,PhysxSchema.PhysxCollisionAPI,PhysxSchema.PhysxConvexDecompositionCollisionAPI,PhysxSchema.PhysxConvexHullCollisionAPI):
   if p.HasAPI(schema):p.RemoveAPI(schema)
 if p.GetTypeName()=='Shader':stage.RemovePrim(p.GetPath())
stage.GetRootLayer().Export(str(OUT/'robot.usdc'))
(OUT/'g2_collision_footprint.json').write_text(json.dumps(collision_footprint(stage,spec['start_position_m'][:2],spec['start_yaw_rad'],pose_source),indent=2))
world=Usd.Stage.CreateInMemory();world.GetRootLayer().subLayerPaths=[str(scene/'physics_only.usdc')];robot=world.DefinePrim('/robot','Xform');robot.GetReferences().AddReference(str(OUT/'robot.usdc'),'/robot');world.GetRootLayer().Export(str(OUT/'scene.usda'))
(OUT/'pose.json').write_text(json.dumps(qpre));print(spec,flush=True)
from g2_start_recovery import prepare_footprint
prepare_footprint(OUT)
app.close()
