from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'r1lite-collision-deps'),str(ROOT.parent/'r1lite-contact-geometry-final-20261110/research/rsi/household_catalog')]
from cooked_geometry import CookedGeometry,ConvexPart
from g2_kinematics import Kinematics
k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf')
def geometry(path):
    d=json.loads(Path(path).read_text());return CookedGeometry([ConvexPart(h['vertices_object_m'],f'hull{i}') for i,h in enumerate(d['hulls'])])
robot={r['name']:geometry(ROOT/'robot_assets'/(r['name']+'_cooked.json')) for r in json.loads((ROOT/'robot_assets/all_link_manifest.json').read_text())}
aid=sys.argv[1]
cache=ROOT/'object_assets'/(aid+'_cache_provenance.json')
path=json.loads(cache.read_text())['path'] if cache.exists() else ROOT/'object_assets'/(aid+'_cooked.json')
obj=geometry(path)
p=ROOT/'strategy15_chair_validated_plans'/(aid+'.json');doc=json.loads(p.read_text());accepted=[];rejected=[]
for index,plan in enumerate(doc['plans']):
    a,b=plan['q_pre'],plan['q_grasp'];names=set(a)|set(b)
    trajectory=[({n:(1-u)*a.get(n,0)+u*b.get(n,0) for n in names},0.,True) for u in np.linspace(0,1,5)]
    trajectory += [(q,.025*(i+1),False) for i,q in enumerate(plan['q_lift'])]
    if 'q_start' in plan:
        start=plan['q_start'];end=plan['q_pre'];trajectory=[({n:(1-u)*start[n]+u*end[n] for n in end},0.,True) for u in np.linspace(0,1,25)]+trajectory
    failure=None
    for step,(q,dz,check_fingers) in enumerate(trajectory):
        fk=k.fk(q,np.array(plan['base_pose']));position=np.array(plan['object_position'])+np.array([0,0,dz])
        for name,geom in robot.items():
            if not check_fingers and name.startswith('gripper_'):continue
            pose=fk[name].copy();pose[:3,3]-=position
            rot=Rotation.from_euler('z',plan.get('object_yaw_deg',0.),degrees=True).as_matrix();pose[:3,3]=rot.T@pose[:3,3];pose[:3,:3]=rot.T@pose[:3,:3]
            # Cheap conservative link-box pruning before exact convex-union query.
            center=geom.bounds.mean(0);extent=np.diff(geom.bounds,axis=0)[0]/2
            center=pose[:3,:3]@center+pose[:3,3];extent=np.abs(pose[:3,:3])@extent
            if np.any(center+extent<obj.bounds[0]) or np.any(center-extent>obj.bounds[1]):continue
            gap=obj.compound_gap(geom,pose)
            if gap<-.001:
                failure={'candidate':index,'original_candidate':plan['original_candidate'],'step':step,'link':name,'gap_m':gap};break
        if failure:break
    if failure:rejected.append(failure)
    else:accepted.append({**plan,'whole_robot_object_clearance_checked':True})
doc['plans']=accepted;doc['whole_robot_object_rejections']=rejected
p.write_text(json.dumps(doc,indent=2));print(aid,'object_clear',len(accepted),'rejected',rejected,flush=True)
