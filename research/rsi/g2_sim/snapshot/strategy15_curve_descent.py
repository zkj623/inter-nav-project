from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
from ideal_constraints import Envelope
R=Path(__file__).resolve().parent
ns={'__file__':str(R/'strategy15_tilt_audit_native_self.py')}
exec((R/'strategy15_tilt_audit_native_self.py').read_text().split('for aid in sys.argv[1:]:')[0],ns)
k=ns['k'];robot=ns['robot'];collision=ns['collision'];Envelope=Envelope()
aid=sys.argv[1];p=R/'strategy15_curve_validated_plans'/(aid+'.json');doc=json.loads(p.read_text())
cache=R/'object_assets'/(aid+'_cache_provenance.json')
path=Path(json.loads(cache.read_text())['path']) if cache.exists() else R/'object_assets'/(aid+'_cooked.json')
if aid=='stool_rectangular':path=R/'strategy15_surface_assets'/(aid+'_cooked.json')
d=json.loads(path.read_text());obj=ns['CookedGeometry']([ns['ConvexPart'](h['vertices_object_m'],str(i)) for i,h in enumerate(d['hulls'])])
def hit(q,plan):
    if not Envelope.check(q)['ok']:return 'posture'
    for n,v in q.items():
        if n in k.by_name and k.by_name[n]['kind']=='revolute' and not k.by_name[n]['lower']<=v<=k.by_name[n]['upper']:return 'joint_limit'
    if collision(q):return 'self'
    fk=k.fk(q,np.array(plan['base_pose']));rot=Rotation.from_euler('z',plan.get('object_yaw_deg',0),degrees=True).as_matrix()
    for name,g in robot.items():
        t=fk[name].copy();t[:3,3]=rot.T@(t[:3,3]-plan['object_position']);t[:3,:3]=rot.T@t[:3,:3]
        center=t[:3,:3]@g.bounds.mean(0)+t[:3,3];extent=np.abs(t[:3,:3])@((g.bounds[1]-g.bounds[0])/2)
        if np.any(center+extent<obj.bounds[0]) or np.any(center-extent>obj.bounds[1]):continue
        if obj.compound_gap(g,t)<-.001:return name
    return None
accepted=[];rejected=[]
for i,plan in enumerate(doc['plans'][:4]):
    end=plan['q_pre'];start={n:(0. if 'body_joint' in n else v) for n,v in end.items()};found=False
    for peak in (20.,10.,25.,0.):
        path=[];bad=None
        for u in np.linspace(0,1,25):
            q={n:(1-u)*start[n]+u*end[n] for n in end}
            q['idx03_body_joint3']-=np.radians(peak)*np.sin(np.pi*u)
            bad=hit(q,plan)
            if bad:break
            path.append(q)
        if not bad:
            plan.update(q_start=start,descent_waypoints=path,descent_curve_peak_deg=peak,descent_self_audit=[]);accepted.append(plan);found=True;print('CURVE_CLEAR',aid,i,peak,flush=True);break
    if not found:rejected.append(dict(candidate=i,last_failure=bad))
doc['plans']=accepted;doc['curve_descent_rejections']=rejected;p.write_text(json.dumps(doc,indent=2));print('CURVE_RESULT',aid,len(accepted),rejected,flush=True)
