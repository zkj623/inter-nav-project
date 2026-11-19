from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'caseyaw_live_trial.py').read_text()
s=s.replace("ROOT/'caseyaw_trials'","ROOT/'caseyaw_diag_trials'")
needle='            contact_witnesses=[]'
patch="""            diagnostic_contacts={}
            for i,n in enumerate(fname):
                if not n.endswith('_link4'):continue
                ob=fcl.CollisionObject(oshape,fcl.Transform(ot[:3,:3],ot[:3,3]));fb=fcl.CollisionObject(fshapes[n],fcl.Transform(ft[i][:3,:3],ft[i][:3,3]))
                gap=max(0.,float(fcl.distance(fb,ob,fcl.DistanceRequest(),fcl.DistanceResult())))
                pts=[]
                for col in range(counts.shape[1]):
                    start=int(starts[i,col]);count=int(counts[i,col]);pts.extend(points[start:start+count].tolist())
                error=999.
                if pts:
                    pts=np.array(pts);oi=(pts-ot[:3,3])@ot[:3,:3];fi=(pts-ft[i][:3,3])@ft[i][:3,:3]
                    _,od,_=trimesh.proximity.closest_point(omesh_mm,oi*1000.);_,fd,_=trimesh.proximity.closest_point(fmeshes_mm[n],fi*1000.)
                    error=float(max(od.max(),fd.max())/1000.)
                diagnostic_contacts[n]=dict(gap_m=gap,contact_count=len(pts),max_visible_witness_error_m=error,available=bool(gap<=.0015 and len(pts)>0 and error<=.0025))
"""
assert needle in s;s=s.replace(needle,patch+needle)
s=s.replace("            rec['gravity_torques_nm']=","            rec['diagnostic_contacts']=diagnostic_contacts\n            rec['gravity_torques_nm']=")
s=s.replace("if plan.get('mode')=='bimanual_spread' and stable<3:raise TrialRejected('FOUR_FINGER_PRELOAD_NOT_ESTABLISHED')","if plan.get('mode')=='bimanual_spread' and stable<3:result['diagnostic_preload_force_gate_failed']=True")
s=s.replace('ready=four_contact(recent,1.) and velocity<.015','ready=all(x[\'available\'] for x in recent[\'diagnostic_contacts\'].values()) and velocity<.015')
needle="    result['max_frozen_joint_error_rad']="
idx=s.index(needle)
s=s[:idx]+"    result['diagnostic_only']=True\n    result['diagnostic_note']='force-preload gate bypassed to test actual contact availability; original final four-force acceptance retained; not an accepted baseline unless original success also passes'\n    hold_diag=[x for x in samples if x['phase']=='hold']\n    result['diagnostic_four_available_fraction']=sum(all(v['available'] for v in x['diagnostic_contacts'].values()) for x in hold_diag)/max(1,len(hold_diag))\n"+s[idx:]
ast.parse(s);(R/'caseyaw_contactdiag_trial.py').write_text(s)
