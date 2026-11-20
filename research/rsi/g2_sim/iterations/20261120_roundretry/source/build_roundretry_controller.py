from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'directprep_white_validation_trial.py').read_text();s=s.replace("ROOT/'directprep_trials'","ROOT/'roundretry_trials'")
s=s.replace("if max(loads)<6.:squeeze[side]=min(.012,squeeze[side]+.0003)\n                elif max(loads)>35.:squeeze[side]=max(0.,squeeze[side]-.0003)","feedback_dt=.4375\n                force_error=28.-sum(loads)\n                rate=0. if abs(force_error)<2. else float(np.clip(force_error*.00008,-.0008,.0008))\n                if max(loads)>35.:rate=-.0008\n                squeeze[side]=float(np.clip(squeeze[side]+rate*feedback_dt,0.,.020))")
s=s.replace("if min(loads)<2. and max(loads)>4.:\n                    balance[side]=float(np.clip(balance[side]+np.radians(.08)*np.sign(loads[1]-loads[0]),-np.radians(8),np.radians(8)))","if sum(loads)>4. and abs(loads[1]-loads[0])>.5:\n                    imbalance=(loads[1]-loads[0])/max(4.,sum(loads))\n                    balance[side]=float(np.clip(balance[side]+np.radians(.4)*feedback_dt*imbalance,-np.radians(8),np.radians(8)))")
s=s.replace("center_world=target[:3,3]+target[:3,:3]@center","center_world=target[:3,3]+target[:3,:3]@(center-tcp[:3,3])")
s=s.replace("target[:3,3]+=squeeze[side]*normal","target[:3,3]+=squeeze[side]*normal\n                lateral=np.array(obj.get_world_pose()[0])-np.array(plan['object_position']);lateral[2]=0.\n                target[:3,3]+=np.clip(lateral,-.02,.02)")
s=s.replace("reference_closed =", "reference_closed =")
s=s.replace("    result['contact_definition']=", "    result['roundretry_controller']='correct wrist contact pivot (remove duplicated TCP offset);28N/hand damped force; normalized0.4deg/s finger load balance; bounded20mm measured-object XY follow; original acceptance retained'\n    result['contact_definition']=")
ast.parse(s);(R/'roundretry_trial.py').write_text(s);print('CONTROLLER_BUILT',flush=True)
