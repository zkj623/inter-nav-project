from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'plan_strategy15_frame.py').read_text()
s=s.replace('from audit_self_collision import intersections','from audit_self_collision import intersections\nfrom liftall_collision import check')
s=s.replace('max_plans=6','max_plans=3').replace('deadline=started+420.','deadline=started+600.')
start=s.index('    profiles=');end=s.index('    armnames=',start)
s=s[:start]+"    profiles=[json.loads((ROOT/'ground_lift_training_config.json').read_text())['fixed_joint_positions_rad']]\n"+s[end:]
s=s.replace('        nonlocal attempts','        nonlocal attempts\n        allow_body=False',1)
s=s.replace('>.016 or np.linalg.norm(r[i+3:i+6])>.04','>.008 or np.linalg.norm(r[i+3:i+6])>.02')
s=s.replace("'case_handheld':(.45,.92)","'case_handheld':(.72,1.)")
s=s.replace('for gap in (.025,.08):','for gap in (.06,.09,.03):')
s=s.replace('        return dict(asset=aid,mode=mode,','        result=dict(asset=aid,mode=mode,')
needle="scope='fixed torso; arms-only IK/lift; initial stance search; no navigation or object attachment')"
s=s.replace(needle,needle+"\n        result['object_yaw_deg']=float(yaw)\n        bad=check(aid,result)\n        if bad:reject['native_'+bad['kind']]+=1;return None\n        return result")
s=s.replace("out=ROOT/'strategy15_frame_plans'","out=ROOT/'liftfix_feature_plans'")
ast.parse(s);(R/'plan_liftfix_feature.py').write_text(s)
