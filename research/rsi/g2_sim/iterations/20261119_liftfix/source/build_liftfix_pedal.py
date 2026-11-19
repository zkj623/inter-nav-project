from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'plan_strategy15_tilt.py').read_text()
s=s.replace('from audit_self_collision import intersections','from audit_self_collision import intersections\nfrom liftall_collision import check')
s=s.replace('max_plans=6','max_plans=4').replace('deadline=started+360.','deadline=started+900.')
start=s.index('    profiles=');end=s.index('    armnames=',start)
s=s[:start]+"    profiles=[json.loads((ROOT/'ground_lift_training_config.json').read_text())['fixed_joint_positions_rad']]\n"+s[end:]
s=s.replace('        nonlocal attempts','        nonlocal attempts\n        allow_body=False',1)
s=s.replace('>.016 or np.linalg.norm(r[i+3:i+6])>.04','>.008 or np.linalg.norm(r[i+3:i+6])>.02')
s=s.replace('for theta in (35,50,20):','for theta in (35,20,50,65):')
s=s.replace("fractions.get(aid,(.75,.60,.45))","fractions.get(aid,(.85,.95,.70,.55))")
s=s.replace('for distance in (-.06,.0,.06):','for distance in (.06,.09,.03,.12,0.,-.03):')
s=s.replace('        return dict(asset=aid,mode=mode,','        result=dict(asset=aid,mode=mode,')
needle="scope='fixed torso; arms-only IK/lift; initial stance search; no navigation or object attachment')"
assert needle in s
s=s.replace(needle,needle+"\n        result['object_yaw_deg']=candidate['object_yaw_deg']\n        bad=check(aid,result)\n        if bad:reject['native_'+bad['kind']]+=1;return None\n        return result")
s=s.replace("out=ROOT/'strategy15_tilt_plans'","out=ROOT/'liftfix_pedal_plans'")
ast.parse(s);(R/'plan_liftfix_pedal.py').write_text(s)
