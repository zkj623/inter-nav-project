from pathlib import Path
import ast
R=Path(__file__).resolve().parent
for label,yaw,opening in [('rear',180,-.6),('other',270,-.6),('narrow90',90,-.4)]:
 s=(R/'plan_caseyaw.py').read_text().replace('for yaw in (90,):',f'for yaw in ({yaw},):').replace('openq=-.6;',f'openq={opening};').replace("'case_handheld':(.65,.50,.78)","'case_handheld':(.78,.50,.65)").replace("out=ROOT/'caseyaw_raw_plans'",f"out=ROOT/'caseyaw_{label}_raw_plans'")
 ast.parse(s);(R/f'plan_caseyaw_{label}.py').write_text(s)
 s=(R/'connect_caseyaw.py').read_text().replace("R/'caseyaw_raw_plans/case_handheld.json'",f"R/'caseyaw_{label}_raw_plans/case_handheld.json'").replace("R/'caseyaw_plans'",f"R/'caseyaw_{label}_plans'").replace("f'caseyaw_planning_{a}.json'",f"f'caseyaw_{label}_planning_{{a}}.json'")
 (R/f'connect_caseyaw_{label}.py').write_text(s)
s=(R/'caseyaw_trial.py').read_text();needle='            samples.append(rec)'
s=s.replace(needle,needle+"\n            if step%120==0:(OUT/'progress.json').write_text(json.dumps(dict(t=rec['t'],phase=phase,lift_m=rec['lift_m'],tilt_deg=tilt,finger_forces=rec['finger_forces'])))")
ast.parse(s);(R/'caseyaw_live_trial.py').write_text(s)
