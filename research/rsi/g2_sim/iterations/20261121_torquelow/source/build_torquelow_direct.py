from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'torquelow_single_trial.py').read_text().replace("ROOT/'torquelow_single_trials'","ROOT/'torquelow_direct_trials'");s=s.replace("    def retarget(nominal,command,extra=0.):","    def retarget(nominal,command,extra=0.):\n        if plan.get('mode')=='single_top':return nominal.copy()  # replay collision-audited joint/null-space path")
s=s.replace("        vals=vector(target)","        globals()['current_phase']=phase\n        vals=vector(target)");s=s.replace("dict(actor0=a,actor1=b,impulse=","dict(t=step/120,phase=globals().get('current_phase','initialization'),actor0=a,actor1=b,impulse=")
s=s.replace("            if environment_contacts and", "            if self_contacts and max(x['impulse'] for x in self_contacts)>.001:failure=failure or 'UNINTENDED_SELF_SUPPORT'\n            if environment_contacts and")
s=s.replace("seconds=max(seconds,need)*(1.75 if phase!='hold' else 1.)","seconds=max(seconds,need)*(3. if phase in ('lift','contact_settle') else 1.75 if phase!='hold' else 1.)")
s=s.replace("left-arm nominal IK, planned counterweight-arm commands, native gripper close; no object following or attachments","collision-audited joint/null-space replay for both arms; original-cap native gripper; no object following/attachment; self/environment support rejected")
s=s.replace("motion duration x1.75; 10fps video","motion x1.75; lift/settle x3 for strict50Nm;10fps video")
ast.parse(s);(R/'torquelow_single_direct_trial.py').write_text(s);print('STRICT50_DIRECT_SOURCE_READY')
