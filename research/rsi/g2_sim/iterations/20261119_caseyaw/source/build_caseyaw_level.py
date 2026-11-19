from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'caseyaw_live_trial.py').read_text().replace("ROOT/'caseyaw_trials'","ROOT/'caseyaw_level_trials'")
diag=(R/'caseyaw_contactdiag_trial.py').read_text();block=diag[diag.index('            diagnostic_contacts={}'):diag.index('            contact_witnesses=[]')]
s=s.replace('            contact_witnesses=[]',block+'            contact_witnesses=[]')
s=s.replace("            rec['gravity_torques_nm']=","            rec['diagnostic_contacts']=diagnostic_contacts\n            rec['gravity_torques_nm']=")
s=s.replace("    balance = {'l': 0., 'r': 0.}","    balance = {'l': 0., 'r': 0.}\n    level_vector=np.zeros(3);last_level_time=None")
s=s.replace('    def retarget(nominal,command,extra=0.):','    def retarget(nominal,command,extra=0.):\n        global level_vector,last_level_time')
needle='        delta=np.eye(3)  # Maintain the planned upright support planes; avoid following object tipping'
level="""        measured_rotation=Rotation.from_quat(np.array(obj.get_world_pose()[1])[[1,2,3,0]]).as_matrix()
        orientation_error=Rotation.from_matrix(measured_rotation@reference_rotation.T).as_rotvec();orientation_error[2]=0.
        wanted=-.4*orientation_error;angle=np.linalg.norm(wanted)
        if angle>np.radians(8):wanted*=np.radians(8)/angle
        now_level=step/120.;dt_level=.25 if last_level_time is None else min(.5,max(.01,now_level-last_level_time));last_level_time=now_level
        change=wanted-level_vector;size=np.linalg.norm(change);limit=np.radians(1.)*dt_level
        if size>limit:change*=limit/size
        level_vector+=change;delta=Rotation.from_rotvec(level_vector).as_matrix()"""
assert needle in s;s=s.replace(needle,level)
s=s.replace('# Read object orientation only: object position is never imposed or followed.','# Read object orientation for bounded opposite tilt feedback; no object state is written.')
s=s.replace('        stable=0\n        for refine in range(96):','        stable=0\n        pressure_stable=0;test_lift_done=False\n        for refine in range(96):')
start=s.index("            if plan.get('mode')=='bimanual_spread':\n                achieved=",s.index('        for refine in range(96):'));end=s.index('            stable=stable+1 if achieved else 0',start)
s=s[:start]+"""            bilateral=all(sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_'))>12. for side in ('l','r'))
            pressure_stable=pressure_stable+1 if bilateral else 0
            if not test_lift_done and pressure_stable>=3:
                raised=nominal_closed.copy();fk_test=k.fk(nominal_closed,np.array(plan['base_pose']))
                for side in ('l','r'):
                    target=fk_test[f'arm_{side}_end_link'].copy();target[2,3]+=.015
                    raised,pe,re=k.solve(side,target,raised,np.array(plan['base_pose']),np.eye(4))
                    if pe>.002 or re>.02:raise TrialRejected('TEST_LIFT_IK_UNREACHABLE')
                if not envelope.check(raised)['ok']:raise TrialRejected('TEST_LIFT_STATIC_LIMIT')
                corrected=retarget(raised,closed);segment(closed,corrected,3.,'test_lift');closed=corrected;nominal_closed=raised;test_lift_done=True
                result['test_lift_m']=.015;forces=samples[-1]['finger_forces']
                bilateral=all(sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_'))>12. for side in ('l','r'))
            available=samples[-1]['diagnostic_contacts']
            achieved=test_lift_done and bilateral and len(available)==4 and all(x['available'] for x in available.values())
"""+s[end:]
s=s.replace("'FOUR_FINGER_PRELOAD_NOT_ESTABLISHED'","'FOUR_VISIBLE_CONTACT_CAPTURE_NOT_ESTABLISHED'")
s=s.replace('ready=four_contact(recent,1.) and velocity<.015',"ready=len(recent['diagnostic_contacts'])==4 and all(x['available'] for x in recent['diagnostic_contacts'].values()) and recent['tilt_deg']<=20. and velocity<.015")
s=s.replace("    result['contact_definition']=","    result['capture_gate']='bilateral >12N for 3 updates, 15mm physical test lift, four actual contact manifolds within original visible tolerances; final hold still requires original four positive forces >=90%'\n    result['orientation_feedback']='opposite measured tilt gain0.4, total<=8deg, rate<=1deg/s; no object position control or attachment'\n    result['contact_definition']=")
ast.parse(s);(R/'caseyaw_level_trial.py').write_text(s)
