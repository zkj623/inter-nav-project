from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_transfer_trials'")
s=s.replace('        stable=0\n        for refine in range(96):','        stable=0\n        transferred=False\n        for refine in range(96):')
needle="            forces=samples[-1]['finger_forces']\n            if plan.get('mode')=='bimanual_spread':"
patch="""            forces=samples[-1]['finger_forces']
            if not transferred and all(sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_'))>2. for side in ('l','r')):
                shifted_nominal=nominal_closed.copy();fk_transfer=k.fk(nominal_closed,np.array(plan['base_pose']))
                for side in ('l','r'):
                    target=fk_transfer[f'arm_{side}_end_link'].copy();target[:3,3]+=np.array([.02,0.,.01])
                    shifted_nominal,pe,re=k.solve(side,target,shifted_nominal,np.array(plan['base_pose']),np.eye(4))
                    if pe>.002 or re>.02:raise TrialRejected('LOAD_TRANSFER_IK_UNREACHABLE')
                if not envelope.check(shifted_nominal)['ok']:raise TrialRejected('LOAD_TRANSFER_STATIC_LIMIT')
                corrected=retarget(shifted_nominal,closed)
                segment(closed,corrected,3.,'load_transfer');closed=corrected;nominal_closed=shifted_nominal;transferred=True;stable=0
                result['load_transfer_m']=[.02,0.,.01];result['load_transfer_trigger']='first bilateral contact >2N each during refinement'
                forces=samples[-1]['finger_forces']
            if plan.get('mode')=='bimanual_spread':"""
assert needle in s;s=s.replace(needle,patch,1)
ast.parse(s);(R/'commonfix_transfer_case_handheld_trial.py').write_text(s)
