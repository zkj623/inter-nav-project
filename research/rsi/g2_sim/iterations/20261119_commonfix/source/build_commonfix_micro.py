from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'commonfix_solver_case_handheld_trial.py').read_text().replace("ROOT/'commonfix_solver_trials'","ROOT/'commonfix_micro_trials'")
needle='        nominal_closed=closed.copy()'
patch="""        # Short forward/up load transfer before four-finger refinement.
        initial_forces=samples[-1]['finger_forces']
        if all(sum(v for n,v in initial_forces.items() if n.startswith(f'gripper_{side}_'))>1. for side in ('l','r')):
            shifted=closed.copy();base_now=mat(*robot.get_world_pose());fk_micro=k.fk(closed,np.array(plan['base_pose']))
            for side in ('l','r'):
                target=fk_micro[f'arm_{side}_end_link'].copy();target[:3,3]+=np.array([.02,0.,.01])
                shifted,pe,re=k.solve(side,target,shifted,base_now,np.eye(4))
                if pe>.002 or re>.02:raise TrialRejected('MICRO_LOAD_IK_UNREACHABLE')
            if not envelope.check(shifted)['ok']:raise TrialRejected('MICRO_LOAD_STATIC_LIMIT')
            segment(closed,shifted,3.,'micro_load');closed=shifted
            result['micro_load_transfer_m']=[.02,0.,.01]
"""
assert needle in s;s=s.replace(needle,patch+needle)
ast.parse(s);(R/'commonfix_micro_case_handheld_trial.py').write_text(s)
