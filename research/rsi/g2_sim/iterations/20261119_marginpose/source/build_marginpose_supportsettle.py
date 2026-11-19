from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'marginpose_support_trial.py').read_text().replace("'marginpose_support_trials'","'marginpose_supportsettle_trials'")
old="""        for _ in range(8):
            corrected=retarget(nominal,last);segment(last,corrected,.2,'support_settle');last=corrected
        segment(last,last,3.,'hold')"""
new="""        settled=0
        for _ in range(60):
            corrected=retarget(nominal,last);segment(last,corrected,.2,'support_settle');last=corrected
            recent=samples[-1];previous=samples[-2]
            speed=np.linalg.norm(np.array(recent['object_position'])-previous['object_position'])/(recent['t']-previous['t'])
            forces=recent['finger_forces']
            ready=all(sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_{finger}_'))>.75 for side in ('l','r') for finger in ('inner','outer')) and speed<.01
            settled=settled+1 if ready else 0
            if settled>=4:break
        result['settled_support_updates']=settled
        if settled<4:raise TrialRejected('NO_SETTLED_FOUR_BRANCH_SUPPORT')
        segment(last,last,3.,'hold')"""
assert old in s;s=s.replace(old,new);ast.parse(s);(R/'marginpose_supportsettle_trial.py').write_text(s)
