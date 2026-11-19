from pathlib import Path
import ast
R=Path(__file__).resolve().parent
s=(R/'fixedpose_supportv2_trial.py').read_text().replace("'fixedpose_supportv2_trials'","'fixedpose_supportstable_trials'")
old="""            loads=[sum(v for n,v in forces.items() if n.startswith(f'gripper_{side}_{finger}_')) for finger in ('inner','outer')]
            if max(loads)>.5:
                balance[side]=float(np.clip(balance[side]+np.radians(.15)*(loads[1]-loads[0])/(sum(loads)+1.),-np.radians(4),np.radians(4)))"""
new="""            loads=[float(np.mean([sum(v for n,v in s['finger_forces'].items() if n.startswith(f'gripper_{side}_{finger}_')) for s in samples[-3:]])) for finger in ('inner','outer')]
            # Preserve established two-branch support instead of chasing equality.
            if max(loads)>.5 and min(loads)<.75:
                balance[side]=float(np.clip(balance[side]+np.radians(.05)*(loads[1]-loads[0])/(sum(loads)+1.),-np.radians(4),np.radians(4)))"""
assert old in s;s=s.replace(old,new).replace('for _ in range(48):','for _ in range(96):')
ast.parse(s);(R/'fixedpose_supportstable_trial.py').write_text(s)
