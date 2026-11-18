from pathlib import Path
import subprocess,sys,os
R=Path(__file__).resolve().parent;aid=sys.argv[1]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
py='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python'
subprocess.run([py,str(R/'plan_strategy15_chaircolumnshared.py'),'--asset',aid],cwd=R,env=env,check=True)
for name in ['validate_plans','audit_native_self','audit_object_clearance']:
    subprocess.run([py,str(R/f'strategy15_chaircolumnshared_{name}.py'),aid],cwd=R,env=env,check=True)
