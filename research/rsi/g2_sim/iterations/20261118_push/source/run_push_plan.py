from pathlib import Path
import os,subprocess,sys
R=Path(__file__).resolve().parent;aid=sys.argv[1];py='/data20t/embodied/share/inter-nav/envs/isaaclab/bin/python';env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for cmd in [['plan_strategy15_push.py','--asset',aid],['strategy15_push_validate_plans.py',aid],['strategy15_push_audit_native_self.py',aid]]:subprocess.run([py,str(R/cmd[0]),*cmd[1:]],cwd=R,env=env,check=True)
print('PUSH_AUDITS_COMPLETE',aid,flush=True)
