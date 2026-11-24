from pathlib import Path
import json,subprocess,sys,concurrent.futures
R=Path(__file__).resolve().parent;assets=list(json.loads((R/'standby_preparation_config.json').read_text())['execution'])
def run(a):
 p=subprocess.run([sys.executable,str(R/'plan_realpose.py'),'--asset',a],cwd=R,capture_output=True,text=True);(R/'realpose_plans').mkdir(exist_ok=True);(R/'realpose_plans'/(a+'_planner.log')).write_text(p.stdout+p.stderr);print(a,p.returncode,p.stdout[-500:] or p.stderr[-500:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,assets))
rows=[json.loads(p.read_text()) for p in (R/'realpose_plans').glob('*_planning.json')];(R/'realpose_planning_summary.json').write_text(json.dumps(rows,indent=2));print('PLANNING_DONE',sum(r['success'] for r in rows),len(rows),flush=True)
