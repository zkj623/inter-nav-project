"""Run one stage1-go2-simulation-release task from this checkout and relocated assets, without historic experiment directories."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
POLICY='8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8'
def main():
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=[f'T{i:02d}' for i in range(1,11)],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--record-video',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();out=a.output.resolve();assets=a.assets.resolve()
 if out.exists():raise SystemExit('Refusing existing output: '+str(out))
 for n in ['after_scene.usda','go2_training_flattened.usdc','asset-preparation.json']:
  if not (assets/n).is_file():raise SystemExit('Missing prepared asset: '+n)
 if hashlib.sha256((R/'delivery/model/actor_checkpoint.pt').read_bytes()).hexdigest()!=POLICY:raise SystemExit('Policy hash mismatch')
 out.mkdir(parents=True);inputs=out/'benchmark_inputs';shutil.copytree(R/'delivery/inputs'/a.task,inputs)
 profile=json.loads((inputs/'scene_profile.json').read_text());profile['scene_asset_path']=str(assets/'after_scene.usda');profile.pop('static_obstacle_metadata_path',None);(inputs/'scene_profile.json').write_text(json.dumps(profile,indent=2))
 policy=out/'policy';shutil.copytree(R/'delivery/model',policy)
 support=R/'delivery/runtime-support'
 args=[sys.executable,'-u',str(support/'capture_go2_policy_state_v193.py'),str(support/'contact_report_launch.py'),str(support/'launch_online_checked.py'),str(R/'grutopia/demo/go2_online_interactive_navigation.py'),'--benchmark-dir',str(inputs),'--output',str(out),'--task-ids',a.task,'--gpu',str(a.gpu),'--max-steps','40000','--policy-path',str(policy/'actor_checkpoint.pt'),'--robot-usd',str(assets/'go2_training_flattened.usdc'),'--robot-physics-profile','isaaclab_training','--physics-backend','gpu_dynamics','--actuator-mode','explicit_dc','--global-map-hz','1','--local-map-hz','10','--floor-top','.01','--map-source','online','--interaction-source','ground_truth','--path-controller','continuous','--material-mode','preview','--navigation-attitude-recovery','--navigation-diagnostics','--third-person-height','1.35','--third-person-forward','.25','--third-person-fov','100','--route-planner','voronoi','--navigation-heading','path','--roadmap-shape','circle','--route-clearance','.30','--navigation-half-width','.22','--voronoi-clearance-weight','.30','--push-start-speed','.60','--push-max-speed','.95','--turn-speed','1.0','--contact-centering-speed','.25','--contact-lateral-speed','.35','--contact-yaw-tolerance','.10','--contact-front-gap','.35']
 if a.record_video:args+=['--record-video','--record-every','1']
 (out/'command.json').write_text(json.dumps(args,indent=2));print('Output:',out,flush=True)
 if a.prepare_only:return
 env=os.environ.copy();env['PYTHONPATH']=str(R);env['MALLOC_ARENA_MAX']='2';env['GRUTOPIA_ASSETS_PATH']=str(assets)
 lock=Path(os.environ.get('GO2_REPRO_LOCK','/tmp/go2-simulation-reproduction.lock'))
 with lock.open('a') as lf:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
  stable=None
  while True:
   ram=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/1024**2
   gpu=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
   if ram>=56 and gpu>=14336:stable=stable or time.monotonic()
   else:stable=None
   if stable and time.monotonic()-stable>=60:break
   print('Waiting resources:',round(ram,1),'GiB RAM,',gpu,'MiB GPU',flush=True);time.sleep(10)
  scope=['systemd-run','--user','--scope','--quiet','--collect','--unit=codex-go2-repro-'+str(os.getpid()),'--property=MemoryMax=24G','--property=MemorySwapMax=0','--',sys.executable,str(R/'tools/scoped_go2_exec.py'),str(out),*args]
  with (out/'run.log').open('w') as log:result=subprocess.run(scope,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  summary={'exit_code':result.returncode,'task':a.task,'record_video':a.record_video}
  f=out/a.task/'result.json'
  if f.exists():summary['result']=json.loads(f.read_text())
  (out/'reproduction-process.json').write_text(json.dumps(summary,indent=2))
  if result.returncode:raise SystemExit(result.returncode)
  subprocess.run([sys.executable,str(R/'reproduction/check_result.py'),str(out),a.task],cwd=R,env=env,check=True)
if __name__=='__main__':main()
