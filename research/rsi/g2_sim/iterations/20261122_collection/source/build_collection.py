from pathlib import Path
import json,ast
R=Path(__file__).resolve().parent;cfg=json.loads((R/'standby_preparation_config.json').read_text());home=json.loads((R/'ground_lift_training_config.json').read_text())
for asset,e in cfg['execution'].items():
 s=(R/e['harness']).read_text(encoding='utf-8-sig');old="ROOT/'roundretry_trials'" if asset=='bin_round_lidded' else "ROOT/'directprep_trials'";assert old in s;s=s.replace(old,"ROOT/'collection_trials'")
 s=s.replace("plan=plan_doc['plans'][args.candidate] if plan_doc['plans'] else None", "plan=plan_doc['plans'][args.candidate] if plan_doc['plans'] else None\nfrom collection_recorder import Recorder\nrecorder=Recorder(OUT,"+repr(home['policy_action_joint_names'])+",dict(asset=args.asset,placement=plan.get('collection_placement',{}),plan_dir=args.plan_dir,plan_sha256=hashlib.sha256((ROOT/args.plan_dir/(args.asset+'.json')).read_bytes()).hexdigest(),baseline_harness="+repr(e['harness'])+",baseline_commit='52f4d1d1141f2634a4d1cfa26d3fe668e8cf149f'))")
 needle="camera.set_focal_length(24.);camera.set_horizontal_aperture(20.955)";assert needle in s
 s=s.replace(needle,needle+"\n    train_camera=camera;train_camera._frequency=-1\n    set_camera_view(eye=np.array([2.5,1.8,1.65]),target=np.array([.45,0.,.48]),camera_prim_path='/World/TrainingWorkspace');recorder.bind(train_camera)")
 needle='        vals=vector(target)';assert needle in s;s=s.replace(needle,'        target=recorder.command(target,phase,robot,world)\n'+needle)
 s=s.replace("G2 STRATEGY15 IDEAL",'G2 VARIED-PLACEMENT COLLECTION')
 needle="    result['max_frozen_joint_error_rad']=";s=s.replace(needle,"    result['collection_initialization']=plan.get('collection_initialization','full_standby_preparation')\n"+needle)
 needle="    (OUT/'result.json').write_text(json.dumps(result,indent=2))";assert needle in s;s=s.replace(needle,"    recorder.finish(result)\n"+needle)
 s=s.replace("resolution=(1280,720),frequency=30","resolution=(640,480)").replace("10,(1280,720)","10,(640,480)")
 s=s.replace("[3.4,3.4,2.3]","[2.5,1.8,1.65]").replace("[.1,.15,.95]","[.45,0.,.48]")
 s=s.replace("camera_prim_path='/World/TrainingWorkspace'","camera_prim_path='/World/GraspReview'")
 s=s.replace("if step%12==0:world.render()","if step%12==0 and recorder.ticks==0:world.render()")
 s=s.replace("recorder.finish(result)","recorder.finish(result)")
 s=s.replace("            samples.append(rec)","            rec['physics_time_s']=float(world.current_time)\n            samples.append(rec)")
 # Reduce idle/free-space time without changing contact/lift control or velocity bound computation.
 for phase in ('default_settle','initial_settle','settle'):
  s=s.replace("segment(qpre,qpre,2.,'"+phase+"')","segment(qpre,qpre,.5,'"+phase+"')")
 s=s.replace("segment(last,waypoint,6.,'common_approach')","segment(last,waypoint,6.,'common_approach')")
 s=s.replace("segment(q_grasp_pre,q_grasp_pre,2.,'pregrasp_hold')","segment(q_grasp_pre,q_grasp_pre,.75,'pregrasp_hold')")
 s=s.replace("    result['collection_initialization']=", "    result['collection_timing_profile']='shorter static settles only; original free-space/contact/lift durations and original velocity bound retained'\n    result['collection_initialization']=")
 s=s.replace("from isaacsim import SimulationApp", "sys.argv.extend(['--/plugins/carb.tasking.plugin/threadCount=16','--/plugins/omni.tbb.globalcontrol/maxThreadCount=16','--/persistent/physics/numThreads=16'])\nfrom isaacsim import SimulationApp")
 s=s.replace("    world.reset()", "    world.reset()\n    import carb\n    carb.settings.get_settings().set_bool('/app/runLoops/main/rateLimitEnabled',False)\n    result['execution_runtime']={'carb_task_workers':16,'tbb_workers':16,'physx_workers':16,'wall_clock_rate_limit':False,'physics_dt':1/120}")
 s=s.replace("from gravity_compensation import GravityCompensation","from fast_gravity_compensation import GravityCompensation")
 s=s.replace("    model_gravity=GravityCompensation()", "    model_gravity=GravityCompensation()\n    result['gravity_implementation_sha256']=model_gravity.implementation_sha256\n    result['gravity_equivalence']='102 pose tests vs archived scalar code; max error2.85e-13Nm; same URDF, mass, axes and torque caps'")
 ast.parse(s);(R/f'collect_{asset}_trial.py').write_text(s)
print('COLLECTION_SOURCES',len(cfg['execution']))







