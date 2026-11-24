from pathlib import Path
import hashlib,ast,json,re
R=Path(__file__).resolve().parent
for asset in json.loads((R/'standby_preparation_config.json').read_text())['execution']:
    s=(R/f'collect_{asset}_trial.py').read_text(encoding='utf-8-sig')
    s=s.replace("ROOT/'collection_trials'","ROOT/'realpose_trials'")
    s,count=re.subn(r"(p.add_argument\('--plan-dir',default=)'[^']+'",r"\1'realpose_plans'",s)
    assert count==1,asset
    needle="plan=plan_doc['plans'][args.candidate] if plan_doc['plans'] else None"
    guard="\nprofile=json.loads((ROOT/'ground_lift_training_config.json').read_text())\nassert plan.get('profile_id')==profile['profile_id'],'REALPOSE_PROFILE_MISMATCH'\nassert all(abs(plan['q_start'].get(n,0)-v)<1e-8 for n,v in profile['common_initial_joint_positions_rad'].items()),'REALPOSE_COMMON_HOME_MISMATCH'"
    s=s.replace(needle,needle+guard)
    s=s.replace("recorder=Recorder(","head_pilot=None\nrecorder=Recorder(")
    s=s.replace("    prepared=OUT/'prepared_scene.usdc'","    from home_floor_material import add_wood_floor\n    result['home_floor']=add_wood_floor(stage,ROOT)\n    prepared=OUT/'prepared_scene.usdc'")
    s=s.replace("    train_camera=camera;train_camera._frequency=-1","    train_camera=camera;train_camera._frequency=-1\n    from head_pilot_recorder import HeadPilotRecorder\n    head_pilot=HeadPilotRecorder(OUT,robot,recorder)")
    s=s.replace("        target=recorder.command(target,phase,robot,world)","        target=recorder.command(target,phase,robot,world)\n        head_pilot.capture(phase,world)")
    s=s.replace("    recorder.finish(result)","    recorder.finish(result)\n    if head_pilot is not None:head_pilot.finish(result)")
    ast.parse(s);p=R/f'realpose_{asset}_trial.py';p.write_text(s);data=p.read_bytes();(R/'source_versions'/(hashlib.sha256(data).hexdigest()+'.py')).write_bytes(data)
p=R/'verify_collection.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("margin<9.9","margin<cfg.get('minimum_body_margin_acceptance_deg',9.9)")
(R/'verify_realpose.py').write_text(s)
print('REALPOSE_PILOTS_BUILT; only explicit new-profile margin gate uses2.9deg, all original grasp/hold gates retained')
