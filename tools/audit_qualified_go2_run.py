"""Audit one recorded qualified task against its frozen definition, not old T04/T05 coordinates."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
from tools.audit_actual_passage import crossing_evidence,final_stop_evidence,yaw_tilt


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def audit(root,task,qualification_path,expected_commit):
    folder=root/task;qualification=read(qualification_path)
    definition=next(t for t in qualification['qualified_candidates'] if t['id']==task)
    result=read(folder/'result.json');config=read(root/'recording_config.json');model=read(root/'robot_model_contract.json')
    source=read(root/'source_version_audit.json');manifest=read(root/'source_manifest.json');binding=read(root/'qualified_input_manifest.json')
    video=read(folder/'video_qa.json');coverage=read(folder/'recording_coverage_qa.json')
    events=rows(folder/'events.jsonl');control=rows(folder/'control_trace.jsonl')
    policy=read(root/'policy_source_snapshot.json')['files']
    checks={
        'difficulty_tortuosity_strict_gt_1_3':definition['tortuosity'] is not None and math.isfinite(definition['tortuosity']) and definition['tortuosity']>1.3,
        'difficulty_clutter_strict_gt_020':qualification['clutter']['fraction']>.20,
        'robot_connectivity_screen':definition['robot_clearance_grid_path_m'] is not None,
        'frozen_qualification_hash':binding['qualification_sha256']==sha(qualification_path),
        'frozen_evaluation_map_hash':binding['evaluation_map_sha256']==sha(qualification_path.parent/'evaluation_map.npz'),
        'input_snapshot_hashes':all(sha(root/'benchmark_inputs'/name)==value for name,value in binding['benchmark_files'].items()),
        'scene_and_layout_unchanged':all(binding['benchmark_files'][name]==qualification['inputs'][name] for name in ['scene_profile.json','layout.json']),
        'declared_task_coordinates':all(np.allclose(result[k],definition[k],atol=1e-8,rtol=0) for k in ['start','goal']),
        'automatic_task_success':result['success'] is True and result['reason']=='goal_reached' and not (root/'STOP').exists() and not (root/'operator_stop_reason.json').exists(),
        'goal_error_within_user_050m':float(np.linalg.norm(np.asarray(result['final_position'][:2])-definition['goal']))<=.50,
        'scene_scale_one':config['scene_scale']==[1,1,1],
        'robot_scale_one':np.allclose(model['robot_scale'],[1,1,1],atol=1e-8,rtol=0),
        'floor_one_cm':abs(result['ground_height']-.01)<1e-8,
        'no_teleports_flag':result['teleports_during_rollout'] is False,
        'rollout_health':result['rollout_health']['healthy'] and result['maximum_tilt_degrees']<30,
        'model_contract':model['training_contract_valid'] and all(model['training_contract_checks'].values()),
        'robot_geometry':model['body_count']==19 and model['collider_count']==27 and abs(model['total_mass_kg']-15.019)<.01,
        'robot_asset_hash':model['robot_asset_sha256']=='75958c8286d2dfdf7617108a685110ac831dfc47f2546e71a8cc2ccaecbd83a5',
        'policy_hash':model['policy_sha256']=='8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8',
        'policy_files':set(policy)=={'actor_checkpoint.pt','deployment_contract.json'} and all(sha(root/'policy'/name)==digest for name,digest in policy.items()),
        'physics_200hz_dc':config['physics_dt']==.005 and config['actuator_mode']==model['actuator_mode']=='explicit_dc',
        'online_map_1_10hz':config['map_source']=='online' and config['global_map_hz']==1 and config['local_map_hz']==10,
        'source_version':source['valid'] and source['commit']==expected_commit and set(source['files'])==set(manifest),
        'source_snapshot':bool(manifest) and all(sha(root/'source'/name)==digest for name,digest in manifest.items()),
        'full_media_decode':video['valid'],
        'recording_coverage_and_hold':coverage['valid_final_recording_coverage'],
    }
    with np.load(folder/'initial_unknown/layered_map.npz') as initial:
        checks['empty_initial_map']=not initial['observed'].any() and not initial['static'].any() and not initial['global_observed'].any()
    guard=read(root/'online_input_audit.json')
    checks['no_offline_map_reads']=guard['guard_armed'] and not guard['prior_read_attempts'] and {'qualification.json','evaluation_map.npz'}.issubset(guard['scene_prior_reads_forbidden'])
    startup=next(e for e in events if e['event']=='startup')
    checks['startup_at_declared_start']=np.linalg.norm(np.asarray(startup['position'][:2])-definition['start'])<.08 and startup['tilt_degrees']<15
    layout=read(root/'benchmark_inputs/layout.json')['objects'];initial_boxes={b['id']:b for b in startup['boxes']}
    checks['initial_box_layout_matches']=all(o['id'] in initial_boxes and np.linalg.norm(np.asarray(initial_boxes[o['id']]['xy'])-o['position'][:2])<.02 for o in layout)
    physics_times=[];startup_steps=0;in_start=True
    for r in control:
        count=int(r['physics_substeps']);physics_times.extend(r['command_time_s']+(j+1)*.005 for j in range(count))
        if r['phase'] not in ['start','map_warmup']:in_start=False
        if in_start:startup_steps+=count
    with np.load(qualification_path.parent/'evaluation_map.npz') as m:
        support=m['support'];static_occupied=m['static_occupied'];origin=m['origin'];resolution=float(m['resolution'])
    raw=[r for r in rows(root/'control_trace.jsonl') if r.get('task')==task]
    max_tilt=0.;startup_max=0.;height=math.inf;max_jump=0.;outside=0;static_crossings=[];previous=None
    for i,r in enumerate(raw):
        _,tilt=yaw_tilt(r['orientation']);max_tilt=max(max_tilt,tilt)
        if i<startup_steps:startup_max=max(startup_max,tilt)
        pos=np.asarray(r['position']);height=min(height,float(pos[2]))
        if previous is not None:max_jump=max(max_jump,float(np.linalg.norm(pos[:2]-previous[:2])))
        previous=pos
        col,row=np.floor((pos[:2]-origin)/resolution).astype(int)
        if not(0<=row<support.shape[0] and 0<=col<support.shape[1] and support[row,col]):outside+=1
        elif static_occupied[row,col]:static_crossings.append({'tick':i,'xy':pos[:2].tolist()})
    checks.update(raw_physics_complete=len(raw)==len(physics_times) and all(r['env_tick']==i for i,r in enumerate(raw)),
        raw_health=max_tilt<30 and height>=.21,raw_startup_below15=startup_max<15,
        no_large_pose_jumps=max_jump<.05,actor_50hz=abs(int(control[-1]['policy_inferences'])-math.ceil(len(raw)/4))<=1,
        stays_in_evaluation_scene_domain=outside==0)
    stopped=final_stop_evidence(control,definition['goal']);checks['measured_final_stop']=stopped['valid']
    truth=read(root/'interaction_ground_truth.json');truth_sha=sha(root/'interaction_ground_truth.json')
    starts=[e for e in events if e['event']=='skill_start'];ends=[e for e in events if e['event']=='skill_end']
    checks['interaction_count_consistent']=result['push_count']==len(starts)==len(ends)
    effects=[];contacts=rows(root/'contacts.jsonl');trajectory=read(folder/'trajectory.json')
    main_index=next(i for i,r in enumerate(control) if r['phase'] not in ['start','map_warmup'])
    for start in starts:
        permission=start['permission'];object_id=permission['object_id'];end=next((e for e in ends if e['object_id']==object_id),None)
        obj=truth.get('objects',{}).get(object_id,{})
        permitted=permission.get('source')=='ground_truth_layout' and permission.get('source_digest')==truth_sha and obj.get('category')=='box' and 'push_aside' in obj.get('skills',[]) and not permission.get('evidence_frames') and not permission.get('classifier_models')
        if end is None or end['status']!='succeeded':effects.append({'object_id':object_id,'valid':False,'reason':'skill_not_completed'});continue
        snapshot=folder/'interaction'/object_id/'effect_confirmation';observed=read(snapshot/'robot_state.json');mapped=read(snapshot/'layered_map.json');cert=observed['certificate']
        fresh=cert.get('confirming_observation_time_s',-math.inf)>cert.get('last_box_motion_time_s',math.inf) and observed['time_s']>=cert['confirming_observation_time_s']
        original=dict(initial_boxes[object_id]);original['xy']=start['plan']['initial_xy'];final=next(b for b in result['final_boxes'] if b['id']==object_id)
        observations=[]
        for sample in trajectory:
            index=main_index+sample['step']
            if object_id in sample['boxes'] and index<len(control):observations.append({'time_s':control[index]['command_time_s'],'xy':sample['boxes'][object_id]})
        crossing=crossing_evidence(control,cert,original,observations,observed['time_s'],np.maximum(original['dimensions'],final['dimensions']).tolist())
        impulses=[]
        for contact in contacts:
            actors=[contact['actor0'],contact['actor1']];tick=int(contact['env_tick'])
            if any(p.endswith('/InteractiveBoxes/'+object_id) for p in actors) and any('/robots/go2/' in p for p in actors) and tick<len(physics_times) and start['time_s']<=physics_times[tick]<=observed['time_s']:
                if any(np.linalg.norm(p.get('impulse',0.))>1e-8 for p in contact.get('points',[])):impulses.append(tick)
        valid=permitted and fresh and object_id in mapped['visible_box_ids'] and cert['side_parked'] and end['actual_push_distance_m']>.25 and bool(impulses) and crossing['valid']
        effects.append({'object_id':object_id,'valid':bool(valid),'permission_valid':permitted,'fresh_map':fresh,'impulse_samples':len(impulses),'crossing':crossing})
    checks['any_interaction_physically_completed']=all(e['valid'] for e in effects)
    incidental=[]
    for contact in contacts:
        actors=[contact['actor0'],contact['actor1']]
        if not any('/robots/go2/' in p for p in actors):continue
        if any('/InteractiveBoxes/' in p or '/objects/grscene_go2_floor' in p or '/Base/ground/' in p for p in actors):continue
        if all('/robots/go2/' in p for p in actors):continue
        if any(np.linalg.norm(p.get('impulse',0.))>1e-8 for p in contact.get('points',[])):incidental.append(contact)
    report={'task':task,'run':root.name,'server_checks_passed':all(bool(v) for v in checks.values()),
        'new_task_accepted':False,'local_hash_and_visual_review_outstanding':True,'checks':{k:bool(v) for k,v in checks.items()},
        'definition':definition,'scene_clutter':qualification['clutter'],'source_commit':source['commit'],'policy_sha256':model['policy_sha256'],
        'goal_error_m':result['goal_error_m'],'final_stop':stopped,'frames_per_stream':coverage['frames'],'fps':25.,
        'duration_s':video['videos']['combined']['duration_s'],'final_goal_hold_cycles':coverage['last_goal_hold_cycles'],
        'raw_physics':{'rows':len(raw),'expected_rows':len(physics_times),'maximum_tilt_deg':max_tilt,'startup_maximum_tilt_deg':startup_max,'minimum_height_m':height,'maximum_planar_step_m':max_jump,'outside_scene_domain_steps':outside},
        'interactions':effects,'interaction_required':False,'incidental_nonfloor_nonbox_contact_records':len(incidental),
        'static_evaluation_occupied_center_steps':len(static_crossings),'static_evaluation_discrepancy_samples':static_crossings[::max(1,len(static_crossings)//20)][:20],
        'videos_sha256':{p.name:sha(p) for p in (folder/'video').glob('*.mp4')},
        'audit_tool_sha256':sha(__file__),
        'scope':'New finite-initial-path task definition; no push is required, but every attempted interaction must have physical and post-scan traversal evidence.'}
    (folder/'qualified_run_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'task':task,'server_checks_passed':report['server_checks_passed'],'failed_checks':[k for k,v in checks.items() if not v],'goal_error_m':result['goal_error_m'],'raw_physics':report['raw_physics'],'incidental_contacts':len(incidental)}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('task');p.add_argument('--qualification',type=Path,required=True);p.add_argument('--expected-commit',required=True);a=p.parse_args()
    result=audit(a.root,a.task,a.qualification,a.expected_commit);raise SystemExit(0 if result['server_checks_passed'] else 1)
