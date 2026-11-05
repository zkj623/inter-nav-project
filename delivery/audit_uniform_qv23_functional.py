"""Audit physical success without requiring a video or pretending to review one."""
from pathlib import Path
import argparse,collections,hashlib,json,math,sys
import numpy as np
R=Path('/data20t/embodied/zkj/inter-nav-textured-qv23-20261104')
sys.path.insert(0,str(R))
from tools.audit_actual_passage import crossing_evidence,final_stop_evidence,yaw_tilt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('task');p.add_argument('--recorded-reference',action='store_true');a=p.parse_args()
HEAD='514cd51dc7a4400a3e7db431b1f18a7aba7e3f50'
POLICY='8862682da34ebb090a2e56f39254af430d5645f20cb5e3e104c72a4720671db8'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
root=a.root;f=root/a.task;base=root.parents[1]
q=read(base/'assessment/qualification.json');definition=next(t for t in q['qualified_candidates'] if t['id']==a.task)
result=read(f/'result.json');control=rows(f/'control_trace.jsonl');events=rows(f/'events.jsonl');raw=rows(root/'control_trace.jsonl')
starts=[e for e in events if e['event']=='skill_start'];ends=[e for e in events if e['event']=='skill_end']
startup=next(e for e in events if e['event']=='startup');initial={b['id']:b for b in startup['boxes']}
times=[]
for c in control:times.extend(c['command_time_s']+(j+1)*.005 for j in range(int(c['physics_substeps'])))
incidental=collections.Counter();impulses=collections.Counter()
for line in (root/'contacts.jsonl').open():
    c=json.loads(line);actors=[c['actor0'],c['actor1']]
    if sum('/robots/go2/' in s for s in actors)!=1:continue
    points=[v for v in c.get('points',[]) if np.linalg.norm(v.get('impulse',0.))>1e-8]
    if not points:continue
    other=next(s for s in actors if '/robots/go2/' not in s);tick=int(c['env_tick'])
    if '/InteractiveBoxes/' in other:
        box=other.rsplit('/',1)[-1]
        for start in starts:
            if start['permission']['object_id']!=box:continue
            end=next((e for e in ends if e['object_id']==box),{})
            if tick<len(times) and start['time_s']<=times[tick]<=end.get('time_s',-1):impulses[box]+=1
    elif not any(s in other for s in ['/objects/grscene_go2_floor','/Base/ground/']):incidental[other]+=1
effects=[];truth=read(root/'interaction_ground_truth.json')
trajectory=read(f/'trajectory.json');main_index=next(i for i,r in enumerate(control) if r['phase'] not in ['start','map_warmup'])
for start in starts:
    box=start['permission']['object_id'];end=next((e for e in ends if e['object_id']==box),{})
    evidence=f/'interaction'/box/'effect_confirmation';effect_checks={'skill_succeeded':end.get('status')=='succeeded','actual_push_over_025m':end.get('actual_push_distance_m',0)>.25,'real_contact_impulses':impulses[box]>0}
    crossing={'valid':False,'crossings':[]};cert={}
    if (evidence/'robot_state.json').exists():
        state=read(evidence/'robot_state.json');mapped=read(evidence/'layered_map.json');cert=state['certificate']
        original=dict(initial[box]);original['xy']=start['plan']['initial_xy'];final=next(b for b in result['final_boxes'] if b['id']==box)
        observations=[{'time_s':control[main_index+r['step']]['command_time_s'],'xy':r['boxes'][box]} for r in trajectory if box in r['boxes'] and main_index+r['step']<len(control)]
        crossing=crossing_evidence(control,cert,original,observations,state['time_s'],np.maximum(original['dimensions'],final['dimensions']).tolist())
        effect_checks.update(fresh_scan_after_box_motion=cert.get('confirming_observation_time_s',-math.inf)>cert.get('last_box_motion_time_s',math.inf),side_parked_and_visible=bool(cert.get('side_parked') and box in mapped['visible_box_ids']))
    else:effect_checks.update(fresh_scan_after_box_motion=False,side_parked_and_visible=False)
    obj=truth.get('objects',{}).get(box,{})
    effect_checks.update(actual_body_crossing=crossing['valid'],ground_truth_permission=start['permission'].get('source')=='ground_truth_layout' and start['permission'].get('source_digest')==sha(root/'interaction_ground_truth.json') and obj.get('category')=='box' and 'push_aside' in obj.get('skills',[]))
    effects.append({'box_id':box,'checks':effect_checks,'valid':all(effect_checks.values()),'actual_push_distance_m':end.get('actual_push_distance_m'),'impulse_samples':impulses[box],'crossing':crossing,'certificate':cert})
stop=final_stop_evidence(control,definition['goal'])
# The user accepts 0.5 m; retain all independent measured quiet/stability gates.
stop['goal_within_tolerance']=stop['goal_error_m']<=.5
stop['valid']=all(v for k,v in stop.items() if k not in ['valid','window_seconds','goal_error_m'])
flow_path=f/'fluency_review.json';flow=read(flow_path) if flow_path.exists() else {}
source=read(root/'source_version_audit.json');manifest=read(root/'source_manifest.json');binding=read(root/'qualified_input_manifest.json');profile=read(root/'benchmark_inputs/scene_profile.json')
policy=read(root/'policy_source_snapshot.json')['files'];guard=read(root/'online_input_audit.json');scope=read(root/'scope_identity.json');outcome=read(root/'supervisor_outcome.json')
floor_geometry=base.parent/'existing-five/runs/qualified-qv13-t05-try1/T05/floor_cover_contact_geometry.json'
blanket='/Furnitures/blanket/model_924aeb98d1adf1f47c4032e1b174f820_0/Instance'
floor_only=not incidental or (len(incidental)==1 and next(iter(incidental)).endswith(blanket) and floor_geometry.exists() and read(floor_geometry)['max_m'][2]<.03)
geometry=flow.get('geometry_consistency',{})
required=definition.get('interaction_qualification',{}).get('box_id') if definition.get('requires_interaction') else None
checks={'native_goal_reached':bool(result.get('success') and result.get('reason')=='goal_reached'),
        'declared_start_and_goal':all(np.allclose(result[k],definition[k],atol=1e-8,rtol=0) for k in ['start','goal']),
        'measured_final_stop':stop['valid'],'goal_error_within_050m':result['goal_error_m']<=.5,
        'startup_stable':bool(startup['passed'] and startup['tilt_degrees']<15),
        'every_physics_step_present':len(raw)==len(times) and all(r['env_tick']==i and r['task']==a.task for i,r in enumerate(raw)),
        'every_physics_step_tilt_below30':all(yaw_tilt(r['orientation'])[1]<30 for r in raw),
        'rollout_healthy':bool(result.get('rollout_health',{}).get('healthy')),
        'geometry_consistent':geometry.get('source')=='every_physics_step' and geometry.get('samples',0)>0 and geometry.get('outside_domain')==geometry.get('static_occupied')==0,
        'all_attempted_skills_completed':len(starts)==len(ends)==result['push_count'] and all(e['valid'] for e in effects),
        'required_interaction_completed':required is None or any(e['box_id']==required and e['valid'] for e in effects),
        'only_verified_floor_cover_incidental_contacts':floor_only,
        'frozen_qv23_source':source['valid'] and source['commit']==HEAD and bool(manifest) and all(sha(root/'source'/name)==h for name,h in manifest.items()),
        'same_policy':sha(root/'policy/actor_checkpoint.pt')==POLICY and all(sha(root/'policy'/name)==h for name,h in policy.items()),
        'fixed_inputs':all(sha(root/'benchmark_inputs'/name)==h for name,h in binding['benchmark_files'].items()) and binding['qualification_sha256']==sha(base/'assessment/qualification.json') and binding['evaluation_map_sha256']==sha(base/'assessment/evaluation_map.npz'),
        'scene_asset_unchanged':sha(Path(profile['scene_asset_path']))==q['geometry_asset_sha256'],
        'online_without_forbidden_prior':guard['guard_armed'] and not guard['prior_read_attempts'],
        'resource_limits_preserved':scope['memory_max_bytes']==24*1024**3 and scope['swap_max_bytes']==0 and outcome['exit_code']==0 and not outcome['resource_stop'],
        'recording_mode_matches':bool(any((f/'video').glob('*.mp4')))==a.recorded_reference}
report={'functional_validation':True,'video_required':False,'recording_accepted':False,'task':a.task,'version':'qv23','source_commit':HEAD,'policy_sha256':POLICY,
        'functional_checks_passed':all(checks.values()),'checks':checks,'fluency_checks_passed':flow.get('flow_comparison_passed',False),
        'failed_fluency_checks':[k for k,v in flow.get('checks',{}).items() if not v],
        'goal_error_m':result['goal_error_m'],'duration_to_goal_braking_s':flow.get('duration_to_goal_braking_s'),
        'maximum_physics_tilt_deg':max(yaw_tilt(r['orientation'])[1] for r in raw),'final_stop':stop,'effects':effects,
        'incidental_contacts':dict(incidental),'floor_cover_geometry_sha256':sha(floor_geometry) if incidental and floor_only else None,
        'recorded_reference':a.recorded_reference,
        'scope':'One physical trial on the frozen QV23 implementation; recording mode explicitly identified; no new video acceptance or universal success-rate claim.'}
(f/('functional_reference_validation.json' if a.recorded_reference else 'functional_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['task','functional_checks_passed','fluency_checks_passed','goal_error_m','maximum_physics_tilt_deg']}))
print(json.dumps({'failed_checks':[k for k,v in checks.items() if not v]}))
raise SystemExit(0 if report['functional_checks_passed'] else 1)
