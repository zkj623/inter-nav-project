"""Supplemental trajectory evidence for an original online task, after existing QA.

This checks actual vacated-volume traversal and the measured final stop. It
does not replace source/map audits, video decoding, or visual review.
"""
from pathlib import Path
import argparse,bisect,json,math
import numpy as np


def yaw_tilt(q):
    q=np.asarray(q,float);norm=np.linalg.norm(q)
    if not np.isfinite(q).all() or norm<1e-8:raise ValueError('Invalid recorded quaternion')
    w,x,y,z=q/norm
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),math.degrees(math.acos(np.clip(1-2*(x*x+y*y),-1,1)))


def rectangle(xy,heading,front,rear,half_width):
    c,s=math.cos(heading),math.sin(heading)
    return np.array([[front,half_width],[-rear,half_width],[-rear,-half_width],[front,-half_width]])@np.array([[c,s],[-s,c]])+xy


def overlap(a,b):
    for polygon in [a,b]:
        for edge in np.roll(polygon,-1,axis=0)-polygon:
            normal=np.array([-edge[1],edge[0]])
            x,y=a@normal,b@normal
            if max(x.min(),y.min())>=min(x.max(),y.max())-1e-9:return False
    return True


def box_polygon(box,xy=None):
    return rectangle(np.asarray(box['xy'] if xy is None else xy),box.get('yaw',0.),box['dimensions'][0]/2,box['dimensions'][0]/2,box['dimensions'][1]/2)


def crossing_evidence(trace,certificate,original_box,box_observations,after_time=0.,current_box_dimensions=None):
    axis=np.asarray(certificate['axis'],float);axis/=np.linalg.norm(axis)
    boundary=float(certificate.get('boundary',np.dot(certificate['center'],axis)))
    before=False;previous_rear=None;previous_row=None;candidates=[]
    old_polygon=box_polygon(original_box);observation_times=[p['time_s'] for p in box_observations]
    current_geometry={**original_box,'dimensions':original_box['dimensions'] if current_box_dimensions is None else current_box_dimensions}
    tangent=np.array([-axis[1],axis[0]])
    for index,row in enumerate(trace):
        heading,_=yaw_tilt(row['orientation']);position=np.asarray(row['position'][:2])
        body=rectangle(position,heading,.50,.32,.25);projection=body@axis-boundary
        rear=float(projection.min());front=float(projection.max())
        if front<0:before=True
        crossed=before and previous_rear is not None and previous_rear<=.02<rear and row['time_s']>=after_time
        if crossed:
            j=bisect.bisect_left(observation_times,row['time_s'])
            bracketing=0<j<len(box_observations)
            previous_box=box_observations[j-1] if bracketing else None;next_box=box_observations[j] if bracketing else None
            timely=bracketing and next_box['time_s']-previous_box['time_s']<=.40
            stable=bool(timely and np.linalg.norm(np.array(previous_box['xy'])-next_box['xy'])<.02)
            sideways=min(abs(np.dot(np.asarray(b['xy'])-original_box['xy'],tangent)) for b in [previous_box,next_box]) if bracketing else 0.
            box_free=bool(stable and sideways>.25 and not overlap(body,box_polygon(current_geometry,previous_box['xy'])) and not overlap(body,box_polygon(current_geometry,next_box['xy'])))
            # The original task also requires4cm body clearance. A crossing
            # that needed this space can benefit from pushing even if the
            # physical body would narrowly miss the old box itself.
            protected_body=rectangle(position,heading,.54,.36,.29)
            original_blocked=overlap(protected_body,old_polygon)
            delta=np.linalg.norm(position-np.asarray(previous_row['position'][:2]));dt=row['time_s']-previous_row['time_s']
            continuous=0<dt<=.02 and delta<=max(.15,5.*dt)
            valid=bool(original_blocked and box_free and continuous)
            candidates.append({'row':index,'time_s':row['time_s'],'position':row['position'],'rear_beyond_boundary_m':rear,
                'original_box_would_block_body_clearance':bool(original_blocked),'counterfactual_clearance_margin_m':.04,'recorded_box_stable_near_crossing':stable,
                'actual_body_clear_of_relocated_box':box_free,'observed_side_displacement_m':float(sideways),'continuous_local_trace':bool(continuous),'valid':valid,
                'bracketing_box_observations':None if not bracketing else [previous_box,next_box]})
        previous_rear=rear;previous_row=row
    return {'valid':any(c['valid'] for c in candidates),'crossings':candidates}


def final_stop_evidence(trace,goal):
    if not trace:return {'valid':False,'reason':'missing_trace'}
    end=trace[-1]['time_s'];tail=[r for r in trace if r['time_s']>=end-.4-1e-6]
    span=tail[-1]['time_s']-tail[0]['time_s'];positions=np.array([r['position'] for r in tail])
    velocities=np.array([r['body_velocity'] for r in tail]);angular=np.array([r['body_angular_velocity'] for r in tail])
    tilts=np.array([yaw_tilt(r['orientation'])[1] for r in tail]);error=float(np.linalg.norm(positions[-1,:2]-goal))
    checks={'window_seconds':span,'window_long_enough':span>=.4-1e-6,'goal_error_m':error,
        'goal_within_tolerance':error<.25,'linear_xy_quiet':bool(np.max(np.linalg.norm(velocities[:,:2],axis=1))<.08),
        'vertical_quiet':bool(np.max(np.abs(velocities[:,2]))<.05),'angular_quiet':bool(np.max(np.abs(angular))<.10),
        'tilt_quiet':bool(np.max(tilts)<15.),'position_stable':bool(np.max(np.linalg.norm(positions[:,:2]-positions[0,:2],axis=1))<.025),
        'height_stable':bool(np.max(np.abs(positions[:,2]-positions[0,2]))<.015),'tilt_stable':bool(np.max(np.abs(tilts-tilts[0]))<1.5)}
    keys=[k for k in checks if k not in ['window_seconds','goal_error_m']]
    return {'valid':all(checks[k] for k in keys),**checks}


def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(folder):
    result=json.loads((folder/'result.json').read_text());trace=rows(folder/'control_trace.jsonl');events=rows(folder/'events.jsonl')
    canonical={'T04':([2.11,-.78],[10.3,1.75]),'T05':([13.6,-.4],[3.3,1.4])}
    if folder.name not in canonical:raise ValueError('Only original T04/T05 accepted')
    start,goal=canonical[folder.name]
    coordinates=np.allclose(result['start'],start,atol=1e-8,rtol=0.) and np.allclose(result['goal'],goal,atol=1e-8,rtol=0.)
    skills=[e for e in events if e['event']=='skill_end' and e.get('status')=='succeeded'];evidence=[]
    trajectory=json.loads((folder/'trajectory.json').read_text())
    main_index=next(i for i,r in enumerate(trace) if r['phase'] not in ['start','map_warmup'])
    initial=next(e for e in events if e['event']=='startup')['boxes']
    for skill in skills:
        object_id=skill['object_id'];begin=next(e for e in events if e['event']=='skill_start' and e['permission']['object_id']==object_id)
        original=dict(next(b for b in initial if b['id']==object_id));original['xy']=begin['plan']['initial_xy']
        final=next(b for b in result['final_boxes'] if b['id']==object_id)
        # BoxState dimensions are the tracked AABB; retain the larger recorded envelope.
        current_dimensions=np.maximum(original['dimensions'],final['dimensions']).tolist()
        observations=[]
        for sample in trajectory:
            index=main_index+sample['step']
            if object_id not in sample['boxes'] or index>=len(trace):continue
            observations.append({'time_s':trace[index]['command_time_s'],'xy':sample['boxes'][object_id]})
        crossing=crossing_evidence(trace,skill['passage_certificate'],original,observations,begin['time_s'],current_dimensions)
        evidence.append({'object_id':object_id,'actual_push_distance_m':skill['actual_push_distance_m'],**crossing})
    stopped=final_stop_evidence(trace,goal);maximum=max(yaw_tilt(r['orientation'])[1] for r in trace)
    checks={'result_reports_success':result['success'],'original_coordinates':bool(coordinates),'maximum_tilt_below_30':maximum<30.,
        'measured_final_stop':stopped['valid'],'successful_skill_and_actual_crossing':bool(evidence) and all(e['valid'] and e['actual_push_distance_m']>.25 for e in evidence)}
    report={'supplemental_only':True,'does_not_replace_map_source_video_audits':True,'task':folder.name,'valid':all(checks.values()),
        'checks':checks,'maximum_tilt_degrees_from_trace':maximum,'final_stop':stopped,'passages':evidence}
    (folder/'actual_passage_qa.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('task_directory',type=Path)
    report=audit(parser.parse_args().task_directory);print(json.dumps(report,indent=2));raise SystemExit(0 if report['valid'] else 1)
