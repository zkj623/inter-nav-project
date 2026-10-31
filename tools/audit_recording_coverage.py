"""Verify complete video timeline and final recorded goal hold, independently."""
from pathlib import Path
import argparse,json,math
p=argparse.ArgumentParser();p.add_argument('task_directory',type=Path);a=p.parse_args();folder=a.task_directory
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
control=rows(folder/'control_trace.jsonl');frames=rows(folder/'video/video_frames.jsonl')
camera=rows(folder/'video/camera_heading.jsonl');native=read(folder/'result.json');video=read(folder/'video_qa.json')
start=next((i for i,row in enumerate(control) if row['phase']=='map_warmup'),None)
steps=[row['step'] for row in frames];times=[];states=[]
if start is not None:
    for frame in frames:
        index=start+int(frame['step'])
        if not 0<=index<len(control):break
        times.append(control[index]['time_s'])
        states.append(frame['state'].split(maxsplit=1)[-1]==control[index]['phase'])
names={'combined','third_person','robot_rgb','map_topdown','pointcloud_map'}
checks={
    'recording_starts_at_map_warmup':start is not None and bool(steps) and 0<=steps[0]<5 and frames[0]['state'].endswith('map_warmup'),
    'recording_covers_task_end':start is not None and bool(steps) and 0<=len(control)-start-1-steps[-1]<5,
    'contiguous_frame_indices':bool(frames) and all(row['frame_index']==i for i,row in enumerate(frames)),
    'all_frame_steps_in_control':len(times)==len(frames) and bool(times),
    'all_frame_states_match_control':bool(states) and all(states),
    'continuous_25fps_sim_timeline':len(times)>1 and all(abs((b-a)-.04)<1e-5 for a,b in zip(times,times[1:])),
    'exact_five_streams':set(video['videos'])==names and {f.stem for f in (folder/'video').glob('*.mp4')}==names,
    'all_decoded_streams_match_timeline':bool(frames) and all(v['valid'] and abs(v['fps']-25.)<1e-5 and v['decoded_frames']==v['declared_frames']==len(frames) for v in video['videos'].values()),
    'camera_records_cover_all_frames':len(camera)==len(frames) and all(c['frame_index']==f['frame_index'] and c['step']==f['step'] for c,f in zip(camera,frames)),
}
tail=len(control)-1
while tail>=0 and control[tail]['phase']==native['reason']:tail-=1
hold=0
while tail>=0 and control[tail]['phase']=='goal_settling':hold+=1;tail-=1
goal_events=[r for r in rows(folder/'events.jsonl') if r.get('event')=='goal_settled']
goal_hold_valid=bool(native['success'] and native['reason']=='goal_reached' and hold>=260 and goal_events
    and goal_events[-1].get('measured_stopped') and goal_events[-1].get('full_body_crossed')
    and goal_events[-1]['distance']<.25 and native['goal_error_m']<.25)
report={'coverage_only':True,'original_task_accepted':False,'task':folder.name,
    'coverage_valid':all(checks.values()),'coverage_checks':checks,
    'recording_control_start_index':start,'frames':len(frames),'first_recorded_step':steps[0] if steps else None,
    'last_recorded_step':steps[-1] if steps else None,'last_goal_hold_cycles':hold,
    'goal_hold_valid':goal_hold_valid,'valid_final_recording_coverage':bool(all(checks.values()) and goal_hold_valid),
    'scope':'Timeline and recorded hold only; physical push/map/traversal/source/policy/local hashes require separate acceptance.'}
(folder/'recording_coverage_qa.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
