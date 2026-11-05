"""Measure pauses and post-push backtracking from a real recorded rollout.

This supplements, and cannot replace, physical/map/goal/video acceptance.
"""
import argparse,json,math
from pathlib import Path
import numpy as np


def audit(folder):
    trace=[json.loads(line) for line in (folder/'control_trace.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    t=np.array([r['time_s'] for r in trace]);xy=np.array([r['position'][:2] for r in trace])
    q=np.array([r['orientation'] for r in trace]);q/=np.linalg.norm(q,axis=1)[:,None]
    yaw=np.unwrap(np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2)))
    command=np.array([r.get('policy_command',r['command']) for r in trace])
    intervals=[];start=None;last=None
    # Windows test retained physical progress, not only velocity or a nonzero
    # command. Intentional settling is excluded explicitly.
    for right in np.searchsorted(t,np.arange(t[0]+1.5,t[-1],.2)):
        left=np.searchsorted(t,t[right]-1.5)
        phase=trace[right]['phase']
        excluded=any(word in phase for word in ['settle','brake','start','warmup','hold'])
        motion=np.median(np.linalg.norm(command[left:right+1,:2],axis=1))>.04 or np.median(np.abs(command[left:right+1,2]))>.10
        no_progress=np.linalg.norm(np.ptp(xy[left:right+1],axis=0))<.015 and np.ptp(yaw[left:right+1])<.04
        if motion and no_progress and not excluded:
            if start is None:start=float(t[left])
            last=float(t[right])
        elif start is not None:
            intervals.append({'start_s':start,'end_s':last,'duration_s':last-start});start=None
    if start is not None:intervals.append({'start_s':start,'end_s':last,'duration_s':last-start})
    passages=[]
    for event in events:
        if event['event']!='skill_end' or event.get('status')!='succeeded':continue
        proof=event['passage_certificate'];axis=np.asarray(proof['axis'],float);axis/=np.linalg.norm(axis)
        center=np.asarray(proof['center'],float);left=np.searchsorted(t,event['time_s'])
        # Match the protected physical body's rear crossing plane. Full
        # collision/counterfactual validation remains in the original audit.
        c=np.cos(yaw[left:]);s=np.sin(yaw[left:])
        forward=c*axis[0]+s*axis[1];side=-s*axis[0]+c*axis[1]
        rear=np.minimum(-.32*forward,.50*forward)-.25*np.abs(side)
        projection=(xy[left:]-center)@axis
        crosses=np.flatnonzero(projection+rear>.02)
        if not len(crosses):
            passages.append({'object_id':event['object_id'],'crossed':False});continue
        right=left+int(crosses[0]);indices=np.unique(np.searchsorted(t,np.arange(t[left],t[right]+.0001,.1)))
        indices=indices[indices<=right]
        path_length=float(np.linalg.norm(np.diff(xy[indices],axis=0),axis=1).sum())
        passages.append({'object_id':event['object_id'],'crossed':True,'release_time_s':float(t[left]),
            'crossing_time_s':float(t[right]),'release_to_crossing_seconds':float(t[right]-t[left]),
            'path_length_until_crossing_m':path_length,
            'maximum_backtrack_along_corridor_m':float(max(0.,projection[0]-projection[:right-left+1].min()))})
    turn_indices=np.array([i for i,r in enumerate(trace) if r['phase']=='skill_turn_at_stage'],dtype=int)
    turn_summary=None
    if len(turn_indices):
        turns=[]
        for indices in np.split(turn_indices,np.flatnonzero(np.diff(turn_indices)>1)+1):
            moving=indices[np.abs(command[indices,2])>.10]
            pauses=[];pause_start=None
            if len(moving):
                for i in range(int(moving[0]),int(moving[-1])+1):
                    quiet=abs(command[i,2])<.08
                    if quiet and pause_start is None:pause_start=float(t[i])
                    elif not quiet and pause_start is not None:
                        if t[i]-pause_start>=.20:pauses.append(float(t[i]-pause_start))
                        pause_start=None
            turns.append({'start_s':float(t[indices[0]]),'end_s':float(t[indices[-1]]),
                'duration_s':float(t[indices[-1]]-t[indices[0]]),
                'retained_angle_deg':float(np.degrees(yaw[indices[-1]]-yaw[indices[0]])),
                'internal_zero_command_pauses_s':pauses,'internal_pause_seconds':sum(pauses)})
        turn_summary=turns
    return {'supplemental_only':True,'task':folder.name,'maximum_ineffective_motion_seconds':max([x['duration_s'] for x in intervals],default=0.),
            'ineffective_motion_intervals':intervals,'passages':passages,'sampled_duration_s':float(t[-1]-t[0]),
            'box_preparation_turns':turn_summary,
            'does_not_establish_task_or_video_acceptance':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();report=audit(a.folder);text=json.dumps(report,indent=2)
    if a.output:a.output.write_text(text+'\n',encoding='utf-8')
    print(text)
