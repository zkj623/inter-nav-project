"""Supplemental recorded hold/permission/action-history audit, not task acceptance."""
from pathlib import Path
import argparse
import json
import math
import hashlib


def audit(folder):
    folder=Path(folder);root=folder.parent
    model=json.loads((root/'robot_model_contract.json').read_text())
    recording=json.loads((root/'recording_config.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    traces=[json.loads(s) for s in (folder/'control_trace.jsonl').read_text().splitlines() if s.strip()]
    forbidden={'skill_engage','skill_push','skill_settle_push_progress','skill_brake_after_push',
               'skill_retreat','skill_settle_after_retreat','skill_recover_release_pose','skill_settle_released_corridor'}
    errors=[];captures=[];releases=[];segments=[];active=[];last=None;false_since=None
    def same(a,b):return len(a)==len(b)==12 and all(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-6 for x,y in zip(a,b))
    def flush():
        if not active:return
        first,end=active[0],active[-1];duration=end['time_s']-first['time_s']
        inferences=end['policy_inferences']-first['policy_inferences']
        valid=duration<.08 or abs(inferences-duration/.02)<=2
        segments.append({'start_s':first['time_s'],'end_s':end['time_s'],'duration_s':duration,
            'capture_count':first['stationary_hold']['capture_count'],'actor_inferences_during_hold':inferences,
            'actor_cadence_valid':valid,'frames':len(active)})
        if not valid:errors.append({'time_s':first['time_s'],'reason':'actor_cadence_during_hold'})
        active.clear()
    for row in traces:
        hold=row.get('stationary_hold');target=row.get('motor_joint_target',[])
        if hold is None:
            errors.append({'time_s':row['time_s'],'reason':'missing_hold_report'});continue
        allowed=row.get('stationary_hold_allowed')
        if row['phase'] in forbidden and allowed is not False:
            errors.append({'time_s':row['time_s'],'reason':'contact_phase_permission_not_disabled','phase':row['phase']})
        previous_count=0 if last is None else last['stationary_hold']['capture_count']
        previous_release=0 if last is None else last['stationary_hold']['release_count']
        if hold['capture_count']<previous_count or hold['release_count']<previous_release:
            errors.append({'time_s':row['time_s'],'reason':'unexpected_counter_reset'})
        if hold['capture_count']>previous_count:
            captures.append({'time_s':row['time_s'],'phase':row['phase'],'allowed':allowed,'count':hold['capture_count']})
            if not allowed:errors.append({'time_s':row['time_s'],'reason':'capture_while_disallowed'})
        if hold['release_count']>previous_release:
            releases.append({'time_s':row['time_s'],'phase':row['phase'],'reason':hold['reason'],'count':hold['release_count']})
            if last is not None and last['stationary_hold']['state']=='holding' and not same(target,last['motor_joint_target']):
                errors.append({'time_s':row['time_s'],'reason':'initial_release_target_jump'})
        if allowed is False:
            if false_since is None:false_since=row['time_s']
            if hold['state']=='holding' and row['time_s']-false_since>.03:
                errors.append({'time_s':row['time_s'],'reason':'hold_not_released_after_permission_change'})
        else:false_since=None
        if hold['state']=='holding':
            if active and active[0]['stationary_hold']['capture_count']!=hold['capture_count']:flush()
            if active and not same(target,active[0]['motor_joint_target']):
                errors.append({'time_s':row['time_s'],'reason':'held_joint_target_changed'})
            active.append(row)
        else:flush()
        last=row
    flush()
    checks={'model_flag_enabled':model.get('stationary_joint_hold') is True,
        'recording_flag_enabled':recording.get('stationary_joint_hold') is True,
        'result_flag_enabled':result.get('stationary_joint_hold') is True,
        'hold_and_release_observed':bool(captures and releases),
        'holding_segments_observed':bool(segments),'no_trace_violations':not errors,
        'final_counts_match_result':bool(last and last['stationary_hold']==result.get('stationary_hold_report'))}
    report={'feature_only':True,'does_not_replace_task_map_video_or_source_audits':True,'task':folder.name,
        'valid':all(checks.values()),'checks':checks,'captures':captures,'releases':releases,
        'holding_segments':segments,'violation_count':len(errors),'first_violations':errors[:20],
        'permission_response_tolerance_s':.03,'final_state':None if last is None else last['stationary_hold'],
        'audit_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'stationary_hold_qa.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('task_folder',type=Path)
    args=parser.parse_args();report=audit(args.task_folder)
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['valid'] else 1)
