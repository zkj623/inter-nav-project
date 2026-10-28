"""Audit declared GT/vision permission -> physical skill -> goal evidence.

Run after video decoding QA has created video_qa.json. This checker does not
replace physical execution and fails closed on incomplete or old-format runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(folder):
    root=folder.parent;result=json.loads((folder/'result.json').read_text())
    events=rows(folder/'events.jsonl');checks={};details=[]
    config=json.loads((root/'recording_config.json').read_text())
    mode=config.get('interaction_source','vision')
    checks['interaction_source_declared']=result.get('interaction_source',mode)==mode and mode in ['ground_truth','vision']
    truth={};truth_digest=None
    if mode=='ground_truth':
        truth_path=root/'interaction_ground_truth.json'
        if truth_path.exists():
            raw=truth_path.read_bytes();truth=json.loads(raw);truth_digest=hashlib.sha256(raw).hexdigest()
        checks['ground_truth_manifest']=truth.get('source')=='ground_truth_layout' and truth.get('version')==1
    checks['goal_reached']=result['success'] and result['goal_error_m']<.25
    checks['original_scale_and_floor']=json.loads((root/'recording_config.json').read_text())['scene_scale']==[1,1,1] and abs(result['ground_height']-.01)<1e-8
    checks['no_rollout_teleport']=result['teleports_during_rollout'] is False
    checks['healthy']=result['rollout_health']['healthy'] and result['maximum_tilt_degrees']<30.
    initial=np.load(folder/'initial_unknown/layered_map.npz')
    checks['initial_unknown']=int(initial['observed'].sum())==0 and result['map_source']=='online'
    guard=json.loads((root/'online_input_audit.json').read_text())
    checks['no_prior_reads']=guard['guard_armed'] and not guard['prior_read_attempts']
    canonical={'T04':([2.11,-.78],[10.3,1.75]),'T05':([13.6,-.4],[3.3,1.4])}
    checks['original_task_coordinates']=folder.name in canonical and all(np.allclose(result[key],value) for key,value in zip(['start','goal'],canonical[folder.name]))
    starts=[(i,e) for i,e in enumerate(events) if e['event']=='skill_start']
    checks['skill_executed']=bool(starts) and result['push_count']==len(starts)
    for index,start in starts:
        permission=start['permission'];object_id=permission['object_id'];frames=set(permission['evidence_frames'])
        ended=next((e for e in events[index+1:] if e['event']=='skill_end' and e['object_id']==object_id),None)
        valid_frames=set()
        if mode=='ground_truth':
            obj=truth.get('objects',{}).get(object_id,{})
            admitted=(permission.get('source')=='ground_truth_layout' and permission.get('source_digest')==truth_digest
                      and Path(permission.get('source_reference','')).name=='interaction_ground_truth.json'
                      and obj.get('category')=='box' and 'push_aside' in obj.get('skills',[])
                      and not frames and not permission.get('classifier_models'))
        else:
            evidence={}
            for event in events[:index]:
                if event['event']!='visual_recognition':continue
                debug=event['details']
                for detection in debug.get('detections',[]):
                    if (detection.get('status')=='classified' and detection.get('object_id')==object_id
                        and detection.get('category') in ['box','cardboard box','plastic storage box','wooden crate']
                        and detection.get('detector_score',0)>=.30 and detection.get('classifier_model')
                        and debug.get('semantic_source')=='rgb_dino_qwen_vl'):
                        crop=folder/'recognition'/Path(detection['crop_path']).name
                        if crop.is_file() and crop.stat().st_size>0:evidence[debug['frame_id']]=detection
            valid_frames=frames.intersection(evidence)
            admitted=len(valid_frames)>=2 and permission.get('source','rgb_dino_qwen_vl')=='rgb_dino_qwen_vl'
        okay=(admitted and permission['category']=='box' and permission['skill_name']=='push_aside'
              and ended is not None and ended['status']=='succeeded' and ended['actual_push_distance_m']>.25
              and bool(ended.get('passage_certificate')))
        if mode=='ground_truth' and ended:
            okay=okay and ended.get('interaction_source')=='ground_truth_layout' and ended.get('source_digest')==truth_digest
        if ended and ended.get('passage_certificate'):
            direction=np.asarray(start['plan']['direction']);axis=np.asarray(ended['passage_certificate']['axis'])
            okay=okay and abs(float(np.dot(direction,axis)))<.4
        details.append({'object_id':object_id,'interaction_source':mode,'visual_frames_before_skill':sorted(valid_frames),'valid':bool(okay)})
    checks['interaction_permission_effect_chain']=bool(details) and all(d['valid'] for d in details)
    checks['full_body_passage']=result.get('full_body_passage_crossed') is True
    video_qa=json.loads((folder/'video_qa.json').read_text())
    checks['video_decoding']=bool(video_qa.get('videos')) and all(v.get('valid',False) for v in video_qa['videos'].values())
    manifest=json.loads((root/'source_manifest.json').read_text())
    checks['source_snapshot']=all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest for name,digest in manifest.items())
    required=['interaction_catalog.py','rgb_object_recognition.py','interaction_recognition_runtime.py','interaction_skill.py','push_aside_skill.py','object_inspection.py']
    if mode=='ground_truth':required.append('ground_truth_interaction.py')
    checks['new_source_present']=all(any(Path(name).name==needed for name in manifest) for needed in required)
    report={'task':folder.name,'interaction_source':mode,'valid':all(checks.values()),'checks':checks,'skills':details,
            'audit_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'visual_skill_qa.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task_directory',type=Path)
    report=audit(parser.parse_args().task_directory)
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['valid'] else 1)
