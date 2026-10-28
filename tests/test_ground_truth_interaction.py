import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionCatalog,InteractionPermission
from grutopia_extension.interactive_navigation.ground_truth_interaction import GroundTruthInteractionRuntime
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill


def runtime(tmp_path):
    path=tmp_path/'layout.json'
    path.write_text(json.dumps({'objects':[
        {'id':'B1','interactions':['push']},
        {'id':'B2','interactions':['push'],'interactable':False},
        {'id':'C1','category':'chair','interactions':['push']},
        {'id':'B3','interactions':[]},
    ]}))
    return GroundTruthInteractionRuntime(InteractionCatalog(),path,tmp_path/'output')


def test_ground_truth_allows_only_declared_push_boxes_without_visual_frames(tmp_path):
    ground_truth=runtime(tmp_path);ground_truth.reset('T04')
    assert ground_truth.eligible_ids(0.)=={'B1'}
    permission=ground_truth.permission('B1','push_aside',0.)
    assert permission.source=='ground_truth_layout'
    assert not permission.evidence_frames and not permission.classifier_models
    assert permission.source_digest==hashlib.sha256(ground_truth.reference.read_bytes()).hexdigest()
    assert not ground_truth.wants_frame(0.,[object()]) and ground_truth.poll() is None
    assert ground_truth.permission('not-in-layout','push_aside',0.) is None
    saved=ground_truth.save(10000.)
    assert saved['required'] is False and saved['semantic_source']=='ground_truth_layout'
    assert saved['objects']['B1']['frames']==0


def test_skill_accepts_explicit_ground_truth_but_keeps_vision_requirements(tmp_path):
    gt=runtime(tmp_path)
    permission=gt.permission('B1','push_aside',0.)
    plan=SimpleNamespace(box_id='B1',direction=(1.,0.),push_distance=.8)
    skill=PushAsideSkill(permission,plan,(0.,0.),(2.,0.))
    assert skill.report()['interaction_source']=='ground_truth_layout'
    with pytest.raises(ValueError):PushAsideSkill(replace(permission,source='rgb_dino_qwen_vl'),plan,(0.,0.),(2.,0.))
    with pytest.raises(ValueError):PushAsideSkill(replace(permission,evidence_frames=('fake',)),plan,(0.,0.),(2.,0.))
    with pytest.raises(ValueError):PushAsideSkill(replace(permission,source_digest=''),plan,(0.,0.),(2.,0.))
    with pytest.raises(ValueError):PushAsideSkill(replace(permission,object_id='B2'),plan,(0.,0.),(2.,0.))


def test_truth_snapshot_contains_eligibility_and_does_not_supply_a_map(tmp_path):
    gt=runtime(tmp_path)
    data=json.loads(gt.reference.read_text())
    assert data['objects']['B1']=={'category':'box','skills':['push_aside']}
    assert not any(key in data for key in ['grid','free','occupied','positions'])
    with pytest.raises(ValueError):gt.reset('../outside')
