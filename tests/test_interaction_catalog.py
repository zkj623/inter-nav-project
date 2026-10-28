from grutopia_extension.interactive_navigation.interaction_catalog import (
    InteractionCatalog,RecognitionStore,RecognitionEvidence)


def evidence(label='cardboard box',frame='frame1',source='rgb_dino_qwen_vl'):
    return RecognitionEvidence('B06',label,frame,1.,.7,'Qwen3-VL','frame.jpg',source)


def test_box_id_and_simulator_label_do_not_imply_interaction_permission():
    store=RecognitionStore();catalog=InteractionCatalog()
    assert not store.observe(evidence(source='simulator'))
    assert catalog.permission('B06','push_aside',store,1.) is None
    assert catalog.eligible_ids('push_aside',store,1.)==frozenset()


def test_distinct_visual_observations_admit_box_to_registered_skill():
    store=RecognitionStore();catalog=InteractionCatalog()
    store.observe(evidence())
    assert catalog.permission('B06','push_aside',store,1.) is None
    assert not store.observe(evidence())
    store.observe(evidence(label='plastic storage box',frame='frame2'))
    permission=catalog.permission('B06','push_aside',store,2.)
    assert permission.category=='box' and len(permission.evidence_frames)==2
    assert catalog.permission('B06','lift',store,2.) is None


def test_recognized_non_box_stays_outside_current_whitelist():
    store=RecognitionStore();catalog=InteractionCatalog()
    store.observe(evidence('chair','f1'));store.observe(evidence('chair','f2'))
    assert store.confirmed('B06',2.).category=='chair'
    assert catalog.permission('B06','push_aside',store,2.) is None


def test_stale_or_conflicting_evidence_does_not_authorize_new_skill():
    store=RecognitionStore(max_age_s=5);catalog=InteractionCatalog()
    store.observe(evidence(frame='f1'));store.observe(evidence(frame='f2'))
    assert catalog.permission('B06','push_aside',store,7.) is None
    store.observe(evidence('cabinet','f3'))
    assert catalog.permission('B06','push_aside',store,2.) is None
