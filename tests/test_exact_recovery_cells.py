from pathlib import Path
import json,math
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    _recovery_translation_samples,body_translation_recovery_clear,calibrated_buffer_translation,ForwardExplorationPathTracker)
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter,smooth_recovery_action


def snapshot():
    with np.load(Path(__file__).parent/'fixtures/t05_skipped_recovery_cell_v89.npz',allow_pickle=False) as z:
        p=SidePushPlanner(z['local_free'].copy(),z['limits'].tolist(),resolution=.05)
        p.box_free=z['box_free'].copy();p.planning_observed=z['observed'].copy()
        position=z['position'][:2].copy();heading=float(z['heading']);boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
        target=z['old_target'].copy()
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True;p.planning_scope='local';p.navigation_route_radius=.30
    return p,boxes,position,heading,target


def test_long_stride_cannot_skip_the_blocked_cell_that_rejects_its_short_prefix():
    p,boxes,position,heading,target=snapshot();unit=(target-position)/np.linalg.norm(target-position)
    assert not body_translation_recovery_clear(p,boxes,position,heading,position+unit*.015,require_clear_end=False)
    assert not body_translation_recovery_clear(p,boxes,position,heading,target)
    assert calibrated_buffer_translation(p,boxes,position,heading,target) is None


def test_real_t05_snapshot_selects_an_exit_with_executable_acceleration_and_prefixes():
    p,boxes,position,heading,old_target=snapshot();free=p.free.copy();observed=p.planning_observed.copy()
    box=next(b for b in boxes if b.id=='B07');goal=np.asarray(box.xy)-np.array([box.dimensions[0]/2+.33+.35,0.])
    tracker=ForwardExplorationPathTracker(position,[goal],guard_motion=False)
    limiter=VelocitySlewLimiter();limiter.accept([0.,0.,0.],0.);limiter.observe([0.,0.,0.],0.)
    for i in range(1,81):
        now=i*.01;p.control_time=now
        action,state=tracker.action(position,heading,p,boxes)
        applied=smooth_recovery_action(action,position,heading,p,boxes,limiter,now)
        limiter.accept(applied['move_by_speed'],now);limiter.observe([0.,0.,0.],now)
    assert np.linalg.norm(np.asarray(state['target'])-old_target)>.2
    assert np.linalg.norm(limiter.command)>.35  # old rollout was capped near.02
    unit=np.asarray(state['target'])-position;distance=np.linalg.norm(unit);unit/=distance
    for length in np.linspace(0.,distance,101):
        assert body_translation_recovery_clear(p,boxes,position,heading,position+unit*length,require_clear_end=False)
    np.testing.assert_array_equal(p.free,free);np.testing.assert_array_equal(p.planning_observed,observed)


def test_original_samples_are_retained_and_no_distance_bound_is_relaxed():
    position=np.array([.043,.017]);delta=np.array([.313,-.19]);resolution=.05
    samples=dict(_recovery_translation_samples(position,delta,[[0,2],[-2,2]],resolution))
    old=np.linspace(0.,1.,max(2,int(np.ceil(np.linalg.norm(delta)/resolution))+1))
    assert all(float(f) in samples for f in old)
    assert all(limit>=fraction for fraction,limit in samples.items())


def test_skipped_diagonal_cell_is_now_evaluated():
    p,_,position,_,target=snapshot();unit=(target-position)/np.linalg.norm(target-position);delta=unit*.3630769230769231
    samples=_recovery_translation_samples(position,delta,p.limits,.05)
    cells={p.cell(position+fraction*delta) for fraction,_ in samples}
    assert (133,32) in cells


def test_cell_interval_end_enforces_progress_even_after_its_midpoint():
    position=np.array([.043,.025]);delta=np.array([.12,0.])
    samples=_recovery_translation_samples(position,delta,[[0,2],[0,2]],.05)
    assert any(fraction*.12<.1<limit*.12 for fraction,limit in samples)
