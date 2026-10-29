from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import json
import math
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState,PushPlan
from grutopia_extension.interactive_navigation.interaction_catalog import InteractionPermission
from grutopia_extension.interactive_navigation.push_aside_skill import PushAsideSkill
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_rotation_clear


def fixture(heading_offset=0.):
    with np.load(Path(__file__).parent/'fixtures/t04_reuse_heading_wrap_v104.npz') as z:
        p=SidePushPlanner(z['free'].copy(),z['limits'].tolist(),resolution=.05)
        p.planning_observed=z['observed'].copy();position=z['position'].copy()
        heading=float(z['heading'])+heading_offset
        boxes=[BoxState(**b) for b in json.loads(str(z['boxes_json']))]
        permission=InteractionPermission(**json.loads(str(z['permission_json'])))
        plan=PushPlan(**json.loads(str(z['plan_json'])))
    p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    p.planning_scope='local';p.robot_heading=heading
    box=next(b for b in boxes if b.id=='B06')
    @contextmanager
    def planning():yield boxes
    skill=PushAsideSkill(permission,plan,[2.11,-.78],[10.3,1.75])
    ctx=NS(position=position,heading=heading,planner=p,boxes=boxes,box=box,layers=NS(global_planning=planning))
    return skill,ctx


def test_real_retreat_pose_reuses_short_checked_rotation_not_full_revolution():
    skill,ctx=fixture();before=ctx.planner.free.copy()
    assert abs(skill._box_frame(ctx)[2])<.003
    assert not body_rotation_clear(ctx.planner,ctx.boxes,ctx.position,ctx.heading,skill.push_heading,margin=.025)
    assert skill._reuse_alignment_pose(ctx)
    np.testing.assert_array_equal(skill.stage,ctx.position[:2])
    np.testing.assert_array_equal(ctx.planner.free,before)


def test_equivalent_yaw_representations_produce_same_reused_pose():
    first,ctx=fixture();second,other=fixture(2*math.pi)
    assert first._reuse_alignment_pose(ctx) and second._reuse_alignment_pose(other)
    np.testing.assert_allclose(first.stage,second.stage,atol=1e-9)
    np.testing.assert_allclose(first.anchor,second.anchor,atol=1e-9)


def test_stage_search_is_invariant_to_equivalent_arrival_heading():
    first,ctx=fixture();second,other=fixture(2*math.pi)
    assert first._select_stage(ctx) and second._select_stage(other)
    np.testing.assert_allclose(first.stage,second.stage,atol=1e-8)
    np.testing.assert_allclose(first.anchor,second.anchor,atol=1e-8)


def test_short_arc_does_not_bypass_real_collision_checks():
    skill,ctx=fixture();ctx.planner.free[:]=False
    assert not skill._reuse_alignment_pose(ctx)
