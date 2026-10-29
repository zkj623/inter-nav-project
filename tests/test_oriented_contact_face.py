"""An expanded world AABB must not masquerade as a rotated contact face."""
from dataclasses import replace
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_contact_control import observed_contact_face_clear
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner


def yaw_quaternion(angle):return (math.cos(angle/2),0.,0.,math.sin(angle/2))


def test_real_axis_aligned_face_keeps_four_centimetres_at_the_head_edges():
    args=dict(position=(.1,.049,.42),orientation=(1,0,0,0),box_xy=(1.3,0),center_z=.36,
        local_dimensions=(.85,.28,.7),box_orientation=(1,0,0,0),direction=(1,0))
    assert observed_contact_face_clear(**args)
    args['position']=(.1,.051,.42)
    assert not observed_contact_face_clear(**args)


def test_rotated_face_rejects_a_world_centerline_that_misses_its_contact_center():
    args=dict(position=(.1,0.,.42),orientation=(1,0,0,0),box_xy=(1.3,0),center_z=.36,
        local_dimensions=(.85,.65,.7),box_orientation=yaw_quaternion(.2),direction=(1,0))
    assert not observed_contact_face_clear(**args)
    # Center the actual padded head envelope in the box's rotated frame.
    head_center_x=.1+(.239+.341)/2
    args['position']=(.1,math.tan(.2)*(head_center_x-1.3),.42)
    assert observed_contact_face_clear(**args)


def test_pitched_or_rolled_box_uses_its_full_observed_orientation():
    args=dict(position=(.1,.06,.45),orientation=(1,0,0,0),box_xy=(1.3,0),center_z=.36,
        local_dimensions=(.85,.28,.7),box_orientation=(1,0,0,0),direction=(1,0))
    assert not observed_contact_face_clear(**args)
    args['local_dimensions']=(.85,.65,.7)
    args['box_orientation']=(math.cos(.1),math.sin(.1),0,0)
    assert observed_contact_face_clear(**args)


def test_metadata_does_not_double_rotate_the_planners_world_aabb():
    angle=.2;local=np.array([.85,.65,.7])
    rotation=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    bounds=tuple(np.abs(rotation)@local)
    old=BoxState('B07',(1.3,0.),bounds,0.,5.)
    observed=replace(old,local_dimensions=tuple(local),orientation=yaw_quaternion(angle),center_z=.36)
    planner=SidePushPlanner(np.ones((160,160),bool),[[-4,4],[-4,4]],resolution=.05)
    np.testing.assert_array_equal(planner.box_mask(old),planner.box_mask(observed))
    moved=replace(observed,xy=(1.5,0.))
    assert moved.orientation==observed.orientation and moved.local_dimensions==observed.local_dimensions


def test_invalid_or_missing_observed_orientation_never_grants_contact():
    args=dict(position=(.1,0.,.42),orientation=(1,0,0,0),box_xy=(1.3,0),center_z=.36,
        local_dimensions=(.85,.65,.7),box_orientation=(0,0,0,0),direction=(1,0))
    assert not observed_contact_face_clear(**args)
    args['box_orientation']=(1,0,0,0);args['orientation']=(float('nan'),0,0,0)
    assert not observed_contact_face_clear(**args)
