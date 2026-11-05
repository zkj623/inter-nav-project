import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grutopia_extension.interactive_navigation.online_navigation_decision import OnlineGoalSelector


def test_frontier_turn_uses_short_angular_distance_across_pi():
    layers=SimpleNamespace(robot_yaw=math.pi-.1)
    path=[[-math.cos(.1),-math.sin(.1),0.]]
    assert OnlineGoalSelector.initial_turn_cost(layers,(0.,0.),path)==pytest.approx(.06)


def test_frontier_turn_ignores_zero_length_prefix_and_unavailable_heading():
    path=[[0.,0.,0.],[.01,0.,0.],[0.,1.,0.]]
    assert OnlineGoalSelector.initial_turn_cost(SimpleNamespace(robot_yaw=math.pi/2),(0.,0.),path)==pytest.approx(0.)
    assert OnlineGoalSelector.initial_turn_cost(SimpleNamespace(),(0.,0.),path)==0.


def test_recorded_t07_prefers_observed_exit_with_less_turning_without_changing_map(monkeypatch):
    from grutopia_extension.interactive_navigation.mapping import MappingConfig
    from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
    from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
    from grutopia_extension.interactive_navigation.online_navigation_map import OnlineNavigationMap
    f=Path(__file__).parent/'fixtures/go2_t07_frontier_turn_cost'
    meta=json.loads(f.with_suffix('.json').read_text())
    with np.load(f.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
    cfg=MappingConfig(x_limits=tuple(meta['limits'][0]),y_limits=tuple(meta['limits'][1]),grid_resolution=.05,robot_radius=.25,obstacle_inflation_radius=.30)
    p=SidePushPlanner(np.zeros(a['global_observed'].shape,bool),meta['limits'],resolution=.05)
    p.navigation_route_radius=.30;p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    layers=OnlineNavigationMap(p,cfg)
    layers.global_observed[:]=a['global_observed'];layers.observed[:]=a['observed'];layers.static_occupied[:]=a['static']
    layers.global_boxes=[BoxState(**x) for x in meta['global_boxes']];layers.global_tracked=layers._mask(layers.global_boxes)
    w,x,y,z=meta['orientation'];layers.robot_yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));p.robot_heading=layers.robot_yaw
    before=[v.copy() for v in [layers.global_observed,layers.observed,layers.static_occupied,layers.global_tracked]]
    with monkeypatch.context() as patch:
        patch.setattr(OnlineGoalSelector,'initial_turn_cost',staticmethod(lambda *args:0.))
        old,old_endpoint=OnlineGoalSelector().decide(layers,meta['start'],meta['goal'])
    chosen,endpoint=OnlineGoalSelector().decide(layers,meta['start'],meta['goal'])
    assert old.kind==chosen.kind=='explore'
    assert np.allclose(old_endpoint,[9.075,2.975]) and np.allclose(endpoint,[11.325,.025])
    assert chosen.explanation['initial_turn_cost_m']<OnlineGoalSelector.initial_turn_cost(layers,meta['start'],old.path)
    with layers.global_planning() as boxes:
        assert p.path(meta['start'],endpoint,boxes)
        assert layers.global_observed[p.cell(endpoint)]
    for initial,current in zip(before,[layers.global_observed,layers.observed,layers.static_occupied,layers.global_tracked]):
        np.testing.assert_array_equal(initial,current)
