from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation.side_push_navigation import SidePushPlanner
from grutopia_extension.interactive_navigation.local_turn_maneuver import turn_placement_clear
from grutopia_extension.interactive_navigation.local_staging_control import LocalStagingControl
from grutopia_extension.interactive_navigation.smooth_go2_navigation import VelocitySlewLimiter

def case():
    with np.load(Path(__file__).parent/'fixtures/go2_qualified_t02_placement_boundary.npz') as f:a={k:f[k] for k in f.files}
    p=SidePushPlanner(a['free'],a['limits'],resolution=.05);p.box_free=a['box_free'];p.planning_observed=a['observed'];p.planning_scope='local'
    p.navigation_route_radius=.30;p.navigation_half_width=.22;p.full_navigation_footprint=True;p.layered_local_control=True
    return a,p,a['position'],float(a['heading']),float(a['goal_heading'])

def test_grid_boundary_cannot_hide_unclear_neighbor_cells():
    a,p,pos,heading,goal=case()
    for perturb in [-1e-12,0.,1e-12]:
        assert not turn_placement_clear(p,[],a['old_target']+perturb,heading,goal,.05)
    assert turn_placement_clear(p,[],pos,heading,goal,.025)

def test_recorded_near_target_pose_turns_after_measured_stop_instead_of_micro_chasing():
    a,p,pos,heading,goal=case();control=LocalStagingControl(pos,heading,goal,p,[],0.)
    assert not control.failed
    limiter=VelocitySlewLimiter();limiter.policy_command=np.zeros(3)
    for now in [.01,.02,.3,.31]:command,state=control.update(pos,heading,p,[],limiter,now)
    assert state['staging_phase']=='staging_turn' and state['turn_placement_latched']
    assert state['turn_clearance_margin_m']>=.025 and abs(command[2])==1. and np.linalg.norm(command[:2])<1e-9
    assert np.array_equal(p.free,a['free']) and np.array_equal(p.planning_observed,a['observed'])
