from pathlib import Path
import numpy as np
from grutopia_extension.interactive_navigation.route_progress import trim_initial_attachment
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


def test_recorded_t07_does_not_turn_back_to_the_nearest_skeleton_spur():
    with np.load(Path(__file__).parent/'fixtures/go2_t07_initial_attachment.npz') as data:
        start=data['start'];path=data['path'];blocked=data['blocked'];limits=data['limits'];resolution=float(data['resolution'])
    before=blocked.copy()
    cell=lambda xy:(int((xy[1]-limits[1,0])/resolution),int((xy[0]-limits[0,0])/resolution))
    clear=lambda a,b:AStarMapPlanner.segment_clear(cell(a),cell(b),blocked)
    assert path[0,0]<start[0] and path[0,1]>start[1]
    route,skipped=trim_initial_attachment(start,path,clear)
    assert skipped==5 and route[0,0]>start[0] and route[0,1]<start[1]
    assert clear(start,route[0]) and np.array_equal(route,path[skipped:])
    assert np.array_equal(before,blocked)


def test_blocked_or_unknown_chord_keeps_original_corridor():
    path=((0.,.3),(0.,.6),(.4,.6),(.8,.6))
    route,skipped=trim_initial_attachment((0.,0.),path,lambda a,b:False)
    assert skipped==0 and route==path


def test_attachment_never_rewrites_later_skeleton_bends():
    path=((0.,.3),(-.3,.3),(-.3,0.),(.3,0.),(3.,0.),(3.,3.))
    route,skipped=trim_initial_attachment((0.,0.),path,lambda a,b:True)
    assert skipped==3 and route==path[3:]
    assert route[-2:]==((3.,0.),(3.,3.))


def test_straight_initial_route_is_unchanged():
    path=((.3,0.),(.6,0.),(.9,0.))
    assert trim_initial_attachment((0.,0.),path,lambda a,b:True)==(path,0)
