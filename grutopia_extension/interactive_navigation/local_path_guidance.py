"""Connect a safe local pose back to a forward waypoint on the global route."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap,AStarMapPlanner,PlanningError
from grutopia_extension.interactive_navigation.semantic_voronoi import VoronoiPathPlanner
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked


def local_guidance_path(planner,start,global_path,boxes,heading,local_radius=3.):
    if not global_path:return (),{'reason':'no_global_guidance'}
    points=np.asarray(global_path,dtype=float)
    distances=np.linalg.norm(points[:,:2]-np.asarray(start[:2]),axis=1)
    nearest=int(np.argmin(distances))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1))]
    anchors=[i for i in range(nearest,len(points)) if .15<distances[i]<local_radius-.65 and arc[i]-arc[nearest]<=1.0]
    if not anchors:anchors=[nearest]
    blocked=navigation_body_blocked(planner,boxes,heading)
    row,col=planner.cell(start);extent=int(local_radius/planner.resolution)
    window=np.zeros_like(blocked)
    window[max(0,row-extent):min(blocked.shape[0],row+extent+1),
           max(0,col-extent):min(blocked.shape[1],col+extent+1)]=True
    blocked |= ~window
    # Stay near the chosen global branch instead of taking a shortcut through
    # the opposite (narrow) side of the same obstacle.
    guide=np.zeros_like(blocked)
    for a,b in zip(points,points[1:]):
        for xy in np.linspace(a[:2],b[:2],max(2,int(np.linalg.norm(b[:2]-a[:2])/planner.resolution)+1)):
            cell=planner.cell(xy)
            if planner.valid_cell(cell):guide[cell]=True
    for xy in points[:,:2]:
        cell=planner.cell(xy)
        if planner.valid_cell(cell):guide[cell]=True
    tube=distance_transform_edt(~guide)*planner.resolution<=.60
    rr,cc=np.ogrid[:blocked.shape[0],:blocked.shape[1]]
    join_radius=min(local_radius-.5,float(distances[nearest])+.25)
    tube |= (rr-row)**2+(cc-col)**2 <= (join_radius/planner.resolution)**2
    blocked |= ~tube
    config=replace(planner.config,robot_radius=planner.resolution/10,
        obstacle_inflation_radius=planner.resolution/10,static_obstacle_inflation_radius=planner.resolution/10)
    occupancy=OccupancyGridMap(config);h,w=occupancy.log_odds.shape
    occupancy.static_occupied[:]=blocked[:h,:w];occupancy.log_odds[:]=-4;occupancy.observed[:]=True
    astar=AStarMapPlanner(occupancy,observed_only=True)
    sampler=VoronoiPathPlanner(SimpleNamespace(occupancy=occupancy),waypoint_spacing=.20,observed_only=True)
    for index in reversed(anchors):
        anchor=tuple(map(float,points[index]))
        try:raw=astar.plan((*start[:2],0.),anchor)
        except PlanningError:continue
        cells=[occupancy.world_to_cell(start[:2])]+[occupancy.world_to_cell(point[:2]) for point in raw]
        sampled=sampler._sample_along_path(cells,occupancy.inflated_mask(),planner.resolution)
        prefix=tuple((*occupancy.cell_to_world(cell),0.) for cell in sampled[1:])
        prefix=prefix[:-1]+(anchor,) if prefix else (anchor,)
        return prefix+tuple(tuple(map(float,p)) for p in points[index+1:]),{
            'engine':'local_grid_connector','anchor_index':index,'anchor':anchor,
            'anchor_path_distance_m':float(arc[index]-arc[nearest]),
            'local_prefix':prefix,'local_radius_m':local_radius}
    return (),{'engine':'local_grid_connector','reason':'no_safe_local_connection'}
