"""Reuse exploration Voronoi routes and forward-facing tracking for Go2."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.mapping import (
    AStarMapPlanner, OccupancyGridMap, PlanningError,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import (
    SemanticVoronoiGraph, VoronoiPathPlanner,
)
from grutopia_extension.interactive_navigation.go2_exploration_tracking import ExplorationPathTracker
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime


class ConnectedVoronoiPathPlanner(VoronoiPathPlanner):
    """Attach both endpoints to a shared reachable exploration skeleton.

    Independent nearest attachments can select two small disconnected spurs
    even when both endpoints can reach the main corridor skeleton.
    """
    def plan(self,start,goal):
        self.shared_component_used=False
        try:return super().plan(start,goal)
        except PlanningError as original:
            if 'different skeleton components' not in str(original):raise
            failure=original
        snapshot=self.graph.cached_snapshot();occupancy=self.graph.occupancy
        blocked=occupancy.inflated_mask() | ~occupancy.observed
        remaining={cell for cell in snapshot.skeleton_cells if not blocked[cell]}
        components=[]
        while remaining:
            seed=remaining.pop();component={seed};queue=[seed]
            while queue:
                row,col=queue.pop()
                for dr,dc,_ in self._NEIGHBORS:
                    neighbor=(row+dr,col+dc)
                    if neighbor not in remaining:continue
                    if dr and dc and (blocked[row,col+dc] or blocked[row+dr,col]):continue
                    remaining.remove(neighbor);component.add(neighbor);queue.append(neighbor)
            components.append(component)
        for component in sorted(components,key=len,reverse=True):
            if len(component)<2:continue
            filtered=replace(snapshot,skeleton_cells=tuple(sorted(component)))
            class ComponentGraph:
                def cached_snapshot(self):return filtered
            graph=ComponentGraph();graph.occupancy=occupancy
            try:
                path=VoronoiPathPlanner(graph,attach_radius=self.attach_radius,
                    waypoint_spacing=self.waypoint_spacing,observed_only=self.observed_only).plan(start,goal)
            except PlanningError:continue
            self.shared_component_used=True
            return path
        raise failure


class ExplorationVoronoiRoutes:
    """The exploration skeleton planner over the current known collision layer.

    Contact/parking feasibility remains the SidePushPlanner's responsibility.
    Ordinary movement follows 30 cm skeleton waypoints without A* chord cuts.
    """
    def __init__(self, planner, cache_size=8):
        self.planner=planner
        self.cache_size=cache_size;self._cache=OrderedDict()
        self.voronoi_plans=0;self.grid_fallbacks=0;self.graph_builds=0
        self.last_plan={}

    def plan(self,start,goal,boxes):
        p=self.planner
        # Reuse the existing body clearance instead of adding another circular
        # inflation on top; that extra inflation disconnects real narrow paths.
        blocked=route_blocked(p,boxes)
        key=hashlib.blake2b(blocked.tobytes(),digest_size=16).digest()
        if key not in self._cache:
            # This mask already includes clearance: do not inflate it twice.
            cfg=replace(p.config,robot_radius=p.resolution/10,
                obstacle_inflation_radius=p.resolution/10,
                static_obstacle_inflation_radius=p.resolution/10)
            occupancy=OccupancyGridMap(cfg);h,w=occupancy.log_odds.shape
            occupancy.static_occupied[:]=blocked[:h,:w]
            occupancy.log_odds[:]=-4;occupancy.observed[:]=True
            graph=SemanticVoronoiGraph(occupancy);graph.update()
            engine=ConnectedVoronoiPathPlanner(graph,waypoint_spacing=.30,observed_only=True)
            self._cache[key]=(engine,blocked[:h,:w].copy())
            self.graph_builds+=1
            if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        engine,_=self._cache[key]
        try:
            path=engine.plan((*start[:2],0.),(*goal[:2],0.))
        except PlanningError as error:
            self.grid_fallbacks+=1
            path=p.path(start,goal,boxes)
            self.last_plan={'engine':'grid_astar_fallback','fallback_reason':str(error)}
            return path
        self.voronoi_plans+=1
        self.last_plan={'engine':'exploration_voronoi','waypoint_spacing_m':.30,
            'skeleton_cells':len(engine.graph.cached_snapshot().skeleton_cells),
            'shared_component_attachment':engine.shared_component_used}
        return path

    def statistics(self):
        return {'voronoi_plans':self.voronoi_plans,'grid_fallbacks':self.grid_fallbacks,
            'graph_builds':self.graph_builds,'route_radius_m':getattr(self.planner,'navigation_route_radius',self.planner.radius),
            'roadmap_pose_heading':getattr(self.planner,'navigation_pose_heading',None),
            'execution_clearance':'oriented_Go2_body_motion_sweep',
            'map_source':'known_collision_prior_and_simulator_box_poses'}


class ForwardExplorationPathTracker(ExplorationPathTracker):
    """Use the exploration carrot/yaw law, facing the path before walking."""
    def __init__(self,start,path,lookahead=.55,guard_motion=True):
        super().__init__(start,path,lookahead=lookahead,fixed_heading=None)
        self.guard_motion=guard_motion
        self._buffer_replan=False
        self._buffer_active=False;self._buffer_target=None
        self._turn_ramp_ticks=0;self._turn_ramp_sign=0.
        self._heading_recovery=False

    def action(self,position,heading,planner,boxes):
        planner.robot_heading=heading
        self._route_blocked=route_blocked(planner,boxes)
        combined=self._route_blocked | navigation_body_blocked(planner,boxes,heading)
        cell=planner.cell(position)
        nominal=getattr(planner,'navigation_pose_heading',None)
        if nominal is not None and planner.valid_cell(cell):
            error=(nominal-heading+math.pi)%(2*math.pi)-math.pi
            nominal_free=not navigation_body_blocked(planner,boxes,nominal)[cell]
            current_blocked=navigation_body_blocked(planner,boxes,heading)[cell]
            if self._heading_recovery and not current_blocked and abs(error)<.06:
                self._heading_recovery=False;self._buffer_replan=True
                return {'move_by_speed':[0.,0.,0.]},{'target':list(map(float,position[:2])),'recovering_buffer':False,
                    'needs_replan':True,'buffer_cleared_replan':True,'controller':'footprint_heading_recovery'}
            if (self._heading_recovery or current_blocked) and nominal_free and abs(error)>.03:
                if turn_reduces_overlap(planner,boxes,position,heading,heading+error):
                    self._heading_recovery=True;self._buffer_active=False;self._buffer_target=None;self._buffer_replan=False
                    rotation=math.copysign(min(1.,.40+.01*self._turn_ramp_ticks),error)
                    self._turn_ramp_ticks+=1
                    return {'move_by_speed':[0.,0.,rotation]},{'target':list(map(float,position[:2])),'recovering_buffer':True,
                        'needs_replan':False,'controller':'footprint_heading_recovery','heading_error_rad':float(error)}
        if planner.valid_cell(cell) and (combined[cell] or self._buffer_active):
            clearance=distance_transform_edt(~combined)*planner.resolution
            if self._buffer_active and not combined[cell] and clearance[cell]>.08:
                self._buffer_active=False;self._buffer_replan=True
                return {'move_by_speed':[0.,0.,0.]},{'target':self._buffer_target.tolist(),
                    'recovering_buffer':False,'needs_replan':True,'buffer_cleared_replan':True,
                    'controller':'calibrated_navigation_buffer_exit'}
            raw=planner.free.copy()
            for box in boxes:raw &= ~planner.box_mask(box)
            safe=distance_transform_edt(~combined)*planner.resolution>.12
            if not safe.any():safe=~combined
            _,nearest=distance_transform_edt(~safe,return_indices=True)
            row,col=nearest[:,cell[0],cell[1]]
            candidate=np.array([planner.limits[0][0]+(col+.5)*planner.resolution,
                planner.limits[1][0]+(row+.5)*planner.resolution])
            if self._buffer_target is None or combined[planner.cell(self._buffer_target)]:self._buffer_target=candidate
            target=self._buffer_target;self._buffer_active=True
            state={'target':target.tolist(),'recovering_buffer':True,'needs_replan':False,
                'controller':'calibrated_navigation_buffer_exit'}
            if not raw[cell] or not AStarMapPlanner.segment_clear(cell,planner.cell(target),~raw):
                state['needs_replan']=True
                return {'move_by_speed':[0.,0.,0.]},state
            delta=target-np.asarray(position[:2]);world=.25*delta/max(np.linalg.norm(delta),1e-9)
            c,s=math.cos(heading),math.sin(heading)
            command=[float(c*world[0]+s*world[1]),float(-s*world[0]+c*world[1]),0.]
            return {'move_by_speed':command},state
        action,state=super().action(position,heading,planner,boxes)
        state['controller']='exploration_carrot_and_path_heading'
        if self._buffer_replan:
            state['needs_replan']=True;state['buffer_cleared_replan']=True
            return {'move_by_speed':[0.,0.,0.]},state
        if state['recovering_buffer']:
            raw=planner.free.copy()
            for box in boxes:raw &= ~planner.box_mask(box)
            if not AStarMapPlanner.segment_clear(planner.cell(position),planner.cell(state['target']),~raw):
                state['needs_replan']=True
                return {'move_by_speed':[0.,0.,0.]},state
            command=action['move_by_speed'];norm=np.linalg.norm(command[:2])
            if norm>1e-9:command[:2]=[float(v*.25/norm) for v in command[:2]]
            command[2]=0.;state['controller']='calibrated_clearance_buffer_exit'
            return action,state
        if state['needs_replan']:return action,state
        if not self.guard_motion:
            state['controller']='pure_exploration_velocity_action_for'
            vf,vl,rotation=action['move_by_speed']
            if not body_motion_clear(planner,boxes,position,heading,vf,rotation,lateral=vl):
                if np.linalg.norm([vf,vl])>1e-8 and body_motion_clear(planner,boxes,position,heading,vf,0.,lateral=vl):
                    action={'move_by_speed':[vf,vl,0.]};state['foot_clearance_translation']=True
                else:
                    delta=np.asarray(state['target'])-np.asarray(position[:2]);c,s=math.cos(heading),math.sin(heading)
                    body_delta=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])
                    candidates=[]
                    for axis in [0,1]:
                        if abs(body_delta[axis])<1e-5:continue
                        command=np.zeros(2);command[axis]=math.copysign(.25,body_delta[axis])
                        if body_motion_clear(planner,boxes,position,heading,float(command[0]),0.,lateral=float(command[1])):
                            candidates.append((abs(body_delta[axis]),command))
                    if candidates:
                        _,command=max(candidates,key=lambda entry:entry[0])
                        action={'move_by_speed':[float(command[0]),float(command[1]),0.]}
                        state['foot_clearance_axis_step']=True
                        return action,state
                    if abs(rotation)>1e-8 and body_motion_clear(planner,boxes,position,heading,0.,rotation):
                        action={'move_by_speed':[0.,0.,rotation]};state['foot_clearance_turn']=True
                    else:
                        state['needs_replan']=True;state['foot_motion_blocked']=True
                        action={'move_by_speed':[0.,0.,0.]}
            return action,state
        delta=np.asarray(state['target'])-np.asarray(position[:2])
        distance=float(np.linalg.norm(delta))
        if distance<1e-6:return action,state
        desired=math.atan2(delta[1],delta[0])
        error=(desired-heading+math.pi)%(2*math.pi)-math.pi
        state['heading_error_rad']=float(error)
        rotation=float(action['move_by_speed'][2])
        # Keep the original exploration world direction, including its small
        # turn corrections. Scale both body components together so the
        # checked carrot chord is preserved while yaw catches up smoothly.
        factor=max(0.,math.cos(error))**2 if abs(error)<math.pi/2 else 0.
        forward=float(action['move_by_speed'][0])*factor
        lateral=float(action['move_by_speed'][1])*factor
        coupled=body_motion_clear(planner,boxes,position,heading,forward,rotation,lateral=lateral)
        translation=body_motion_clear(planner,boxes,position,heading,forward,0.,lateral=lateral)
        if not coupled or not translation:
            turn=math.copysign(max(abs(rotation),1.0),rotation) if abs(rotation)>1e-8 else 0.
            if abs(turn)>1e-8 and body_motion_clear(planner,boxes,position,heading,0.,turn):
                forward=0.;lateral=0.;rotation=turn;state['motion_adjustment']='turn_before_advancing'
            elif np.linalg.norm([forward,lateral])>1e-8 and translation:
                rotation=0.;state['motion_adjustment']='advance_to_turn_clearance'
            else:
                state['needs_replan']=True;state['blocked_motion_sweep']=True
                return {'move_by_speed':[0.,0.,0.]},state
        if np.linalg.norm([forward,lateral])<1e-8 and abs(rotation)>1e-8:
            sign=math.copysign(1.,rotation)
            if sign!=self._turn_ramp_sign:self._turn_ramp_ticks=0
            self._turn_ramp_sign=sign
            limit=.40+.01*self._turn_ramp_ticks
            rotation=sign*min(abs(rotation),limit)
            self._turn_ramp_ticks+=1
            state['ramped_in_place_turn']=True
        else:self._turn_ramp_ticks=0;self._turn_ramp_sign=0.
        return {'move_by_speed':[forward,lateral,rotation]},state

    def _tracking_segment_clear(self,start,target):
        return AStarMapPlanner.segment_clear(self._planner.cell(start),self._planner.cell(target),self._route_blocked)

    def _lookahead_target(self,position):
        if not self.guard_motion:
            original=self.velocity_lookahead_distance
            try:
                for distance in [original,.35,.20,.10,.05,.025]:
                    if distance>original:continue
                    self.velocity_lookahead_distance=distance
                    target=MapNavigationRuntime._lookahead_target(self,np.asarray(position[:2]))
                    if self._tracking_segment_clear(position,target):return target
                return target
            finally:self.velocity_lookahead_distance=original
        return super()._lookahead_target(position)


def route_blocked(planner,boxes):
    """Use the existing heading-independent navigation clearance for the graph."""
    pose=getattr(planner,'navigation_pose_heading',None)
    if pose is not None:return navigation_body_blocked(planner,boxes,pose)
    radius=getattr(planner,'navigation_route_radius',planner.radius)
    if radius!=planner.radius:
        raw=planner.free.copy()
        for box in boxes:
            mask=planner.box_mask(box)
            if mask is None:raise ValueError('Box outside navigation grid')
            raw &= ~mask
        return distance_transform_edt(raw)*planner.resolution<=radius
    if not hasattr(planner,'robot_heading'):return planner.blocked(boxes)
    saved=planner.robot_heading
    try:
        planner.robot_heading=None
        return planner.blocked(boxes)
    finally:planner.robot_heading=saved


def navigation_body_blocked(planner,boxes,heading):
    """Protect the full standing/gait footprint, including front paws."""
    import cv2
    raw=planner.free.copy()
    for box in boxes:raw &= ~planner.box_mask(box)
    xs=np.arange(-.32,.501,planner.resolution);ys=np.arange(-.22,.221,planner.resolution)
    xx,yy=np.meshgrid(xs,ys);c,s=math.cos(heading),math.sin(heading)
    dx=np.rint((c*xx-s*yy)/planner.resolution).astype(int).ravel()
    dy=np.rint((s*xx+c*yy)/planner.resolution).astype(int).ravel()
    x0,x1=dx.min(),dx.max();y0,y1=dy.min(),dy.max()
    kernel=np.zeros((y1-y0+1,x1-x0+1),np.uint8);kernel[dy-y0,dx-x0]=1
    valid=cv2.erode(raw.astype(np.uint8),kernel,anchor=(-x0,-y0),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return ~valid.astype(bool)


def turn_reduces_overlap(planner,boxes,position,first,last):
    """Permit recovery only when modeled foot overlap does not worsen."""
    raw=planner.free.copy()
    for box in boxes:raw &= ~planner.box_mask(box)
    row,col=planner.cell(position)
    xs=np.arange(-.32,.501,planner.resolution);ys=np.arange(-.22,.221,planner.resolution)
    xx,yy=np.meshgrid(xs,ys);counts=[]
    for angle in np.linspace(first,last,19):
        c,s=math.cos(angle),math.sin(angle)
        cols=col+np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        rows=row+np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        if rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]:return False
        counts.append(int((~raw[rows,cols]).sum()))
    return counts[-1]==0 and max(counts)<=counts[0]


def body_motion_clear(planner,boxes,position,heading,forward,rotation,horizon=.25,lateral=0.):
    """Check the body over the next coupled forward/yaw movement, not a full
    future turn about a stationary center. Pose spacing is below one map cell.
    """
    raw=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return False
        raw &= ~mask
    xs=np.arange(-.32,.501,planner.resolution);ys=np.arange(-.22,.221,planner.resolution)
    xx,yy=np.meshgrid(xs,ys)
    for t in np.linspace(0.,horizon,19):
        angle=heading+rotation*t
        if abs(rotation)>1e-8:
            delta=forward/rotation*np.array([math.sin(angle)-math.sin(heading),math.cos(heading)-math.cos(angle)])
            delta+=lateral/rotation*np.array([math.cos(angle)-math.cos(heading),math.sin(angle)-math.sin(heading)])
        else:delta=t*np.array([forward*math.cos(heading)-lateral*math.sin(heading),
            forward*math.sin(heading)+lateral*math.cos(heading)])
        row,col=planner.cell(np.asarray(position[:2])+delta)
        c,s=math.cos(angle),math.sin(angle)
        cols=col+np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        rows=row+np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        if rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]:return False
        if not raw[rows,cols].all():return False
    return True


def body_rotation_clear(planner,boxes,position,first_heading,last_heading):
    """Check the same oriented Go2 footprint locally through a heading sweep."""
    raw=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return False
        raw &= ~mask
    row,col=planner.cell(position)
    xs=np.arange(-.32,.501,planner.resolution)
    ys=np.arange(-.22,.221,planner.resolution)
    xx,yy=np.meshgrid(xs,ys)
    for heading in np.linspace(first_heading,last_heading,19):
        c,s=math.cos(heading),math.sin(heading)
        cols=col+np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        rows=row+np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        if rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]:return False
        if not raw[rows,cols].all():return False
    return True


def update_task_cameras(first_camera,third_camera,position,orientation):
    """Same actual-yaw update as the pure exploration recording session."""
    first_camera.update_pose(position,orientation)
    third_camera.update_pose(position,orientation)
