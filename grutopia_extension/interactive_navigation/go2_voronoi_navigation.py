"""Reuse exploration Voronoi routes and forward-facing tracking for Go2."""
from collections import OrderedDict
from dataclasses import replace
import hashlib
import math
from heapq import heappop,heappush
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


class ClearanceWeightedVoronoiPlanner(VoronoiPathPlanner):
    """Prefer usable clearance over a marginally shorter skeleton branch."""
    def __init__(self,*args,clearance_weight=.30,**kwargs):
        super().__init__(*args,**kwargs)
        self.clearance_weight=float(clearance_weight)
        if self.clearance_weight<0:raise ValueError('clearance_weight must be nonnegative')
        self.last_weighted_cost=0.

    def _skeleton_path(self,start,goal,skeleton,blocked):
        resolution=self.graph.occupancy.config.grid_resolution
        clearance=distance_transform_edt(~blocked)*resolution
        cost=1.+self.clearance_weight/np.maximum(clearance,resolution)
        frontier=[(0.,start)];scores={start:0.};parents={}
        while frontier:
            _,current=heappop(frontier)
            if current==goal:
                path=[goal]
                while path[-1]!=start:path.append(parents[path[-1]])
                self.last_weighted_cost=scores[goal]*resolution
                return list(reversed(path))
            for dr,dc,step_cost in self._NEIGHBORS:
                neighbor=(current[0]+dr,current[1]+dc)
                if neighbor not in skeleton:continue
                if dr and dc and (blocked[current[0],neighbor[1]] or blocked[neighbor[0],current[1]]):continue
                candidate=scores[current]+step_cost*.5*(cost[current]+cost[neighbor])
                if candidate>=scores.get(neighbor,float('inf')):continue
                scores[neighbor]=candidate;parents[neighbor]=current
                estimate=math.hypot(goal[0]-neighbor[0],goal[1]-neighbor[1])
                heappush(frontier,(candidate+estimate,neighbor))
        raise PlanningError('start and goal connect to different skeleton components')


class ConnectedVoronoiPathPlanner(ClearanceWeightedVoronoiPlanner):
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
                path=ClearanceWeightedVoronoiPlanner(graph,attach_radius=self.attach_radius,
                    waypoint_spacing=self.waypoint_spacing,observed_only=self.observed_only,
                    clearance_weight=self.clearance_weight).plan(start,goal)
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
            engine=ConnectedVoronoiPathPlanner(graph,waypoint_spacing=.30,observed_only=True,
                clearance_weight=getattr(p,'voronoi_clearance_weight',.30))
            self._cache[key]=(engine,blocked[:h,:w].copy())
            self.graph_builds+=1
            if len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        engine,_=self._cache[key]
        try:
            path=engine.plan((*start[:2],0.),(*goal[:2],0.))
        except PlanningError as error:
            self.grid_fallbacks+=1
            # Keep the exact same configured roadmap occupancy for fallback;
            # using the older short-body mask could invent an unsafe shortcut.
            occupancy=engine.graph.occupancy
            try:
                raw=AStarMapPlanner(occupancy,observed_only=True).plan((*start[:2],0.),(*goal[:2],0.))
                cells=[occupancy.world_to_cell(start[:2])]+[occupancy.world_to_cell(point[:2]) for point in raw]
                sampled=engine._sample_along_path(cells,occupancy.inflated_mask(),p.resolution)
                path=tuple((*occupancy.cell_to_world(cell),0.) for cell in sampled[1:])
            except PlanningError:path=()
            self.last_plan={'engine':'grid_astar_fallback','fallback_reason':str(error),
                'same_roadmap_occupancy':True}
            return path
        self.voronoi_plans+=1
        self.last_plan={'engine':'exploration_voronoi','waypoint_spacing_m':.30,
            'clearance_weight_m':engine.clearance_weight,
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
        self._local_turn_ticks=0;self._local_turn_sign=0.
        self._corner_maneuver=None
        self._gait_turn_target=None
        self._gait_brake_ticks=0
        self._gait_brake_positions=[]
        self._gait_brake_headings=[]

    def action(self,position,heading,planner,boxes):
        planner.robot_heading=heading
        if self._gait_turn_target is not None:
            self._gait_brake_ticks+=1
            self._gait_brake_positions.append(np.asarray(position[:2],float).copy())
            self._gait_brake_headings.append(float(heading))
            self._gait_brake_positions=self._gait_brake_positions[-21:]
            self._gait_brake_headings=self._gait_brake_headings[-21:]
            settled=(len(self._gait_brake_positions)==21 and
                max(np.linalg.norm(p-self._gait_brake_positions[-1]) for p in self._gait_brake_positions)<.02 and
                max(abs((h-heading+math.pi)%(2*math.pi)-math.pi) for h in self._gait_brake_headings)<.035)
            state={'target':self._gait_turn_target.tolist(),'recovering_buffer':True,
                'needs_replan':False,'local_turn_maneuver':True,'controller':'gait_turn_braking',
                'braking_ticks':self._gait_brake_ticks,'pose_settled':bool(settled)}
            if self._gait_brake_ticks>=400 and not settled:
                self._gait_turn_target=None
                state.update(needs_replan=True,controller='gait_turn_braking_timeout')
                return {'move_by_speed':[0.,0.,0.]},state
            if self._gait_brake_ticks<100 or not settled:
                return {'move_by_speed':[0.,0.,0.]},state
            target=self._gait_turn_target.copy();self._gait_turn_target=None
            maneuver=self._begin_corner_maneuver(position,heading,planner,boxes,target=target)
            if maneuver is not None:return maneuver
            delta=target-np.asarray(position[:2])
            error=(math.atan2(delta[1],delta[0])-heading+math.pi)%(2*math.pi)-math.pi
            if np.linalg.norm(delta)>.05 and abs(error)>.12:
                state.update(needs_replan=True,controller='gait_turn_space_unavailable')
                return {'move_by_speed':[0.,0.,0.]},state
        if self._corner_maneuver is not None:
            result=self._corner_maneuver.action(position,heading,planner,boxes)
            if result is not None:
                if self._corner_maneuver.failed:self._corner_maneuver=None
                return result
            self._corner_maneuver=None
        actual_body=navigation_body_blocked(planner,boxes,heading)
        if getattr(planner,'layered_local_control',False):
            # Global circular clearance guides route selection. Its halo is
            # not a second rigid robot: local motion uses the actual oriented
            # footprint, followed by the coupled translation/rotation sweep.
            self._route_blocked=actual_body
        else:self._route_blocked=route_blocked(planner,boxes)
        combined=self._route_blocked | actual_body
        cell=planner.cell(position)
        nominal=getattr(planner,'navigation_pose_heading',None)
        if nominal is not None and planner.valid_cell(cell):
            error=(nominal-heading+math.pi)%(2*math.pi)-math.pi
            nominal_free=not navigation_body_blocked(planner,boxes,nominal)[cell]
            current_blocked=navigation_body_blocked(planner,boxes,heading)[cell]
            if self._heading_recovery and not current_blocked:
                # Once the current footprint is free, let the path controller
                # choose the next heading. Continuing to force the original
                # nominal yaw can turn away from a safe corridor approach.
                self._heading_recovery=False;self._buffer_replan=False
                self._turn_ramp_ticks=0
            if (self._heading_recovery or current_blocked) and nominal_free and abs(error)>.03:
                if turn_reduces_overlap(planner,boxes,position,heading,heading+error):
                    self._heading_recovery=True;self._buffer_active=False;self._buffer_target=None;self._buffer_replan=False
                    rotation=math.copysign(min(1.,.40+.01*self._turn_ramp_ticks),error)
                    self._turn_ramp_ticks+=1
                    return {'move_by_speed':[0.,0.,rotation]},{'target':list(map(float,position[:2])),'recovering_buffer':True,
                        'needs_replan':False,'controller':'footprint_heading_recovery','heading_error_rad':float(error)}
        if planner.valid_cell(cell) and (combined[cell] or self._buffer_active):
            clearance=distance_transform_edt(~combined)*planner.resolution
            if self._buffer_active and not combined[cell] and (getattr(planner,'layered_local_control',False) or clearance[cell]>.08):
                released_target=self._buffer_target.tolist()
                self._buffer_active=False;self._buffer_target=None
                self._buffer_replan=not getattr(planner,'layered_local_control',False)
                return {'move_by_speed':[0.,0.,0.]},{'target':released_target,
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
            if getattr(planner,'layered_local_control',False) and not body_translation_recovery_clear(planner,boxes,position,heading,self._buffer_target):
                # The old recovery checked only the centre ray and could drag
                # a paw through furniture on its way to a free destination.
                r,c=cell;span=int(1./planner.resolution)
                candidates=[candidate]
                for rr in range(max(0,r-span),min(safe.shape[0],r+span+1),4):
                    for cc in range(max(0,c-span),min(safe.shape[1],c+span+1),4):
                        if safe[rr,cc]:candidates.append(np.array([planner.limits[0][0]+(cc+.5)*planner.resolution,
                            planner.limits[1][0]+(rr+.5)*planner.resolution]))
                candidates.sort(key=lambda point:np.linalg.norm(point-np.asarray(position[:2])))
                target=next((point for point in candidates[:50] if body_translation_recovery_clear(
                    planner,boxes,position,heading,point)),None)
                if target is None:
                    goal=np.asarray(self._planned_path[-1][:2])
                    desired=math.atan2(goal[1]-position[1],goal[0]-position[0])
                    trials=[desired,heading-math.pi/8,heading+math.pi/8,heading-math.pi/4,heading+math.pi/4]
                    trials.sort(key=lambda theta:abs((theta-heading+math.pi)%(2*math.pi)-math.pi))
                    for theta in trials:
                        delta=(theta-heading+math.pi)%(2*math.pi)-math.pi
                        if abs(delta)<.03:continue
                        if turn_reduces_overlap(planner,boxes,position,heading,heading+delta):
                            sign=math.copysign(1.,delta)
                            if sign!=self._turn_ramp_sign:self._turn_ramp_ticks=0
                            self._turn_ramp_sign=sign
                            rotation=sign*min(.8,.4+.01*self._turn_ramp_ticks);self._turn_ramp_ticks+=1
                            self._buffer_target=None;self._buffer_active=False
                            return {'move_by_speed':[0.,0.,rotation]},{'target':list(map(float,position[:2])),
                                'recovering_buffer':True,'needs_replan':False,'controller':'body_safe_recovery_turn'}
                    return {'move_by_speed':[0.,0.,0.]},{'target':list(map(float,position[:2])),
                        'recovering_buffer':True,'needs_replan':True,'controller':'no_body_safe_buffer_exit'}
                self._buffer_target=target
            target=self._buffer_target;self._buffer_active=True
            state={'target':target.tolist(),'recovering_buffer':True,'needs_replan':False,
                'controller':'calibrated_navigation_buffer_exit'}
            if not raw[cell] or not AStarMapPlanner.segment_clear(cell,planner.cell(target),~raw):
                state['needs_replan']=True
                return {'move_by_speed':[0.,0.,0.]},state
            delta=target-np.asarray(position[:2]);world=.25*delta/max(np.linalg.norm(delta),1e-9)
            c,s=math.cos(heading),math.sin(heading)
            command=[float(c*world[0]+s*world[1]),float(-s*world[0]+c*world[1]),0.]
            if getattr(planner,'layered_local_control',False):
                vx,vy,_=command
                if vx<-.05 and abs(vx)>=abs(vy):factor=.35/abs(vx)
                elif abs(vy)>abs(vx) and abs(vy)>.05:factor=.30/abs(vy)
                elif vx>.05:factor=.25/vx
                else:factor=1.
                factor=max(1.,min(factor,.40/max(np.linalg.norm(command[:2]),1e-9)))
                if factor>1.:
                    distance=max(np.linalg.norm(delta),np.linalg.norm(world)*factor*.5)
                    braking_target=np.asarray(position[:2])+delta/max(np.linalg.norm(delta),1e-9)*distance
                    if body_translation_recovery_clear(planner,boxes,position,heading,braking_target):
                        command[0]*=factor;command[1]*=factor
                        state['calibrated_buffer_motion']=True
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
        if state['needs_replan']:
            if getattr(planner,'layered_local_control',False):
                maneuver=self._begin_corner_maneuver(position,heading,planner,boxes)
                if maneuver is not None:return maneuver
                delta=np.asarray(state['target'])-np.asarray(position[:2])
                desired=math.atan2(delta[1],delta[0])
                error=(desired-heading+math.pi)%(2*math.pi)-math.pi
                rotation=self._local_turn_request(math.copysign(1.,error))
                if abs(error)>.08 and body_motion_clear(planner,boxes,position,heading,0.,rotation):
                    state={**state,'needs_replan':False,'controller':'align_to_global_guidance','foot_clearance_turn':True}
                    return self._calibrate_local_action({'move_by_speed':[0.,0.,rotation]},state,position,heading,planner,boxes)
            return action,state
        delta=np.asarray(state['target'])-np.asarray(position[:2])
        heading_error=(math.atan2(delta[1],delta[0])-heading+math.pi)%(2*math.pi)-math.pi
        if (getattr(planner,'layered_local_control',False) and abs(heading_error)>.5 and
                np.linalg.norm(action['move_by_speed'][:2])>.15 and
                body_rotation_clear(planner,boxes,position,heading,heading+heading_error,margin=.04)):
            # A geometrically clear curved chord can still excite the rough
            # gait when lateral motion and yaw rise together. Finish braking
            # before using the existing body-checked turning maneuver. Tight
            # corners without room for a full heading change retain their
            # incremental, collision-checked turn/translation recovery.
            self._gait_turn_target=np.asarray(state['target'],float).copy()
            self._gait_brake_ticks=0
            self._gait_brake_positions=[np.asarray(position[:2],float).copy()]
            self._gait_brake_headings=[float(heading)]
            return {'move_by_speed':[0.,0.,0.]},{**state,'recovering_buffer':True,
                'needs_replan':False,'local_turn_maneuver':True,'controller':'gait_turn_braking',
                'heading_error_rad':float(heading_error),'braking_ticks':0,'pose_settled':False}
        if not self.guard_motion:
            state['controller']='pure_exploration_velocity_action_for'
            vf,vl,rotation=action['move_by_speed']
            if not body_motion_clear(planner,boxes,position,heading,vf,rotation,lateral=vl):
                if getattr(planner,'layered_local_control',False):
                    maneuver=self._begin_corner_maneuver(position,heading,planner,boxes)
                    if maneuver is not None:return maneuver
                turn_request=self._local_turn_request(rotation) if abs(rotation)>1e-8 else 0.
                if (getattr(planner,'layered_local_control',False) and abs(rotation)>.2 and
                        body_motion_clear(planner,boxes,position,heading,0.,turn_request)):
                    action={'move_by_speed':[0.,0.,turn_request]};state['foot_clearance_turn']=True
                    state['turn_before_translation']=True
                elif np.linalg.norm([vf,vl])>1e-8 and body_motion_clear(planner,boxes,position,heading,vf,0.,lateral=vl):
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
                        return self._calibrate_local_action(action,state,position,heading,planner,boxes)
                    if abs(rotation)>1e-8 and body_motion_clear(planner,boxes,position,heading,0.,rotation):
                        action={'move_by_speed':[0.,0.,rotation]};state['foot_clearance_turn']=True
                    else:
                        state['needs_replan']=True;state['foot_motion_blocked']=True
                        action={'move_by_speed':[0.,0.,0.]}
            return self._calibrate_local_action(action,state,position,heading,planner,boxes)
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

    def _local_turn_request(self,rotation):
        sign=math.copysign(1.,rotation)
        ticks=self._local_turn_ticks if sign==self._local_turn_sign else 0
        return sign*min(1.,.4+.01*ticks)

    def _begin_corner_maneuver(self,position,heading,planner,boxes,target=None):
        if target is None:target=super()._lookahead_target(np.asarray(position[:2],float))
        delta=np.asarray(target)-np.asarray(position[:2])
        if np.linalg.norm(delta)<.05:return None
        desired=math.atan2(delta[1],delta[0])
        if abs((desired-heading+math.pi)%(2*math.pi)-math.pi)<.12:return None
        from grutopia_extension.interactive_navigation.local_turn_maneuver import LocalTurnManeuver
        self._corner_maneuver=LocalTurnManeuver(desired)
        self._buffer_active=False;self._buffer_target=None;self._buffer_replan=False
        result=self._corner_maneuver.action(position,heading,planner,boxes)
        if result is None or self._corner_maneuver.failed:
            self._corner_maneuver=None
            return None
        return result

    def _calibrate_local_action(self,action,state,position,heading,planner,boxes):
        if not getattr(planner,'layered_local_control',False):return action,state
        vx,vy,rotation=action['move_by_speed']
        if (state.get('foot_clearance_translation') or state.get('foot_clearance_axis_step')) and abs(vx)<.05 and .03<abs(vy)<.30:
            request=math.copysign(.30,vy)
            if body_motion_clear(planner,boxes,position,heading,0.,0.,lateral=request):
                action={'move_by_speed':[0.,request,0.]}
                state['calibrated_local_sidestep']=True
        vx,vy,rotation=action['move_by_speed']
        if abs(vx)+abs(vy)<1e-8 and abs(rotation)>1e-8:
            sign=math.copysign(1.,rotation)
            if sign!=self._local_turn_sign:self._local_turn_ticks=0
            self._local_turn_sign=sign
            request=self._local_turn_request(rotation)
            if body_motion_clear(planner,boxes,position,heading,0.,request):
                action={'move_by_speed':[0.,0.,request]};state['calibrated_local_turn']=True
            self._local_turn_ticks+=1
        else:self._local_turn_ticks=0;self._local_turn_sign=0.
        return action,state

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


def navigation_footprint_samples(planner,margin=0.):
    halfwidth=getattr(planner,'navigation_half_width',.22)+margin
    # Include both extrema; arange omitted the last 2 cm of the front and
    # produced an asymmetric width when dimensions were not grid multiples.
    return np.meshgrid(np.linspace(-.32-margin,.50+margin,int(math.ceil((.82+2*margin)/planner.resolution))+1),
        np.linspace(-halfwidth,halfwidth,int(math.ceil(2*halfwidth/planner.resolution))+1))


def body_translation_recovery_clear(planner,boxes,position,heading,target):
    """Allow existing mild overlap to decrease, never a deeper/new body collision."""
    raw=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return False
        raw &= ~mask
    xx,yy=navigation_footprint_samples(planner);c,s=math.cos(heading),math.sin(heading)
    dx=np.rint((c*xx-s*yy)/planner.resolution).astype(int)
    dy=np.rint((s*xx+c*yy)/planner.resolution).astype(int)
    position=np.asarray(position[:2],float);delta=np.asarray(target[:2])-position
    count=max(2,int(np.ceil(np.linalg.norm(delta)/planner.resolution))+1)
    first=None;last=None
    for fraction in np.linspace(0.,1.,count):
        r,col=planner.cell(position+fraction*delta);rows=r+dy;cols=col+dx
        if rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]:return False
        overlap=int((~raw[rows,cols]).sum())
        if first is None:first=overlap
        if overlap>first:return False
        if first>0 and fraction*np.linalg.norm(delta)>.05 and overlap>=first:return False
        last=overlap
    return last==0


def navigation_body_blocked(planner,boxes,heading):
    """Protect the full standing/gait footprint, including front paws."""
    import cv2
    raw=planner.free.copy()
    for box in boxes:raw &= ~planner.box_mask(box)
    xx,yy=navigation_footprint_samples(planner);c,s=math.cos(heading),math.sin(heading)
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
    xx,yy=navigation_footprint_samples(planner);counts=[]
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
    xx,yy=navigation_footprint_samples(planner)
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


def body_rotation_clear(planner,boxes,position,first_heading,last_heading,margin=0.):
    """Check the same oriented Go2 footprint locally through a heading sweep."""
    raw=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return False
        raw &= ~mask
    row,col=planner.cell(position)
    xx,yy=navigation_footprint_samples(planner,margin=margin)
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
