"""Track supplied routes through the existing Go2 exploration velocity law."""
import numpy as np
from types import SimpleNamespace
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner
from grutopia_extension.interactive_navigation.state_machine import StateMachineDecision,InteractionState,ControllerCommand


def decision_buffer_recovery(position,heading,planner,boxes,goal,fixed_heading,*,local_free=None,local_boxes=None,require_goal_connection=True):
    """Exit a footprint buffer before a new decision declares the route blocked."""
    if (getattr(planner,'layered_local_control',False) and
            getattr(planner,'planning_scope','local')=='global' and
            getattr(planner,'navigation_pose_heading',None) is None):
        return _body_safe_global_buffer_exit(position,heading,planner,boxes,goal,
            local_free=local_free,local_boxes=local_boxes,require_goal_connection=require_goal_connection)
    cell=planner.cell(position)
    raw_free=planner.free.copy()
    for box in boxes:raw_free &= ~planner.box_mask(box)
    if not planner.valid_cell(cell) or not raw_free[cell]:return None
    tracker=ExplorationPathTracker(position,[goal],fixed_heading=fixed_heading)
    action,state=tracker.action(position,heading,planner,boxes)
    if not state['recovering_buffer']:return None
    target_cell=planner.cell(state['target'])
    # Only footprint inflation may be crossed while recovering. The base
    # center must have a clear segment through the unbuffered geometry/boxes.
    if not AStarMapPlanner.segment_clear(cell,target_cell,~raw_free):return None
    command=action['move_by_speed'];norm=float(np.linalg.norm(command[:2]))
    if norm<1e-9:return None
    # Post-contact lateral requests below ~0.16 m/s stall the rough gait.
    # Preserve the original exit direction at the calibrated staging speed.
    command[:2]=[float(v*.25/norm) for v in command[:2]]
    state['controller']='original_buffer_exit_direction_with_gait_staging_speed'
    return action,state


def _body_safe_global_buffer_exit(position,heading,planner,boxes,goal,*,local_free=None,local_boxes=None,require_goal_connection=True):
    """Connect an oriented local pose to a usable circular global-plan seed.

    The nearest global free point can require dragging a paw through another
    obstacle. Search alternatives, check their whole local-body connections,
    and keep a valid selected target until the global buffer has been left.
    """
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
        body_motion_clear,body_translation_recovery_clear,calibrated_buffer_translation)
    position=np.asarray(position[:2],float);goal_key=tuple(map(float,goal[:2]))
    cell=planner.cell(position);global_free=planner.free
    raw=global_free.copy()
    for box in boxes:raw &= ~planner.box_mask(box)
    if not planner.valid_cell(cell) or not raw[cell]:
        planner._global_buffer_exit=None
        return None
    blocked=planner.blocked(boxes)
    if not blocked[cell]:
        planner._global_buffer_exit=None
        return None
    clearance=distance_transform_edt(~blocked)*planner.resolution
    motion_free=global_free if local_free is None else local_free
    motion_boxes=boxes if local_boxes is None else local_boxes
    c,s=np.cos(heading),np.sin(heading)

    def safe_command(target):
        target=np.asarray(target,float);delta=target-position
        distance=float(np.linalg.norm(delta));rc=planner.cell(target)
        if (distance<1e-6 or distance>1.05 or not planner.valid_cell(rc) or
                clearance[rc]<planner.resolution-1e-9):return None
        saved=planner.free;planner.free=motion_free
        try:
            command=calibrated_buffer_translation(planner,motion_boxes,position,heading,target)
        finally:planner.free=saved
        return None if command is None else command.tolist()

    cached=getattr(planner,'_global_buffer_exit',None)
    if cached is not None and cached['goal']==goal_key and cached.get('require_goal_connection',True)==require_goal_connection:
        command=safe_command(cached['target'])
        if command is not None:
            return {'move_by_speed':command},{'target':list(cached['target']),
                'recovering_buffer':True,'needs_replan':False,'controller':'body_safe_global_buffer_exit',
                'reused_exit_target':True,'goal_connection_checked':require_goal_connection}
    planner._global_buffer_exit=None
    candidates=[]
    for angle in np.linspace(-np.pi,np.pi,32,endpoint=False):
        unit=np.array([np.cos(angle),np.sin(angle)])
        for distance in [.10,.15,.20,.30,.45,.60,.80,1.0]:
            target=position+distance*unit;command=safe_command(target)
            if command is None:continue
            score=distance+float(np.linalg.norm(target-np.asarray(goal[:2])))
            candidates.append((score,distance,target,command))
    candidates.sort(key=lambda r:r[:2])
    # Check the best geometric candidates until one connects, rather than
    # solving an entire global route for every candidate on every selection.
    for omit_boxes in [False,True]:
        for _,_,target,command in candidates:
            # A box may still require interaction after the buffer exit.
            # Every motion check above included the real current boxes.
            if require_goal_connection and not planner.path(target,goal,[] if omit_boxes else boxes):continue
            planner._global_buffer_exit={'goal':goal_key,'target':tuple(map(float,target)),
                'require_goal_connection':require_goal_connection}
            return {'move_by_speed':command},{'target':target.tolist(),'recovering_buffer':True,
                'needs_replan':False,'controller':'body_safe_global_buffer_exit','reused_exit_target':False,
                'safe_exit_candidates':len(candidates),'goal_connection_checked':require_goal_connection}
    return None


class ExplorationPathTracker:
    def __init__(self,start,path,lookahead=.55,fixed_heading=None,max_forward_speed=.40,max_lateral_speed=.20):
        self._planned_start=np.asarray(start[:2],dtype=float)
        self._planned_path=tuple(tuple(p[:2]) for p in path)
        self._waypoint_index=0
        self.fixed_heading=fixed_heading
        self.max_forward_speed=max_forward_speed;self.max_lateral_speed=max_lateral_speed
        self.velocity_lookahead_distance=lookahead
        self.final_goal_slow_radius=.75
        self.safe_path_tracking=True
        self.tracking_chord_fallbacks=0;self.tracking_segment_stops=0
        self._needs_replan=False

    def action_for(self,decision,observation,step=None):
        position=np.asarray(observation['position'][:2])
        while self._waypoint_index<len(self._planned_path)-1 and np.linalg.norm(position-self._planned_path[self._waypoint_index])<.28:
            # Dense corners may be near the robot while the next chord cuts
            # through a chair/box. Consume a waypoint only after that chord
            # becomes clear; otherwise the original lookahead can fall back
            # to this waypoint and finish the corner safely.
            if not self._tracking_segment_clear(position,self._planned_path[self._waypoint_index+1]):
                break
            self._waypoint_index+=1
        return {'move_along_path':[]}

    def _lookahead_target(self,position):
        # The supplied A* route has long shortcut segments, unlike the dense
        # exploration routes. Start the unchanged exploration interpolation at
        # the projection onto the active segment so the carrot stays on that
        # segment and lateral gait drift is corrected locally.
        first=(self._planned_start if self._waypoint_index==0
               else np.asarray(self._planned_path[self._waypoint_index-1],dtype=float))
        last=np.asarray(self._planned_path[self._waypoint_index],dtype=float)
        delta=last-first
        fraction=float(np.clip(np.dot(position-first,delta)/max(np.dot(delta,delta),1e-12),0.,1.))
        return MapNavigationRuntime._lookahead_target(self,first+fraction*delta)

    def _tracking_segment_clear(self,start,target):
        return AStarMapPlanner.segment_clear(self._planner.cell(start),self._planner.cell(target),self._blocked)

    def _record_replan(self,*args):
        self._needs_replan=True

    def action(self,position,heading,planner,boxes):
        self._planner=planner
        # The supplied route was checked at this body heading. Preserve it
        # while translating along the route, including sideways segments.
        # Recomputing the footprint at the carrot-facing yaw can make a
        # feasible doorway route disappear halfway through execution.
        planner.robot_heading=heading if self.fixed_heading is None else self.fixed_heading
        self._blocked=planner.blocked(boxes)
        self.navigation_occupancy=SimpleNamespace(config=SimpleNamespace(grid_resolution=planner.resolution))
        self._needs_replan=False
        cell=planner.cell(position)
        if planner.valid_cell(cell) and self._blocked[cell]:
            safe=distance_transform_edt(~self._blocked)*planner.resolution>.12
            if not safe.any():safe=~self._blocked
            _,nearest=distance_transform_edt(~safe,return_indices=True)
            row,col=nearest[:,cell[0],cell[1]]
            target=np.array([planner.limits[0][0]+(col+.5)*planner.resolution,
                             planner.limits[1][0]+(row+.5)*planner.resolution])
            # Same bounded world-velocity law as the exploration buffer exit.
            delta=target-np.asarray(position[:2]);world=delta*min(1.,.15/max(np.linalg.norm(delta),1e-9))
            c,s=np.cos(heading),np.sin(heading)
            return {'move_by_speed':[float(c*world[0]+s*world[1]),float(-s*world[0]+c*world[1]),0.]}, {
                'target':target.tolist(),'recovering_buffer':True,'needs_replan':False,
                'controller':'original_exploration_buffer_exit_velocity_law'}
        observation={'position':tuple(position)+(.55,) if len(position)==2 else position,
                     'orientation':(np.cos(heading/2),0.,0.,np.sin(heading/2))}
        decision=StateMachineDecision(state=InteractionState.NAVIGATE_TO_GOAL,
            command=ControllerCommand('move_along_path',((self._planned_path[-1],),)))
        action=MapNavigationRuntime.velocity_action_for(self,decision,observation,max_forward_speed=self.max_forward_speed,max_lateral_speed=self.max_lateral_speed)
        if self.fixed_heading is not None:
            error=(self.fixed_heading-heading+np.pi)%(2*np.pi)-np.pi
            action['move_by_speed'][2]=float(np.clip(1.8*error,-.45,.45))
        target=getattr(self,'_last_velocity_target',None)
        if target is None:target=self._lookahead_target(np.asarray(position[:2]))
        target=np.asarray(target,float)
        return action,{'target':target.tolist(),'recovering_buffer':False,'needs_replan':self._needs_replan,
                       'planned_heading':self.fixed_heading,
                       'controller':'original_exploration_translation_with_planned_heading' if self.fixed_heading is not None
                                    else 'original_exploration_velocity_action_for'}
