"""Contact-push decisions over the existing occupancy/A* navigation framework.

Movable object poses are provided by simulation in this first integration.
The immutable architectural layer is never cleared when a box moves.
"""
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.mapping import MappingConfig, OccupancyGridMap, AStarMapPlanner, PlanningError


@dataclass(frozen=True)
class BoxState:
    id: str
    xy: tuple
    dimensions: tuple
    yaw: float = 0.0
    mass_kg: float = 1.0


@dataclass(frozen=True)
class PushPlan:
    box_id: str
    initial_xy: tuple
    destination_xy: tuple
    direction: tuple
    approach_xy: tuple
    approach_path: tuple
    push_distance: float
    score: float


@dataclass(frozen=True)
class BoxDecision:
    kind: str
    reason: str
    path: tuple = ()
    push: object = None
    explanation: object = None


class BoxPushPlanner:
    def __init__(self, static_free, limits, resolution=.025, robot_radius=.45,
                 contact_offset=.48, max_push_distance=1.6, detour_ratio=1.6, parking_rule=None):
        self.free = np.asarray(static_free, dtype=bool).copy()
        self.box_free = self.free
        self.limits = limits
        self.resolution = resolution
        self.radius = robot_radius
        self.contact_offset = contact_offset
        self.max_push_distance = max_push_distance
        self.detour_ratio = detour_ratio
        self.parking_rule = parking_rule
        self.config = MappingConfig(x_limits=tuple(limits[0]), y_limits=tuple(limits[1]),
            grid_resolution=resolution, robot_radius=robot_radius,
            obstacle_inflation_radius=robot_radius, static_obstacle_inflation_radius=robot_radius)
        self.static_clear = distance_transform_edt(self.free)*resolution > robot_radius

    def cell(self, xy):
        return (int((xy[1]-self.limits[1][0])/self.resolution),
                int((xy[0]-self.limits[0][0])/self.resolution))

    def valid_cell(self, cell):
        return 0 <= cell[0] < self.free.shape[0] and 0 <= cell[1] < self.free.shape[1]

    def box_mask(self, box, xy=None, padding=0.0):
        xy = np.array(box.xy if xy is None else xy)
        width,depth = np.array(box.dimensions[:2])+2*padding
        yaw=math.radians(box.yaw);c,s=math.cos(yaw),math.sin(yaw)
        extent=np.array([abs(c)*width/2+abs(s)*depth/2,abs(s)*width/2+abs(c)*depth/2])
        r0,c0=self.cell(xy-extent);r1,c1=self.cell(xy+extent)
        mask=np.zeros_like(self.free)
        if r0<0 or c0<0 or r1>=mask.shape[0] or c1>=mask.shape[1]:return None
        yy,xx=np.mgrid[r0:r1+1,c0:c1+1]
        dx=self.limits[0][0]+(xx+.5)*self.resolution-xy[0]
        dy=self.limits[1][0]+(yy+.5)*self.resolution-xy[1]
        local_x=c*dx+s*dy;local_y=-s*dx+c*dy
        mask[r0:r1+1,c0:c1+1]=(abs(local_x)<=width/2)&(abs(local_y)<=depth/2)
        return mask

    def blocked(self, boxes, omit=None):
        raw=np.zeros_like(self.free)
        for box in boxes:
            if box.id==omit:continue
            mask=self.box_mask(box)
            if mask is None:raise ValueError(f'Box {box.id} outside planning grid')
            raw|=mask
        return ~self.static_clear | (distance_transform_edt(~raw)*self.resolution <= self.radius) if raw.any() else ~self.static_clear.copy()

    def path(self, start, goal, boxes, omit=None):
        blocked=self.blocked(boxes,omit)
        sc,gc=self.cell(start),self.cell(goal)
        if not self.valid_cell(sc) or not self.valid_cell(gc) or blocked[sc] or blocked[gc]:return ()
        # The existing planner receives an already inflated mask. Use zero
        # residual inflation and observed-only planning to prohibit corner cuts.
        config=replace(self.config,robot_radius=self.resolution/10,
                       obstacle_inflation_radius=self.resolution/10,
                       static_obstacle_inflation_radius=self.resolution/10)
        grid=OccupancyGridMap(config)
        h,w=grid.log_odds.shape
        grid.static_occupied[:]=blocked[:h,:w]
        grid.observed[:]=True
        grid.log_odds[:]=-4
        try:
            raw=AStarMapPlanner(grid,observed_only=True).plan((*start,0.0),(*goal,0.0))
        except PlanningError:return ()
        # The controller brakes at every waypoint. Retaining each 2.5 cm
        # staircase corner makes a short detour needlessly slow. Shortcut only
        # segments validated by the existing corner-safe collision checker.
        margin_blocked=blocked | (distance_transform_edt(~blocked)*self.resolution <= .04)
        smoothed=[];anchor=tuple(start);index=0
        while index<len(raw):
            chosen=index
            for candidate in range(len(raw)-1,index,-1):
                if AStarMapPlanner.segment_clear(self.cell(anchor),self.cell(raw[candidate][:2]),margin_blocked):
                    chosen=candidate;break
            smoothed.append(raw[chosen]);anchor=raw[chosen][:2];index=chosen+1
        return tuple(smoothed)

    @staticmethod
    def length(start,path):
        points=np.array([(*start,0.0),*path])
        return float(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).sum()) if path else math.inf

    def route_intersects(self, start, path, box, xy=None):
        mask=self.box_mask(box,xy=xy,padding=self.radius+.08)
        if mask is None:return True
        for left,right in zip([(*start,0.0),*path[:-1]],path):
            a,b=np.array(left[:2]),np.array(right[:2])
            for p in np.linspace(a,b,max(2,math.ceil(np.linalg.norm(b-a)/self.resolution)+1)):
                cell=self.cell(p)
                if not self.valid_cell(cell) or mask[cell]:return True
        return False

    def push_candidates(self,start,goal,boxes,box,optimistic,immovable=()):
        other=np.zeros_like(self.free)
        for candidate in boxes:
            if candidate.id!=box.id:other|=self.box_mask(candidate,padding=.06)
        allowed=self.box_free&~other
        robot_blocked=self.blocked(boxes,omit=box.id)
        yaw=math.radians(box.yaw)
        diagonal=math.sqrt(.5)
        for direction in [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.),
                          (diagonal,diagonal),(diagonal,-diagonal),(-diagonal,diagonal),(-diagonal,-diagonal)]:
            direction=np.array(direction)
            local=[math.cos(yaw)*direction[0]+math.sin(yaw)*direction[1],
                   -math.sin(yaw)*direction[0]+math.cos(yaw)*direction[1]]
            extent=sum(abs(local[i])*box.dimensions[i]/2 for i in (0,1))
            approach=np.array(box.xy)-direction*(extent+self.contact_offset+.10)
            approach_path=self.path(start,approach,boxes)
            if not approach_path:continue
            for distance in np.arange(.35,self.max_push_distance+.01,.10):
                destination=np.array(box.xy)+distance*direction
                if self.parking_rule is not None and not self.parking_rule(box,destination,direction):continue
                valid=True
                for offset in np.arange(0,distance+self.resolution/2,self.resolution):
                    center=np.array(box.xy)+offset*direction
                    mask=self.box_mask(box,xy=center,padding=.04)
                    if mask is None or not allowed[mask].all():valid=False;break
                    robot_xy=center-direction*(extent+self.contact_offset)
                    rc=self.cell(robot_xy)
                    if not self.valid_cell(rc) or robot_blocked[rc]:valid=False;break
                if not valid:continue
                moved=replace(box,xy=tuple(destination))
                after=[moved if x.id==box.id else x for x in boxes]
                end_robot=destination-direction*(extent+self.contact_offset+.18)
                route=self.path(end_robot,goal,after)
                remaining_optimistic=self.path(end_robot,goal,immovable)
                if not remaining_optimistic or self.route_intersects(end_robot,remaining_optimistic,box,destination):continue
                score=self.length(start,approach_path)+distance*2.5
                if route:score+=self.length(end_robot,route)
                else:score+=15  # permit sequential clearing, with a finite penalty
                yield PushPlan(box.id,box.xy,tuple(destination),tuple(direction),tuple(approach),
                    approach_path,float(distance),float(score))
                break  # nearest safe parking position for this push direction

    def decide(self,start,goal,boxes,excluded=()):
        direct=self.path(start,goal,boxes)
        eligible=getattr(self,'interactable_ids',None)
        immovable=[box for box in boxes if eligible is not None and box.id not in eligible]
        optimistic=self.path(start,goal,immovable)
        finite=lambda value:float(value) if math.isfinite(value) else None
        details={'start':list(map(float,start)),'goal':list(map(float,goal)),
            'before_path':direct,'without_boxes_path':optimistic,
            'before_length_m':finite(self.length(start,direct)),
            'without_boxes_length_m':finite(self.length(start,optimistic)),
            'candidate_count':0,'cost_formula':'approach + 2.5 * push_distance + remaining_route',
            'map_source':'static_prior_plus_current_obstacles'}
        def decision(kind,reason,path=(),push=None):
            details['kind']=kind;details['reason']=reason
            return BoxDecision(kind,reason,path,push,details)
        if not optimistic:return decision('failed','fixed_geometry_blocks_task')
        if direct and self.length(start,direct)<=self.detour_ratio*self.length(start,optimistic)+.5:
            return decision('navigate','ordinary_path_available',direct)
        candidates=[]
        for box in boxes:
            if box.id in excluded or (eligible is not None and box.id not in eligible) or not self.route_intersects(start,optimistic,box):continue
            candidates.extend(self.push_candidates(start,goal,boxes,box,optimistic,immovable=immovable))
        if not candidates:
            if direct:return decision('navigate','no_feasible_push_use_detour',direct)
            return decision('failed','no_safe_contact_push_or_parking_pose')
        best=min(candidates,key=lambda p:p.score)
        moved=next(replace(box,xy=best.destination_xy) for box in boxes if box.id==best.box_id)
        after=[moved if box.id==best.box_id else box for box in boxes]
        after_path=self.path(start,goal,after)
        before_length=self.length(start,direct);after_length=self.length(start,after_path)
        details.update(candidate_count=len(candidates),box_id=best.box_id,
            initial_xy=list(best.initial_xy),destination_xy=list(best.destination_xy),
            push_direction=list(best.direction),approach_path=best.approach_path,
            approach_length_m=self.length(start,best.approach_path),push_distance_m=best.push_distance,
            weighted_push_cost=best.push_distance*2.5,total_push_cost=best.score,
            after_path=after_path,after_length_m=finite(after_length),
            path_saving_m=finite(before_length-after_length) if direct and after_path else None,
            cost_saving=finite(before_length-best.score) if direct else None,
            opens_connectivity=bool(not direct and after_path),
            passage_clearance=(self.clearance(moved) if hasattr(self,'clearance') else None))
        if direct and self.length(start,direct)<=best.score:
            return decision('navigate','detour_cheaper_than_push',direct)
        return decision('push','movable_box_blocks_efficient_route',push=best)
