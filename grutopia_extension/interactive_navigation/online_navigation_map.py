"""LiDAR-built global/local maps. No scene grid or architectural metadata input.

Box proposals are a simulated perception interface. A proposal is admitted
only when current LiDAR returns see it; hidden proposals never enter planning.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib,json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.layered_navigation_map import LayeredNavigationMap
from grutopia_extension.interactive_navigation.online_lidar_evidence import online_lidar_masks


class OnlineNavigationMap(LayeredNavigationMap):
    def __init__(self,planner,sensor_config,*,floor_height=.01,**kwargs):
        super().__init__(planner,sensor_config,**kwargs)
        self.static_occupied[:]=False
        self.evidence=OccupancyGridMap(replace(self.config,sticky_free_scale=.25))
        assert self.evidence.observed.shape==self.static_occupied.shape
        self.observed=self.evidence.observed
        self.global_observed=np.zeros_like(self.observed)
        self.floor_height=floor_height;self.detected_boxes={}
        self.visible_box_ids=[];self.scan_count=0
        self.ray_statistics={}
        self.initial_observed_cells=0;self.initial_occupied_cells=0
        self._latest_static=np.zeros_like(self.observed)
        self.static_digest=hashlib.sha256(self.static_occupied.tobytes()).hexdigest()
        self._bind_local()

    @property
    def local_occupied(self):
        return self._latest_static | self.tracked_occupied | self.local_dynamic

    @property
    def global_occupied(self):
        return self.static_occupied | self.global_tracked

    def _visible_proposals(self,points,proposals):
        visible=[]
        if points.ndim!=2 or points.shape[1]<3:return visible
        points=points[np.isfinite(points[:,:3]).all(axis=1),:3]
        for box in proposals:
            # Actor identity/size is supplied only after a sensor visibility
            # test. No ray is allowed to reveal a box through a wall.
            xy=points[:,:2]-np.asarray(box.xy)
            a=np.radians(box.yaw);c,s=np.cos(a),np.sin(a)
            local=np.column_stack((c*xy[:,0]+s*xy[:,1],-s*xy[:,0]+c*xy[:,1]))
            inside=(np.abs(local)<=np.asarray(box.dimensions[:2])/2+.04).all(axis=1)
            inside &= (points[:,2]>=self.floor_height+.025)&(points[:,2]<=self.floor_height+box.dimensions[2]+.04)
            if np.count_nonzero(inside)>=3:visible.append(box)
        return visible

    def update(self,now,observation,boxes,*,force=False):
        self.now=float(now)
        if not force and self.now-self.last_local<self.local_period-1e-8:return False
        self.last_local=self.now;self.local_updates+=1
        self.robot_xy=np.asarray(observation['position'][:2],float)
        q=np.asarray(observation.get('orientation',[1,0,0,0]),float);q/=max(np.linalg.norm(q),1e-9)
        self.robot_yaw=float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
        self.local_window=self._window(self.robot_xy)
        forward=np.array([np.cos(self.robot_yaw),np.sin(self.robot_yaw)])
        footprint=BoxState('__robot__',tuple(self.robot_xy+.09*forward),(.82,.44,.5),np.degrees(self.robot_yaw))
        mask=self.planner.box_mask(footprint,padding=self.planner.resolution)
        self.robot_footprint=mask if mask is not None else np.zeros_like(self.observed)
        lidar=observation.get('sensors',{}).get('lidar',{})
        scan=lidar.get('physics_step');points=np.asarray(lidar.get('pointcloud',[]),float)
        fresh=bool(points.size and (scan is None or scan!=self.last_scan))
        self.visible_box_ids=[]
        if fresh:
            origin=np.asarray(lidar.get('position',observation['position']),float)
            distances=np.linalg.norm(points[:,:3]-origin,axis=1)
            returns=points[distances<float(lidar.get('max_range',5.))*.99]
            visible=self._visible_proposals(returns,boxes)
            self.visible_box_ids=[b.id for b in visible]
            self.detected_boxes.update({b.id:b for b in visible})
        self.current_boxes=list(self.detected_boxes.values())
        new_mask=self._mask(self.current_boxes)
        vacated=self.tracked_occupied & ~new_mask
        if vacated.any():
            # A moved box does not prove that the hidden space is free.
            # Discard only its owned evidence; current free rays must confirm
            # the newly opened passage before motion uses it.
            self.evidence.log_odds[vacated]=0.;self.observed[vacated]=False
            self.local_hits[vacated]=False;self.hit_time[vacated]=-np.inf
            self.vacated_time[vacated]=self.now
            self.events.append({'event':'visible_box_moved','time_s':self.now,
                'visible_ids':self.visible_box_ids,'vacated_cells':int(vacated.sum())})
        self.tracked_occupied=new_mask
        self.tracked_sensor_mask=new_mask.copy()
        for box in self.current_boxes:
            part=self.planner.box_mask(box,padding=.04)
            if part is not None:self.tracked_sensor_mask |= part
        if fresh:
            filtered=self.filter_robot_returns(observation,count=True)['sensors']['lidar']
            measured_hit,measured_free,self.ray_statistics=online_lidar_masks(self.config,
                filtered.get('position',observation['position']),np.asarray(filtered['pointcloud']),
                max_range=float(filtered.get('max_range',5.)),min_range=float(filtered.get('min_range',.25)))
            hit=measured_hit & ~self.tracked_sensor_mask & ~self.robot_footprint
            free=measured_free & ~new_mask
            self.ingest_scan_masks(hit,free)
            self.scan_count+=1;self.last_scan=scan
        # The object's full extent is observed by the admitted object sensor,
        # but it is kept in the independent object layer, not structural map.
        self.observed[new_mask]=True;self.evidence.log_odds[new_mask]=-2.
        self.evidence.mark_free(self.robot_xy,min(.28,self.planner.radius))
        self.local_hits[self.now-self.hit_time>self.obstacle_ttl]=False
        self._latest_static=self.evidence.occupied_mask() & ~new_mask
        digest=hashlib.blake2b((self.local_occupied | ~self.observed).tobytes(),digest_size=12).digest()
        if digest!=self._local_digest:self.local_revision+=1;self._local_digest=digest
        if force or self.now-self.last_global>=self.global_period-1e-8:self.refresh_global(reason='initial_scan' if force else 'periodic')
        self._bind_local()
        return True

    def ingest_scan_masks(self,hit,free):
        """Fuse one actual scan, preserving unknown cells outside its rays."""
        hit=np.asarray(hit,bool);free=np.asarray(free,bool)&~hit
        self.evidence.log_odds[hit]+=self.config.occupied_update
        self.evidence.log_odds[free]-=self.config.free_update
        np.clip(self.evidence.log_odds,-4.,4.,out=self.evidence.log_odds)
        self.observed[hit|free]=True
        self.local_hits[free]=False;self.local_hits[hit]=True;self.hit_time[hit]=self.now
        self.seen_time[hit|free]=self.now

    def refresh_global(self,boxes=None,*,reason='interaction_completed'):
        if reason=='decision' and self.now-self.last_global<self.global_period-1e-8:return self.global_revision
        # The optional boxes argument is intentionally ignored. Only boxes
        # admitted from observations may be published to the global planner.
        self.global_boxes=list(self.current_boxes)
        self.global_tracked=self._mask(self.global_boxes)
        self.static_occupied[:]=self.evidence.occupied_mask() & ~self.tracked_occupied
        self.global_observed[:]=self.observed
        self.last_global=self.now;self.global_updates+=1
        digest=hashlib.blake2b((self.global_occupied | ~self.global_observed).tobytes(),digest_size=12).digest()
        if digest!=self._global_digest:
            self.global_revision+=1;self._global_digest=digest
            self.events.append({'event':'global_map_changed','time_s':self.now,'reason':reason,
                'revision':self.global_revision,'observed_cells':int(self.global_observed.sum())})
        return self.global_revision

    def _bind_local(self):
        if not hasattr(self,'evidence'):return
        free=self.observed & ~(self._latest_static|self.local_dynamic)
        self.planner.free=free
        self.planner.box_free=self.observed & ~self._latest_static
        self.planner.static_clear=distance_transform_edt(free)*self.planner.resolution>self.planner.radius
        self.planner.planning_scope='local'
        self.planner.planning_observed=self.observed

    @contextmanager
    def global_planning(self):
        saved=(self.planner.free,self.planner.static_clear,getattr(self.planner,'planning_scope','local'),self.planner.box_free,
               getattr(self.planner,'planning_observed',None))
        free=self.global_observed & ~self.static_occupied
        self.planner.free=free;self.planner.box_free=free
        self.planner.static_clear=distance_transform_edt(free)*self.planner.resolution>self.planner.radius
        self.planner.planning_scope='global'
        self.planner.planning_observed=self.global_observed
        try:yield self.global_boxes
        finally:(self.planner.free,self.planner.static_clear,self.planner.planning_scope,self.planner.box_free,
                 self.planner.planning_observed)=saved

    def frontier_grid(self):
        grid=OccupancyGridMap(replace(self.config,robot_radius=self.planner.radius,
            obstacle_inflation_radius=getattr(self.planner,'navigation_route_radius',.30),
            static_obstacle_inflation_radius=getattr(self.planner,'navigation_route_radius',.30)))
        grid.observed[:]=self.global_observed
        grid.log_odds[:]=-2.
        grid.log_odds[self.global_occupied]=3.
        return grid

    def synchronize_display(self,runtime,step):
        runtime.navigation_layers=self
        occupancy=runtime.map.occupancy
        size=(occupancy.log_odds.shape[1],occupancy.log_odds.shape[0])
        resize=lambda m:cv2.resize(m.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST).astype(bool)
        occupancy.static_occupied[:]=False
        occupancy.log_odds[:]=0.
        occupancy.observed[:]=resize(self.observed)
        occupancy.log_odds[occupancy.observed]=-2.
        occupancy.log_odds[resize(self.local_occupied)]=3.
        occupancy.revision+=1
        vacated=(self.now-self.vacated_time<3.) & ~self.tracked_occupied & self.observed & ~self.local_occupied
        voxels=runtime.map.voxels._voxels;remove=[]
        if vacated.any():
            for key,v in voxels.items():
                if v.last_step>=step:continue
                point=v.position_sum/max(v.count,1);rc=self.planner.cell(point[:2])
                if self.planner.valid_cell(rc) and vacated[rc]:remove.append(key)
        for key in remove:del voxels[key]
        self.cleared_voxels+=len(remove)

    def statistics(self):
        return {'map_source':'online_lidar','uses_scene_collision_prior':False,
            'initial_observed_cells':0,'initial_occupied_cells':0,
            'observed_cells':int(self.observed.sum()),'total_cells':int(self.observed.size),
            'global_observed_cells':int(self.global_observed.sum()),'scan_count':self.scan_count,
            'global_hz':1/self.global_period,'local_hz':1/self.local_period,
            'grid_resolution_m':self.planner.resolution,'limits':self.planner.limits,
            'obstacle_height_band_m':list(self.config.obstacle_height),
            'ray_traversal':self.config.ray_traversal,
            'free_ray_model':'height_clipped_actual_rays_with_conservative_range_endpoints',
            'latest_ray_statistics':self.ray_statistics,
            'local_radius_m':self.local_radius,'global_revision':self.global_revision,
            'local_revision':self.local_revision,'local_updates':self.local_updates,
            'global_updates':self.global_updates,'local_replans':self.local_replans,
            'cleared_voxels':self.cleared_voxels,'filtered_self_points':self.filtered_self_points,
            'visible_box_ids':self.visible_box_ids,'known_box_ids':sorted(self.detected_boxes),
            'movable_pose_source':'simulated object proposals admitted only by LiDAR visibility'}

    def save(self,directory):
        super().save(directory)
        p=Path(directory)/'layered_map.npz'
        with np.load(p) as data:arrays={k:data[k] for k in data.files}
        np.savez_compressed(p,**arrays,observed=self.observed,global_observed=self.global_observed,
                           online_log_odds=self.evidence.log_odds)
