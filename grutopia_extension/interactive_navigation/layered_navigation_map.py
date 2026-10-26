"""Static-prior global planning and a rolling, time-stamped local obstacle layer.

Movable identities/poses currently come from the simulator. LiDAR supplies
untracked local obstacles and ray-clearing evidence. Moving-object occupancy
never enters the immutable architectural prior.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from grutopia_extension.interactive_navigation.mapping import OccupancyGridMap
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState


class LayeredNavigationMap:
    def __init__(self, planner, sensor_config, *, local_hz=10., global_hz=1.,
                 local_radius=3., obstacle_ttl=1.2, promote_after=3., clear_after=1.):
        if min(local_hz,global_hz,local_radius,obstacle_ttl,promote_after,clear_after)<=0:
            raise ValueError('Map rates, ranges and persistence must be positive')
        self.planner=planner
        self.static_occupied=~planner.free.copy()
        self.static_digest=hashlib.sha256(self.static_occupied.tobytes()).hexdigest()
        self.config=replace(sensor_config,grid_resolution=planner.resolution,
            x_limits=tuple(planner.limits[0]),y_limits=tuple(planner.limits[1]))
        self.local_period=1./local_hz;self.global_period=1./global_hz
        self.local_radius=local_radius;self.obstacle_ttl=obstacle_ttl
        self.promote_after=promote_after;self.clear_after=clear_after
        shape=self.static_occupied.shape
        self.local_hits=np.zeros(shape,bool);self.stable_obstacles=np.zeros(shape,bool)
        self.hit_time=np.full(shape,-np.inf);self.seen_time=np.full(shape,-np.inf)
        self.occupied_since=np.full(shape,np.inf);self.free_since=np.full(shape,np.inf)
        self.tracked_occupied=np.zeros(shape,bool);self.global_tracked=np.zeros(shape,bool)
        self.tracked_sensor_mask=np.zeros(shape,bool)
        self.robot_footprint=np.zeros(shape,bool)
        self.self_cleared_cells=0;self.filtered_self_points=0
        self.local_window=np.zeros(shape,bool);self.vacated_time=np.full(shape,-np.inf)
        self.pending_vacated=np.zeros(shape,bool)
        self.current_boxes=[];self.global_boxes=[];self.previous_boxes={}
        self.last_local=-np.inf;self.last_global=-np.inf;self.now=0.
        self.local_revision=0;self.global_revision=0;self.local_updates=0;self.global_updates=0
        self.local_replans=0
        self._local_digest=None;self._global_digest=None;self.last_scan=None
        self.robot_xy=np.zeros(2);self.robot_yaw=0.;self.global_path=();self.local_path=()
        self.decision=None;self.events=[];self.pending_global_event=None
        self.cleared_voxels=0;self._display_revision=-1

    def _mask(self,boxes):
        mask=np.zeros_like(self.static_occupied)
        for box in boxes:
            part=self.planner.box_mask(box)
            if part is None:raise ValueError(f'{box.id} is outside map bounds')
            mask |= part
        return mask

    def _window(self,position):
        result=np.zeros_like(self.static_occupied)
        r,c=self.planner.cell(position)
        extent=int(np.ceil(self.local_radius/self.planner.resolution))
        result[max(0,r-extent):min(result.shape[0],r+extent+1),
               max(0,c-extent):min(result.shape[1],c+extent+1)]=True
        return result

    @property
    def global_static(self):
        return self.static_occupied | self.stable_obstacles

    @property
    def local_dynamic(self):
        return self.local_hits & self.local_window & (self.now-self.hit_time<=self.obstacle_ttl)

    @property
    def local_occupied(self):
        return self.global_static | self.tracked_occupied | self.local_dynamic

    @property
    def global_occupied(self):
        return self.global_static | self.global_tracked

    def update(self,now,observation,boxes,*,force=False):
        self.now=float(now)
        if not force and self.now-self.last_local<self.local_period-1e-8:return False
        self.last_local=self.now;self.local_updates+=1
        self.robot_xy=np.asarray(observation['position'][:2],float)
        q=np.asarray(observation.get('orientation',[1,0,0,0]),float)
        q=q/max(np.linalg.norm(q),1e-9)
        self.robot_yaw=float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))
        forward=np.array([np.cos(self.robot_yaw),np.sin(self.robot_yaw)])
        footprint=BoxState('__robot__',tuple(self.robot_xy+.09*forward),
            (.82,2*getattr(self.planner,'navigation_half_width',.22),.5),np.degrees(self.robot_yaw))
        mask=self.planner.box_mask(footprint,padding=2*self.planner.resolution)
        self.robot_footprint=mask if mask is not None else np.zeros_like(self.static_occupied)
        self.local_window=self._window(self.robot_xy)
        # Only observed nearby boxes receive high-frequency pose updates;
        # the remainder keeps its last global snapshot until the next refresh.
        old={b.id:b for b in self.current_boxes}
        cached={b.id:b for b in self.global_boxes}
        self.current_boxes=[b if force or b.id not in old or
            np.linalg.norm(np.asarray(b.xy)-self.robot_xy)<=self.local_radius+.8 else cached.get(b.id,old[b.id])
            for b in boxes]
        new_mask=self._mask(self.current_boxes)
        vacated=self.tracked_occupied & ~new_mask & ~self.static_occupied
        self.vacated_time[vacated]=self.now;self.pending_vacated |= vacated
        self.local_hits[vacated]=False;self.hit_time[vacated]=-np.inf
        self.stable_obstacles[vacated]=False;self.occupied_since[vacated]=np.inf
        self.tracked_occupied=new_mask
        self.tracked_sensor_mask=new_mask.copy()
        for box in self.current_boxes:
            association=self.planner.box_mask(box,padding=.04)
            if association is not None:self.tracked_sensor_mask |= association
        self.local_hits[self.now-self.hit_time>self.obstacle_ttl]=False
        self.occupied_since[self.now-self.hit_time>self.obstacle_ttl]=np.inf
        self.free_since[self.now-self.seen_time>self.obstacle_ttl]=np.inf
        lidar=self.filter_robot_returns(observation,count=True).get('sensors',{}).get('lidar',{})
        scan=lidar.get('physics_step')
        points=np.asarray(lidar.get('pointcloud',[]),dtype=float)
        if points.size and (scan is None or scan!=self.last_scan):
            frame=OccupancyGridMap(self.config)
            frame.update_lidar(lidar.get('position',observation['position']),points,
                max_range=float(lidar.get('max_range',5.)))
            occupied=frame.occupied_mask()
            hit=np.zeros_like(self.static_occupied);free=np.zeros_like(hit)
            h=min(hit.shape[0],occupied.shape[0]);w=min(hit.shape[1],occupied.shape[1])
            hit[:h,:w]=occupied[:h,:w]
            free[:h,:w]=(frame.observed & ~occupied)[:h,:w]
            self.ingest_evidence(self.now,hit,free)
            self.last_scan=scan
        self.local_hits[self.tracked_occupied|self.static_occupied]=False
        self.occupied_since[self.tracked_occupied]=np.inf
        self.stable_obstacles[self.tracked_occupied]=False
        self.clear_robot_footprint()
        new_positions={b.id:(np.asarray(b.xy),np.asarray(b.dimensions)) for b in self.current_boxes}
        changed=[key for key,(xy,size) in new_positions.items() if key in self.previous_boxes and
            (np.linalg.norm(xy-self.previous_boxes[key][0])>.04 or
             np.linalg.norm(size-self.previous_boxes[key][1])>.04)]
        if changed:
            self.events.append({'event':'dynamic_objects_moved','time_s':self.now,'boxes':changed,
                'vacated_cells':int(vacated.sum())})
            self.previous_boxes=new_positions
        elif not self.previous_boxes:self.previous_boxes=new_positions
        digest=hashlib.blake2b(self.local_occupied.tobytes(),digest_size=12).digest()
        if digest!=self._local_digest:self.local_revision+=1;self._local_digest=digest
        if force or self.now-self.last_global>=self.global_period-1e-8:
            self.refresh_global(boxes,reason='initial' if force else 'periodic')
        # Local collision checking is the default view for all control stages.
        self._bind_local()
        return True

    def ingest_evidence(self,now,hit,free):
        """One scan's hit/free masks; hits take precedence over crossing rays."""
        self.now=float(now)
        hit=np.asarray(hit,bool) & self.local_window & ~self.tracked_sensor_mask & ~self.static_occupied
        free=np.asarray(free,bool) & self.local_window & ~hit & ~self.tracked_occupied
        new=hit & (~self.local_hits | (self.now-self.hit_time>self.obstacle_ttl))
        self.occupied_since[new]=self.now;self.occupied_since[free]=np.inf
        self.free_since[free & ~np.isfinite(self.free_since)]=self.now
        self.free_since[hit]=np.inf
        self.local_hits[free]=False;self.local_hits[hit]=True
        self.hit_time[hit]=self.now;self.seen_time[hit|free]=self.now
        self.clear_robot_footprint()

    def clear_robot_footprint(self):
        """Keep robot returns out of avoidance, preserving known scene objects.

        This is the layered equivalent of MapNavigationRuntime.mark_free.
        It never erases the immutable prior, confirmed stable obstacles, or
        the independent tracked-box occupancy, including intentional contact.
        """
        clear=self.robot_footprint & ~self.global_static & ~self.tracked_occupied
        self.self_cleared_cells+=int((self.local_hits & clear).sum())
        self.local_hits[clear]=False;self.hit_time[clear]=-np.inf
        self.occupied_since[clear]=np.inf;self.seen_time[clear]=self.now

    def filter_robot_returns(self,observation,*,count=False):
        """Remove footprint-internal unclassified hits before voxel fusion."""
        sensors=observation.get('sensors',{});lidar=sensors.get('lidar',{})
        points=np.asarray(lidar.get('pointcloud',[]))
        if points.ndim!=2 or points.shape[1]<3 or not len(points):return observation
        xy=np.asarray(points[:,:2],float);finite=np.isfinite(xy).all(axis=1)
        rows=np.zeros(len(points),int);cols=np.zeros(len(points),int)
        rows[finite]=np.floor((xy[finite,1]-self.planner.limits[1][0])/self.planner.resolution).astype(int)
        cols[finite]=np.floor((xy[finite,0]-self.planner.limits[0][0])/self.planner.resolution).astype(int)
        inside=finite & (rows>=0)&(cols>=0)&(rows<self.static_occupied.shape[0])&(cols<self.static_occupied.shape[1])
        removable=self.robot_footprint & ~self.global_static & ~self.tracked_sensor_mask
        remove=np.zeros(len(points),bool);remove[inside]=removable[rows[inside],cols[inside]]
        if not remove.any():return observation
        if count:self.filtered_self_points+=int(remove.sum())
        return {**observation,'sensors':{**sensors,'lidar':{**lidar,'pointcloud':points[~remove]}}}

    def refresh_global(self,boxes=None,*,reason='interaction_completed'):
        if boxes is not None:self.global_boxes=list(boxes)
        else:self.global_boxes=list(self.current_boxes)
        self.global_tracked=self._mask(self.global_boxes)
        persistent=(self.now-self.occupied_since>=self.promote_after) & self.local_dynamic
        confirmed_free=(self.now-self.free_since>=self.clear_after) & self.local_window
        self.stable_obstacles |= persistent & ~self.global_tracked
        self.stable_obstacles[confirmed_free|self.global_tracked]=False
        self.last_global=self.now;self.global_updates+=1
        digest=hashlib.blake2b(self.global_occupied.tobytes(),digest_size=12).digest()
        if digest!=self._global_digest:
            self.global_revision+=1;self._global_digest=digest
            self.pending_global_event={'event':'global_map_changed','time_s':self.now,
                'revision':self.global_revision,'reason':reason}
            self.events.append(self.pending_global_event)
        return self.global_revision

    def _bind_local(self):
        free=~(self.global_static|self.local_dynamic)
        if not np.array_equal(self.planner.free,free):
            self.planner.free=free
            self.planner.static_clear=distance_transform_edt(free)*self.planner.resolution>self.planner.radius

    @contextmanager
    def global_planning(self):
        saved=(self.planner.free,self.planner.static_clear,getattr(self.planner,'planning_scope','local'))
        free=~self.global_static
        self.planner.free=free
        self.planner.static_clear=distance_transform_edt(free)*self.planner.resolution>self.planner.radius
        self.planner.planning_scope='global'
        try:yield self.global_boxes
        finally:self.planner.free,self.planner.static_clear,self.planner.planning_scope=saved

    def synchronize_display(self,runtime,step):
        """Expose the same map used for control and remove vacated voxel ghosts."""
        runtime.navigation_layers=self
        occupancy=runtime.map.occupancy
        size=(occupancy.log_odds.shape[1],occupancy.log_odds.shape[0])
        resize=lambda m:cv2.resize(m.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST).astype(bool)
        occupancy.static_occupied[:]=resize(self.global_static)
        occupancy.log_odds[:]=-2.
        occupancy.log_odds[resize(self.tracked_occupied|self.local_dynamic)]=3.
        occupancy.observed[:]=True
        occupancy.revision+=1
        voxels=runtime.map.voxels._voxels
        self.pending_vacated=(self.now-self.vacated_time<3.) & ~self.tracked_occupied
        if self.pending_vacated.any():
            removed=[]
            for key,voxel in voxels.items():
                if voxel.last_step>=step:continue
                point=voxel.position_sum/max(1,voxel.count)
                cell=self.planner.cell(point[:2])
                if self.planner.valid_cell(cell) and self.pending_vacated[cell] and not self.static_occupied[cell]:
                    removed.append(key)
            for key in removed:del voxels[key]
            self.cleared_voxels+=len(removed)

    def statistics(self):
        return {'global_hz':1/self.global_period,'local_hz':1/self.local_period,
            'grid_resolution_m':self.planner.resolution,'limits':self.planner.limits,
            'local_radius_m':self.local_radius,'obstacle_ttl_s':self.obstacle_ttl,
            'global_revision':self.global_revision,'local_revision':self.local_revision,
            'local_updates':self.local_updates,'global_updates':self.global_updates,
            'local_replans':self.local_replans,
            'self_cleared_cells':self.self_cleared_cells,'filtered_self_points':self.filtered_self_points,
            'cleared_voxels':self.cleared_voxels,'immutable_prior_preserved':
                hashlib.sha256(self.static_occupied.tobytes()).hexdigest()==self.static_digest,
            'movable_pose_source':'simulator, high-frequency updates within local window'}

    def save(self,directory):
        root=Path(directory);root.mkdir(exist_ok=True)
        np.savez_compressed(root/'layered_map.npz',static=self.static_occupied,
            global_objects=self.global_tracked,stable=self.stable_obstacles,
            local_dynamic=self.local_dynamic,current_objects=self.tracked_occupied,
            vacated_time=self.vacated_time,local_window=self.local_window)
        (root/'layered_map.json').write_text(json.dumps({**self.statistics(),
            'time_s':self.now,'current_boxes':[vars(b) for b in self.current_boxes],
            'global_boxes':[vars(b) for b in self.global_boxes]},indent=2))
        (root/'map_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
