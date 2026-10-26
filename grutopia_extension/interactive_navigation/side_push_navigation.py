"""Require roadside parking and a robot-width passage before resuming navigation."""
import math
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from grutopia_extension.interactive_navigation.box_push_navigation import BoxPushPlanner


class SidePushPlanner(BoxPushPlanner):
    def __init__(self,static_free,limits,passages=None,robot_radius=.25,**kwargs):
        self.passages=passages or {}
        self.active_passages={}
        self.robot_heading=None
        super().__init__(static_free,limits,robot_radius=robot_radius,contact_offset=.55,
            max_push_distance=1.6,parking_rule=self.side_parking_allowed,**kwargs)

    def blocked(self,boxes,omit=None):
        if getattr(self,'planning_scope','local')=='global' and getattr(self,'navigation_pose_heading',None) is None:
            free=self.free.copy()
            for box in boxes:
                if box.id!=omit:free &= ~self.box_mask(box)
            radius=getattr(self,'navigation_route_radius',self.radius)
            return distance_transform_edt(free)*self.resolution<=radius
        if self.robot_heading is None:return super().blocked(boxes,omit)
        if getattr(self,'full_navigation_footprint',False):
            from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked
            return navigation_body_blocked(self,[box for box in boxes if box.id!=omit],self.robot_heading)
        free=self.free.copy()
        for box in boxes:
            if box.id!=omit:free &= ~self.box_mask(box)
        theta=self.robot_heading;c,s=math.cos(theta),math.sin(theta)
        xs=np.arange(-.28,.371,self.resolution);ys=np.arange(-.22,.221,self.resolution)
        yy,xx=np.meshgrid(ys,xs)
        dx=np.rint((c*xx-s*yy)/self.resolution).astype(int).ravel()
        dy=np.rint((s*xx+c*yy)/self.resolution).astype(int).ravel()
        x0,x1=dx.min(),dx.max();y0,y1=dy.min(),dy.max()
        kernel=np.zeros((y1-y0+1,x1-x0+1),np.uint8);kernel[dy-y0,dx-x0]=1
        valid=cv2.erode(free.astype(np.uint8),kernel,anchor=(-x0,-y0),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        return ~valid.astype(bool)

    def configure_route(self,start,goal,boxes):
        path=self.path(start,goal,[])
        self.active_passages={}
        for box in boxes:
            if box.id in self.passages:
                self.active_passages[box.id]=self.passages[box.id]
                continue
            if not path:continue
            best=None
            points=[(*start,0),*path]
            for first,last in zip(points,points[1:]):
                a,b=np.array(first[:2]),np.array(last[:2]);segment=b-a
                if np.linalg.norm(segment)<.05:continue
                projection=a+segment*np.clip(np.dot(np.array(box.xy)-a,segment)/np.dot(segment,segment),0,1)
                distance=np.linalg.norm(np.array(box.xy)-projection)
                if best is None or distance<best[0]:best=(distance,projection,segment/np.linalg.norm(segment))
            if best:self.active_passages[box.id]={'center':best[1].tolist(),'axis':best[2].tolist()}

    def turning_blocked(self, boxes, first_heading, last_heading, margin=.05):
        """Union of body footprints through the turn, including gait clearance."""
        from grutopia_extension.interactive_navigation.go2_voronoi_navigation import navigation_body_blocked
        saved=self.robot_heading
        delta=(last_heading-first_heading+math.pi)%(2*math.pi)-math.pi
        blocked=np.zeros_like(self.free,dtype=bool)
        try:
            for theta in np.linspace(first_heading,first_heading+delta,19):
                self.robot_heading=theta
                blocked |= navigation_body_blocked(self,boxes,theta)
        finally:self.robot_heading=saved
        pixels=int(math.ceil(margin/self.resolution))
        return cv2.dilate(blocked.astype(np.uint8),np.ones((2*pixels+1,2*pixels+1),np.uint8)).astype(bool)

    def turning_staging(self, start, box_id, boxes, first_heading):
        passage=self.passages[box_id];axis=np.array(passage['axis'],float)
        last_heading=math.atan2(axis[1],axis[0])
        blocked=self.turning_blocked(boxes,first_heading,last_heading)
        self.robot_heading=first_heading
        for setback in [1.1,1.25,1.4,1.55]:
            point=np.array(passage['center'])-setback*axis
            rc=self.cell(point)
            if self.valid_cell(rc) and not blocked[rc]:
                path=self.path(start,point,boxes)
                if path:return point,path,blocked
        return None,None,blocked

    def clearance(self,box,xy=None):
        passage=self.active_passages[box.id]
        axis=np.array(passage['axis'],float);axis/=np.linalg.norm(axis)
        normal=np.array([-axis[1],axis[0]])
        yaw=math.radians(box.yaw)
        first=np.array([math.cos(yaw),math.sin(yaw)])
        second=np.array([-math.sin(yaw),math.cos(yaw)])
        half=abs(np.dot(normal,first))*box.dimensions[0]/2+abs(np.dot(normal,second))*box.dimensions[1]/2
        center=np.array(box.xy if xy is None else xy)
        lateral=abs(float(np.dot(center-passage['center'],normal)))
        protected_half=self.radius+.08
        required=half+protected_half
        return {'axis':axis.tolist(),'center':passage['center'],'lateral_offset_m':lateral,
                'required_lateral_offset_m':float(required),'passage_clearance_margin_m':float(lateral-required),
                'robot_passage_width_m':2*(self.radius+.08),'side_parked':bool(lateral>=required)}

    def side_parking_allowed(self,box,destination,direction):
        if box.id not in self.active_passages:return False
        axis=np.array(self.active_passages[box.id]['axis'],float);axis/=np.linalg.norm(axis)
        # Near-perpendicular pushes move the carton to the side of the passage.
        if abs(float(np.dot(direction,axis)))>.4:return False
        gap=self.clearance(box,destination)
        return gap['passage_clearance_margin_m']>=.025

    def decide(self,start,goal,boxes,excluded=()):
        self.configure_route(start,goal,boxes)
        return super().decide(start,goal,boxes,excluded)
