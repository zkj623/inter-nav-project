"""Navigation video maps with explicit task start, goal and passage markers."""
import cv2
import math
import json
import numpy as np
from grutopia_extension.interactive_navigation.visualization import InteractionVideoRecorder


class TaskVideoRecorder(InteractionVideoRecorder):
    def __init__(self,*args,task=None,**kwargs):
        self.task=task
        super().__init__(*args,**kwargs)

    def close(self):
        if self.frame_count==0:
            self._video_frames.close()
            for writer in self._writers.values():writer.release()
            (self.output_dir/'recording_incomplete.json').write_text(json.dumps({'frames':0,'reason':'no_completed_capture'}))
            return
        return super().close()

    def _render_map(self,step,state,observation,mapping_runtime,target_context=None,view='geometry'):
        layers=getattr(mapping_runtime,'navigation_layers',None)
        frame=(self._render_layers(step,state,observation,mapping_runtime,layers) if layers is not None else
            super()._render_map(step,state,observation,mapping_runtime,target_context,view=view))
        if self.task:
            for label,point,color,kind in [('S'+self.task['id'][-1],self.task['start'],(30,165,40),'start'),
                                          ('G'+self.task['id'][-1],self.task['goal'],(35,35,230),'goal')]:
                pixel=self._world_to_pixel(point,mapping_runtime.map)
                if kind=='start':cv2.circle(frame,pixel,10,color,3,cv2.LINE_AA)
                else:cv2.drawMarker(frame,pixel,color,cv2.MARKER_STAR,24,3,cv2.LINE_AA)
                cv2.putText(frame,label,(pixel[0]+12,pixel[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),4,cv2.LINE_AA)
                cv2.putText(frame,label,(pixel[0]+12,pixel[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        return frame

    def _render_layers(self,step,state,observation,runtime,layers):
        frame=np.full((720,1280,3),248,np.uint8)
        p=layers.planner;grid=np.full((*layers.static_occupied.shape,3),242,np.uint8)
        grid[layers.static_occupied]=(83,88,94)
        grid[layers.stable_obstacles]=(135,140,148)
        grid[(layers.now-layers.vacated_time<3.) & ~layers.static_occupied]=(167,239,188)
        grid[layers.global_tracked & ~layers.tracked_occupied]=(216,221,230)
        grid[layers.local_dynamic]=(55,65,223)
        grid[layers.tracked_occupied]=(70,181,245)
        if hasattr(layers,'observed'):
            grid[~layers.observed]=(164,168,175)
        height,width=grid.shape[:2];plot_width=775
        scale=min(plot_width/width,650/height);w,h=int(width*scale),int(height*scale)
        left=(plot_width-w)//2;top=46+(650-h)//2
        self._map_viewport=(left,top,w,h)
        frame[top:top+h,left:left+w]=cv2.resize(np.flipud(grid),(w,h),interpolation=cv2.INTER_NEAREST)
        pixel=lambda xy:self._world_to_pixel(xy,runtime.map)
        def route(start,points,color,width=2,dashed=False):
            vertices=[start,*points]
            for a,b in zip(vertices,vertices[1:]):
                a=np.asarray(pixel(a));b=np.asarray(pixel(b))
                if not dashed:cv2.line(frame,tuple(a),tuple(b),color,width,cv2.LINE_AA)
                else:
                    n=max(1,int(np.linalg.norm(b-a)/7))
                    for i in range(0,n,2):
                        c=a+(b-a)*i/n;d=a+(b-a)*min(i+1,n)/n
                        cv2.line(frame,tuple(c.astype(int)),tuple(d.astype(int)),color,width,cv2.LINE_AA)
        details=layers.decision or {}
        origin=details.get('start',self.task['start'])
        if details.get('before_path'):route(origin,details['before_path'],(20,120,230),2,True)
        if details.get('after_path'):route(origin,details['after_path'],(30,175,50),2,True)
        current_route=runtime._planned_path or ()
        if current_route:route(layers.robot_xy,current_route,(235,105,25),2)
        if self._trajectory:
            for a,b in zip(self._trajectory,self._trajectory[1:]):
                cv2.line(frame,pixel(a),pixel(b),(90,120,70),1,cv2.LINE_AA)
        if details.get('kind')=='push':
            a=pixel(details['initial_xy']);b=pixel(details['destination_xy'])
            cv2.arrowedLine(frame,a,b,(185,45,190),3,cv2.LINE_AA,tipLength=.25)
            cv2.circle(frame,a,10,(185,45,190),1,cv2.LINE_AA)
        for box in layers.current_boxes:
            xy=np.asarray(box.xy);half=np.asarray(box.dimensions[:2])/2
            a=pixel(xy-half);b=pixel(xy+half)
            cv2.rectangle(frame,(a[0],b[1]),(b[0],a[1]),(20,110,165),1)
            cv2.putText(frame,box.id,(a[0],b[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.40,(25,65,110),1,cv2.LINE_AA)
        center=pixel(layers.robot_xy)
        yaw=layers.robot_yaw
        ahead=layers.robot_xy+.45*np.array([math.cos(yaw),math.sin(yaw)])
        halfwidth=getattr(p,'navigation_half_width',.22)
        body=np.array([[-.32,-halfwidth],[.50,-halfwidth],[.50,halfwidth],[-.32,halfwidth]])
        rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        body_world=body@rot.T+layers.robot_xy
        cv2.polylines(frame,[np.array([pixel(xy) for xy in body_world],np.int32)],True,(210,70,25),1,cv2.LINE_AA)
        cv2.circle(frame,center,6,(210,70,25),-1)
        cv2.arrowedLine(frame,center,pixel(ahead),(210,70,25),2,tipLength=.35)
        radius=layers.local_radius
        a=pixel(layers.robot_xy-radius);b=pixel(layers.robot_xy+radius)
        cv2.rectangle(frame,(a[0],b[1]),(b[0],a[1]),(205,110,40),1)
        cv2.line(frame,(785,40),(785,719),(215,215,215),1)
        x=804;y=65
        def line(text,color=(55,55,55),scale=.49):
            nonlocal y
            cv2.putText(frame,str(text),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA);y+=26
        def distance(value):return 'BLOCKED' if value is None else f'{value:.2f} m'
        choice=details.get('kind','at goal' if 'goal' in state else 'deciding').upper()
        line(choice+(' '+details['box_id'] if choice=='PUSH' else ''),(145,40,145),.66)
        reasons={'ordinary_path_available':'Open route: no interaction needed',
            'goal_not_connected_in_observed_map':'Goal unseen / disconnected: observe frontier',
            'no_reachable_observed_frontier':'Waiting for a safe observed route',
            'detour_cheaper_than_push':'Detour costs less than interaction',
            'movable_box_blocks_efficient_route':('Clearing opens a blocked route' if details.get('opens_connectivity') else 'Side push costs less than detour'),
            'no_feasible_push_use_detour':'No safe parking: use detour'}
        line(reasons.get(details.get('reason'),'Evaluating route and parking'),scale=.45)
        execution=details.get('execution_state')
        if execution=='side_parked_pending_observation':line('Parked: corridor observation pending',scale=.44)
        elif execution=='confirming_after_release':line('Released: confirming the observed corridor',scale=.42)
        elif execution=='observed_corridor_clear':line('Observed full-body corridor confirmed',(35,160,45),.43)
        line('Before: '+(distance(details['before_length_m']) if 'before_length_m' in details else 'not evaluated'),(20,120,230))
        if details.get('after_length_m') is not None:
            line('Predicted route: '+distance(details['after_length_m']),(35,160,45))
            gain=details.get('path_saving_m')
            line('Route benefit: '+(f'{gain:.2f} m shorter' if gain is not None else 'disconnected -> reachable'))
            line(f"Approach {details['approach_length_m']:.2f} m | push {details['push_distance_m']:.2f} m")
            line(f"Push plan cost: {details['total_push_cost']:.2f} equiv. m")
            line('Cost = approach + 2.5*push + remaining',scale=.41)
            clearance=details.get('passage_clearance')
            if clearance:line(f"Pred. side clearance: {clearance['passage_clearance_margin_m']:.2f} m",scale=.44)
        line(f"Goal distance: {np.linalg.norm(layers.robot_xy-np.array(self.task['goal'])):.2f} m",(45,70,180),.48)
        line(f'Global {1/layers.global_period:g} Hz | local {1/layers.local_period:g} Hz',scale=.47)
        line(f'Revisions G{layers.global_revision} / L{layers.local_revision}',scale=.44)
        if hasattr(layers,'observed'):
            line(f'Online map: {100*layers.observed.mean():.1f}% observed | grey: unknown',scale=.43)
        line('Gold: box   Red: live local obstacle',scale=.42)
        line('Green: vacated   Blue: current route',scale=.42)
        # The inset uses the exact high-frequency collision layer, not a
        # separately accumulated visualization map.
        r,c=p.cell(layers.robot_xy);extent=int(radius/p.resolution)
        r0=max(0,r-extent);r1=min(height,r+extent+1)
        c0=max(0,c-extent);c1=min(width,c+extent+1)
        inset=grid[r0:r1,c0:c1].copy()
        if inset.size:
            side=min(270,700-y-22);ix=x+95;iy=y+18
            if side>50:
                crop=cv2.resize(np.flipud(inset),(side,side),interpolation=cv2.INTER_NEAREST)
                def local_pixel(xy):
                    rr,cc=p.cell(xy)
                    return (int((cc-c0)/max(c1-c0-1,1)*(side-1)),int((r1-1-rr)/max(r1-r0-1,1)*(side-1)))
                cv2.circle(crop,local_pixel(layers.robot_xy),5,(210,70,25),-1)
                cv2.polylines(crop,[np.array([local_pixel(xy) for xy in body_world],np.int32)],True,(210,70,25),1)
                cv2.arrowedLine(crop,local_pixel(layers.robot_xy),local_pixel(ahead),(210,70,25),2)
                points=[layers.robot_xy,*current_route]
                for aa,bb in zip(points,points[1:]):cv2.line(crop,local_pixel(aa),local_pixel(bb),(235,105,25),1)
                frame[iy:iy+side,ix:ix+side]=crop
                cv2.putText(frame,'LOCAL COLLISION MAP',(x,y+9),cv2.FONT_HERSHEY_SIMPLEX,.43,(65,65,65),1,cv2.LINE_AA)
        self._header(frame,f'Global guidance + live local map | {state} | {layers.now:.1f} s')
        return frame
