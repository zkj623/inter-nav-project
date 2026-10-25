"""Navigation video maps with explicit task start, goal and passage markers."""
import cv2
from grutopia_extension.interactive_navigation.visualization import InteractionVideoRecorder


class TaskVideoRecorder(InteractionVideoRecorder):
    def __init__(self,*args,task=None,**kwargs):
        self.task=task
        super().__init__(*args,**kwargs)

    def _render_map(self,step,state,observation,mapping_runtime,target_context=None,view='geometry'):
        frame=super()._render_map(step,state,observation,mapping_runtime,target_context,view=view)
        if self.task:
            for label,point,color,kind in [('S'+self.task['id'][-1],self.task['start'],(30,165,40),'start'),
                                          ('G'+self.task['id'][-1],self.task['goal'],(35,35,230),'goal')]:
                pixel=self._world_to_pixel(point,mapping_runtime.map)
                if kind=='start':cv2.circle(frame,pixel,10,color,3,cv2.LINE_AA)
                else:cv2.drawMarker(frame,pixel,color,cv2.MARKER_STAR,24,3,cv2.LINE_AA)
                cv2.putText(frame,label,(pixel[0]+12,pixel[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),4,cv2.LINE_AA)
                cv2.putText(frame,label,(pixel[0]+12,pixel[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        return frame
