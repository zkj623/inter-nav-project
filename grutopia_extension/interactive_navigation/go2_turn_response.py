"""Start a stalled yaw gait using a short, observed-free stepping arc."""
import math
import numpy as np
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear


def stepping_arc_clear(planner,boxes,position,heading,rotation):
    return any(body_motion_clear(planner,boxes,position,heading,speed,rotation,horizon=.8)
               for speed in [-.20,.20])


class TurnResponseAssist:
    def __init__(self):
        self.sign=None;self.heading=None;self.last_progress=None
        self.assist_origin=None;self.assist_heading=None;self.assist_speed=None

    def update(self,command,position,heading,planner,boxes,limiter,now,*,maintain_arc=False):
        command=np.asarray(command,float).copy();position=np.asarray(position[:2],float)
        sign=np.sign(command[2])
        if np.linalg.norm(command[:2])>.01 or abs(command[2])<.3:
            self.__init__();return command,{}
        if sign!=self.sign or self.heading is None:
            self.__init__()
            self.sign=sign;self.heading=heading;self.last_progress=now
        progress=sign*((heading-self.heading+math.pi)%(2*math.pi)-math.pi)
        if progress>.04:
            self.heading=heading;self.last_progress=now
        if self.assist_origin is not None:
            rotation=sign*((heading-self.assist_heading+math.pi)%(2*math.pi)-math.pi)
            if (rotation>.18 and not maintain_arc) or np.linalg.norm(position-self.assist_origin)>.20:
                self.assist_origin=None;self.assist_speed=None;self.last_progress=now
                return command,{'turn_assist':'gait_started'}
        stalled=(now-self.last_progress>.8 and abs(limiter.measured[2])<.15)
        if not stalled and self.assist_origin is None:return command,{}
        speeds=[self.assist_speed] if self.assist_speed is not None else [-.20,.20]
        for speed in speeds:
            candidate=np.array([speed,0.,command[2]])
            preview=limiter.preview(candidate,now)
            if (body_motion_clear(planner,boxes,position,heading,speed,command[2],horizon=.8) and
                body_motion_clear(planner,boxes,position,heading,preview[0],preview[2],horizon=.8,lateral=preview[1])):
                if self.assist_origin is None:self.assist_origin=position.copy();self.assist_heading=heading
                self.assist_speed=speed
                return candidate,{'turn_assist':'observed_free_stepping_arc','turn_assist_forward_mps':speed}
        return command,{'turn_assist':'no_observed_safe_arc'}
