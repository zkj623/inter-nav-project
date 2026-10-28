"""Measured settling and recovery coverage for non-contact interaction motion."""
import numpy as np


def navigation_recovery_enabled(phase,skill_state=None):
    return phase in {'navigate','approach','decide','cross_passage',
                     'inspect_navigate','inspect_turn','inspect_wait'} or (
        phase=='skill' and skill_state in {'navigate_to_stage','reobserve'})


class MeasuredStopGate:
    def __init__(self,hold_seconds=.4):
        self.hold_seconds=float(hold_seconds);self.reset()

    def reset(self):
        self.since=None;self.origin=None;self.initial_tilt=None

    def update(self,position,linear_velocity,angular_velocity,tilt_degrees,now):
        p=np.asarray(position,float);v=np.asarray(linear_velocity,float);w=np.asarray(angular_velocity,float)
        quiet=(np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(w).all()
               and np.linalg.norm(v[:2])<.08 and abs(v[2])<.05
               and np.max(np.abs(w))<.10 and np.isfinite(tilt_degrees) and tilt_degrees<15.)
        if not quiet:self.reset();return False
        if self.since is not None and (now<self.since or np.linalg.norm(p[:2]-self.origin[:2])>=.025
                or abs(p[2]-self.origin[2])>=.015 or abs(tilt_degrees-self.initial_tilt)>=1.5):
            self.reset()
        if self.since is None:
            self.since=float(now);self.origin=p.copy();self.initial_tilt=float(tilt_degrees)
        return now-self.since>=self.hold_seconds


class ObservedSubgoalArrival:
    """Latch braking at a nearby subgoal; never hand off a moving robot."""
    def __init__(self,tolerance=.16):
        self.tolerance=float(tolerance);self.stop=MeasuredStopGate();self.reset()

    def reset(self):self.braking=False;self.stop.reset()

    def update(self,target,position,linear_velocity,angular_velocity,tilt_degrees,now,*,heading=None,command=None,acceleration=.65):
        delta=np.asarray(target[:2])-np.asarray(position[:2]);error=float(np.linalg.norm(delta))
        brake_radius=self.tolerance
        if heading is not None and error>1e-9:
            c,s=np.cos(heading),np.sin(heading);rot=np.array([[c,-s],[s,c]])
            toward=max(0.,float(np.dot(rot@np.asarray(linear_velocity[:2]),delta/error)))
            commanded=0. if command is None else max(0.,float(np.dot(rot@np.asarray(command[:2]),delta/error)))
            brake_radius=max(brake_radius,.025+.5*toward+commanded**2/(2*acceleration))
        if error<brake_radius and not self.braking:
            self.braking=True;self.stop.reset()
        detail={'subgoal_error_m':error,'subgoal_brake_radius_m':brake_radius,'subgoal_braking':self.braking,'subgoal_reached':False,'replan':False}
        if not self.braking:return detail
        if not self.stop.update(position,linear_velocity,angular_velocity,tilt_degrees,now):return detail
        if error<self.tolerance:detail['subgoal_reached']=True
        else:
            self.reset();detail.update(subgoal_braking=False,replan=True)
        return detail


def checked_skill_brake(command,context):
    """Slew a requested stop through the same body/momentum safety check."""
    velocity=np.asarray(command.velocity,float)
    if command.status!='running' or not context.healthy or getattr(context,'tilt_degrees',0.)>=18.:return np.zeros(3)
    if (command.navigation_goal is not None or np.any(np.abs(velocity)>1e-9) or context.limiter is None):
        return velocity
    from grutopia_extension.interactive_navigation.go2_contact_turn import checked_contact_turn_command
    boxes=context.boxes
    if command.state in {'settle_push_progress','brake_after_push'} and context.box is not None:
        # Controlled deceleration is still part of the permitted push. The
        # target remains a physical collider, but rejecting that SAME contact
        # here changes a slew-limited stop into a hard zero at every brake.
        # Keep the static map and all other objects in the momentum check.
        boxes=[box for box in boxes if box.id!=context.box.id]
    safe,_=checked_contact_turn_command([0.,0.,0.],context.position,context.heading,
        context.planner,boxes,context.limiter,context.now)
    return safe
