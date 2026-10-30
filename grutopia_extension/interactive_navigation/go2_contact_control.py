"""Place the actual Go2 contact point on the carton push centerline."""
import numpy as np
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


# Contact-pose stopping response is about 0.4 s of current speed. A 1 s
# prediction cancelled small lateral requests before the rough gait stepped.
CONTACT_BRAKING_HORIZON_S = .40
# Reserve for measured post-withdrawal body relaxation before braking.
# The ordinary footprint and every final stopped-pose check remain intact.
CONTACT_RELEASE_BRAKING_MARGIN_M = .04


def observed_release_braking_clear(planner, boxes, position, heading):
    """Require observed body space around the pose before ending withdrawal."""
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_motion_clear, body_rotation_clear
    return (body_motion_clear(planner,boxes,position,heading,0.,0.,horizon=0.)
            and body_rotation_clear(planner,boxes,position,heading,heading,
                                    margin=CONTACT_RELEASE_BRAKING_MARGIN_M))


def approach_alignment_ready(position, approach, planner, blocked, radius=.35):
    """Hand a nearby, footprint-safe approach to the contact gait controller."""
    if np.linalg.norm(np.asarray(position[:2])-np.asarray(approach[:2])) >= radius:
        return False
    start, end = planner.cell(position), planner.cell(approach)
    return (planner.valid_cell(start) and planner.valid_cell(end)
            and not blocked[start] and not blocked[end]
            and AStarMapPlanner.segment_clear(start, end, blocked))


def nose_aligned_body_target(box_xy, dimensions, direction, heading, nose_offset=.33, front_gap=.14):
    direction=np.asarray(direction,dtype=float)
    direction=direction/np.linalg.norm(direction)
    extent=float(np.dot(np.abs(direction),np.asarray(dimensions[:2],dtype=float))/2)
    nose_target=np.asarray(box_xy,dtype=float)-direction*(extent+front_gap)
    forward=np.array([np.cos(heading),np.sin(heading)])
    return nose_target-nose_offset*forward


def contact_centering_velocity(delta, direction, forward_speed, lateral_speed=.35, lateral_tolerance=.035,axis=None):
    """Correct the contact line before advancing toward the carton.

    A normalized diagonal request can leave its lateral component below the
    gait's effective speed. Separate the axes while keeping the nose farther
    from the carton during the sidestep.
    """
    direction=np.asarray(direction,dtype=float)
    direction=direction/np.linalg.norm(direction)
    delta=np.asarray(delta,dtype=float)
    tangent=np.array([-direction[1],direction[0]])
    lateral=float(np.dot(delta,tangent))
    if axis not in [None,'lateral','longitudinal']:raise ValueError('Unknown contact correction axis')
    if axis=='lateral' or (axis is None and abs(lateral)>lateral_tolerance):
        return lateral_speed*np.sign(lateral)*tangent
    longitudinal=float(np.dot(delta,direction))
    speed=forward_speed if longitudinal>=0 else max(forward_speed,.35)
    return speed*np.sign(longitudinal)*direction


def contact_axis_delta(delta,direction,axis):
    """Distance remaining along the axis actually commanded to the gait."""
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    if axis=='lateral':unit=np.array([-direction[1],direction[0]])
    elif axis=='longitudinal':unit=direction
    else:raise ValueError('Contact braking requires a translation axis')
    return unit*float(np.dot(np.asarray(delta,float),unit))


def damped_contact_centering_velocity(delta, direction, velocity, max_speed, axis, *, lateral_speed=None):
    """Use longitudinal feedback and an optional calibrated lateral stride.

    Longitudinal requests decay with measured progress. A calibrated lateral
    stride remains above the policy's measured no-step region and ends using
    the caller's predictive brake and settling gate.
    """
    if not np.isfinite(max_speed) or not .20 <= max_speed <= .50:
        raise ValueError('Contact centering speed must be within 0.20-0.50 m/s')
    correction=contact_axis_delta(delta,direction,axis)
    distance=float(np.linalg.norm(correction))
    if distance<1e-9:return np.zeros(2)
    unit=correction/distance
    if axis=='lateral' and lateral_speed is not None:
        if not np.isfinite(lateral_speed) or not .35<=lateral_speed<=.50:
            raise ValueError('Calibrated lateral stepping request must be within 0.35-0.50 m/s')
        # The cold rough policy leans without stepping at 0.25 m/s sideways.
        # Use a measured effective stride request; the caller brakes on
        # actual travel rather than decaying into that inactive range.
        return float(lateral_speed)*unit
    toward=float(np.dot(np.asarray(velocity,float)[:2],unit))
    speed=float(np.clip(1.5*distance-.6*toward,.20,max_speed))
    return speed*unit


def contact_heading_hold(heading_error, measured_yaw_rate):
    """Counter the measured yaw drift of a centering sidestep before realignment."""
    return float(np.clip(3.*heading_error-.35*measured_yaw_rate,-.35,.35))


class ContactYawServo:
    """Finish a small heading correction with measured braking before reversal."""
    def __init__(self,tolerance=.10):
        self.tolerance=float(tolerance)
        self.braking=False;self.stable_since=None;self.direction=0.

    def update(self,error,measured_rate,now):
        if self.braking:
            if abs(measured_rate)<.06:
                if self.stable_since is None:self.stable_since=now
                if now-self.stable_since>=.20:
                    self.braking=False;self.direction=0.;self.stable_since=None
            else:self.stable_since=None
            return 0.
        if self.direction==0.:self.direction=float(np.sign(error))
        toward=self.direction*measured_rate
        stopping_angle=max(0.,toward)*.55+.025
        if (abs(error)<.7*self.tolerance or error*self.direction<0
                or (toward>.06 and abs(error)<stopping_angle)):
            self.braking=True;self.stable_since=None
            return 0.
        return self.direction*float(np.clip(3.*abs(error)+.25,.40,.65))


def contact_braking_required(delta, velocity, horizon=1.0, stop_margin=.025):
    """Allow for gait travel after a zero-speed request near the contact pose."""
    delta=np.asarray(delta,dtype=float)
    distance=float(np.linalg.norm(delta))
    toward=float(np.dot(delta,velocity)/max(distance,1e-8))
    return toward>0 and distance<stop_margin+horizon*toward


def contact_face_lateral_allowance(dimensions, direction, heading_error, tilt_degrees,
                                   maximum=.10, face_margin=.04):
    """Keep the whole Go2 head envelope inside a cardinal box face.

    The two head colliders in isaaclab_go2.usd fit body-frame bounds
    x=[.239,.341], y=[-.051,.051], z=[-.108,.061] (including 1 mm padding).
    Bound their lateral projection relative to the existing .33 m nose point
    for every roll/pitch compatible with the measured total tilt. This is a
    contact-face margin; all robot-body collision checks remain independent.
    """
    dimensions=np.asarray(dimensions,float);direction=np.asarray(direction,float)
    if (dimensions.size<2 or direction.shape!=(2,) or not np.isfinite(dimensions).all()
            or not np.isfinite(direction).all() or np.any(dimensions[:2]<=0)
            or not np.isfinite([heading_error,tilt_degrees,maximum,face_margin]).all()
            or maximum<0 or face_margin<0 or not 0<=tilt_degrees<90):return 0.
    length=float(np.linalg.norm(direction))
    if length<1e-9:return 0.
    direction=direction/length
    # This skill plans against axis-aligned faces. Do not infer a full face
    # from a diagonal AABB support projection.
    if np.max(np.abs(direction))<1.-1e-6:return 0.
    tangent=np.array([-direction[1],direction[0]])
    half_face=float(np.dot(np.abs(tangent),dimensions[:2]))/2
    tilt=np.deg2rad(tilt_degrees);st,ct=np.sin(tilt),np.cos(tilt)
    sy,cy=abs(np.sin(heading_error)),abs(np.cos(heading_error))
    longitudinal=max(abs(.239*ct-.33),abs(.341-.33))
    projected_head=sy*(longitudinal+st*(.051*st+.108))+cy*(.051+.108*st)
    return float(max(0.,min(maximum,half_face-projected_head-face_margin)))


def observed_contact_face_clear(position, orientation, box_xy, center_z, local_dimensions,
                                box_orientation, direction, maximum=.10, face_margin=.04):
    """Project the padded head envelope onto the observed, oriented box face.

    The world AABB is retained for obstacle checks, but is not a physical
    contact face when the box rotates. This checks face width and bounded
    centering; it does not certify box movement or passage clearance.
    """
    position=np.asarray(position,float);dimensions=np.asarray(local_dimensions,float)
    direction=np.asarray(direction,float);box_xy=np.asarray(box_xy,float)
    if (position.shape!=(3,) or dimensions.shape!=(3,) or direction.shape!=(2,) or box_xy.shape!=(2,)
            or not all(np.isfinite(x).all() for x in [position,dimensions,direction,box_xy])
            or not np.isfinite([center_z,maximum,face_margin]).all() or np.any(dimensions<=0)
            or maximum<0 or face_margin<0):return False
    def rotation(q):
        q=np.asarray(q,float)
        if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-9:return None
        w,x,y,z=q/np.linalg.norm(q)
        return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
            [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
    robot_rotation=rotation(orientation);box_rotation=rotation(box_orientation)
    if robot_rotation is None or box_rotation is None or np.linalg.norm(direction)<1e-9:return False
    normal=np.array([*(direction/np.linalg.norm(direction)),0.])
    alignment=np.abs(box_rotation[:,:2].T@normal);face_axis=int(np.argmax(alignment))
    if alignment[face_axis]<.9:return False  # No near-normal cardinal push on this observed face.
    tangent_axis=1-face_axis
    corners=np.array([[x,y,z] for x in [.239,.341] for y in [-.051,.051] for z in [-.108,.061]])
    head_world=position+corners@robot_rotation.T
    box_center=np.array([*box_xy,float(center_z)])
    local_head=(head_world-box_center)@box_rotation
    projected=local_head[:,tangent_axis];lower=float(projected.min());upper=float(projected.max())
    half_face=dimensions[tangent_axis]/2
    return bool(lower>=-half_face+face_margin and upper<=half_face-face_margin
                and abs((lower+upper)/2)<maximum)


def contact_pose_ready(position, heading, box_xy, dimensions, direction,
                       front_gap, yaw_tolerance, nose_offset=.33, minimum_gap=.20):
    """Accept a safe standoff interval instead of demanding backward gait at one point."""
    direction=np.asarray(direction,dtype=float)
    direction=direction/np.linalg.norm(direction)
    tangent=np.array([-direction[1],direction[0]])
    nose=np.asarray(position[:2])+nose_offset*np.array([np.cos(heading),np.sin(heading)])
    delta=np.asarray(box_xy)-nose
    extent=float(np.dot(np.abs(direction),np.asarray(dimensions[:2]))/2)
    gap=float(np.dot(delta,direction)-extent)
    heading_error=(np.arctan2(direction[1],direction[0])-heading+np.pi)%(2*np.pi)-np.pi
    return (minimum_gap<=gap<=front_gap+.075 and abs(float(np.dot(delta,tangent)))<.055
            and abs(heading_error)<yaw_tolerance)


def safe_contact_retreat(planner, boxes, position, heading, direction, required_distance):
    """Select a whole-body-safe withdrawal, including escape from mild overlap.

    The selected contact box is omitted by the caller. Existing overlap with
    other geometry may only decrease, and the destination must be fully free.
    A fixed 24 cm test incorrectly rejected withdrawals needing one more cell.
    """
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
    position=np.asarray(position[:2],float)
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    tangent=np.array([-direction[1],direction[0]])
    candidates=[]
    minimum=max(.25,min(.50,float(required_distance)+.10))
    for distance in [minimum,minimum+.10,minimum+.20]:
        for lateral in [0.,-.10,.10,-.20,.20,-.30,.30,-.40,.40]:
            delta=-distance*direction+lateral*tangent
            candidates.append((np.linalg.norm(delta)+abs(lateral),delta))
    for _,delta in sorted(candidates,key=lambda entry:entry[0]):
        target=position+delta
        if not body_translation_recovery_clear(planner,boxes,position,heading,target):
            continue
        velocity=.35*delta/max(np.linalg.norm(delta),1e-9)
        return velocity,target
    return None


def observed_contact_retreat(planner, boxes, position, heading, direction, limiter, now, *, speed=.20):
    """Take a bounded observed step when the full withdrawal is still unknown.

    Include the contact box: its existing footprint overlap may decrease but
    may not deepen. Never authorize movement by treating unknown as free.
    The caller keeps sensing and recomputes the remaining withdrawal.
    """
    from grutopia_extension.interactive_navigation.go2_voronoi_navigation import body_translation_recovery_clear
    if not np.isfinite(speed) or speed<=0.:raise ValueError('Retreat speed must be finite and positive')
    position=np.asarray(position[:2],float);direction=np.asarray(direction,float)
    direction/=max(np.linalg.norm(direction),1e-9)
    c,s=np.cos(heading),np.sin(heading);rotation=np.array([[c,-s],[s,c]])
    measured=np.zeros(3) if limiter is None else limiter.measured
    if abs(measured[2])>.15:return None
    measured_target=position+rotation@measured[:2]*.25
    if not body_translation_recovery_clear(planner,boxes,position,heading,measured_target,require_clear_end=False):return None
    tangent=np.array([-direction[1],direction[0]])
    candidates=[-rotation[:,0],-direction]
    candidates.extend(-direction+ratio*tangent for ratio in [.2,-.2,.4,-.4])
    for unit in candidates:
        unit=unit/max(np.linalg.norm(unit),1e-9)
        if float(np.dot(unit,-direction))<.75:continue
        velocity=float(speed)*unit;target=position+velocity*.9
        if not body_translation_recovery_clear(planner,boxes,position,heading,target,require_clear_end=False):continue
        request=np.array([*(rotation.T@velocity),0.])
        applied=request if limiter is None else limiter.preview(request,now)
        if abs(applied[2])>1e-8:continue
        acceleration=.65 if limiter is None else limiter.linear_acceleration
        horizon=.4+max(np.linalg.norm(applied[:2]),np.linalg.norm(measured[:2]))/(2*acceleration)
        endpoint=position+(rotation@applied[:2])*horizon
        if not body_translation_recovery_clear(planner,boxes,position,heading,endpoint,require_clear_end=False):continue
        return applied,target
    return None


def push_centerline_velocity(direction, forward_speed, lateral_error, lateral_velocity):
    """Damp lateral gait drift before it grows into a contact realignment."""
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    tangent=np.array([-direction[1],direction[0]])
    correction=float(np.clip(3.*lateral_error-.6*lateral_velocity,-.20,.20))
    return forward_speed*direction+correction*tangent


def push_completion_state(certificate, clearance, moved_along, online):
    """Stop force at safe side parking; sensor confirmation follows release."""
    if certificate:return 'observed_corridor_clear'
    if (online and moved_along>.25 and clearance.get('side_parked',False)
            and clearance.get('passage_clearance_margin_m',-1.)>=.025):
        return 'side_parked_pending_observation'
    return None


class ContactAlignmentGate:
    """Finish one correction with hysteresis before switching correction axes."""
    def __init__(self):
        self.axis=None
        self.motion_mode=None
        self.settle_until=-1
        self.settling=False

    def settling_required(self,mode,step,settle_ticks=100,measured_speed=None):
        """Brake once on a mode change, without interrupting every yaw tick."""
        if mode!=self.motion_mode:
            self.motion_mode=mode
            self.settle_until=step+settle_ticks
            self.settling=True
        if self.settling:
            if step<self.settle_until or (measured_speed is not None and measured_speed>=.08):return True
            self.settling=False
        return False

    def select(self,lateral_error,yaw_error,yaw_tolerance):
        lateral=abs(float(lateral_error));yaw=abs(float(yaw_error))
        if self.axis=='yaw' and yaw<=.7*yaw_tolerance:self.axis=None
        if self.axis=='lateral' and lateral<=.035:self.axis=None
        # Large orientation errors take precedence over a lateral correction.
        if yaw>.18:self.axis='yaw'
        if self.axis is None:
            if yaw>max(.15,yaw_tolerance+.04):self.axis='yaw'
            elif lateral>.055:self.axis='lateral'
            elif yaw>yaw_tolerance:self.axis='yaw'
        return self.axis


def push_braking_distance(forward_velocity, horizon=.55, margin=.015):
    """Calibrated coast allowance from FILTERED measured push-axis speed.

    Instantaneous gait velocity peaks stop the push too early. The caller
    uses VelocitySlewLimiter.measured (0.15 s filter), then checks the actual
    settled displacement before releasing the box or continuing a shortfall.
    """
    if not np.isfinite(forward_velocity):
        raise ValueError('Push braking requires finite measured velocity')
    return margin + horizon * max(0., float(forward_velocity))
