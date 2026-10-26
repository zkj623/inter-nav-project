"""Place the actual Go2 contact point on the carton push centerline."""
import numpy as np
from grutopia_extension.interactive_navigation.mapping import AStarMapPlanner


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


def contact_centering_velocity(delta, direction, forward_speed, lateral_speed=.35, lateral_tolerance=.035):
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
    if abs(lateral)>lateral_tolerance:
        return lateral_speed*np.sign(lateral)*tangent
    longitudinal=float(np.dot(delta,direction))
    speed=forward_speed if longitudinal>=0 else max(forward_speed,.35)
    return speed*np.sign(longitudinal)*direction


def contact_braking_required(delta, velocity, horizon=1.0, stop_margin=.025):
    """Allow for gait travel after a zero-speed request near the contact pose."""
    delta=np.asarray(delta,dtype=float)
    distance=float(np.linalg.norm(delta))
    toward=float(np.dot(delta,velocity)/max(distance,1e-8))
    return toward>0 and distance<stop_margin+horizon*toward


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


def push_centerline_velocity(direction, forward_speed, lateral_error, lateral_velocity):
    """Damp lateral gait drift before it grows into a contact realignment."""
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    tangent=np.array([-direction[1],direction[0]])
    correction=float(np.clip(3.*lateral_error-.6*lateral_velocity,-.20,.20))
    return forward_speed*direction+correction*tangent


class ContactAlignmentGate:
    """Finish one correction with hysteresis before switching correction axes."""
    def __init__(self):
        self.axis=None

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
