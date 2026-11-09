"""Keep the verified corridor entry as a subgoal after a sideways box push."""
import math
from dataclasses import fields
import numpy as np
from grutopia_extension.interactive_navigation.box_push_navigation import BoxState
from grutopia_extension.interactive_navigation.go2_voronoi_navigation import (
    navigation_footprint_samples, body_rotation_clear, _recovery_translation_samples)
from grutopia_extension.interactive_navigation.go2_interaction_motion import ObservedSubgoalArrival


def _straight_body_connector_clear(planner,boxes,position,heading,target,margin=.04):
    raw=planner.free.copy()
    for box in boxes:
        mask=planner.box_mask(box)
        if mask is None:return False
        raw &= ~mask
    xx,yy=navigation_footprint_samples(planner,margin=margin)
    c,s=math.cos(heading),math.sin(heading)
    dx=np.rint((c*xx-s*yy)/planner.resolution).astype(int)
    dy=np.rint((s*xx+c*yy)/planner.resolution).astype(int)
    position=np.asarray(position[:2],float);delta=np.asarray(target[:2],float)-position
    for fraction,_ in _recovery_translation_samples(position,delta,planner.limits,planner.resolution):
        row,col=planner.cell(position+fraction*delta);rows=row+dy;cols=col+dx
        if (rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]
                or not raw[rows,cols].all()):return False
    return True


class PassageEntryGuide:
    def __init__(self,certificate,goal):
        self.certificate=dict(certificate)
        self.entry=np.asarray(certificate['guide_entry'] if 'guide_entry' in certificate else certificate['corridor_start'],float)[:2].copy()
        self.exit=np.asarray(certificate['guide_exit'] if 'guide_exit' in certificate else certificate['corridor_end'],float)[:2].copy()
        self.goal=np.asarray(goal,float)[:2].copy()
        if not all(np.isfinite(x).all() for x in [self.entry,self.exit,self.goal]):
            raise ValueError('Passage guide coordinates must be finite')
        self.entry_reached=False
        self.arrival=ObservedSubgoalArrival(tolerance=.20)

    def shorten_checked_entry(self,position,heading,planner,boxes):
        """Choose a nearer approach pose inside the existing certified corridor.

        Called only after a fresh settled release fails direct admission.
        It keeps an entry subgoal and its measured-stop lifecycle. A shorter
        translation, a complete turn and the remaining corridor must all be
        free with the original clearance margin; no occupancy is changed.
        """
        proof=self.certificate;position=np.asarray(position[:2],float)
        axis=np.asarray(proof['axis'],float);axis/=np.linalg.norm(axis)
        center=np.asarray(proof['center'],float)
        direction=self.exit-self.entry;direction/=np.linalg.norm(direction)
        course=math.atan2(direction[1],direction[0])
        turn_end=heading+(course-heading+math.pi)%(2*math.pi)-math.pi
        original=self.entry.copy();choices=[]
        # Keep the whole forward envelope on the approach side of the
        # crossing plane: 0.50 m front + 0.04 m clearance + 0.02 m reserve.
        alignment=float(np.dot(direction,axis))
        if alignment<=.5:return {'shortened':False,'reason':'invalid_corridor_direction'}
        available=(-.56-float(np.dot(original-center,axis)))/alignment
        if not np.isfinite(available) or available<=.05:
            return {'shortened':False,'reason':'no_shorter_approach_side_pose'}
        for distance in np.linspace(available,.05,10):
            candidate=original+distance*direction
            if not _straight_body_connector_clear(planner,boxes,position,heading,candidate):continue
            if not body_rotation_clear(planner,boxes,candidate,heading,turn_end,margin=.04):continue
            if not _straight_body_connector_clear(planner,boxes,candidate,course,self.exit):continue
            if not planner.path(position,candidate,boxes) or not planner.path(candidate,self.exit,boxes):continue
            cost=float(np.linalg.norm(position-candidate)+np.linalg.norm(self.exit-candidate))
            choices.append((cost,candidate))
        if not choices:return {'shortened':False,'reason':'no_checked_nearer_entry'}
        _,self.entry=min(choices,key=lambda item:item[0])
        return {'shortened':True,'reason':'nearer_full_body_checked_entry',
                'original_entry':original.tolist(),'entry':self.entry.tolist(),
                'entry_shift_m':float(np.linalg.norm(self.entry-original)),'body_margin_m':.04}

    def try_join_from_pose(self,position,heading,planner,boxes,linear_velocity,angular_velocity,tilt):
        """Skip the distant entry only for a fresh, clear, actual crossing.

        This is a route admission check, never task completion. The live
        controller retains acceleration, braking and full-body checks;
        actual passage crossing is still measured independently.
        """
        proof=self.certificate
        detail={'joined':False,'reason':'certificate_or_stop_not_ready'}
        if (self.entry_reached or not proof.get('side_parked')
                or not proof.get('original_box_would_block_corridor')
                or proof.get('confirming_observation_time_s',-math.inf)<=proof.get('last_box_motion_time_s',math.inf)
                or np.linalg.norm(linear_velocity[:2])>=.06 or abs(linear_velocity[2])>=.05
                or np.max(np.abs(angular_velocity))>=.10 or not 0<=tilt<15.):
            return detail
        position=np.asarray(position[:2],float);delta=self.exit-position
        axis=np.asarray(proof['axis'],float);axis/=np.linalg.norm(axis)
        boundary=float(np.dot(proof['center'],axis))
        length=float(np.linalg.norm(delta))
        if length<.05 or np.dot(delta,axis)<=0:
            return {**detail,'reason':'no_forward_corridor_connection'}
        course=math.atan2(delta[1],delta[0])
        turn_end=heading+(course-heading+math.pi)%(2*math.pi)-math.pi
        if not body_rotation_clear(planner,boxes,position,heading,turn_end,margin=.04):
            return {**detail,'reason':'current_pose_cannot_turn_with_margin'}
        raw=planner.free.copy()
        for box in boxes:
            mask=planner.box_mask(box)
            if mask is None:return {**detail,'reason':'invalid_box_geometry'}
            raw &= ~mask
        # Dense body samples and every crossed base-cell interval: a free
        # endpoint alone cannot admit a diagonal through an unknown sliver.
        xx,yy=navigation_footprint_samples(planner,margin=.04)
        c,s=math.cos(course),math.sin(course)
        dx=np.rint((c*xx-s*yy)/planner.resolution).astype(int)
        dy=np.rint((s*xx+c*yy)/planner.resolution).astype(int)
        def pose_cells(xy):
            row,col=planner.cell(xy)
            return row+dy,col+dx
        for fraction,_ in _recovery_translation_samples(position,delta,planner.limits,planner.resolution):
            rows,cols=pose_cells(position+fraction*delta)
            if (rows.min()<0 or cols.min()<0 or rows.max()>=raw.shape[0] or cols.max()>=raw.shape[1]
                    or not raw[rows,cols].all()):
                return {**detail,'reason':'body_connector_blocked_or_unknown'}
        # The pending rear-boundary crossing must still traverse space that
        # the original box obstructed, instead of taking an unrelated bypass.
        forward=np.array([c,s]);side=np.array([-s,c])
        rear_offset=min(float(np.dot(x*forward+y*side,axis)) for x in [-.32,.50] for y in [-.25,.25])
        fraction=(boundary+.02-rear_offset-float(np.dot(position,axis)))/float(np.dot(delta,axis))
        if not 0<fraction<1 or 'original_box' not in proof:
            return {**detail,'reason':'no_pending_full_body_crossing'}
        names={f.name for f in fields(BoxState)}
        original=BoxState(**{k:v for k,v in proof['original_box'].items() if k in names})
        original_mask=planner.box_mask(original)
        rows,cols=pose_cells(position+fraction*delta)
        if original_mask is None or not original_mask[rows,cols].any():
            return {**detail,'reason':'connector_does_not_use_vacated_space'}
        if not planner.path(position,self.exit,boxes) or not planner.path(self.exit,self.goal,boxes):
            return {**detail,'reason':'route_not_connected'}
        self.entry_reached=True
        return {'joined':True,'reason':'checked_current_pose_corridor_join',
                'position':position.tolist(),'skipped_entry':self.entry.tolist(),
                'exit':self.exit.tolist(),'body_margin_m':.04,'connector_length_m':length}

    @property
    def target(self):
        return self.goal if self.entry_reached else self.entry

    def advance(self,position,tolerance=.20):
        if not self.entry_reached and np.linalg.norm(np.asarray(position[:2])-self.entry)<tolerance:
            self.entry_reached=True
            return True
        return False

    def reset_arrival(self):
        self.arrival.reset()

    def observe_entry(self,position,linear_velocity,angular_velocity,tilt_degrees,now,*,heading,command=None):
        """Advance the guide only after a predicted brake and measured stop."""
        detail=self.arrival.update(self.entry,position,linear_velocity,angular_velocity,
            tilt_degrees,now,heading=heading,command=command)
        detail['entry_advanced']=bool(detail['subgoal_reached'] and self.advance(position))
        return detail

    def path(self,planner,position,boxes):
        if not self.entry_reached:return planner.path(position,self.entry,boxes)
        if float(np.dot(np.asarray(position[:2])-self.exit,self.exit-self.entry))>=0:
            return planner.path(position,self.goal,boxes)
        approach=planner.path(position,self.exit,boxes)
        after=planner.path(self.exit,self.goal,boxes)
        return [*approach,*after] if approach and after else []
