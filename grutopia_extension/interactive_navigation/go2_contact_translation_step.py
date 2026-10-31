"""Bound a corrective lateral input before checking the actually settled pose."""
import math
import numpy as np


class ContactTranslationStep:
    SPEED=.35
    ACK_SECONDS=.04
    MAX_SECONDS=3.
    MAX_TRANSLATION=.25
    PROGRESS_SECONDS=1.5
    PROGRESS_DISTANCE=.025
    TARGET_PROGRESS_CAP=.08

    @classmethod
    def configuration(cls):
        return {'world_speed_mps':cls.SPEED,'acknowledged_hold_seconds':cls.ACK_SECONDS,
                'input_timeout_seconds':cls.MAX_SECONDS,'maximum_translation_m':cls.MAX_TRANSLATION,
                'eligible_observation_profile':'unitree_go2_45','settled_contact_gate_unchanged':True,
                'initial_alignment_plateau_seconds':cls.PROGRESS_SECONDS,
                'initial_alignment_progress_m':cls.PROGRESS_DISTANCE,
                'motion_check':'full_then_existing_finite_target','measured_braking_horizon_s':.8,
                'skill_target_progress_cap_m':cls.TARGET_PROGRESS_CAP,'corrective_yaw_request':0.}

    def __init__(self,position,world_direction,now,*,minimum_progress_m=0.):
        self.origin=np.asarray(position[:2],float).copy()
        self.minimum_progress_m=float(minimum_progress_m)
        if not np.isfinite(self.minimum_progress_m) or not 0.<=self.minimum_progress_m<=self.MAX_TRANSLATION:
            raise ValueError('Required step progress must be finite and within the displacement bound')
        self.input_acknowledged=False;self.maximum_acknowledged_seconds=0.
        direction=np.asarray(world_direction,float).reshape(2)
        if not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-8:
            raise ValueError('Contact translation step needs a finite direction')
        self.direction=direction/np.linalg.norm(direction)
        self.started=float(now);self.acknowledged_since=None
        self.complete=False;self.failed=None
        self.finite_target_check_used=False;self.last_motion_check=None

    def update(self,position,heading,policy_command,now):
        received=np.asarray(policy_command,float) if policy_command is not None else np.empty(0)
        valid=received.shape==(3,) and np.isfinite(received).all()
        distance=float(np.linalg.norm(np.asarray(position[:2],float)-self.origin))
        if not self.complete and self.failed is None:
            if not valid:self.failed='lateral_step_feedback_unavailable'
            elif float(now)-self.started>self.MAX_SECONDS:self.failed=('lateral_step_progress_timeout' if self.input_acknowledged and self.minimum_progress_m>0. else 'lateral_step_input_timeout')
            elif distance>self.MAX_TRANSLATION:self.failed='lateral_step_translation_limit'
        c,s=math.cos(heading),math.sin(heading)
        actual=0. if not valid else float(np.dot(np.array([[c,-s],[s,c]])@received[:2],self.direction))
        progress=float(np.dot(np.asarray(position[:2],float)-self.origin,self.direction))
        if valid and not self.complete and not self.failed:
            if actual>=self.SPEED-1e-4:
                if self.acknowledged_since is None:self.acknowledged_since=float(now)
                held=float(now)-self.acknowledged_since
                self.maximum_acknowledged_seconds=max(self.maximum_acknowledged_seconds,held)
                if held>=self.ACK_SECONDS:self.input_acknowledged=True
            else:self.acknowledged_since=None
            if self.input_acknowledged and progress>=self.minimum_progress_m:self.complete=True
        detail={'world_direction':self.direction.tolist(),'requested_speed_mps':self.SPEED,
            'actual_speed_component_mps':actual,'elapsed_seconds':float(now)-self.started,
            'translation_m':distance,'input_complete':self.complete,'failure':self.failed,
            'input_acknowledged':self.input_acknowledged,'step_complete':self.complete,
            'minimum_progress_m':self.minimum_progress_m,'directional_progress_m':progress,
            'finite_target_check_used':self.finite_target_check_used,'last_motion_check':self.last_motion_check,
            'acknowledged_seconds':self.maximum_acknowledged_seconds}
        command=np.zeros(2) if self.complete or self.failed else self.direction*self.SPEED
        return command,detail
