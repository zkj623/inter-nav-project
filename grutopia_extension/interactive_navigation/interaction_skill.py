"""Contracts for reusable closed-loop robot interaction skills."""
from dataclasses import dataclass,field
from typing import Optional,Any


@dataclass(frozen=True)
class SkillSpec:
    name:str
    version:str
    categories:tuple
    effect:str


@dataclass
class SkillCommand:
    state:str
    velocity:tuple=(0.,0.,0.)
    navigation_goal:Optional[tuple]=None
    status:str='running'
    reason:str=''
    events:list=field(default_factory=list)
    certificate:Any=None


@dataclass
class SkillContext:
    now:float
    position:Any
    heading:float
    body_velocity:Any
    yaw_rate:float
    tilt_degrees:float
    healthy:bool
    box:Any
    boxes:Any
    planner:Any
    layers:Any
    limiter:Any
    navigation_reached:bool=False
    body_angular_velocity:Any=None
    orientation:Any=None
    policy_command:Any=None
    # Deployment supplies a fresh scan based check; command is already bounded.
    # Only the selected, registered target may be contacted. False means stop.
    contact_motion_check:Any=None
    # Original source acquisition times/identities, never processing completion.
    box_observed_at:Any=None
    box_observation_id:Any=None
    map_observed_at:Any=None
    map_observation_id:Any=None


class SkillRegistry:
    def __init__(self):self.skills={}
    def register(self,skill_class):
        if skill_class.spec.name in self.skills:raise ValueError('Duplicate skill name')
        self.skills[skill_class.spec.name]=skill_class
    def create(self,permission,**kwargs):
        if permission is None:raise ValueError('An explicit interaction permission is required')
        skill=self.skills.get(permission.skill_name)
        if skill is None or permission.category not in skill.spec.categories:
            raise ValueError('No registered skill supports this recognized category')
        return skill(permission=permission,**kwargs)
