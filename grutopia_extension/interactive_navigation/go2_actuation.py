"""NumPy implementation of the Isaac Lab Go2 DC actuator contract."""
import numpy as np


def dc_motor_efforts(target,position,velocity,*,stiffness=25.,damping=.5,
                     saturation_effort=23.5,effort_limit=23.5,velocity_limit=30.):
    target=np.asarray(target,np.float32);position=np.asarray(position,np.float32);velocity=np.asarray(velocity,np.float32)
    if target.shape!=position.shape or target.shape!=velocity.shape:
        raise ValueError('Go2 actuator arrays must have identical shapes')
    if not all(np.isfinite(x).all() for x in [target,position,velocity]):
        raise ValueError('Go2 actuator state must be finite')
    raw=stiffness*(target-position)-damping*velocity
    upper=np.clip(saturation_effort*(1.-velocity/velocity_limit),0.,effort_limit)
    lower=np.clip(saturation_effort*(-1.-velocity/velocity_limit),-effort_limit,0.)
    return np.clip(raw,lower,upper).astype(np.float32)


def physics_steps_per_command(index,actuator_mode):
    """Keep legacy 125 Hz average navigation and 25 fps over exact 200 Hz physics.

    Each five command ticks advance eight physical steps. Rendering every
    eighth physical step therefore happens at the end of a command tick.
    The actuator is evaluated again before EVERY physical step.
    """
    if actuator_mode=='implicit_pd':return 1
    if actuator_mode=='explicit_dc':return (1,2,1,2,2)[index%5]
    raise ValueError('Unknown Go2 actuator mode')
