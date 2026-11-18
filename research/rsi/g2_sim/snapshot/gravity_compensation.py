"""Ground-supported quasi-static feed-forward, signed dU/dq from the URDF."""
import numpy as np
from upright_constraints import Envelope
class GravityCompensation:
    def __init__(self):self.e=Envelope();self.k=self.e.k
    def torques(self,q,base=None):
        fk=self.k.fk(q,base);positions={n:(fk[n]@point)[:3] for n,(_,point) in self.e.inertia.items()};out={}
        for j in self.k.joints:
            if j['kind']!='revolute' or j['mimic'] or not ('body_joint' in j['name'] or 'arm_' in j['name']):continue
            t=fk[j['parent']]@j['origin'];axis=t[:3,:3]@j['axis'];origin=t[:3,3]
            external=sum(float(np.dot(np.cross(positions[n]-origin,[0,0,-9.81*self.e.inertia[n][0]]),axis)) for n in self.e.desc[j['name']] if n in positions)
            out[j['name']]=-external
        return out
    def potential(self,q):
        fk=self.k.fk(q)
        return sum(m*9.81*(fk[n]@point)[2] for n,(m,point) in self.e.inertia.items())
if __name__=='__main__':
    g=GravityCompensation();q={'idx01_body_joint1':-.1,'idx02_body_joint2':.2,'idx03_body_joint3':-.1,'idx22_arm_l_joint2':-.6}
    expected=g.torques(q);errors={}
    for n,value in expected.items():
        eps=1e-6;plus={**q,n:q.get(n,0)+eps};minus={**q,n:q.get(n,0)-eps}
        numerical=(g.potential(plus)-g.potential(minus))/(2*eps);errors[n]=abs(value-numerical)
    assert max(errors.values())<1e-5,errors
    print('gravity finite-difference check passed; max error Nm:',max(errors.values()))
