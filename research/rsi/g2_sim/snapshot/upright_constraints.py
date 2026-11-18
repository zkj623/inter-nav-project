"""Conservative test envelope, not a claim about G2 hardware limits."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from g2_kinematics import Kinematics,transform
ROOT=Path(__file__).resolve().parent
MAX_TORSO_TILT_DEG=20.
MAX_BODY_JOINT_DEG=35.
MAX_STATIC_EFFORT_FRACTION=.8
class Envelope:
    def __init__(self):
        self.k=Kinematics(ROOT/'generated/g2_crs_omnipicker.urdf')
        self.inertia={};self.desc={}
        for link in ET.parse(ROOT/'generated/g2_crs_omnipicker.urdf').getroot().findall('link'):
            i=link.find('inertial')
            if i is None:continue
            o=i.find('origin');xyz=[float(v) for v in o.get('xyz','0 0 0').split()] if o is not None else [0,0,0]
            self.inertia[link.get('name')]=(float(i.find('mass').get('value')),np.r_[xyz,1.])
        for j in self.k.joints:
            descendants={j['child']}
            while True:
                more=descendants|{a['child'] for a in self.k.joints if a['parent'] in descendants}
                if more==descendants:break
                descendants=more
            self.desc[j['name']]=descendants
        self.upright=self.k.fk({})['body_link5'][:3,:3]
    def check(self,q,payload=None):
        fk=self.k.fk(q);axis=(fk['body_link5'][:3,:3]@self.upright.T)[:,2]
        tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
        bodymax=float(max(abs(np.degrees(q.get(n,0))) for n in self.k.by_name if 'body_joint' in n))
        torque=[]
        for j in self.k.joints:
            if j['kind']!='revolute' or j['mimic'] or not ('body_joint' in j['name'] or 'arm_' in j['name']):continue
            t=fk[j['parent']]@j['origin'];origin=t[:3,3];axis=t[:3,:3]@j['axis'];desc=self.desc[j['name']]
            moment=sum(float(np.dot(np.cross((fk[n]@self.inertia[n][1])[:3]-origin,[0,0,-9.81*self.inertia[n][0]]),axis)) for n in desc if n in self.inertia)
            for link,mass,point in payload or []:
                if link in desc:moment+=float(np.dot(np.cross((fk[link]@np.r_[point,1])[:3]-origin,[0,0,-9.81*mass]),axis))
            torque.append(dict(joint=j['name'],required_nm=abs(moment),limit_nm=j['effort'],ratio=abs(moment)/max(.001,j['effort'])))
        torque.sort(key=lambda x:x['ratio'],reverse=True)
        return dict(ok=bool(tilt<=MAX_TORSO_TILT_DEG+1e-6 and bodymax<=MAX_BODY_JOINT_DEG+1e-6 and torque[0]['ratio']<=MAX_STATIC_EFFORT_FRACTION),torso_tilt_deg=tilt,max_body_joint_deg=bodymax,maximum_static_effort_fraction=torque[0]['ratio'],worst_joint=torque[0])
def body_profiles():
    # Fixed throughout approach, closure and lifting. Arms do the lifting.
    return [dict(zip(('idx01_body_joint1','idx02_body_joint2','idx03_body_joint3'),np.radians(v))) for v in ((0,0,0),(-5,10,-5),(-10,20,-10),(-15,30,-15),(-10,20,0),(0,0,10),(0,0,20),(0,0,-10),(-5,5,-10),(-10,10,-10),(-10,20,-20),(0,0,-15))]
def grasp_body_profiles():
    # Promote only profiles with physical evidence: three empty-hold passes and
    # the -10 degree torso profile that passed the supported basket lift.
    profiles=body_profiles()
    return [profiles[i] for i in (0,1,7,5)]
if __name__=='__main__':
    import json
    e=Envelope()
    print(json.dumps([dict(q=q,check=e.check(q),shoulder=e.k.fk(q)['arm_l_link1'][:3,3].tolist()) for q in body_profiles()],indent=2))
