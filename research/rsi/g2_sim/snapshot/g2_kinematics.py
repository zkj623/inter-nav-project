from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def transform(xyz=(0,0,0),rpy=(0,0,0)):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz
    return t

class Kinematics:
    def __init__(self,path):
        root=ET.parse(path).getroot()
        self.joints=[];self.by_name={};self.links=[l.attrib['name'] for l in root.findall('link')]
        for j in root.findall('joint'):
            o=j.find('origin');a=j.find('axis');lim=j.find('limit');m=j.find('mimic')
            d=dict(name=j.attrib['name'],kind=j.attrib['type'],parent=j.find('parent').attrib['link'],child=j.find('child').attrib['link'],
                origin=transform(*[[float(v) for v in o.attrib.get(k,'0 0 0').split()] for k in ('xyz','rpy')]) if o is not None else np.eye(4),
                axis=np.array([float(v) for v in a.attrib['xyz'].split()]) if a is not None else np.array([1.,0,0]),
                lower=float(lim.attrib.get('lower',-np.pi)) if lim is not None else -np.pi,
                upper=float(lim.attrib.get('upper',np.pi)) if lim is not None else np.pi,
                effort=float(lim.attrib.get('effort',0)) if lim is not None else 0,
                velocity=float(lim.attrib.get('velocity',1)) if lim is not None else 1,
                mimic=m.attrib if m is not None else None)
            self.joints.append(d);self.by_name[d['name']]=d
        children={j['child'] for j in self.joints};self.root=next(l for l in self.links if l not in children)
    def fk(self,q,base=None):
        out={self.root:np.eye(4) if base is None else base};pending=self.joints.copy()
        while pending:
            count=len(pending)
            for j in pending.copy():
                if j['parent'] not in out:continue
                v=q.get(j['name'],0.)
                if j['mimic']:
                    m=j['mimic'];v=q.get(m['joint'],0)*float(m.get('multiplier',1))+float(m.get('offset',0))
                r=np.eye(4)
                if j['kind'] in ('revolute','continuous'):r[:3,:3]=Rotation.from_rotvec(j['axis']*v).as_matrix()
                elif j['kind']=='prismatic':r[:3,3]=j['axis']*v
                out[j['child']]=out[j['parent']]@j['origin']@r;pending.remove(j)
            assert len(pending)<count,'Disconnected URDF'
        return out
    def solve(self,side,target,seed,base,tcp,body=False):
        names=[j['name'] for j in self.joints if f'arm_{side}_joint' in j['name'] and j['kind']!='fixed']
        if body:names=[n for n in self.by_name if 'body_joint' in n]+names
        lower=np.array([self.by_name[n]['lower'] for n in names]);upper=np.array([self.by_name[n]['upper'] for n in names])
        x0=np.clip([seed.get(n,0) for n in names],lower+1e-6,upper-1e-6)
        def residual(x):
            q={**seed,**dict(zip(names,x))};p=self.fk(q,base)[f'arm_{side}_end_link']@tcp
            return np.r_[(p[:3,3]-target[:3,3])*4,Rotation.from_matrix(target[:3,:3].T@p[:3,:3]).as_rotvec(),.002*(x-x0)]
        fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=180,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q={**seed,**dict(zip(names,fit.x))};actual=self.fk(q,base)[f'arm_{side}_end_link']@tcp
        pe=float(np.linalg.norm(actual[:3,3]-target[:3,3]));re=float(np.linalg.norm(Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()))
        return q,pe,re

if __name__=='__main__':
    import json
    k=Kinematics(Path(__file__).with_name('g2_crs_omnipicker.urdf'))
    for n,t in k.fk({}).items():
        if 'arm_' in n or 'gripper_l' in n:print(n,np.round(t[:3,3],4).tolist(),np.round(t[:3,:3],3).tolist())
