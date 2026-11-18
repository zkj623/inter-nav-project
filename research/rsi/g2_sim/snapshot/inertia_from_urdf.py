"""Re-author inertial frames from source URDF, including fixed-link merging."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from g2_kinematics import Kinematics,transform
def apply_inertials(stage,urdf):
    from pxr import UsdPhysics,Gf
    k=Kinematics(urdf);fk=k.fk({});parent={j['child']:j for j in k.joints}
    rigid={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)};groups={n:[] for n in rigid}
    for link in ET.parse(urdf).getroot().findall('link'):
        inert=link.find('inertial')
        if inert is None:continue
        name=link.get('name');body=name
        while body not in groups:
            joint=parent[body];assert joint['kind']=='fixed';body=joint['parent']
        o=inert.find('origin');xyz=[float(v) for v in o.get('xyz','0 0 0').split()] if o is not None else [0]*3;rpy=[float(v) for v in o.get('rpy','0 0 0').split()] if o is not None else [0]*3
        t=np.linalg.inv(fk[body])@fk[name]@transform(xyz,rpy);a=inert.find('inertia');v={n:float(a.get(n,'0')) for n in ('ixx','iyy','izz','ixy','ixz','iyz')}
        I=np.array([[v['ixx'],v['ixy'],v['ixz']],[v['ixy'],v['iyy'],v['iyz']],[v['ixz'],v['iyz'],v['izz']]])
        groups[body].append((float(inert.find('mass').get('value')),t[:3,3],t[:3,:3]@I@t[:3,:3].T))
    rows=[]
    for name,items in groups.items():
        assert items,name
        mass=sum(m for m,_,_ in items);com=sum(m*c for m,c,_ in items)/mass
        I=sum(i+m*(np.dot(c-com,c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,i in items)
        eig,R=np.linalg.eigh(I);assert min(eig)>0,(name,eig)
        if np.linalg.det(R)<0:R[:,0]*=-1
        q=Rotation.from_matrix(R).as_quat();api=UsdPhysics.MassAPI.Apply(rigid[name])
        api.CreateMassAttr(float(mass));api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,com)));api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,eig)));api.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
        rows.append(dict(link=name,mass=mass,com=com.tolist(),diagonal=eig.tolist(),principal_axes_xyzw=q.tolist()))
    return rows
