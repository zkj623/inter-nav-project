"""Vectorized URDF gravity feed-forward; algebraically identical to archived implementation."""
from pathlib import Path
import hashlib
import numpy as np
from upright_constraints import Envelope
class GravityCompensation:
 def __init__(self):
  self.e=Envelope();self.k=self.e.k;self.implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();self.links=list(self.e.inertia);self.masses=np.array([self.e.inertia[n][0] for n in self.links]);self.points=np.array([self.e.inertia[n][1] for n in self.links]);self.js=[j for j in self.k.joints if j['kind']=='revolute' and not j['mimic'] and ('body_joint' in j['name'] or 'arm_' in j['name'])];self.mask=np.array([[n in self.e.desc[j['name']] for n in self.links] for j in self.js],dtype=np.float64);self.groupmass=self.mask@self.masses;self.gravity=np.array([0.,0.,-9.81])
 def torques(self,q,base=None):
  fk=self.k.fk(q,base);poses=np.array([fk[n] for n in self.links]);positions=np.einsum('nij,nj->ni',poses,self.points)[:,:3];origins=[];axes=[]
  for j in self.js:
   t=fk[j['parent']]@j['origin'];origins.append(t[:3,3]);axes.append(t[:3,:3]@j['axis'])
  moment=self.mask@(positions*self.masses[:,None])-self.groupmass[:,None]*np.array(origins);external=np.cross(moment,self.gravity);values=-np.einsum('ij,ij->i',external,np.array(axes));return dict(zip([j['name'] for j in self.js],map(float,values)))
 def potential(self,q):
  fk=self.k.fk(q);poses=np.array([fk[n] for n in self.links]);p=np.einsum('nij,nj->ni',poses,self.points);return float(np.dot(self.masses*9.81,p[:,2]))
