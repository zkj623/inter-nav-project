import unittest
import numpy as np
from build import pathfind,length,mask_objects
class GeometryTests(unittest.TestCase):
 def test_no_corner_cutting(self):
  m=np.array([[True,False],[False,True]])
  self.assertIsNone(pathfind(m,(0,0),(1,1)))
 def test_obstacle_disconnects_then_restores(self):
  free=np.zeros((50,50),bool);free[20:30,1:49]=True
  objects=[{'cell':[25,25],'dimensions_m':[.8,.8,.5]}]
  self.assertIsNone(pathfind(mask_objects(free,objects),(25,8),(25,42)))
  route=pathfind(mask_objects(free,objects,skip=0),(25,8),(25,42))
  self.assertAlmostEqual(length(route),3.4)
 def test_unreachable(self):
  m=np.ones((20,20),bool);m[10]=False
  self.assertIsNone(pathfind(m,(1,1),(18,18)))
if __name__=='__main__':unittest.main()
