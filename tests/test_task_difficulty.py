import math
import numpy as np
from shapely.geometry import Polygon,box
from tools.audit_go2_task_difficulty import polygon_shortest_path,free_polygons,component_polygon,scene_clutter,grid_graph
from scipy.sparse.csgraph import dijkstra


def test_empty_region_has_unit_tortuosity():
    result=polygon_shortest_path(box(0,0,10,10),[1,1],[7,9],20.)
    assert abs(result['length_m']-10.)<1e-9


def test_visibility_shortest_path_goes_around_wall_without_clearance_weights():
    domain=box(0,0,10,10).difference(box(4,0,6,8))
    result=polygon_shortest_path(domain,[2,2],[8,2],30.)
    expected=2*math.hypot(2,6)+2
    assert abs(result['length_m']-expected)<1e-8
    assert result['length_m']/6>1.3


def test_disconnected_regions_are_not_infinite_qualifying_routes():
    free=np.ones((10,10),bool);free[:,5]=False
    polys=free_polygons(free,[0,0],1.)
    assert component_polygon(polys,[1.5,1.5],[8.5,1.5]) is None
    assert polygon_shortest_path(None,[1.5,1.5],[8.5,1.5]) is None


def test_clutter_excludes_exterior_void_and_does_not_inflate_obstacles():
    support=np.zeros((10,10),bool);support[:6]=True
    free=support.copy();free[0]=False
    result=scene_clutter(support,free)
    assert result['support_cells']==60 and result['blocked_cells']==10
    assert result['fraction']==1/6


def test_robot_grid_forbids_diagonal_corner_cutting():
    free=np.array([[True,False],[False,True]])
    ids,graph=grid_graph(free,1.)
    distance=dijkstra(graph,indices=int(ids[0,0]))
    assert math.isinf(distance[int(ids[1,1])])
