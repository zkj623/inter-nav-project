"""Known solid/cavity cases for the offline USD mesh evaluator, without USD."""
import ast
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).parents[1]/'tools/build_go2_mesh_evaluation.py'
tree=ast.parse(SOURCE.read_text())
scope={'np':np,'shape':(40,40),'origin':np.array([0.,0.]),'res':.1}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(SOURCE),'exec'),scope)

def cube(lo,hi):
    p=np.array([[lo[0],lo[1],lo[2]],[hi[0],lo[1],lo[2]],[hi[0],hi[1],lo[2]],[lo[0],hi[1],lo[2]],
        [lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],[hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
    f=np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return p[f]

def test_cap_recovers_solid_interior_when_top_and_bottom_outside_robot_band():
    mask=np.zeros((40,40),np.uint8)
    assert scope['cap_fill'](cube([1,1,0],[3,3,2]),.31,mask)==0
    assert mask.sum()==400 and np.all(mask[10:30,10:30])

def test_cap_keeps_hollow_cabinet_void_instead_of_whole_instance_rectangle():
    mask=np.zeros((40,40),np.uint8)
    tri=np.concatenate([cube([.5,.5,0],[3.5,3.5,2]),cube([1.5,1.5,0],[2.5,2.5,2])[:,::-1]])
    assert scope['cap_fill'](tri,.31,mask)==0
    assert mask.sum()==800 and not mask[15:25,15:25].any()

def test_two_separate_solids_do_not_fill_gap_between_them():
    mask=np.zeros((40,40),np.uint8)
    tri=np.concatenate([cube([.5,.5,0],[1.5,1.5,2]),cube([2.5,.5,0],[3.5,1.5,2])])
    assert scope['cap_fill'](tri,.31,mask)==0
    assert mask.sum()==200 and not mask[:,15:25].any()

def test_overhead_geometry_does_not_fill_low_height_cap():
    mask=np.zeros((40,40),np.uint8)
    assert scope['cap_fill'](cube([1,1,.7],[3,3,2]),.31,mask)==0
    assert not mask.any()

def test_slab_clips_triangle_at_actual_intersection_not_whole_projection():
    p=scope['clip'](np.array([[0.,0.,0.],[2.,0.,2.],[0.,2.,2.]]),1.,False)
    assert np.allclose(p,[[0,0,0],[1,0,1],[0,1,1]])
