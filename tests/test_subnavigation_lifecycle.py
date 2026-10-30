"""Exercise the real runner's subnavigation closure/dispatch without Isaac Sim."""
import ast
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace as NS

import numpy as np

from grutopia_extension.interactive_navigation.go2_interaction_motion import ObservedSubgoalArrival


def driver():
    source=Path(__file__).parents[1]/'grutopia/demo/go2_side_clearance_navigation.py'
    tree=ast.parse(source.read_text())
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='subnavigation']
    branches=[n for n in ast.walk(tree) if isinstance(n,ast.If)
        and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Attribute)
        and isinstance(n.test.left.value,ast.Name) and n.test.left.value.id=='command'
        and n.test.left.attr=='navigation_goal' and isinstance(n.test.ops[0],ast.IsNot)]
    assert len(functions)==len(branches)==1
    # Execute production code, including its timeout and no-goal dispatch.
    # Place the mock robot at the requested subgoal so actual arrival logic
    # returns before route planning in this bookkeeping regression.
    code='''def make():
    subtracker=None;subtarget=None;sub_started=None
    sub_arrival=ObservedSubgoalArrival()
    ctrl=NS(body_lin_vel=np.zeros(3),body_ang_vel=np.zeros(3))
    robot=NS(_go2_locomotion_control=ctrl)
    p=np.zeros(3);heading=0.;current=[];tilt=0.;smooth_limiter=None
    planner=NS(control_time=0.);ctx=NS();stop={'move_by_speed':[0.,0.,0.]}
    skill=NS(failures=[])
    skill._fail=lambda context,reason:skill.failures.append(reason)
    def observation_brake(*args):return stop
'''+indent(ast.unparse(functions[0]),'    ')+'''
    def state():
        return {'target':None if subtarget is None else subtarget.copy(),
            'started':sub_started,'tracker':subtracker,'arrival_braking':sub_arrival.braking}
    def seed_route_cache():
        nonlocal subtracker
        subtracker=object()
    def step(target,now,requested_action=None):
        nonlocal subtracker,subtarget,sub_started,p
        planner.control_time=now
        if target is not None:p=np.array([*target,.4])
        command=NS(navigation_goal=target)
        action=stop if requested_action is None else requested_action
        sub_reached=False;iteration_tracking={}
'''+indent(ast.unparse(branches[0]),'        ')+'''
        return {'action':action,'arrived':sub_reached,'tracking':iteration_tracking,**state()}
    return step,state,seed_route_cache
'''
    namespace={'np':np,'NS':NS,'ObservedSubgoalArrival':ObservedSubgoalArrival}
    exec(compile(code,str(source)+':subnavigation-test','exec'),namespace)
    return namespace['make']()


def test_revisiting_same_goal_gets_a_fresh_visit_budget():
    step,_,_=driver();goal=np.array([2.255911,-.283875])
    assert 'failed' not in step(goal,51.2)['tracking']
    step(None,53.4)  # alignment, heading trim and recovery occur here
    restarted=step(goal,136.97)
    assert 'failed' not in restarted['tracking']
    assert restarted['started']==136.97


def test_leaving_navigation_clears_route_and_arrival_but_preserves_skill_motion():
    step,state,seed_route=driver();step([2.,0.],1.);seed_route()
    assert state()['tracker'] is not None and state()['arrival_braking']
    sidestep={'move_by_speed':[0.,.35,0.]}
    answer=step(None,2.,sidestep)
    assert answer['action'] is sidestep
    assert answer['target'] is None and answer['started'] is None
    assert answer['tracker'] is None and not answer['arrival_braking']


def test_an_uninterrupted_visit_still_times_out_after_eighty_seconds():
    step,_,_=driver();goal=[2.,0.]
    step(goal,10.)
    assert 'failed' not in step(goal,89.9)['tracking']
    assert step(goal,90.001)['tracking']['failed']=='subnavigation_timeout'


def test_small_target_jitter_cannot_keep_an_active_visit_alive_forever():
    step,_,_=driver();step([2.,0.],10.)
    result=step([2.01,0.],90.001)
    assert result['started']==10.
    assert result['tracking']['failed']=='subnavigation_timeout'


def test_a_different_target_resets_its_budget_and_owns_a_copy():
    step,state,_=driver();step([2.,0.],10.)
    target=np.array([2.1,0.]);result=step(target,100.)
    assert result['started']==100. and 'failed' not in result['tracking']
    target[:]=100.
    np.testing.assert_array_equal(state()['target'],[2.1,0.])
