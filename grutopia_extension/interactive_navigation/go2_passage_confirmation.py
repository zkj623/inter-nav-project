"""Confirm an observed body corridor, not just one point planner's crossing."""
import numpy as np


def observed_passage_routes(layers,box,start,goal,robot_position,current_boxes):
    planner=layers.planner
    with layers.global_planning() as global_boxes:
        route=planner.path(robot_position,goal,global_boxes)
        full=planner.path(start,goal,global_boxes)
    proof=planner.certify_cleared_passage(box,full,start,current_boxes)
    if proof and route:return proof,route,full
    passage=planner.passages.get(box.id)
    if passage is None:return None,route,full
    axis=np.asarray(passage['axis'],float);axis/=np.linalg.norm(axis)
    normal=np.array([-axis[1],axis[0]]);nominal=np.asarray(passage['center'],float)
    goal_offset=float(np.dot(np.asarray(goal[:2])-nominal,axis))
    if goal_offset<=.36:return None,route,full
    for offset in [0.,-.05,.05,-.10,.10,-.15,.15,-.20,.20]:
        center=nominal+offset*normal
        entry=center-axis;exit_point=center+min(.7,goal_offset)*axis
        # First test the full local footprint at this crossing. This probe
        # alone is never a certificate: global connections are required below.
        probe=[entry,exit_point,np.asarray(goal[:2])]
        if not planner.certify_cleared_passage(box,probe,start,current_boxes):continue
        with layers.global_planning() as global_boxes:
            before=planner.path(start,entry,global_boxes)
            body_corridor=planner.path(entry,exit_point,global_boxes)
            after=planner.path(exit_point,goal,global_boxes)
            approach=planner.path(robot_position,entry,global_boxes)
        if not all([before,body_corridor,after,approach]):continue
        candidate=[*before,*body_corridor,*after]
        proof=planner.certify_cleared_passage(box,candidate,start,current_boxes)
        if proof:
            proof={**proof,'confirmation_source':'connected_observed_corridor',
                   'guide_entry':entry.tolist(),'guide_exit':exit_point.tolist(),
                   'nominal_offset_m':float(offset)}
            return proof,[*approach,*body_corridor,*after],candidate
    return None,route,full
