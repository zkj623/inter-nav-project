"""Confirm a measured body corridor and refine its direction from the new route."""
import numpy as np


def _route_passages(passage,route,start):
    """Bounded hypotheses; each still needs the complete collision certificate."""
    yield dict(passage)
    axis=np.asarray(passage['axis'],float);axis/=np.linalg.norm(axis)
    center=np.asarray(passage['center'],float)
    points=[np.asarray(start[:2]),*[np.asarray(point[:2]) for point in route]]
    for first,last in zip(points,points[1:]):
        delta=last-first;length=np.linalg.norm(delta)
        if length<.10:continue
        direction=delta/length
        a=float(np.dot(first-center,axis));b=float(np.dot(last-center,axis))
        if not a<=0<b or np.dot(direction,axis)<np.cos(np.pi/4):continue
        crossing=first+delta*(-a)/(b-a)
        if np.linalg.norm(crossing-center)>.50:continue
        if np.dot(direction,axis)>.999 and np.linalg.norm(crossing-center)<.01:continue
        yield {**passage,'axis':direction.tolist(),'center':crossing.tolist(),
               'boundary':float(np.dot(crossing,direction)),
               'axis_source':'observed_route_tangent','previous_axis':axis.tolist()}


def observed_passage_routes(layers,box,start,goal,robot_position,current_boxes):
    planner=layers.planner
    with layers.global_planning() as global_boxes:
        route=planner.path(robot_position,goal,global_boxes)
        full=planner.path(start,goal,global_boxes)
    original=planner.passages.get(box.id)
    if original is None:return None,route,full
    accepted=False
    try:
        for passage in _route_passages(original,full,start):
            planner.passages[box.id]=passage
            if passage.get('axis_source')=='observed_route_tangent':
                # A view target is not a corridor certificate or free-map edit.
                layers.passage_observation_hint={**passage,'object_id':box.id,'confirmed':False}
            proof=planner.certify_cleared_passage(box,full,start,current_boxes)
            if proof and route:
                accepted=True;planner.active_passages[box.id]=passage
                return {**proof,'axis_source':passage.get('axis_source','initial_observed_route')},route,full
            axis=np.asarray(passage['axis'],float);axis/=np.linalg.norm(axis)
            normal=np.array([-axis[1],axis[0]]);nominal=np.asarray(passage['center'],float)
            goal_offset=float(np.dot(np.asarray(goal[:2])-nominal,axis))
            if goal_offset<=.36:continue
            for offset in [0.,-.025,.025,-.05,.05,-.10,.10,-.15,.15,-.20,.20]:
                center=nominal+offset*normal
                entry=center-axis;exit_point=center+min(.7,goal_offset)*axis
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
                    accepted=True;planner.active_passages[box.id]=passage
                    proof={**proof,'confirmation_source':'connected_observed_corridor',
                           'axis_source':passage.get('axis_source','initial_observed_route'),
                           'guide_entry':entry.tolist(),'guide_exit':exit_point.tolist(),
                           'nominal_offset_m':float(offset)}
                    return proof,[*approach,*body_corridor,*after],candidate
    finally:
        if not accepted:planner.passages[box.id]=original
    return None,route,full
