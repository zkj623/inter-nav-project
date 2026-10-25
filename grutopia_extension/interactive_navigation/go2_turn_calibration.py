"""Independent physical yaw trials before the actual navigation episode."""
import json
import math
from pathlib import Path

import cv2
import numpy as np


def heading_and_tilt(observation):
    q=np.asarray(observation['orientation'],float);q=q/np.linalg.norm(q)
    heading=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
    tilt=math.degrees(math.acos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
    return heading,tilt


def calibrate_turn_requests(robot,tick,camera,planner,boxes,output,speeds=(.50,.75,1.0),ground_height=.15):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    blocked=planner.turning_blocked(boxes,math.pi,math.pi/2,margin=.08)
    cells=np.argwhere(~blocked)
    if not len(cells):raise RuntimeError('No collision-checked yaw calibration site')
    points=np.column_stack((planner.limits[0][0]+(cells[:,1]+.5)*planner.resolution,
                            planner.limits[1][0]+(cells[:,0]+.5)*planner.resolution))
    site=points[np.argmin(np.linalg.norm(points-np.array([4.3,-.5]),axis=1))]
    stop={'move_by_speed':[0.,0.,0.]};results=[]
    for index,speed in enumerate(speeds,1):
        case=f'C{index:02d}';folder=output/case;folder.mkdir()
        art=robot.isaac_robot;root,_=art.get_world_pose();base,_=robot.get_world_pose()
        art.set_world_pose(position=root+np.array((*site,ground_height+.40))-base,orientation=np.array([0.,0.,0.,1.]))
        art.set_joint_positions(art.get_joints_default_state().positions)
        art.set_joint_velocities(np.zeros(art.num_dof))
        robot.get_robot_base().set_linear_velocity(np.zeros(3));robot.get_robot_base().set_angular_velocity(np.zeros(3))
        robot._go2_locomotion_control.reset();camera._smoothed_position=None;camera._smoothed_yaw=None
        for _ in range(400):obs=tick(stop)
        start=np.asarray(obs['position'][:2]);writer=None;frame_count=0;tick_count=0;trace=[]
        maximum_tilt=0.;maximum_drift=0.;success=False;reason='yaw_timeout'
        frame_index=(folder/'video_frames.jsonl').open('w')
        def advance(action):
            nonlocal writer,frame_count,tick_count
            observation=tick(action)
            if observation.get('render',False):
                rgba=camera.get_data()['rgba']
                if rgba is None:raise RuntimeError('Missing rendered yaw calibration frame')
                frame=cv2.cvtColor(np.asarray(rgba,dtype=np.uint8),cv2.COLOR_RGBA2BGR)
                if writer is None:
                    writer=cv2.VideoWriter(str(folder/'third_person.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(frame.shape[1],frame.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Cannot open yaw calibration video')
                cv2.putText(frame,f'{case} | yaw request {speed:.2f} rad/s | calibration only',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
                writer.write(frame);frame_index.write(json.dumps({'frame_index':frame_count,'tick':tick_count})+'\n')
                frame_count+=1
            tick_count+=1
            return observation
        try:
            for step in range(5000):
                heading,tilt=heading_and_tilt(obs);p=np.asarray(obs['position'][:2])
                error=(math.pi/2-heading+math.pi)%(2*math.pi)-math.pi
                drift=float(np.linalg.norm(p-start));maximum_tilt=max(maximum_tilt,tilt);maximum_drift=max(maximum_drift,drift)
                if step%20==0:trace.append({'step':step,'heading_rad':heading,'error_rad':error,'position':list(map(float,obs['position'])),'tilt_degrees':tilt})
                if step%200==0:print(json.dumps({'event':'turn_calibration_progress','case':case,'speed':speed,'step':step,'error_deg':math.degrees(error),'drift_m':drift,'tilt_deg':tilt}),flush=True)
                if tilt>30 or obs['position'][2]<ground_height+.20:reason='unstable';break
                if drift>.50:reason='excessive_translation';break
                planner.robot_heading=heading
                if step%20==0 and planner.blocked(boxes)[planner.cell(p)]:reason='footprint_left_clear_site';break
                if abs(error)<.08:
                    for _ in range(120):obs=advance(stop)
                    final_heading,final_tilt=heading_and_tilt(obs)
                    final_error=(math.pi/2-final_heading+math.pi)%(2*math.pi)-math.pi
                    final_drift=float(np.linalg.norm(np.asarray(obs['position'][:2])-start))
                    maximum_tilt=max(maximum_tilt,final_tilt);maximum_drift=max(maximum_drift,final_drift)
                    planner.robot_heading=final_heading
                    final_clear=not planner.blocked(boxes)[planner.cell(obs['position'][:2])]
                    if abs(final_error)<.12 and final_tilt<30 and obs['position'][2]>ground_height+.20 and final_drift<=.50 and final_clear:
                        success=True;reason='yaw_aligned_and_settled';break
                    continue
                obs=advance({'move_by_speed':[0.,0.,math.copysign(speed,error)]})
            if not success:
                for _ in range(120):obs=advance(stop)
        finally:
            if writer is not None:writer.release()
            frame_index.close()
        final_heading,final_tilt=heading_and_tilt(obs)
        row={'id':case,'calibration_only':True,'ground_height':ground_height,'success':success,'reason':reason,'speed':speed,'control_steps':step+1,
             'frames':frame_count,'fps':25,'site':site.tolist(),'final_heading_rad':final_heading,
             'final_tilt_degrees':final_tilt,'maximum_tilt_degrees':maximum_tilt,'maximum_translation_m':maximum_drift}
        (folder/'result.json').write_text(json.dumps(row,indent=2));(folder/'trajectory.json').write_text(json.dumps(trace))
        results.append(row);(output/'summary.json').write_text(json.dumps({'calibration_only':True,'complete':False,'cases':results},indent=2))
        print(json.dumps({'event':'turn_calibration_case_end',**row}),flush=True)
    passed=[r for r in results if r['success']]
    selected=min(passed,key=lambda r:r['control_steps'])['speed'] if passed else None
    report={'calibration_only':True,'complete':True,'selected_turn_speed':selected,'cases':results}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    if selected is None:raise RuntimeError('No yaw request passed physical calibration; navigation has not started')
    return report
