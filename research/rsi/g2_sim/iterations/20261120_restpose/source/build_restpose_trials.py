from pathlib import Path
import json,copy,ast,re
R=Path(__file__).resolve().parent;search=json.loads((R/'restpose_search.json').read_text());chosen=search['candidates'][0];cfg=json.loads((R/'ground_lift_training_config.json').read_text());out=R/'restpose_v2_plans';out.mkdir(exist_ok=True)
for f in (R/'symhome_plans').glob('*.json'):
 d=copy.deepcopy(json.loads(f.read_text()))
 for p in d['plans']:p.update(q_default=chosen['q'],q_prepare=chosen['waypoints'],standby_label='custom simulation arms-down standby; NOT official factory standby')
 (out/f.name).write_text(json.dumps(d,indent=2))
for a,b in cfg['validated_simulation_baselines'].items():
 s=(R/b['harness']).read_text();s=re.sub(r"OUT=ROOT/'[^']+'","OUT=ROOT/'restpose_v2_trials'",s)
 s=s.replace("official flat G2 op scene defaults -> symmetric tucked common home", "CUSTOM arms-down symmetric simulation standby -> unchanged symmetric tucked common home")
 needle="    segment(qpre,qpre,2.,'default_settle')";assert s.count(needle)==1
 capture="""
    result['standby_label']='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL'
    actual_standby=dict(zip(names,robot.get_joint_positions()))
    result['standby_target_q']=qpre.copy()
    result['standby_actual_q']={n:float(v) for n,v in actual_standby.items()}
    result['standby_max_tracking_error_rad']=max(abs(actual_standby.get(n,0.)-qpre.get(n,0.)) for n in qpre)
    standby_fk=k.fk(actual_standby,mat(*robot.get_world_pose()))
    result['standby_link_positions_m']={n:standby_fk[n][:3,3].tolist() for side in ('l','r') for n in [f'arm_{side}_link2',f'arm_{side}_link4',f'arm_{side}_end_link']}
    saved_standby=camera.get_world_pose(camera_axes='usd')
    for label,eye in [('standby_front',[3.8,0.,1.05]),('standby_oblique',[3.4,3.4,2.3])]:
        set_camera_view(eye=np.array(eye),target=np.array([0.,0.,.95]),camera_prim_path='/World/GraspReview')
        for _ in range(12):world.render()
        rgba=camera.get_rgba()
        if rgba is not None and rgba.size:cv2.imwrite(str(OUT/(label+'.png')),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
    camera.set_world_pose(position=saved_standby[0],orientation=saved_standby[1],camera_axes='usd')
    for _ in range(12):world.render()
"""
 s=s.replace(needle,needle+capture)
 s=s.replace('            samples.append(rec)',"            samples.append(rec)\n            if step%120==0:(OUT/'live.json').write_text(json.dumps(dict(t=rec['t'],phase=phase,max_tracking_error_rad=rec['max_tracking_error_rad'],lift_m=rec['lift_m'])))")
 ast.parse(s);(R/f'restpose_v2_{a}_trial.py').write_text(s)
(R/'restpose_candidate_config.json').write_text(json.dumps(dict(label='CUSTOM_SIMULATION_STANDBY_NOT_OFFICIAL',standby_joint_positions_rad=chosen['q'],work_joint_positions_rad=search['work_home'],preparation_waypoints=chosen['waypoints'],preparation_seconds_per_segment=20,work_pose_changed=False,validation_scope='standby holding and preparation under existing pinned-root ideal-torso simulation; old per-asset lift validation does not automatically transfer to the new start',previous_work_config='iterations/20261120_symhome/fixtures/symhome_ground_lift_config.json'),indent=2));print('RESTPOSE_BUILT',len(cfg['validated_assets']))
