from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'g2-act-collect-20261122';O=R/'head_visibility_head27';O.mkdir(exist_ok=False)
sys.argv.extend(['--/plugins/carb.tasking.plugin/threadCount=12','--/plugins/omni.tbb.globalcontrol/maxThreadCount=12'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':5,'multi_gpu':False})
try:
    import cv2,omni.usd,omni.timeline,omni.replicator.core as rep
    from scipy.spatial.transform import Rotation
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    from omni.isaac.core import World
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.semantics import add_update_semantics
    from g2_kinematics import Kinematics,transform
    from home_floor_material import add_wood_floor
    cfg=json.loads((R/'ground_lift_training_config.json').read_text());prep=json.loads((R/'baseline_configs/standby_preparation_config.json').read_text());assets=list(prep['execution']);k=Kinematics(R/'generated/g2_crs_omnipicker.urdf')
    def visual_only(source,destination):
        if destination.exists():return
        import re
        src=Usd.Stage.Open(str(source))
        text=src.Flatten().ExportToString()
        text=text.replace('"Physics','"DisabledPhysics').replace('"Physx','"DisabledPhysx')
        text=re.sub(r'\bdef Physics\w+\s', 'def Xform ',text)
        layer=Sdf.Layer.CreateAnonymous('.usda');layer.ImportFromString(text);layer.Export(str(destination))
    C=R/'visibility_visual_assets';C.mkdir(exist_ok=True)
    seed=OLD/'collection_trials/collection_plans/placement_0023/case_handheld/candidate0_repeat1/prepared_scene.usdc'
    visual_only(seed,C/'robot_scene.usdc')
    for aid in assets:
        folder='strategy15_surface_assets' if aid in ['stool_rectangular','stool_wood_four_leg','table_side','basket_rectangular'] else 'object_assets'
        visual_only(R/folder/(aid+'.usdc'),C/(aid+'.usdc'))
    omni.usd.get_context().open_stage(str(C/'robot_scene.usdc'));stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer());omni.timeline.get_timeline_interface().stop()
    links={p.GetName():p for p in stage.Traverse() if str(p.GetPath()).startswith('/robot/') and p.GetName() in k.links}
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/TestObject')).MakeInvisible();add_wood_floor(stage,R)
    targets={}
    for aid in assets:
        prim=stage.DefinePrim('/World/VisibilityTargets/'+aid,'Xform');prim.GetReferences().AddReference(str(C/(aid+'.usdc')));add_update_semantics(prim,'target');UsdGeom.Imageable(prim).MakeInvisible();targets[aid]=prim
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30);camera=Camera('/World/HeadVisibility',resolution=(640,400));camera.initialize();camera._frequency=-1
    camera.set_focal_length(20.);camera.set_horizontal_aperture(20.*640/305.6425170898);camera.set_vertical_aperture(20.*400/305.6007080078);camera.set_clipping_range(.01,100.)
    usd=UsdGeom.Camera(stage.GetPrimAtPath('/World/HeadVisibility'));usd.GetHorizontalApertureOffsetAttr().Set((320-318.7979431152)*20/305.6425170898);usd.GetVerticalApertureOffsetAttr().Set((202.4943847656-200)*20/305.6007080078)
    segmentation=rep.AnnotatorRegistry.get_annotator('semantic_segmentation');segmentation.attach(camera.get_render_product_path())
    mount=np.eye(4);mount[:3,:3]=Rotation.from_quat(np.array([-.5945104121873723,-.3868316461705513,.37872914728340207,.5945442627086478])[[1,2,3,0]]).as_matrix();mount[:3,3]=[.10237,.02375,.10256]
    report=[]
    for aid in assets:
        planfile=R/'realpose_plans'/(aid+'.json');exists=planfile.exists()
        if not exists:planfile=R/prep['execution'][aid]['plan_dir']/(aid+'.json')
        plan=json.loads(planfile.read_text())['plans'][0]
        for oldtarget in targets.values():UsdGeom.Imageable(oldtarget).MakeInvisible()
        target=targets[aid];UsdGeom.Imageable(target).MakeVisible()
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',plan.get('object_yaw_deg',0),degrees=True).as_matrix();t[:3,3]=plan['object_position'];UsdGeom.Xformable(target).MakeMatrixXform().Set(Gf.Matrix4d(t.T.tolist()))
        vertices=np.load(R/'object_assets'/(aid+'.npz'))['vertices'];worldvertices=vertices@t[:3,:3].T+t[:3,3]
        for head_deg in ([25,27,29] if aid=='case_handheld' else [27]):
            q={**cfg['common_initial_joint_positions_rad'],'idx13_head_joint3':float(np.radians(head_deg))};fk=k.fk(q,transform((0,0,.04)))
            for name,prim in links.items():UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(fk[name].T.tolist()))
            pose=fk['head_link3']@mount;quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]];camera.set_world_pose(position=pose[:3,3],orientation=quat,camera_axes='usd')
            for _ in range(4):world.render()
            rep.orchestrator.step(rt_subframes=4,pause_timeline=True);rgba=camera.get_rgba();assert rgba.shape==(400,640,4)
            rgb=rgba[:,:,:3].astype(np.uint8);filename=f'{aid}_head{head_deg}.jpg';cv2.imwrite(str(O/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,82])
            seg=segmentation.get_data();ids=[int(i) for i,v in seg.get('info',{}).get('idToLabels',{}).items() if v.get('class')=='target'];a=seg.get('data');mask=np.isin(a,ids) if a is not None else np.zeros((400,640),bool)
            if mask.ndim>2:mask=mask.any(-1)
            ys,xs=np.nonzero(mask);box=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None
            v=(worldvertices-pose[:3,3])@pose[:3,:3];z=-v[:,2];px=318.7979431152+305.6425170898*v[:,0]/z;py=202.4943847656-305.6007080078*v[:,1]/z;bbox=[float(px.min()),float(py.min()),float(px.max()),float(py.max())];inside=bool((z>.01).all() and bbox[0]>=0 and bbox[1]>=0 and bbox[2]<640 and bbox[3]<400)
            row=dict(asset=aid,head_deg=head_deg,image=filename,profile_id=cfg['profile_id'],object_position_m=plan['object_position'],uses_new_retargeted_plan=exists,geometry_projection_bbox_px=bbox,fully_in_fov_by_model_vertices=inside,target_visible_pixels=int(mask.sum()),semantic_visible_bbox_px=box,semantic_ids=ids,head_world_pose=pose.tolist(),scope='static initial-state rendering and semantic pixels; grasp/hold occlusion not inferred')
            report.append(row);(O/'visibility_report.json').write_text(json.dumps(report,indent=2));print('HEAD_VISIBILITY',aid,head_deg,inside,int(mask.sum()),bbox,flush=True)
    print('VISIBILITY_RENDER_COMPLETE',len(report),flush=True)
except BaseException:
    import traceback;traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
finally:app.close()
