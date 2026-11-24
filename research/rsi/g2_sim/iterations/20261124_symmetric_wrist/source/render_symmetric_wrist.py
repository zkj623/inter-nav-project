from pathlib import Path
import json,sys,numpy as np
R=Path(__file__).resolve().parent;OLD=R.parent/'g2-act-collect-20261122';O=R/'wrist_review_official_padded';O.mkdir(exist_ok=True)
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
    UsdPhysics.Scene.Define(stage,'/physicsScene')
    world=World(stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=1/30)
    camera=Camera('/World/WristPreview',resolution=(1280,1056));camera.initialize();camera._frequency=-1
    camera.set_focal_length(20.);camera.set_horizontal_aperture(32.);camera.set_vertical_aperture(24.);camera.set_clipping_range(.005,100.)
    segmentation=rep.AnnotatorRegistry.get_annotator('semantic_segmentation');segmentation.attach(camera.get_render_product_path())
    official=json.loads((R/'official_wrist_config.json').read_text());(O/'camera_config.json').write_text(json.dumps(official,indent=2))
    report=[]
    for aid in assets:
        plan=json.loads((R/'realpose_plans'/(aid+'.json')).read_text())['plans'][0]
        for oldtarget in targets.values():UsdGeom.Imageable(oldtarget).MakeInvisible()
        target=targets[aid];UsdGeom.Imageable(target).MakeVisible()
        for phase,q,dz in [('initial',plan['q_start'],0),('pregrasp',plan['q_pre'],0),('grasp',plan['q_close'],0),('lift',plan['q_lift'][-1],.025*len(plan['q_lift']))]:
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',plan.get('object_yaw_deg',0),degrees=True).as_matrix();t[:3,3]=np.array(plan['object_position'])+[0,0,dz];UsdGeom.Xformable(target).MakeMatrixXform().Set(Gf.Matrix4d(t.T.tolist()))
            fk=k.fk(q,np.array(plan['base_pose']))
            for name,prim in links.items():UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(fk[name].T.tolist()))
            for side in ['l','r']:
                c=official['cameras'][side];mount=np.eye(4);mount[:3,3]=c['xyz'];mount[:3,:3]=Rotation.from_quat(np.array(c['wxyz'])[[1,2,3,0]]).as_matrix();pose=fk[c['parent']]@mount
                fx,fy,cx,cy=[c[n][0]/2 for n in ['fx','fy','cx','cy']];camera.set_focal_length(20.);camera.set_horizontal_aperture(20*1280/fx);camera.set_vertical_aperture(20*1056/fy)
                usd=UsdGeom.Camera(stage.GetPrimAtPath('/World/WristPreview'));usd.GetHorizontalApertureOffsetAttr().Set((640-cx-320)*20/fx);usd.GetVerticalApertureOffsetAttr().Set((cy+264-528)*20/fy)
                camera.set_world_pose(position=pose[:3,3],orientation=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]],camera_axes='usd')
                K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]);dist=np.array([c[n][0] for n in ['k1','k2','p1','p2','k3']]);yy,xx=np.mgrid[:528,:640];points=np.stack([xx,yy],-1).astype(np.float32);und=cv2.undistortPoints(points.reshape(-1,1,2),K,dist,P=K).reshape(528,640,2);mx=(und[:,:,0]+320).astype(np.float32);my=(und[:,:,1]+264).astype(np.float32);assert mx.min()>=0 and mx.max()<1279 and my.min()>=0 and my.max()<1055,'LENS_PADDING_INSUFFICIENT'
                for _ in range(3):world.render()
                rep.orchestrator.step(rt_subframes=3,pause_timeline=True);rgb=camera.get_rgba()[:,:,:3].astype(np.uint8);rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR);file=f'{aid}_{phase}_{side}.jpg';cv2.imwrite(str(O/file),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                seg=segmentation.get_data();ids=[int(i) for i,v in seg.get('info',{}).get('idToLabels',{}).items() if v.get('class')=='target'];mask=np.isin(seg['data'],ids).astype(np.uint8);mask=cv2.remap(mask,mx,my,cv2.INTER_NEAREST);pixels=int(mask.sum());report.append(dict(asset=aid,phase=phase,side=side,image=file,target_visible_pixels=pixels,scope='nominal planned pose static render; not synchronized physical rollout'))
        (O/'report.json').write_text(json.dumps(report,indent=2));print('WRIST_DONE',aid,flush=True)
    print('WRIST_COMPLETE',len(report),flush=True)
except BaseException:
    import traceback;traceback.print_exc()
finally:app.close()
